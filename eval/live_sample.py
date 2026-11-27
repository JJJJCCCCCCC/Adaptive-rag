"""Execute a bounded online evaluation and keep server-side model and usage receipts."""

import argparse
import asyncio
import json
import os
import random
import tempfile
from pathlib import Path
from unittest.mock import patch

from pydantic import SecretStr

from app.config.settings import get_settings
from app.providers.deepseek import DeepSeekProvider
from eval import intent_study
from eval.live_synthetic_policy import CASES

BASE = Path("eval/results/nov26/deepseek-live")


def save(path, value):
    payload = json.dumps(value, ensure_ascii=False, indent=2)
    temp = path.with_suffix(".tmp")
    temp.write_text(payload, encoding="utf8")
    temp.replace(path)


async def policy(args, out):
    from app.services.container import build_container
    from eval.strategy_study import ARMS, trial

    with tempfile.TemporaryDirectory(prefix="synthetic-shop-") as folder:
        cases = []
        for ident, policy_text, question, facts, _, _ in CASES:
            Path(folder, ident + ".md").write_text(
                "# " + policy_text.split("的")[0] + "\n\n" + policy_text, encoding="utf8"
            )
            cases.append(
                {
                    "id": ident,
                    "query": question,
                    "group": "single_document",
                    "facts": facts,
                    "docs": [ident],
                }
            )
        cases.append(
            {
                "id": "s7",
                "group": "two_documents",
                "query": "蓝鸥玩具城的银票第几天失效，同时松针集市的雾灯箱押金要多少枚贝币？",
                "facts": ["17", "23"],
                "docs": ["s1", "s3"],
            }
        )
        settings = get_settings().model_copy(
            update={
                "knowledge_dir": folder,
                "database_url": "sqlite+aiosqlite:///:memory:",
                "redis_url": "",
                "qdrant_url": "",
                "retrieval_cache_ttl_seconds": 0,
                "rate_limit_per_minute": 10000,
                "embedding_api_key": SecretStr(""),
                "local_embedding_enabled": False,
                "cross_encoder_enabled": False,
                "unstructured_enabled": False,
                "mcp_enabled": False,
                "retrieval_top_k_simple": 5,
                "retrieval_top_k_complex": 5,
            }
        )
        container = await build_container(settings, seed=False)
        try:
            backend = container.retriever._vector.embedder.backend
            subset = [c for c in cases if c["id"] in args.ids]
            if len(subset) != len(set(args.ids)):
                raise ValueError("Unknown synthetic id")
            report = {
                "dense": backend,
                "model_requested": settings.deepseek_model,
                "case_ids": args.ids,
                "simple_k": settings.retrieval_top_k_simple,
                "multi_k": settings.retrieval_top_k_complex,
                "records": [],
            }
            for case in subset:
                arms = ARMS.copy()
                random.Random(case["id"]).shuffle(arms)
                for arm in arms:
                    result = await trial(container, case, arm, True)
                    report["records"].append(result)
                    save(out, report)
                    print(
                        case["id"],
                        arm,
                        result["provider_responses"],
                        result["provider_tokens"],
                        flush=True,
                    )
        finally:
            await container.aclose()


async def main(args):
    BASE.mkdir(parents=True, exist_ok=True)
    name = args.task if not args.ids else args.task + "-" + "-".join(args.ids)
    out = BASE / (name + ".json")
    audit = BASE / (name + "-usage.jsonl")
    if out.exists() or audit.exists():
        raise FileExistsError(f"Use a fresh output directory for a new run: {BASE}")
    init = DeepSeekProvider.__init__

    def instrument(instance, *params, **kwargs):
        init(instance, *params, **kwargs)

        async def receipt(response):
            data = {}
            if response.is_success:
                await response.aread()
                data = response.json()
            row = {
                "http_status": response.status_code,
                "model_returned": data.get("model"),
                "completion_id": data.get("id"),
                "usage": data.get("usage"),
            }
            with audit.open("a", encoding="utf8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        instance._client.event_hooks["response"].append(receipt)

    with patch.object(DeepSeekProvider, "__init__", instrument):
        if args.task == "intent":
            await intent_study.study(
                argparse.Namespace(live=True, datasets=["intent_paraphrases"], out=out)
            )
        else:
            await policy(args, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=["intent", "synthetic-graph"])
    parser.add_argument("--ids", nargs="+", default=[])
    parser.add_argument("--out-dir", type=Path, default=BASE)
    arguments = parser.parse_args()
    BASE = arguments.out_dir
    os.environ["DEEPSEEK_TRUST_ENV"] = "true"
    asyncio.run(main(arguments))
