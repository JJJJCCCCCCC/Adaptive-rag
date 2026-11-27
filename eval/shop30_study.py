"""Run the three existing graph policies against the isolated fictional shop30 benchmark."""

import argparse
import asyncio
import hashlib
import json
import random
import tempfile
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from pydantic import SecretStr

from app.config.settings import get_settings
from app.providers.deepseek import DeepSeekProvider
from app.retrieval.self_rag import SelfRAGLoop
from app.services.container import build_container
from eval.live_sample import save
from eval.runner import load_jsonl
from eval.shop30_preflight import verify
from eval.strategy_study import ARMS, trial

DATA = Path("eval/dataset/shop30")
OUT = Path("eval/results/shop30-live")


def dataset(directory=DATA):
    questions = directory / "questions.jsonl"
    policies = directory / "corpus.json"
    fingerprint = hashlib.sha256(questions.read_bytes() + policies.read_bytes()).hexdigest()
    cases = load_jsonl(questions)
    corpus = {doc["id"]: doc for doc in json.loads(policies.read_text(encoding="utf8"))}
    assert len(cases) == len({c["id"] for c in cases})
    assert len(corpus) == len({doc["title"] for doc in corpus.values()})
    if directory == DATA:
        assert (len(cases), len(corpus)) == (30, 40)
    return cases, corpus, fingerprint


def prepare(folder, corpus):
    for doc in corpus.values():
        title = doc["title"]
        Path(folder, title + ".md").write_text(
            f"# {title}\n\n{title}。{doc['body']}", encoding="utf8"
        )
    return get_settings().model_copy(
        update={
            "knowledge_dir": folder,
            "database_url": "sqlite+aiosqlite:///:memory:",
            "redis_url": "",
            "qdrant_url": "",
            "retrieval_cache_ttl_seconds": 0,
            "embedding_api_key": SecretStr(""),
            "local_embedding_enabled": False,
            "cross_encoder_enabled": False,
            "unstructured_enabled": False,
            "mcp_enabled": False,
            "deepseek_trust_env": True,
            "retrieval_top_k_simple": 3,
            "retrieval_top_k_complex": 3,
            "rerank_top_n": 3,
            "max_retrieval_hops": 3,
            "rate_limit_per_minute": 10000,
        }
    )


async def evidence_trial(container, case, arm):
    original = SelfRAGLoop.retrieve_by_evidence
    low, high = {"single": (1, 1), "fixed_three": (3, 3), "adaptive": (1, 3)}[arm]

    async def bounded(loop, query, *, top_k):
        return await original(loop, query, top_k=top_k, min_rounds=low, max_rounds=high)

    with patch.object(SelfRAGLoop, "retrieve_by_evidence", bounded):
        return await trial(container, case, arm, True, diagnostics=True, shared_router=True)


async def run(shard, shards, *, directory=DATA, out=OUT, evidence_guided=False):
    cases, corpus, fingerprint = dataset(directory)
    if evidence_guided:
        digest = hashlib.sha256(fingerprint.encode())
        for path in sorted(
            [*Path("app").rglob("*.py"), Path(__file__), Path("eval/strategy_study.py")]
        ):
            digest.update(path.read_bytes())
        fingerprint = digest.hexdigest()
    out.mkdir(exist_ok=True, parents=True)
    with tempfile.TemporaryDirectory(prefix="fictional-shop30-") as folder:
        settings = prepare(folder, corpus)
        settings.evidence_guided_retrieval = evidence_guided
        confirmation = await verify(settings)
        save(out / f"preflight-{shard}.json", confirmation)
        container = await build_container(settings, seed=False)
        try:
            if not isinstance(container.llm, DeepSeekProvider):
                raise RuntimeError("DeepSeek API provider required")
            current = {}
            receipts = []

            async def receipt(response):
                await response.aread()
                payload = response.json() if response.is_success else {}
                row = {
                    **current,
                    "status": response.status_code,
                    "model": payload.get("model"),
                    "completion_id": payload.get("id"),
                    "usage": payload.get("usage"),
                }
                receipts.append(row)
                with (out / f"receipts-{shard}.jsonl").open("a", encoding="utf8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            container.llm._client.event_hooks["response"].append(receipt)
            for case in cases[shard::shards]:
                case["docs"] = [corpus[key]["title"] for key in case["docs"]]
                arms = ARMS.copy()
                random.Random(case["id"]).shuffle(arms)
                for arm in arms:
                    path = out / f"{case['id']}-{arm}.json"
                    if path.exists():
                        if (
                            json.loads(path.read_text(encoding="utf8"))["fingerprint"]
                            != fingerprint
                        ):
                            raise RuntimeError("Use fresh output after dataset change")
                        continue
                    current.update(id=case["id"], arm=arm, attempt=uuid4().hex)
                    receipts.clear()
                    if evidence_guided:
                        result = await evidence_trial(container, case, arm)
                    else:
                        result = await trial(container, case, arm, True, diagnostics=True)
                    successful = [r for r in receipts if r["status"] == 200]
                    if (
                        len(successful) != result["provider_responses"]
                        or any(not r["completion_id"] or not r["model"] for r in successful)
                        or sum(r["usage"]["total_tokens"] for r in successful)
                        != result["provider_tokens"]
                    ):
                        raise RuntimeError("Provider receipts and graph accounting do not agree")
                    result.update(
                        fingerprint=fingerprint,
                        model_requested=settings.deepseek_model,
                        backend=container.retriever._vector.embedder.backend,
                        attempt=current["attempt"],
                    )
                    save(path, result)
                    print(case["id"], arm, result["route"], result["rounds"], flush=True)
        finally:
            await container.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard", type=int)
    parser.add_argument("shards", type=int)
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards:
        parser.error("shard must be in [0, shards)")
    asyncio.run(run(args.shard, args.shards))
