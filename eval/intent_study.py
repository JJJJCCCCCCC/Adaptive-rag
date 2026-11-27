"""Score rules and paired v3 raw / corrected predictions. Provider failures count as errors."""

import argparse
import asyncio
import json
from pathlib import Path

from app.agents.nodes.intent import rule_based_intent
from app.agents.nodes.intent_overlay import apply_intent_overlay, high_precision_intent
from app.config.settings import get_settings
from app.core.constants import PROMPT_VERSION_INTENT
from app.core.exceptions import LLMOutputError, LLMProviderError
from app.prompts.intent import INTENT_SYSTEM, INTENT_USER
from app.providers.deepseek import DeepSeekProvider
from app.providers.json_parser import parse_model
from app.schemas.agent import IntentResult
from eval.runner import load_jsonl


async def study(args):
    provider = DeepSeekProvider(get_settings()) if args.live else None
    report = {}
    try:
        for dataset in args.datasets:
            rows = []
            for case in load_jsonl(Path("eval/dataset", dataset + ".jsonl")):
                query = case["query"]
                raw_rule = rule_based_intent(query).label
                overlay = high_precision_intent(query)
                row = {
                    **case,
                    "rules": raw_rule.value,
                    "overlay_only": overlay.value if overlay else None,
                    "offline_combined": apply_intent_overlay(query, raw_rule)[0].value,
                }
                if provider:
                    try:
                        response = await provider.generate(
                            [
                                {"role": "system", "content": INTENT_SYSTEM},
                                {
                                    "role": "user",
                                    "content": INTENT_USER.format(
                                        query=query, prompt_version=PROMPT_VERSION_INTENT
                                    ),
                                },
                            ],
                            temperature=0,
                            max_tokens=256,
                            response_format={"type": "json_object"},
                        )
                        label = parse_model(response.content, IntentResult).label
                        row.update(
                            v3_raw=label.value,
                            v3_overlay=apply_intent_overlay(query, label)[0].value,
                        )
                    except (LLMOutputError, LLMProviderError) as exc:
                        if not rows:
                            raise RuntimeError(
                                "First live request failed; aborting without a model score"
                            ) from exc
                        row.update(v3_raw=None, v3_overlay=None, error=type(exc).__name__)
                rows.append(row)
            keys = ["rules", "overlay_only", "offline_combined"] + (
                ["v3_raw", "v3_overlay"] if provider else []
            )
            report[dataset] = {
                "n": len(rows),
                "correct": {k: sum(r[k] == r["intent"] for r in rows) for k in keys},
                "errors": sum("error" in r for r in rows),
                "rows": rows,
            }
    finally:
        if provider:
            await provider.aclose()
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")
    for dataset, r in report.items():
        print(dataset, r["n"], r["correct"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["intent_100", "intent_paraphrases"],
        default=["intent_100", "intent_paraphrases"],
    )
    parser.add_argument("--out", type=Path, required=True)
    asyncio.run(study(parser.parse_args()))
