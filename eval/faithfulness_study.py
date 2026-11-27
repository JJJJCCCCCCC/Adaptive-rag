"""Evaluate the project's keyword metric on polarity/condition counterexamples and optionally its real judge."""

import argparse
import asyncio
import json
from pathlib import Path

from app.config.settings import get_settings
from app.guardrails.hallucination import FaithfulnessChecker
from app.providers.deepseek import DeepSeekProvider
from app.schemas.retrieval import RetrievedChunk
from eval.runner import _keyword_hit, load_jsonl


async def run(live):
    provider = DeepSeekProvider(get_settings()) if live else None
    checker = FaithfulnessChecker(provider) if provider else None
    records = []
    try:
        for case in load_jsonl(Path("eval/dataset/faithfulness_pairs.jsonl")):
            source = Path("knowledge", case["doc"] + ".md").read_text(encoding="utf8")
            chunk = RetrievedChunk(
                chunk_id=case["id"],
                document_id=case["doc"],
                content=source,
                title=case["doc"],
                score=1,
            )
            for label in ["supported", "unsupported"]:
                answer = case[label]
                row = {
                    "id": case["id"],
                    "expected": label == "supported",
                    "keyword_pass": _keyword_hit(answer, case["facts"]),
                }
                if checker:
                    verdict = await checker.check(
                        query=case["query"], answer=answer, chunks=[chunk], tool_results_text=""
                    )
                    if verdict.rationale == "checker_unavailable":
                        raise RuntimeError("Judge unavailable; live run incomplete")
                    row.update(judge=verdict.supported, rationale=verdict.rationale)
                records.append(row)
    finally:
        if provider:
            await provider.aclose()
    keys = ["keyword_pass"] + (["judge"] if checker else [])
    summary = {
        k: {
            "tp": sum(r[k] and r["expected"] for r in records),
            "fp": sum(r[k] and not r["expected"] for r in records),
            "tn": sum(not r[k] and not r["expected"] for r in records),
            "fn": sum(not r[k] and r["expected"] for r in records),
        }
        for k in keys
    }
    return {"summary": summary, "rows": records}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(
        json.dumps(asyncio.run(run(args.live)), ensure_ascii=False, indent=2), encoding="utf8"
    )
