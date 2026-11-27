"""Run ranked-document evaluation and save the effective model configuration."""

import argparse
import asyncio
import json
from pathlib import Path

from eval.runner import eval_retrieval

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = asyncio.run(eval_retrieval())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")
    for row in report["systems"]:
        print(
            row["backend"],
            row["precision_at_1"]["mean"],
            row["recall_at_3"]["mean"],
            row["mrr"]["mean"],
        )
