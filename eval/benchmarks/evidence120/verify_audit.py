"""Recompute the Evidence120 tables from published records without API calls."""

import argparse
import json
from pathlib import Path

from study import ARMS, sha256, summarize, validate


def verify(cohort: str, base: Path) -> None:
    raw = (base / f"{cohort}.json").read_bytes()
    cases = {case["id"]: case for case in json.loads(raw)["cases"]}
    expected = json.loads((base / f"{cohort}-summary.json").read_bytes())
    records = [
        json.loads(line)
        for line in (base / f"{cohort}-audit.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    input_digest = sha256(raw)
    run_digest = sha256(raw + (base / "study.py").read_bytes())
    assert expected["input_sha256"] == input_digest
    assert expected["experiment_sha256"] == run_digest
    assert len(records) == len(cases) == len({row["id"] for row in records})
    assert {row["id"] for row in records} == set(cases)
    receipts = {}
    answer_ids = set()
    for row in records:
        case = cases[row["id"]]
        assert row["input_sha256"] == input_digest
        assert row["experiment_sha256"] == run_digest
        assert set(row["arms"]) == set(ARMS)
        for result in row["arms"].values():
            assert validate(case, result["answer"]) == result["passed"]
            if result["calls"]:
                answer_id = result["calls"][-1]["id"]
                assert answer_id not in answer_ids
                answer_ids.add(answer_id)
            for receipt in result["calls"]:
                usage = receipt["usage"]
                assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
                assert receipt["id"] and receipts.setdefault(receipt["id"], receipt) == receipt
    assert len(receipts) == expected["physical_receipts"]
    assert summarize(records, cohort, input_digest, run_digest) == expected
    print(f"{cohort}: {len(records)} paired cases and {len(receipts)} unique model receipts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", choices=("dev", "holdout"))
    args = parser.parse_args()
    verify(args.cohort, Path(__file__).resolve().parent)
