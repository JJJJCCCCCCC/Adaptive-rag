"""Verify raw trial files and write a compact audit suitable for version control."""

import argparse
import json
from collections import Counter
from pathlib import Path

from study import ARMS, sha256, summarize, validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", choices=("dev", "holdout"))
    parser.add_argument("directory")
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    raw = (base / f"{args.cohort}.json").read_bytes()
    source = base / args.directory
    dataset = json.loads(raw)
    expected = json.loads((source / "summary.json").read_bytes())
    assert expected["input_sha256"] == sha256(raw)
    assert expected["experiment_sha256"] == sha256(raw + (base / "study.py").read_bytes())
    records = [json.loads((source / f"{row['id']}.json").read_bytes()) for row in dataset["cases"]]
    assert Counter(row["group"] for row in records) == expected["group_sizes"]
    case_by_id = {case["id"]: case for case in dataset["cases"]}
    receipts = {}
    generated_ids = set()
    for row in records:
        case = case_by_id[row["id"]]
        assert row["input_sha256"] == expected["input_sha256"]
        assert row["experiment_sha256"] == expected["experiment_sha256"]
        assert set(row["arms"]) == set(ARMS)
        for result in row["arms"].values():
            assert validate(case, result["answer"]) == result["passed"]
            if result["calls"]:
                final_id = result["calls"][-1]["id"]
                assert final_id not in generated_ids
                generated_ids.add(final_id)
            for receipt in result["calls"]:
                usage = receipt["usage"]
                assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
                assert receipt["id"] not in receipts or receipts[receipt["id"]] == receipt
                receipts[receipt["id"]] = receipt
    assert len(receipts) == expected["physical_receipts"]
    assert len(generated_ids) == (len(records) - expected["group_sizes"]["outside"]) * len(ARMS)
    expected2 = summarize(
        records, args.cohort, expected["input_sha256"], expected["experiment_sha256"]
    )
    assert expected2 == expected
    with (source / "audit.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(
        f"Validated {len(records)} paired cases and {len(receipts)} unique official usage receipts"
    )


if __name__ == "__main__":
    main()
