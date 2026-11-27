"""Re-score saved responses without calling models. Pair by question across runs."""

import json
import math
import statistics
from pathlib import Path

from eval.runner import load_jsonl
from eval.strategy_study import ARMS, spans_present

ROOT = Path("eval/results/nov26")


def load(path):
    return json.loads(path.read_text(encoding="utf8"))


def compare(left, right):
    assert left.keys() == right.keys()
    wins = sum(right[k] and not left[k] for k in left)
    losses = sum(left[k] and not right[k] for k in left)
    n = wins + losses
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2**n)
    return {"wins": wins, "losses": losses, "ties": len(left) - n, "exact_two_sided_p": round(p, 5)}


def summarize():
    cells = {}
    prediction = {}
    for path in sorted(ROOT.glob("*/strategy.json")):
        run = path.parent.name
        dataset = "strategy_holdout" if "holdout" in run else "strategy_policy"
        cases = {r["id"]: r for r in load_jsonl(Path("eval/dataset", dataset + ".jsonl"))}
        report = load(path)
        rows = report["records"]
        for r in rows:
            case = cases[r["id"]]
            # Require a listed reference source for evidence credit; answer credit is phrase-only.
            evidence = "\n".join(
                c["text"] for c in r["evidence"] if c["source"].rsplit("_", 1)[0] in case["docs"]
            )
            r["reference_evidence"] = spans_present(case["facts"], evidence)
            r["answer_coverage"] = spans_present(case["facts"], r["answer"])
        cells[run] = []
        for group in ["simple", "compound", "all"]:
            for arm in ARMS:
                subset = [r for r in rows if r["arm"] == arm and group in ("all", r["group"])]
                prediction[run, group, arm] = {r["id"]: r["answer_coverage"] for r in subset}
                cells[run].append(
                    {
                        "group": group,
                        "arm": arm,
                        "n": len(subset),
                        **{
                            k: sum(r[k] for r in subset)
                            for k in ["reference_evidence", "answer_coverage", "rounds", "searches"]
                        },
                        "median_ms": round(statistics.median(r["ms"] for r in subset), 2),
                        "mean_answer_chars": round(
                            statistics.mean(len(r["answer"]) for r in subset), 1
                        ),
                    }
                )
    pairs = {}
    for before, after in [("policy", "policy-after"), ("holdout-before", "holdout-after")]:
        pairs[after] = compare(
            prediction[before, "all", "adaptive"], prediction[after, "all", "adaptive"]
        )
    intent = load(ROOT / "intent-bge.json")["paraphrases"]
    intent_rules = load(ROOT / "intent.json")["intent_paraphrases"]["rows"]
    pairs["intent_paraphrases_BGE"] = compare(
        {r["id"]: r["offline_combined"] == r["intent"] for r in intent_rules},
        {r["id"]: r["combined"] == r["intent"] for r in intent},
    )
    return {
        "scoring": "NFKC, whitespace-insensitive reference phrase coverage; exact paired sign/binomial test, exploratory",
        "strategies": cells,
        "paired": pairs,
    }


if __name__ == "__main__":
    (ROOT / "summary.json").write_text(
        json.dumps(summarize(), ensure_ascii=False, indent=2), encoding="utf8"
    )
