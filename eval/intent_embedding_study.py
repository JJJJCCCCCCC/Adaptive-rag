"""Nearest-neighbour intent baseline using only labelled examples in intent_100 as its index."""

import asyncio
import json
import sys
from pathlib import Path

import numpy as np

from app.agents.nodes.intent_overlay import high_precision_intent
from app.config.settings import get_settings
from app.providers.embeddings import BgeM3EmbeddingProvider
from eval.runner import load_jsonl


async def run():
    train = load_jsonl(Path("eval/dataset/intent_100.jsonl"))
    novel = load_jsonl(Path("eval/dataset/intent_paraphrases.jsonl"))
    encoder = BgeM3EmbeddingProvider(get_settings())
    if not encoder.available:
        raise RuntimeError(encoder.load_error)
    queries = [r["query"] for r in train + novel]
    vectors = np.array(await encoder.embed(queries))
    records = []
    for i, row in enumerate(train + novel):
        eligible = [j for j, t in enumerate(train) if t["query"] != row["query"]]
        scores = vectors[eligible] @ vectors[i]
        closest = eligible[int(scores.argmax())]
        pred = train[closest]["intent"]
        overlay = high_precision_intent(row["query"])
        records.append(
            {
                **row,
                "index_neighbor": train[closest]["id"],
                "similarity": round(float(scores.max()), 4),
                "nearest": pred,
                "combined": overlay.value if overlay else pred,
            }
        )
    return {
        "encoder": encoder.backend,
        "index_n": len(train),
        "leave_identical_queries_out": records[: len(train)],
        "paraphrases": records[len(train) :],
    }


if __name__ == "__main__":
    result = asyncio.run(run())
    for key in ["leave_identical_queries_out", "paraphrases"]:
        rows = result[key]
        print(
            key,
            len(rows),
            {p: sum(r[p] == r["intent"] for r in rows) for p in ["nearest", "combined"]},
        )
    Path(sys.argv[1]).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
