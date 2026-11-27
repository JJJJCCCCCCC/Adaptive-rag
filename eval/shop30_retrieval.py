"""Measure evidence coverage and an identifier-following second lookup without an LLM."""

import asyncio
import json
import re
import tempfile
from collections import Counter
from pathlib import Path

from app.services.container import build_retrieval
from eval.runner import _doc_stem, _unique_stems
from eval.shop30_study import dataset, prepare

OUTPUT = Path("eval/results/shop30-retrieval.json")
CODE = re.compile(r"\b[A-Z]\d{2}\b")


async def follow_profile(query, first, retriever):
    for profile in first:
        title = _doc_stem(profile.document_id)
        if not title.endswith("服务档案") or title.removesuffix("服务档案") not in query:
            continue
        codes = set(CODE.findall(profile.content))
        if len(codes) != 1:
            continue
        code = codes.pop()
        hits = await retriever.retrieve(code, top_k=3)
        for hit in hits:
            if hit.document_id != profile.document_id and code in CODE.findall(hit.content):
                unique = {c.document_id: c for c in [profile, hit, *first]}
                return list(unique.values())[:3], code, _unique_stems(hits)
        return first, code, _unique_stems(hits)
    return first, None, []


async def main():
    cases, corpus, fingerprint = dataset()
    with tempfile.TemporaryDirectory(prefix="fictional-shop30-") as folder:
        settings = prepare(folder, corpus)
        retriever, reranker, _ = await build_retrieval(settings, None)
        rows = []
        for case in cases:
            query = case["query"]
            raw = await retriever.retrieve(query, top_k=3)
            first = await reranker.rerank(query, raw, top_n=3)
            final, code, hits = await follow_profile(query, first, retriever)
            required = {corpus[key]["title"] for key in case["docs"]}
            initial_titles = _unique_stems(first)
            final_titles = _unique_stems(final)
            rows.append(
                {
                    "id": case["id"],
                    "group": case["group"],
                    "query": query,
                    "gold": sorted(required),
                    "initial": initial_titles,
                    "initial_complete": required.issubset(initial_titles),
                    "lookup": code,
                    "lookup_candidates": hits,
                    "final": final_titles,
                    "final_complete": required.issubset(final_titles),
                }
            )
        summary = {}
        for group in ("simple", "complex", "multihop", "all"):
            subset = [r for r in rows if group == "all" or r["group"] == group]
            summary[group] = {
                "n": len(subset),
                "first_all": sum(r["initial_complete"] for r in subset),
                "second_lookup_triggered": sum(bool(r["lookup"]) for r in subset),
                "after_all": sum(r["final_complete"] for r in subset),
                "paired": dict(
                    Counter(
                        f"{int(r['initial_complete'])}->{int(r['final_complete'])}" for r in subset
                    )
                ),
            }
        report = {
            "dataset_sha256": fingerprint,
            "dense": retriever._vector.embedder.backend,
            "reranker": reranker._provider.backend,
            "top_k": 3,
            "summary": summary,
            "records": rows,
        }
        OUTPUT.parent.mkdir(exist_ok=True, parents=True)
        OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")
        print(json.dumps(summary))


if __name__ == "__main__":
    asyncio.run(main())
