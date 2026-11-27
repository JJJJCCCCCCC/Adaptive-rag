"""Replay the two historical M08 rewrites and first grade; no generation or new live scores."""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

from app.retrieval.self_rag import GradeResult, SelfRAGLoop
from app.services.container import build_retrieval
from eval.live_sample import save
from eval.runner import _unique_stems
from eval.shop30_study import dataset, prepare


async def main():
    path = Path("eval/results/shop30-live/M08-fixed_three.json")
    record = json.loads(path.read_text(encoding="utf8"))
    cases, corpus, _ = dataset()
    case = next(c for c in cases if c["id"] == "M08")
    old = record["trace"]["rounds"]
    with tempfile.TemporaryDirectory() as folder:
        settings = prepare(folder, corpus)
        retriever, reranker, _ = await build_retrieval(settings, None)
        searches = []
        original = retriever.retrieve

        async def logged(term, *, top_k):
            hits = await original(term, top_k=top_k)
            searches.append({"query": term, "hits": _unique_stems(hits)})
            return hits

        retriever.retrieve = logged
        rewriter = AsyncMock()
        rewriter.rewrite.side_effect = [r["suggestions"] for r in old[:2]]
        loop = SelfRAGLoop(retriever, reranker, rewriter, None, settings)
        loop._grade = AsyncMock(return_value=GradeResult.model_validate(old[0]["grade"]))
        evidence = await loop.retrieve_by_evidence(case["query"], top_k=3, max_rounds=2)
        titles = _unique_stems(evidence)
        report = {
            "case": case["id"],
            "source": str(path),
            "mode": "historical prefix replay; no new LLM calls",
            "original_live_final_docs": record["docs"],
            "original_live_answer": record["answer"],
            "new_searches": searches,
            "new_final_docs": titles,
            "new_all_docs": all(corpus[k]["title"] in titles for k in case["docs"]),
            "new_evidence": [c.content for c in evidence],
        }
        save(Path("eval/results/evidence-replay-M08.json"), report)
        print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
