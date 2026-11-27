"""Paired policy benchmark. Run from project root with python -m eval.strategy_study --out PATH."""

import argparse
import asyncio
import csv
import hashlib
import json
import random
import statistics
import time
import unicodedata
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from app.config.settings import get_settings
from app.core.enums import QueryComplexity, RetrievalStrategy
from app.core.exceptions import LLMProviderError
from app.core.request_scope import request_session
from app.providers.reranker import CrossEncoderRerankProvider
from app.retrieval.adaptive import AdaptiveRouter
from app.retrieval.query_rewrite import QueryRewriter
from app.retrieval.self_rag import SelfRAGLoop
from app.schemas.agent import ComplexityResult
from app.schemas.chat import ChatRequest
from app.services.container import build_container
from eval.runner import _unique_stems, load_jsonl

CASES = Path("eval/dataset/strategy_policy.jsonl")
ARMS = ["single", "fixed_three", "adaptive"]


def spans_present(facts, text):
    """Compare distinctive source phrases after Unicode and whitespace normalization."""

    def compact(value):
        return "".join(unicodedata.normalize("NFKC", value).split())

    target = compact(text)
    return all(compact(fact) in target for fact in facts)


async def trial(container, case, arm, live, *, diagnostics=False, shared_router=False):
    original_route = AdaptiveRouter.route
    original_grade = SelfRAGLoop._grade
    original_rewrite = QueryRewriter.rewrite
    counts = Counter()
    trace = {"rounds": []}

    async def route(router, query, intent):
        if diagnostics:
            trace["intent"] = intent.model_dump(mode="json")
        if arm == "adaptive" or shared_router:
            decision = await original_route(router, query, intent)
            if diagnostics:
                trace["decision"] = decision.model_dump(mode="json")
            return decision
        multi = arm == "fixed_three"
        return ComplexityResult(
            complexity=QueryComplexity.MULTI_HOP if multi else QueryComplexity.SINGLE_HOP,
            strategy=RetrievalStrategy.MULTI if multi else RetrievalStrategy.SINGLE,
            confidence=1,
            rationale="benchmark_assignment",
        )

    async def grade(loop, query, chunks):
        result = await original_grade(loop, query, chunks)
        if diagnostics:
            trace["rounds"][-1]["grade"] = result.model_dump(mode="json")
            trace["rounds"][-1]["accumulated"] = _unique_stems(chunks)
        if arm == "fixed_three" and not shared_router:
            result.sufficient = False
        return result

    async def rewrite(rewriter, query, *, missing_aspect=""):
        counts["rounds"] += 1
        suggestions = await original_rewrite(rewriter, query, missing_aspect=missing_aspect)
        if diagnostics:
            trace["rounds"].append(
                {
                    "input": query,
                    "missing": missing_aspect,
                    "suggestions": suggestions,
                    "searches": [],
                }
            )
        return suggestions

    query = case["query"]
    session = f"{case['id']}-{arm}"
    request = ChatRequest(message=query, user_id="u_demo", session_id=session)
    retrieve = container.retriever.retrieve
    generate = container.llm.generate

    async def count_generation(*args, **kwargs):
        try:
            response = await generate(*args, **kwargs)
        except LLMProviderError:
            counts["provider_errors"] += 1
            raise
        counts["provider_tokens"] += response.usage.total_tokens
        counts["provider_responses"] += 1
        return response

    async def count_search(query, *, top_k):
        counts["searches"] += 1
        hits = await retrieve(query, top_k=top_k)
        if diagnostics:
            trace["rounds"][-1]["searches"].append({"query": query, "hits": _unique_stems(hits)})
        return hits

    with (
        patch.object(AdaptiveRouter, "route", route),
        patch.object(SelfRAGLoop, "_grade", grade),
        patch.object(QueryRewriter, "rewrite", rewrite),
        patch.object(container.retriever, "retrieve", count_search),
        patch.object(container.llm, "generate", count_generation),
    ):
        async with request_session(container.session_factory) as db:
            state = await container.chat_service._initial_state(db, request, session, session)
            start = time.perf_counter()
            result = await container.chat_service._invoke(state)
            elapsed = 1000 * (time.perf_counter() - start)
    if live and (counts["provider_errors"] or not counts["provider_responses"]):
        raise RuntimeError("Incomplete live provider run; refusing to publish mixed scores")
    chunks = result.get("retrieved_chunks") or []
    evidence = "\n".join(c.content for c in chunks[:6])
    answer = result.get("final_answer") or ""
    return {
        "id": case["id"],
        "group": case["group"],
        "arm": arm,
        "route": result["strategy"].value if result.get("strategy") else "unassigned",
        **({"trace": trace, "refused": bool(result.get("refused"))} if diagnostics else {}),
        "rounds": (
            sum(bool(r["searches"]) for r in trace["rounds"])
            if shared_router and diagnostics
            else counts["rounds"]
        ),
        "searches": counts["searches"],
        "ms": round(elapsed, 2),
        "docs": _unique_stems(chunks[:6]),
        "provider_tokens": counts["provider_tokens"] if live else None,
        "provider_responses": counts["provider_responses"] if live else None,
        "all_docs": set(case["docs"]).issubset(_unique_stems(chunks[:6])),
        "all_evidence_spans": spans_present(case["facts"], evidence),
        "all_answer_spans": spans_present(case["facts"], answer),
        "answer": answer,
        "evidence": [{"source": c.document_id, "text": c.content} for c in chunks[:6]],
    }


async def study(args):
    settings = get_settings().model_copy(
        update={
            "database_url": "sqlite+aiosqlite:///:memory:",
            "redis_url": "",
            "qdrant_url": "",
            "retrieval_cache_ttl_seconds": 0,
            "rate_limit_per_minute": 10000,
        }
    )
    if not args.live:
        from pydantic import SecretStr

        settings.deepseek_api_key = SecretStr("")
    elif not settings.llm_configured:
        raise RuntimeError("Live study requires a configured provider")

    if args.equal_k:
        settings.retrieval_top_k_simple = settings.retrieval_top_k_complex
    container = await build_container(settings, seed=True)
    rows = []
    rng = random.Random(2611)
    try:
        for case in load_jsonl(args.cases):
            arms = ARMS.copy()
            rng.shuffle(arms)
            for arm in arms:
                rows.append(await trial(container, case, arm, args.live))
    finally:
        await container.aclose()
    summary = []
    for group in ["simple", "compound", "all"]:
        for arm in ARMS:
            records = [r for r in rows if r["arm"] == arm and group in ("all", r["group"])]
            summary.append(
                {
                    "group": group,
                    "arm": arm,
                    "n": len(records),
                    **{
                        k: sum(r[k] for r in records)
                        for k in [
                            "all_docs",
                            "all_evidence_spans",
                            "all_answer_spans",
                            "rounds",
                            "searches",
                        ]
                    },
                    "median_ms": round(statistics.median(r["ms"] for r in records), 2),
                    "routes": dict(Counter(r["route"] for r in records)),
                }
            )
    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "llm": type(container.llm).__name__,
        "dense": container.retriever._vector.embedder.backend,
        "reranker": CrossEncoderRerankProvider(settings).backend,
        "simple_k": settings.retrieval_top_k_simple,
        "multi_k": settings.retrieval_top_k_complex,
        "case_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        "summary": summary,
        "records": rows,
    }
    (args.out / "strategy.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8"
    )
    rng.shuffle(rows)
    cases = {r["id"]: r for r in load_jsonl(args.cases)}
    key = {}
    with (args.out / "blind-review.csv").open("w", newline="", encoding="utf-8-sig") as out:
        writer = csv.writer(out)
        writer.writerow(
            [
                "编号",
                "问题",
                "候选回答",
                "参考政策全文",
                "品类适用",
                "必要条件完整",
                "无额外错误",
                "评审备注",
            ]
        )
        for i, r in enumerate(rows, 1):
            case = cases[r["id"]]
            sources = "\n".join(
                Path("knowledge", d + ".md").read_text(encoding="utf8") for d in case["docs"]
            )
            writer.writerow([i, case["query"], r["answer"], sources, "", "", "", ""])
            key[i] = [r["id"], r["arm"]]
    (args.out / "review-key.json").write_text(json.dumps(key, indent=2), encoding="utf8")
    print(json.dumps(summary, ensure_ascii=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=CASES)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--equal-k", action="store_true")
    asyncio.run(study(parser.parse_args()))
