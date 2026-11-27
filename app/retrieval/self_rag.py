"""Self-RAG style retrieve → grade → optionally hop again.

Reflection is implemented as structured LLM JSON, not special tokens,
so it stays schema-validated and testable.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from app.config.settings import Settings
from app.core.exceptions import LLMOutputError, LLMProviderError
from app.prompts.self_rag import GRADE_SYSTEM, GRADE_USER
from app.providers.base import LLMProvider
from app.providers.json_parser import parse_model
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.query_rewrite import QueryRewriter
from app.retrieval.reranker import DocumentReranker
from app.schemas.retrieval import RetrievedChunk


class GradeResult(BaseModel):
    relevant: bool
    sufficient: bool
    missing_aspect: str = ""
    rationale: str = ""


def followup_queries(
    missing: str, suggestions: list[str], evidence: list[RetrievedChunk], previous: set[str]
) -> list[str]:
    """Prefer identifiers present in both the missing fact and evidence, then unseen searches."""
    text = "\n".join(c.content for c in evidence).casefold()
    identifiers = re.findall(r"(?<![A-Za-z0-9])[A-Za-z]{1,8}[-_]?\d{1,8}(?![A-Za-z0-9])", missing)
    grounded = [
        code
        for code in identifiers
        if re.search(rf"(?<![a-z0-9]){re.escape(code.casefold())}(?![a-z0-9])", text)
    ]
    named = [
        name for name in re.findall(r"《([^《》\n]{2,40})》", text) if name in missing.casefold()
    ]
    candidates = [*grounded, *named, missing, *suggestions]
    return [q for q in dict.fromkeys(candidates) if q.strip() and q not in previous][:2]


class SelfRAGLoop:
    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: DocumentReranker,
        rewriter: QueryRewriter,
        llm: LLMProvider,
        settings: Settings,
    ) -> None:
        self._retriever = retriever
        self._reranker = reranker
        self._rewriter = rewriter
        self._llm = llm
        self._settings = settings

    async def retrieve_by_evidence(
        self, query: str, *, top_k: int, min_rounds: int = 1, max_rounds: int | None = None
    ) -> list[RetrievedChunk]:
        """Use the evidence available to generation to decide whether another lookup is needed."""
        limit = self._settings.max_retrieval_hops if max_rounds is None else max_rounds
        if not 1 <= min_rounds <= limit:
            raise ValueError("Expected 1 <= min_rounds <= max_rounds")
        budget = self._settings.rerank_top_n
        leaders: dict[str, RetrievedChunk] = {}
        accumulated: dict[str, RetrievedChunk] = {}
        previous: set[str] = set()
        selected: list[RetrievedChunk] = []
        missing = ""
        for number in range(1, limit + 1):
            rewritten = await self._rewriter.rewrite(query, missing_aspect=missing)
            queries = list(dict.fromkeys([query, *rewritten]))[:2]
            if number > 1:
                fresh = followup_queries(missing, rewritten, selected, previous)
                if not fresh and number > min_rounds:
                    break
                queries = fresh or queries
            previous.update(queries)
            hits: dict[str, RetrievedChunk] = {}
            for term in queries:
                for chunk in await self._retriever.retrieve(term, top_k=top_k):
                    hits.setdefault(chunk.chunk_id, chunk)
            focus = queries[0] if number > 1 else query
            ranked = await self._reranker.rerank(focus, list(hits.values()), top_n=budget)
            if ranked:
                lead = next((c for c in ranked if c.chunk_id not in leaders), ranked[0])
                leaders.setdefault(lead.chunk_id, lead)
            for chunk in ranked:
                accumulated.setdefault(chunk.chunk_id, chunk)
            # Reserve one place per round, then fill to the same final context cap in all arms.
            selected = list((leaders | accumulated).values())[:budget]
            if number == limit:
                break
            grade = await self._grade(query, selected)
            if grade.sufficient and number >= min_rounds:
                break
            missing = grade.missing_aspect.strip()
        return selected

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int,
    ) -> list[RetrievedChunk]:
        accumulated: dict[str, RetrievedChunk] = {}
        current_query = query
        missing = ""
        for _hop in range(self._settings.max_retrieval_hops):
            rewritten = await self._rewriter.rewrite(current_query, missing_aspect=missing)
            queries = list(dict.fromkeys([query, *rewritten]))[:2]
            candidates_by_id: dict[str, RetrievedChunk] = {}
            for search_query in queries:
                for candidate in await self._retriever.retrieve(search_query, top_k=top_k):
                    candidates_by_id.setdefault(candidate.chunk_id, candidate)
            reranked = await self._reranker.rerank(
                query, list(candidates_by_id.values()), top_n=self._settings.rerank_top_n
            )
            for chunk in reranked:
                accumulated[chunk.chunk_id] = chunk
            grade = await self._grade(query, list(accumulated.values()))
            if grade.sufficient:
                break
            missing = grade.missing_aspect or "more specific policy conditions"
            current_query = query
        return list(accumulated.values())[: self._settings.rerank_top_n * 2]

    async def _grade(self, query: str, chunks: list[RetrievedChunk]) -> GradeResult:
        evidence = "\n\n".join(f"[{c.chunk_id}] {c.content}" for c in chunks[:6]) or "(empty)"
        try:
            response = await self._llm.generate(
                [
                    {"role": "system", "content": GRADE_SYSTEM},
                    {"role": "user", "content": GRADE_USER.format(query=query, evidence=evidence)},
                ],
                temperature=0.0,
                max_tokens=256,
                response_format={"type": "json_object"},
            )
            return parse_model(response.content, GradeResult)
        except (LLMProviderError, LLMOutputError):
            return GradeResult(relevant=bool(chunks), sufficient=bool(chunks), rationale="fallback")
