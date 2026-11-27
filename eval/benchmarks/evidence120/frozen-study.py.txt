"""Paired component-level benchmark with real DeepSeek usage receipts."""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import random
import re
import statistics
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import httpx
from app.providers.embeddings import HashEmbeddingProvider
from app.providers.reranker import LexicalRerankProvider
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import DocumentReranker
from app.retrieval.vector import InMemoryVectorStore, VectorRetriever, build_records
from app.schemas.retrieval import RetrievedChunk
from probe import Credentials

ARMS = ("single", "fixed_three", "adaptive")
TOP_K = 8
PREVIEW = 3
CONTROL_N = 10000
MODEL_PROMPT = """你是商店政策问答组件。只使用本次提供的资料，重视当前政策。
若提问明显非电商政策，decision 用 outside；有相关请求但资料缺失或不明确，decision 用 missing。
若证据齐全，decision 用 answer，并给出一个数值 value（JSON 整数）、准确的中文单位 unit 和
从商品索引一路到数值条款的全部且仅相关的文档 ID 列表 sources。必须把连接记录、不是只把
最后一份金额条款列为来源。作废或限于样品的资料不能作为当前政策证据。
输出 JSON 对象 {"decision":"answer","value":2,"unit":"元","sources":["P00000000"]}。
拒答时 value 和 unit 均为 null，sources 为 []。"""
REFLECT = """你是检索查询改写器。结合用户问题与现有文档，给出下一轮一个最有用的查询。
优先使用当前有效条款中的引用、名称、编号或别名。资料看似足够时，补查可能影响结论的现行约定。
仅输出 JSON 对象 {"query":"下一个检索词"}，不要回答用户问题。"""
DECLARATION_CODE = re.compile(r"编号\s*([A-Z]{2}-\d{4,7})")
LINK_CODE = re.compile(r"(?<![A-Za-z0-9])([A-Z]{2}-\d{4,7})(?![A-Za-z0-9])")
QUOTED = re.compile(r"《([^》]+)》|「([^」]+)」")
NATURAL = re.compile(r"(?:叫做|名为)([^。；，]+?)的那(?:张|份)")
DONE = re.compile(r"现行政策|仍未发布")


def dump(value):
    return json.dumps(value, ensure_ascii=False, indent=2)


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def body(chunk):
    return chunk.content.split("。", 2)[-1]


def references(chunk):
    fragment = body(chunk)
    quoted = [next(item for item in pair if item) for pair in QUOTED.findall(fragment)]
    return [*dict.fromkeys([*LINK_CODE.findall(fragment), *quoted, *NATURAL.findall(fragment)])]


def declared_keys(chunk):
    prefix = chunk.content.split("。", 2)
    header = prefix[1] if len(prefix) >= 3 else ""
    return [
        chunk.title,
        *DECLARATION_CODE.findall(header),
        *[next(item for item in pair if item) for pair in QUOTED.findall(header)],
    ]


def eligible(chunk):
    return not any(word in chunk.content for word in ("作废", "仅用于样品"))


def select(anchor, chain, pool):
    preferred = [anchor, *chain, *pool]
    chosen = {}
    for chunk in preferred:
        if eligible(chunk):
            chosen.setdefault(chunk.document_id, chunk)
    return list(chosen.values())[:PREVIEW]


class Corpus:
    def __init__(self, documents):
        self.documents = documents
        self.merchants = {}
        self.lexical = DocumentReranker(LexicalRerankProvider())

    async def initialize(self):
        embedder = HashEmbeddingProvider(128)
        per_store = defaultdict(list)
        for document in self.documents:
            per_store[document["store"]].append(
                RetrievedChunk(
                    chunk_id=document["id"], document_id=document["id"],
                    title=document["title"], content=document["text"], score=0,
                )
            )
        for store, chunks in per_store.items():
            memory = InMemoryVectorStore()
            memory.upsert(build_records(chunks, await embedder.embed([c.content for c in chunks])))
            engine = HybridRetriever(BM25Retriever(chunks), VectorRetriever(memory, embedder), rrf_k=60)
            aliases = defaultdict(list)
            for chunk in chunks:
                if eligible(chunk):
                    for key in declared_keys(chunk):
                        aliases[key].append(chunk)
            self.merchants[store] = (engine, aliases)

    async def retrieve(self, store, query):
        engine, aliases = self.merchants[store]
        results = await engine.retrieve(query, top_k=TOP_K)
        ranked = await self.lexical.rerank(query, results, top_n=TOP_K)
        exact = aliases.get(query, [])
        return [*dict(
            (chunk.document_id, chunk)
            for chunk in [*ranked, *exact]
        ).values()] if not exact else [
            *exact, *[c for c in ranked if c.document_id not in {e.document_id for e in exact}]
        ]


def outside_query(store, query):
    return store not in query


def evidence(chunks):
    return [{"id": c.document_id, "text": c.content} for c in chunks]


def validate(case, answer):
    if not isinstance(answer, dict) or set(answer) != {"decision", "value", "unit", "sources"}:
        return False
    if answer["decision"] != case["decision"]:
        return False
    if case["decision"] != "answer":
        return answer["value"] is None and answer["unit"] is None and answer["sources"] == []
    return (
        type(answer["value"]) is int and answer["value"] == case["value"]
        and answer["unit"] == case["unit"] and isinstance(answer["sources"], list)
        and all(isinstance(k, str) for k in answer["sources"])
        and sorted(answer["sources"]) == sorted(case["sources"])
    )


class Service:
    def __init__(self, client, credentials):
        self.client = client
        self.credentials = credentials

    async def ask(self, system, payload, max_tokens):
        request = {
            "model": self.credentials.deepseek_model,
            "temperature": 0,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        }
        for attempt in range(4):
            try:
                response = await self.client.post(
                    "/chat/completions",
                    headers={"Authorization": "Bearer " + self.credentials.deepseek_api_key},
                    json=request,
                )
                if response.status_code not in (429, 500, 502, 503, 504):
                    response.raise_for_status()
                    break
                if attempt == 3:
                    response.raise_for_status()
            except httpx.RequestError:
                if attempt == 3:
                    raise
            await asyncio.sleep(1.5 * 2**attempt)
        body = response.json()
        raw = body["choices"][0]["message"]["content"]
        receipt = {key: body[key] for key in ("id", "model", "usage")}
        if not receipt["id"] or receipt["usage"]["total_tokens"] <= 0:
            raise ValueError("invalid service receipt")
        receipt["finish_reason"] = body["choices"][0]["finish_reason"]
        receipt["response"] = raw
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            parsed = {}
        return parsed, receipt


async def measure(case, corpus, service, input_hash):
    query, store = case["query"], case["store"]
    start = time.perf_counter()
    if outside_query(store, query):
        return {
            "id": case["id"], "store": store, "group": case["group"], "input_sha256": input_hash,
            "trajectory": [], "arms": {
                arm: {"answer": dict(decision="outside", value=None, unit=None, sources=[]),
                      "calls": [], "evidence": [], "rounds": 0, "ms": 0.0,
                      "passed": case["decision"] == "outside"} for arm in ARMS
            },
        }
    pool = await corpus.retrieve(store, query)
    anchor = next((c for c in pool if eligible(c)), pool[0])
    chain = []
    seen = {anchor.document_id}
    trail = anchor
    calls, provenance, snapshots = [], [], []
    stop_at = None
    for iteration in range(3):
        selected = select(anchor, chain, pool)
        snapshots.append((selected, list(calls), (time.perf_counter() - start)*1000))
        keys = references(trail)
        keys = [k for k in keys if k not in {step["query"] for step in provenance}]
        if stop_at is None and (DONE.search(body(trail)) or not keys):
            stop_at = iteration
        if iteration == 2:
            break
        # Same search policy on the shared prefix. A live rewrite is used where a
        # direct link is absent, ambiguous, or was exhausted by the forced baseline.
        planned = keys[0] if keys else ""
        if not planned:
            parsed, receipt = await service.ask(
                REFLECT, {"question": query, "evidence": evidence(selected)}, 90
            )
            calls.append(receipt)
            planned = parsed.get("query", "") if isinstance(parsed, dict) else ""
            planned = planned if isinstance(planned, str) and planned.strip() else query
        candidates = await corpus.retrieve(store, planned)
        exact = [c for c in candidates if planned in declared_keys(c) and c.document_id not in seen]
        if len(exact) == 1 and eligible(exact[0]):
            trail = exact[0]
            chain.append(trail)
            seen.add(trail.document_id)
        pool = [*candidates, *pool]
        provenance.append({"query": planned, "picked": trail.document_id,
                           "hits": [c.document_id for c in candidates]})
    targets = {"single": 0, "fixed_three": 2, "adaptive": 2 if stop_at is None else stop_at}
    arms = {}
    # Each arm submits its own answer request, even if evidence happens to match.
    order = list(ARMS)
    random.Random("arm-order-" + case["id"]).shuffle(order)
    for arm in order:
        index = targets[arm]
        chunks, prefix_calls, retrieval_ms = snapshots[index]
        answer_start = time.perf_counter()
        answer, receipt = await service.ask(
            MODEL_PROMPT, {"question": query, "evidence": evidence(chunks)}, 192
        )
        arms[arm] = {
            "answer": answer,
            "calls": [*prefix_calls, receipt],
            "evidence": [c.document_id for c in chunks],
            "rounds": index + 1,
            "ms": round(retrieval_ms + (time.perf_counter() - answer_start) * 1000, 2),
            "passed": validate(case, answer),
        }
    return {"id": case["id"], "store": store, "group": case["group"],
            "input_sha256": input_hash, "trajectory": provenance, "arms": arms}


def pass_rate(records, arm):
    return sum(record["arms"][arm]["passed"] for record in records) / len(records)


def paired_interval(records, baseline):
    per_shop = defaultdict(list)
    for record in records:
        if record["group"] in {"direct", "two", "three"}:
            per_shop[record["store"]].append(
                int(record["arms"]["adaptive"]["passed"]) - int(record["arms"][baseline]["passed"])
            )
    samples = list(per_shop.values())
    rng = random.Random(1159)
    bootstrap = []
    for _ in range(CONTROL_N):
        drawn = [values for group in rng.choices(samples, k=len(samples)) for values in group]
        bootstrap.append(sum(drawn) / len(drawn))
    bootstrap.sort()
    return [bootstrap[249], bootstrap[9749]]


def summarize(records, cohort, input_hash, fingerprint):
    answerable = [r for r in records if r["group"] in {"direct", "two", "three"}]
    controls = [r for r in records if r not in answerable]
    totals = {}
    physical = {}
    for arm in ARMS:
        rows = [record["arms"][arm] for record in records]
        receipts = [receipt for row in rows for receipt in row["calls"]]
        for receipt in receipts:
            physical[receipt["id"]] = receipt
        total = sum(r["usage"]["total_tokens"] for r in receipts)
        totals[arm] = {
            "passed": sum(row["passed"] for row in rows),
            "answerable_passed": sum(r["arms"][arm]["passed"] for r in answerable),
            "controls_passed": sum(r["arms"][arm]["passed"] for r in controls),
            "passed_by_group": dict(Counter(r["group"] for r in records if r["arms"][arm]["passed"])),
            "rounds": sum(row["rounds"] for row in rows),
            "calls_accounted": len(receipts),
            "prompt_tokens": sum(r["usage"]["prompt_tokens"] for r in receipts),
            "completion_tokens": sum(r["usage"]["completion_tokens"] for r in receipts),
            "total_tokens": total,
            "p50_ms": statistics.median(row["ms"] for row in rows),
        }
    advantage = pass_rate(answerable, "adaptive") - pass_rate(answerable, "single")
    versus_three = pass_rate(answerable, "adaptive") - pass_rate(answerable, "fixed_three")
    savings = 1 - totals["adaptive"]["total_tokens"] / totals["fixed_three"]["total_tokens"]
    ci_single = paired_interval(records, "single")
    ci_three = paired_interval(records, "fixed_three")
    return {
        "cohort": cohort, "input_sha256": input_hash, "experiment_sha256": fingerprint,
        "n": len(records), "merchants": len({r["store"] for r in records}),
        "answerable_n": len(answerable), "controls_n": len(controls),
        "model": sorted({r["model"] for r in physical.values()}),
        "physical_receipts": len(physical),
        "actual_billed_tokens": sum(r["usage"]["total_tokens"] for r in physical.values()),
        "group_sizes": dict(Counter(r["group"] for r in records)),
        "totals": totals, "quality_gain_vs_single": advantage,
        "quality_gain_vs_single_ci95": ci_single,
        "quality_difference_vs_fixed_three": versus_three,
        "quality_difference_ci95": ci_three, "savings_vs_fixed_three": savings,
        "quality_goal": advantage >= 0.10 and ci_single[0] > 0,
        "cost_goal": savings >= 0.20 and ci_three[0] > -0.05,
        "stretch_cost_goal": savings >= 0.276 and ci_three[0] > -0.05,
    }


async def run(args):
    directory = Path(__file__).parent
    dataset = directory / f"{args.cohort}.json"
    raw = dataset.read_bytes()
    fingerprint = sha256(raw + Path(__file__).read_bytes())
    data = json.loads(raw)
    digest = sha256(raw)
    assert data["cohort"] == args.cohort
    out = directory / args.output
    out.mkdir(exist_ok=True)
    corpus = Corpus(data["documents"])
    await corpus.initialize()
    credentials = Credentials(_env_file=Path(os.environ["RAG_EVAL_MODEL_CONFIG"]))
    queue = asyncio.Semaphore(args.parallel)
    async with httpx.AsyncClient(
        base_url="https://api.deepseek.com",
        trust_env=os.environ.get("RAG_EVAL_DIRECT") != "1",
        limits=httpx.Limits(max_connections=args.parallel),
        timeout=httpx.Timeout(55, connect=12),
        follow_redirects=False,
    ) as client:
        service = Service(client, credentials)

        async def one(case):
            path = out / (case["id"] + ".json")
            if path.is_file():
                saved = json.loads(path.read_text(encoding="utf-8"))
                if saved["experiment_sha256"] != fingerprint:
                    raise ValueError("experiment changed; run in separate output directory")
                return saved
            async with queue:
                record = await measure(case, corpus, service, digest)
            record["experiment_sha256"] = fingerprint
            path.write_text(dump(record) + "\n", encoding="utf-8")
            print(case["id"], [record["arms"][arm]["passed"] for arm in ARMS], flush=True)
            return record

        records = await asyncio.gather(*[one(case) for case in data["cases"]])
    summary = summarize(records, args.cohort, digest, fingerprint)
    (out / "summary.json").write_text(dump(summary) + "\n", encoding="utf-8")
    print(dump(summary), flush=True)


if __name__ == "__main__":
    logging.getLogger("jieba").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", choices=("dev", "holdout"))
    parser.add_argument("output")
    parser.add_argument("--parallel", type=int, default=12)
    asyncio.run(run(parser.parse_args()))
