"""Separate component regression on the historical, already inspected Evidence36 data."""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
from collections import Counter
from pathlib import Path

import httpx
import study
from probe import Credentials

LEGACY_CODE = re.compile(r"(?<![A-Za-z0-9])[A-Z]{2}-?\d{2,4}(?![A-Za-z0-9])")
NUMERIC = re.compile(r"^(\d+)([^\d].*)$")
DEDICATED_UNKNOWN = re.compile(r"(?:待发布|未承诺|未公布)")


def load_original(source):
    directory = source / "eval" / "dataset" / "evidence36"
    query_data = (directory / "questions.jsonl").read_bytes()
    corpus_data = (directory / "corpus.json").read_bytes()
    queries = [json.loads(row) for row in query_data.decode().splitlines()]
    docs = json.loads(corpus_data)
    cases = []
    for row in queries:
        n, unit = (None, None)
        if row["kind"] == "answer":
            match = NUMERIC.fullmatch(row["gold"])
            if match is None:
                raise ValueError("Reference does not match the rubric schema")
            n, unit = int(match[1]), match[2]
        kind = {"answer": "answer", "unknown": "missing", "outside": "outside"}[row["kind"]]
        group = {
            "direct": "direct",
            "code_link": "two",
            "named_link": "two",
            "chain": "three",
            "unknown": "missing",
            "outside": "outside",
        }[row["group"]]
        cases.append(
            {
                "id": row["id"],
                "store": "evidence36",
                "group": group,
                "query": row["query"],
                "value": n,
                "unit": unit,
                "sources": row["docs"],
                "decision": kind,
            }
        )
    corpus = [
        {
            "id": d["id"],
            "store": "evidence36",
            "title": d["title"],
            "text": d["title"] + "。项目政策。" + d["body"],
        }
        for d in docs
    ]
    return cases, corpus, hashlib.sha256(query_data + corpus_data).hexdigest()


def old_aliases(chunk):
    fragment = study.body(chunk)
    codes = LEGACY_CODE.findall(fragment)
    return [chunk.title, *dict.fromkeys(codes)]


def old_refs(chunk):
    fragment = study.body(chunk)
    if DEDICATED_UNKNOWN.search(fragment):
        return []
    names = re.findall(r"《([^》]+)》", fragment)
    codes = list(dict.fromkeys(LEGACY_CODE.findall(fragment)))
    # A table with multiple codes is terminal evidence. A lone code points to a
    # table only when the document states a cross-reference rather than a rate.
    if len(codes) == 1 and not re.search(r"\d+(?:元|天|个月|小时|枚|次)", fragment):
        return [*names, *codes]
    return names


async def main(args):
    source = Path(os.environ["RAG_EVAL_SOURCE"])
    credentials = Credentials(_env_file=Path(os.environ["RAG_EVAL_MODEL_CONFIG"]))
    cases, docs, dataset_hash = load_original(source)
    study.references = old_refs
    study.declared_keys = old_aliases
    study.outside_query = lambda _merchant, _query: False
    corpus = study.Corpus(docs)
    await corpus.initialize()
    directory = Path(__file__).parent / args.output
    directory.mkdir(exist_ok=True)
    sem = asyncio.Semaphore(args.parallel)
    fingerprint = hashlib.sha256(
        dataset_hash.encode() + Path(__file__).read_bytes() + Path(study.__file__).read_bytes()
    ).hexdigest()
    async with httpx.AsyncClient(
        base_url="https://api.deepseek.com",
        trust_env=os.environ.get("RAG_EVAL_DIRECT") != "1",
        limits=httpx.Limits(max_connections=args.parallel),
        timeout=httpx.Timeout(55, connect=12),
        follow_redirects=False,
    ) as client:
        service = study.Service(client, credentials)

        async def evaluate(case):
            path = directory / (case["id"] + ".json")
            if path.exists():
                record = json.loads(path.read_bytes())
                if record["experiment_sha256"] != fingerprint:
                    raise ValueError("Different configuration; use another output directory")
                return record
            async with sem:
                result = await study.measure(case, corpus, service, dataset_hash)
            result["experiment_sha256"] = fingerprint
            path.write_text(study.dump(result) + "\n", encoding="utf-8")
            print(case["id"], [result["arms"][arm]["passed"] for arm in study.ARMS], flush=True)
            return result

        rows = await asyncio.gather(*(evaluate(case) for case in cases))
    final = study.summarize(rows, "historical-component-regression", dataset_hash, fingerprint)
    final.pop("quality_gain_vs_single_ci95")
    final.pop("quality_difference_ci95")
    final.pop("quality_goal")
    final.pop("cost_goal")
    final.pop("stretch_cost_goal")
    final["source_group_counts"] = dict(Counter(row["group"] for row in cases))
    final["scope"] = (
        "Historical original cases and corpus, component-level execution, known-data regression "
        "with typed value/unit/source adjudication. Not a replication of the historical end-to-end graph."
    )
    (directory / "summary.json").write_text(study.dump(final) + "\n", encoding="utf-8")
    print(study.dump(final), flush=True)


if __name__ == "__main__":
    logging.getLogger("jieba").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("--parallel", type=int, default=10)
    asyncio.run(main(parser.parse_args()))
