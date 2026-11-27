"""Isolated workflow, load, embedded vector-store and optional loopback service probes."""

import argparse
import asyncio
import json
import socket
import tempfile
import time
import uuid
from pathlib import Path

from pydantic import SecretStr
from qdrant_client import QdrantClient
from sqlalchemy import func, select

from app.config.settings import Settings
from app.core.exceptions import InvalidRequestError, RateLimitError
from app.core.request_scope import request_session
from app.infra.rate_limit import RedisRateLimiter
from app.mcp.runtime import call_mcp_tool
from app.repositories.models import RefundRow
from app.repositories.session_repository import SessionRepository
from app.retrieval.vector import QdrantVectorStore, VectorRecord
from app.schemas.chat import ChatRequest
from app.schemas.retrieval import RetrievedChunk
from app.services.container import build_container


def available(port):
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def vector_case(client):
    name = "rag_study_" + uuid.uuid4().hex
    store = QdrantVectorStore("http://127.0.0.1:6333", name, 2)
    store._client = client
    try:
        chunks = [
            RetrievedChunk(chunk_id=k, document_id=k, title=k, content=k, score=0)
            for k in ["a", "b"]
        ]
        pairs = [VectorRecord(chunks[0], [1.0, 0.0]), VectorRecord(chunks[1], [0.0, 1.0])]
        store.upsert(pairs)
        store.upsert(pairs)
        return {
            "ping": store.ping(),
            "count_after_double_upsert": client.count(name, exact=True).count,
            "rank1": store.search([0.0, 1.0], top_k=1)[0].chunk_id,
        }
    finally:
        if client.collection_exists(name):
            client.delete_collection(name)
        client.close()


async def run():
    results = {"qdrant_embedded": vector_case(QdrantClient(":memory:"))}
    results["qdrant_http"] = (
        vector_case(QdrantClient(url="http://127.0.0.1:6333"))
        if available(6333)
        else {"status": "no_listener"}
    )
    if available(6379):
        from redis.asyncio import Redis

        client = Redis(host="127.0.0.1", socket_connect_timeout=1)
        try:
            limiter = RedisRateLimiter(client, limit=3)
            key = uuid.uuid4().hex
            results["redis"] = {
                "ping": await client.ping(),
                "allowed_of_8": sum(await asyncio.gather(*(limiter.allow(key) for _ in range(8)))),
            }
        finally:
            await client.aclose()
    else:
        results["redis"] = {"status": "no_listener"}
    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings(
            _env_file=None,
            app_env="testing",
            redis_url="",
            qdrant_url="",
            deepseek_api_key=SecretStr(""),
            embedding_api_key=SecretStr(""),
            rate_limit_per_minute=8,
            database_url="sqlite+aiosqlite:///" + (Path(tmp) / "run.db").as_posix(),
        )
        c = await build_container(settings, seed=True)
        try:

            async def chat(text, session, user="u_demo", **kwargs):
                return await c.chat_service.chat(
                    ChatRequest(message=text, session_id=session, user_id=user, **kwargs),
                    trace_id=uuid.uuid4().hex,
                )

            await chat("查询订单 ORD10001", "memory")
            follow = await chat("那物流呢？", "memory")
            fresh = await chat("那物流呢？", "new")
            unsafe_write = await chat("帮我取消订单", "memory")
            async with request_session(c.session_factory) as db:
                repo = SessionRepository(db)
                await repo.save_summary("memory", "回访订单配送进度")
                await db.commit()
                state = await c.chat_service._initial_state(
                    db, ChatRequest(message="你好", user_id="u_demo"), "memory", "m"
                )
            results["memory"] = {
                "same_session_read": follow.answer,
                "fresh_session_read": fresh.answer,
                "write_without_explicit_id_proposed": unsafe_write.pending_action is not None,
                "restored_summary": state["summary"],
            }
            # Submit an actual confirmation, replay it and check the durable business result.
            prop = await chat("取消订单 ORD10003", "cancel")
            action = prop.pending_action.action_id
            first = await chat("确认执行", "cancel", confirm_action_id=action)
            second = await chat("确认执行", "cancel", confirm_action_id=action)
            results["sequential_cancel"] = {"first": first.answer, "replay": second.answer}
            other = await build_container(
                settings.model_copy(update={"rate_limit_per_minute": 1000})
            )
            try:
                service = other.chat_service
                proposal = await service.chat(
                    ChatRequest(
                        message="帮我申请退款 ORD10002", user_id="u_demo", session_id="race"
                    ),
                    trace_id="propose",
                )
                confirm = ChatRequest(
                    message="确认执行",
                    user_id="u_demo",
                    session_id="race",
                    confirm_action_id=proposal.pending_action.action_id,
                )
                try:
                    await c.chat_service.chat(confirm, trace_id="different-worker")
                except InvalidRequestError:
                    cross_worker = "unknown_confirmation"
                else:
                    cross_worker = "accepted"
                replies = await asyncio.gather(
                    *(service.chat(confirm, trace_id=f"same-confirm-{i}") for i in range(4)),
                    return_exceptions=True,
                )
                async with c.session_factory() as db:
                    count = await db.scalar(
                        select(func.count())
                        .select_from(RefundRow)
                        .where(RefundRow.order_id == "ORD10002")
                    )
                results["shared_confirmation"] = {
                    "other_instance": cross_worker,
                    "same_instance_responses": [
                        type(r).__name__ if isinstance(r, BaseException) else r.answer
                        for r in replies
                    ],
                    "persisted_refunds": count,
                }
            finally:
                await other.aclose()

            async def burst(i):
                start = time.perf_counter()
                try:
                    await chat("你好", f"load{i}", "load-user")
                    code = "ok"
                except RateLimitError:
                    code = "limited"
                except Exception as exc:
                    code = type(exc).__name__
                return code, round((time.perf_counter() - start) * 1000, 1)

            samples = await asyncio.gather(*(burst(i) for i in range(32)))
            results["burst"] = {
                "n": 32,
                "limit": 8,
                "ok": sum(s[0] == "ok" for s in samples),
                "limited": sum(s[0] == "limited" for s in samples),
                "records": samples,
            }
            results["local_mcp"] = await call_mcp_tool(
                name="mcp_store_hours", arguments={}, transport="local"
            )
        finally:
            await c.aclose()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(
        json.dumps(asyncio.run(run()), ensure_ascii=False, indent=2), encoding="utf8"
    )
