"""Race independent workers against synthetic orders in a fresh database."""

import argparse
import asyncio
import json
import os
import secrets
import subprocess
import sys
import tempfile
from pathlib import Path

from sqlalchemy import func, select

from app.config.settings import Settings
from app.core.exceptions import AppError
from app.repositories.db import create_engine, create_session_factory, init_schema
from app.repositories.models import AuditLogRow, RefundRow
from app.repositories.order_repository import OrderRepository
from app.services.seed import seed_demo_data


async def compete(url, action, i):
    engine = create_engine(Settings(database_url=url))
    try:
        async with create_session_factory(engine)() as session:
            repo = OrderRepository(session)
            print("ready", flush=True)
            if input() != "go":
                raise RuntimeError("Worker did not receive start signal")
            if action == "refund":
                await repo.create_refund(order_id="ORD10002", reason="synthetic", trace_id=str(i))
            else:
                await repo.cancel_order("ORD10003", trace_id=str(i))
            await session.commit()
        return "committed"
    except AppError as exc:
        return exc.error_code.value
    finally:
        await engine.dispose()


async def inspect_db(url, setup=False):
    engine = create_engine(Settings(database_url=url))
    try:
        if setup:
            await init_schema(engine)
        async with create_session_factory(engine)() as session:
            if setup:
                await seed_demo_data(session)
                await session.commit()
            return {
                "refund_requests": await session.scalar(
                    select(func.count()).select_from(RefundRow)
                ),
                "cancellations": await session.scalar(
                    select(func.count())
                    .select_from(AuditLogRow)
                    .where(AuditLogRow.action == "cancel_order")
                ),
            }
    finally:
        await engine.dispose()


def measure(url):
    asyncio.run(inspect_db(url, setup=True))
    results = {}
    for action in ["refund", "cancel"]:
        workers = [
            subprocess.Popen(
                [sys.executable, "-m", "eval.write_study", "worker", action, str(i)],
                env={**os.environ, "RAG_STUDY_DSN": url},
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
            )
            for i in range(4)
        ]
        try:
            for p in workers:
                if p.stdout.readline().strip() != "ready":
                    raise RuntimeError("Worker startup failed")
            for p in workers:
                p.stdin.write("go\n")
                p.stdin.flush()
            results[action] = [p.communicate(timeout=60)[0].strip() for p in workers]
            if any(p.returncode for p in workers):
                raise RuntimeError("Worker terminated unexpectedly")
        finally:
            for p in workers:
                if p.poll() is None:
                    p.kill()
                    p.wait()
    results.update(asyncio.run(inspect_db(url)))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["postgres", "sqlite"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="rag-write-study-") as temp:
        root = Path(temp)
        if args.backend == "sqlite":
            result = measure("sqlite+aiosqlite:///" + (root / "study.db").as_posix())
        else:
            # Dedicated synthetic DB on the local test server. Never touch application tables.
            name = "rag_eval_" + secrets.token_hex(12)
            role = os.environ.get("PGUSER", "postgres")
            host = "127.0.0.1"
            base = ["-w", "-h", host, "-U", role]
            subprocess.run(["createdb", *base, name], check=True, timeout=20)
            try:
                result = measure(f"postgresql+asyncpg://{role}@{host}/{name}")
            finally:
                subprocess.run(["dropdb", *base, name], check=True, timeout=20)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"backend": args.backend, "workers": 4, **result}, indent=2), encoding="utf8"
    )
    print(result)


if __name__ == "__main__":
    if sys.argv[1:2] == ["worker"]:
        print(asyncio.run(compete(os.environ["RAG_STUDY_DSN"], sys.argv[2], sys.argv[3])))
    else:
        main()
