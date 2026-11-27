"""Exercise the offline diagnostics and their persisted business outcomes."""

import json
import sys
from contextlib import asynccontextmanager
from io import StringIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call

import pytest
from sse_starlette.sse import AppStatus

from app.api.routes.health import capabilities, readiness
from app.config.settings import clear_settings_cache
from eval import runtime_study, study_summary, transport_study, write_study


@pytest.mark.asyncio
async def test_runtime_diagnostics_verify_memory_writes_and_throttling(monkeypatch):
    monkeypatch.setattr(runtime_study, "available", lambda port: False)
    results = await runtime_study.run()

    assert results["qdrant_http"] == results["redis"] == {"status": "no_listener"}
    assert results["qdrant_embedded"] == {
        "ping": True,
        "count_after_double_upsert": 2,
        "rank1": "b",
    }
    memory = results["memory"]
    assert "ORD10001" in memory["same_session_read"]
    assert "运输中" in memory["same_session_read"]
    assert "ORD10001" not in memory["fresh_session_read"]
    assert memory["write_without_explicit_id_proposed"] is False
    assert memory["restored_summary"] == "回访订单配送进度"
    cancellation = results["sequential_cancel"]
    assert cancellation["first"] == cancellation["replay"]
    assert "已取消" in cancellation["first"]

    shared = results["shared_confirmation"]
    assert shared["other_instance"] == "unknown_confirmation"
    assert shared["persisted_refunds"] == 1
    assert len(shared["same_instance_responses"]) == 4
    assert all("退款申请已提交" in answer for answer in shared["same_instance_responses"])
    assert (results["burst"]["ok"], results["burst"]["limited"]) == (8, 24)
    assert results["local_mcp"]["source"] == "mcp_local"


@pytest.mark.asyncio
async def test_write_workers_count_one_record_per_order(tmp_path, monkeypatch, capsys):
    database = "sqlite+aiosqlite:///" + (tmp_path / "study.db").as_posix()
    assert await write_study.inspect_db(database, setup=True) == {
        "refund_requests": 0,
        "cancellations": 0,
    }

    monkeypatch.setattr("builtins.input", lambda: "go")
    assert await write_study.compete(database, "refund", 0) == "committed"
    assert await write_study.compete(database, "refund", 1) == "INVALID_REQUEST"
    assert await write_study.compete(database, "cancel", 0) == "committed"
    assert await write_study.compete(database, "cancel", 1) == "INVALID_REQUEST"

    monkeypatch.setattr("builtins.input", lambda: "unexpected")
    with pytest.raises(RuntimeError, match="start signal"):
        await write_study.compete(database, "refund", 2)
    assert capsys.readouterr().out.splitlines() == ["ready"] * 5
    assert await write_study.inspect_db(database) == {"refund_requests": 1, "cancellations": 1}


def worker(stdout="ready\n", *, status="committed\n", returncode=0, alive=False):
    process = Mock()
    process.stdin = StringIO()
    process.stdout = StringIO(stdout)
    process.communicate.return_value = (status, None)
    process.returncode = returncode
    process.poll.return_value = None if alive else returncode
    return process


def test_write_measure_releases_workers_only_after_ready(monkeypatch):
    database = "sqlite+aiosqlite:///:memory:"
    workers = [
        worker(status="committed\n" if i % 4 == 0 else "INVALID_REQUEST\n") for i in range(8)
    ]
    popen = Mock(side_effect=workers)
    inspect = AsyncMock(
        side_effect=[
            {"refund_requests": 0, "cancellations": 0},
            {"refund_requests": 1, "cancellations": 1},
        ]
    )
    monkeypatch.setattr(write_study.subprocess, "Popen", popen)
    monkeypatch.setattr(write_study, "inspect_db", inspect)

    result = write_study.measure(database)

    expected = ["committed", "INVALID_REQUEST", "INVALID_REQUEST", "INVALID_REQUEST"]
    assert result == {
        "refund": expected,
        "cancel": expected,
        "refund_requests": 1,
        "cancellations": 1,
    }
    assert inspect.await_args_list == [call(database, setup=True), call(database)]
    assert [item.args[0][-2] for item in popen.call_args_list] == ["refund"] * 4 + ["cancel"] * 4
    assert all(item.kwargs["env"]["RAG_STUDY_DSN"] == database for item in popen.call_args_list)
    for process in workers:
        assert process.stdin.getvalue() == "go\n"
        process.communicate.assert_called_once_with(timeout=60)
        process.kill.assert_not_called()


@pytest.mark.parametrize(("ready", "message"), [(False, "startup failed"), (True, "terminated")])
def test_write_measure_cleans_up_workers_on_failure(monkeypatch, ready, message):
    failed = worker("ready\n" if ready else "unexpected\n", returncode=1)
    other_workers = [worker(alive=True) for _ in range(3)]
    monkeypatch.setattr(write_study.subprocess, "Popen", Mock(side_effect=[failed, *other_workers]))
    monkeypatch.setattr(write_study, "inspect_db", AsyncMock(return_value={}))

    with pytest.raises(RuntimeError, match=message):
        write_study.measure("sqlite+aiosqlite:///:memory:")

    for process in other_workers:
        process.kill.assert_called_once()
        process.wait.assert_called_once()
        assert process.stdin.getvalue() == ("go\n" if ready else "")


@pytest.mark.parametrize("backend", ["postgres", "sqlite"])
def test_write_cli_writes_a_report_and_disposes_postgres_database(tmp_path, monkeypatch, backend):
    report = tmp_path / "reports" / "write.json"
    measure = Mock(return_value={"refund_requests": 1, "cancellations": 1})
    command = Mock()
    monkeypatch.setattr(write_study, "measure", measure)
    monkeypatch.setattr(write_study.subprocess, "run", command)
    monkeypatch.setattr(write_study.secrets, "token_hex", lambda n: "fixture")
    monkeypatch.setenv("PGUSER", "test_role")
    monkeypatch.setattr(sys, "argv", ["write_study", "--backend", backend, "--out", str(report)])

    write_study.main()

    assert json.loads(report.read_bytes()) == {
        "backend": backend,
        "workers": 4,
        "refund_requests": 1,
        "cancellations": 1,
    }
    args = ["-w", "-h", "127.0.0.1", "-U", "test_role", "rag_eval_fixture"]
    if backend == "postgres":
        assert command.call_args_list == [
            call(["createdb", *args], check=True, timeout=20),
            call(["dropdb", *args], check=True, timeout=20),
        ]
        measure.assert_called_once_with("postgresql+asyncpg://test_role@127.0.0.1/rag_eval_fixture")
    else:
        command.assert_not_called()
        assert measure.call_args.args[0].startswith("sqlite+aiosqlite:///")
        assert measure.call_args.args[0].endswith("/study.db")


def test_transport_diagnostics_preserves_sse_answer_after_validation(tmp_path, monkeypatch):
    report = tmp_path / "transport.json"
    monkeypatch.setattr(sys, "argv", ["transport_study", str(report)])
    # TestClient starts a new event loop for this diagnostic.
    monkeypatch.setattr(AppStatus, "should_exit_event", None)

    try:
        transport_study.main()
    finally:
        clear_settings_cache()

    diagnosis = json.loads(report.read_bytes())
    assert diagnosis["http_tool"]["valid_reply"] == {"name": "echo_fixture", "id": "synthetic"}
    assert "503" in diagnosis["http_tool"]["upstream_503"]
    sse = diagnosis["sse"]
    assert sse["event_types"][-2:] == ["answer", "done"]
    assert sse["first_token_after_check_node"]
    assert sse["joined_chunks_match_answer"]
    assert sse["ended_with_done"]


def test_study_summary_replays_evidence_and_pairs_identical_questions():
    result = study_summary.summarize()

    rows = result["strategies"]["holdout-after"]
    all_adaptive = next(row for row in rows if row["group"] == "all" and row["arm"] == "adaptive")
    assert all_adaptive["n"] > 0
    assert all_adaptive["reference_evidence"] <= all_adaptive["n"]
    assert all_adaptive["answer_coverage"] <= all_adaptive["n"]
    assert all_adaptive["searches"] >= all_adaptive["rounds"]
    comparison = result["paired"]["holdout-after"]
    assert sum(comparison[k] for k in ("wins", "losses", "ties")) == all_adaptive["n"]
    assert 0 <= comparison["exact_two_sided_p"] <= 1


@pytest.mark.asyncio
@pytest.mark.parametrize("database_ok", [False, True])
async def test_readiness_treats_only_database_as_required(database_ok):
    execute = AsyncMock(side_effect=None if database_ok else RuntimeError("db down"))

    @asynccontextmanager
    async def connect():
        yield SimpleNamespace(execute=execute)

    redis = SimpleNamespace(ping=AsyncMock(side_effect=OSError("optional redis down")))
    vector = SimpleNamespace(ping=Mock(side_effect=OSError("optional qdrant down")))
    llm = SimpleNamespace(ping=AsyncMock(return_value=False))
    dependencies = SimpleNamespace(
        engine=SimpleNamespace(connect=connect), llm=llm, redis=redis, vector_store=vector
    )
    result = await readiness(dependencies)

    assert result.ready is database_ok
    assert {dep.name: dep.ready for dep in result.dependencies} == {
        "database": database_ok,
        "llm": False,
        "redis": False,
        "vector_store": False,
    }
    assert str(execute.await_args.args[0]) == "SELECT 1"


@pytest.mark.asyncio
async def test_capabilities_reports_full_runtime_backends():
    settings = SimpleNamespace(
        llm_configured=True,
        embedding_configured=False,
        local_embedding_configured=True,
        use_sqlite=False,
    )
    container = SimpleNamespace(
        settings=settings,
        redis=object(),
        vector_store=object(),
        chat_service=SimpleNamespace(
            _reranker=SimpleNamespace(_provider=SimpleNamespace(backend="cross_encoder"))
        ),
    )
    status = await capabilities(container)

    assert (status.database, status.llm, status.retriever) == ("postgresql", "deepseek", "hybrid")
    assert (status.cache, status.reranker, status.degraded) == ("redis", "cross_encoder", False)
