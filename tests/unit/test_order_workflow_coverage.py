from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.agents.nodes.executor import executor_node
from app.agents.state import AgentState
from app.config.settings import Settings
from app.core.enums import OrderStatus
from app.core.request_scope import request_session
from app.infra.rate_limit import InMemoryRateLimiter
from app.providers.embeddings import HashEmbeddingProvider
from app.providers.fake import OfflineLLMProvider
from app.providers.reranker import LexicalRerankProvider
from app.repositories.db import init_schema
from app.repositories.idempotency import IdempotencyRepository
from app.repositories.models import AuditLogRow, RefundRow, ReturnRow
from app.repositories.order_repository import OrderRepository
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import DocumentReranker
from app.retrieval.vector import InMemoryVectorStore, VectorRetriever
from app.schemas.agent import AgentPlan, PlanStep
from app.schemas.chat import ChatRequest
from app.services.chat_service import ChatService
from app.services.seed import seed_demo_data
from app.tools.aftersale_tools import CancelOrderTool
from app.tools.factory import build_tool_registry
from app.tools.order_tools import GetOrderTool
from app.tools.registry import ToolRegistry


@pytest.fixture
async def database():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    try:
        await init_schema(engine)
        factory = async_sessionmaker(engine, expire_on_commit=False)
        async with factory() as db:
            await seed_demo_data(db)
            await db.commit()
        yield factory
    finally:
        await engine.dispose()


@pytest.fixture
def retriever():
    vector = VectorRetriever(InMemoryVectorStore(), HashEmbeddingProvider(dim=32))
    return HybridRetriever(BM25Retriever(), vector, rrf_k=60)


@pytest.fixture
def service(database, retriever):
    return ChatService(
        settings=Settings(),
        llm=OfflineLLMProvider(),
        retriever=retriever,
        reranker=DocumentReranker(LexicalRerankProvider()),
        session_factory=database,
        rate_limiter=InMemoryRateLimiter(limit=100),
    )


@pytest.mark.parametrize(
    ("query", "order_id", "tool_name", "answer", "before", "after", "audit_action", "record_type"),
    [
        (
            "取消订单 ORD10003",
            "ORD10003",
            "cancel_order",
            "已取消",
            OrderStatus.PAID,
            OrderStatus.CANCELLED,
            "cancel_order",
            None,
        ),
        (
            "帮我退款 ORD10003",
            "ORD10003",
            "create_refund_request",
            "退款申请已提交",
            OrderStatus.PAID,
            OrderStatus.REFUNDING,
            "create_refund",
            RefundRow,
        ),
        (
            "帮我退货 ORD10002",
            "ORD10002",
            "create_return_request",
            "退货申请已提交",
            OrderStatus.DELIVERED,
            OrderStatus.DELIVERED,
            "create_return",
            ReturnRow,
        ),
    ],
    ids=["cancel", "refund", "return"],
)
async def test_order_write_waits_for_confirmation_and_replays_without_mutating_twice(
    service, database, query, order_id, tool_name, answer, before, after, audit_action, record_type
):
    request = ChatRequest(message=query, user_id="u_demo", session_id="order-session")
    proposal = await service.chat(request, trace_id="proposal")
    pending = proposal.pending_action
    assert pending is not None
    assert pending.tool_name == tool_name
    assert pending.arguments["order_id"] == order_id
    assert proposal.answer == pending.summary
    async with database() as db:
        assert (await OrderRepository(db).get_order(order_id)).status == before
        assert (await db.scalars(select(AuditLogRow))).all() == []

    confirm = request.model_copy(
        update={"message": "确认执行", "confirm_action_id": pending.action_id}
    )
    first = await service.chat(confirm, trace_id="confirmed")
    replay = await service.chat(confirm, trace_id="replayed")

    assert first.pending_action is None
    assert order_id in first.answer and answer in first.answer
    assert replay.answer == first.answer
    async with database() as db:
        assert (await OrderRepository(db).get_order(order_id)).status == after
        audit = (await db.scalars(select(AuditLogRow))).one()
        assert (audit.action, audit.payload, audit.trace_id) == (
            audit_action,
            order_id,
            "confirmed",
        )
        if record_type:
            record = (await db.scalars(select(record_type))).one()
            assert (record.order_id, record.status) == (order_id, "requested")


async def test_read_followup_resolves_prior_order_but_new_write_needs_explicit_id(service):
    request = ChatRequest(
        message="帮我查订单 ORD10002 的物流", user_id="u_demo", session_id="shipping-session"
    )
    initial = await service.chat(request, trace_id="initial")
    assert "已签收" in initial.answer

    followup = await service.chat(
        request.model_copy(update={"message": "那它到哪了？"}), trace_id="read"
    )
    assert "ORD10002" in followup.answer and "已签收" in followup.answer

    write = await service.chat(request.model_copy(update={"message": "帮我退货"}), trace_id="write")
    assert write.pending_action is None
    assert "请提供要处理的订单号" in write.answer


def _execute_state(*steps, **changes) -> AgentState:
    return {
        "plan": AgentPlan(steps=list(steps)),
        "user_id": "u_demo",
        "session_id": "session",
        "trace_id": "trace",
        "user_query": "查订单",
        **changes,
    }


def _order_step(tool_name: str, order_id: str = "ORD10003", step_id: int = 1) -> PlanStep:
    return PlanStep(
        step_id=step_id, goal="处理订单", tool_name=tool_name, arguments={"order_id": order_id}
    )


async def test_executor_rejects_write_without_confirmation(database):
    registry = ToolRegistry()
    registry.register(CancelOrderTool())
    state = _execute_state(_order_step("cancel_order"))
    async with request_session(database) as db:
        result = await executor_node(state, registry=registry, settings=Settings())
        order = await OrderRepository(db).get_order("ORD10003")

    failure = result["tool_results"][0]
    assert failure.success is False
    assert "requires explicit user confirmation" in failure.error
    assert result["step_count"] == 1
    assert order.status == OrderStatus.PAID


async def test_executor_rejects_confirmation_key_reuse_with_different_order(database):
    registry = ToolRegistry()
    registry.register(CancelOrderTool())
    settings = Settings()
    async with request_session(database) as db:
        first = await executor_node(
            _execute_state(_order_step("cancel_order"), confirm_action_id="act_test"),
            registry=registry,
            settings=settings,
        )
        await db.commit()
    async with request_session(database) as db:
        second = await executor_node(
            _execute_state(_order_step("cancel_order", "ORD10001"), confirm_action_id="act_test"),
            registry=registry,
            settings=settings,
        )
        original = await OrderRepository(db).get_order("ORD10003")
        other = await OrderRepository(db).get_order("ORD10001")

    assert first["tool_results"][0].success is True
    assert second["tool_results"][0].success is False
    assert "idempotency key reused with different payload" in second["tool_results"][0].error
    assert original.status == OrderStatus.CANCELLED
    assert other.status == OrderStatus.SHIPPED


@pytest.mark.parametrize("ledger_visible", [True, False], ids=["replay-found", "no-replay"])
async def test_executor_checks_for_matching_result_after_a_rejected_write(
    database, monkeypatch, ledger_visible
):
    # Simulate when a competing cancellation's ledger entry becomes visible
    # while the same request attempts the real, guarded write.
    replay = {"order_id": "ORD10003", "status": "cancelled"}
    read_ledger = AsyncMock(side_effect=[None, replay if ledger_visible else None])
    monkeypatch.setattr(IdempotencyRepository, "get_if_matches", read_ledger)
    registry = ToolRegistry()
    registry.register(CancelOrderTool())

    async with request_session(database) as db:
        await OrderRepository(db).update_status("ORD10003", OrderStatus.CANCELLED)
        result = await executor_node(
            _execute_state(_order_step("cancel_order"), confirm_action_id="act_test"),
            registry=registry,
            settings=Settings(),
        )
        assert (await db.scalars(select(AuditLogRow))).all() == []

    assert read_ledger.await_count == 2
    assert read_ledger.await_args_list[0] == read_ledger.await_args_list[1]
    tool_result = result["tool_results"][0]
    assert tool_result.success is ledger_visible
    if ledger_visible:
        assert tool_result.output == replay
    else:
        assert "only unpaid or paid-unshipped orders can be cancelled" in tool_result.error


@pytest.mark.parametrize(
    ("step_count", "expected_order_ids"),
    [(0, ["ORD10003"]), (8, [])],
    ids=["call-budget", "step-budget"],
)
async def test_executor_stops_at_the_configured_budget(database, step_count, expected_order_ids):
    registry = ToolRegistry()
    registry.register(GetOrderTool())
    state = _execute_state(
        PlanStep(step_id=1, goal="consider next step"),
        _order_step("get_order", step_id=2),
        _order_step("get_order", "ORD10002", step_id=3),
        step_count=step_count,
    )
    async with request_session(database):
        result = await executor_node(
            state, registry=registry, settings=Settings(max_tool_calls=1, max_agent_steps=8)
        )

    assert [item.output["order_id"] for item in result["tool_results"]] == expected_order_ids
    assert result["step_count"] == step_count + len(expected_order_ids)


async def test_product_read_falls_back_to_search_and_escapes_catalog_wildcards(database, retriever):
    registry = build_tool_registry(retriever)
    state = _execute_state(
        PlanStep(
            step_id=1,
            goal="find product",
            tool_name="get_product_detail",
            arguments={"sku": "UNKNOWN"},
        ),
        PlanStep(
            step_id=2,
            goal="literal wildcard",
            tool_name="search_products",
            arguments={"query": "%"},
        ),
        user_query="降噪耳机",
    )
    async with request_session(database):
        result = await executor_node(state, registry=registry, settings=Settings())

    fallback, wildcard = result["tool_results"]
    assert fallback.name == "search_products"
    assert fallback.output["fallback_from"] == "get_product_detail"
    assert [product["sku"] for product in fallback.output["products"]] == ["SKU-HEADSET"]
    assert wildcard.success is True
    assert wildcard.output == {"products": []}
