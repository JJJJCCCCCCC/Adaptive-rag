import json

import pytest

from app.agents.nodes.faithfulness import faithfulness_node
from app.agents.nodes.generate import generate_node
from app.agents.nodes.planner import Planner, planner_node
from app.core.enums import IntentLabel, RetrievalStrategy
from app.core.exceptions import LLMProviderError
from app.guardrails.hallucination import FaithfulnessChecker
from app.providers.base import LLMResponse
from app.providers.fake import OfflineLLMProvider
from app.schemas.agent import AgentPlan, IntentResult, PlanStep, ToolResult
from app.schemas.chat import ChatMessage
from app.schemas.common import TokenUsage
from app.schemas.retrieval import RetrievedChunk
from app.tools.aftersale_tools import (
    CancelOrderTool,
    CreateRefundRequestTool,
    CreateReturnRequestTool,
    GetRefundStatusTool,
)
from app.tools.order_tools import GetOrderTool
from app.tools.registry import ToolRegistry


class ScriptedLLM:
    def __init__(self, *responses: LLMResponse | LLMProviderError) -> None:
        self.responses = iter(responses)
        self.calls: list[tuple[list[dict[str, str]], dict]] = []

    async def generate(self, messages, **options) -> LLMResponse:
        self.calls.append((messages, options))
        response = next(self.responses)
        if isinstance(response, LLMProviderError):
            raise response
        return response


def _json_response(**payload) -> LLMResponse:
    return LLMResponse(json.dumps(payload, ensure_ascii=False))


def _chunk(chunk_id: str, title: str, content: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=f"policy-{chunk_id}",
        title=title,
        content=content,
        score=1,
    )


async def test_generation_sends_current_evidence_and_bounded_history():
    chunks = [_chunk(f"c{i}", f"Policy {i}", f"Evidence {i}") for i in range(1, 8)]
    payload = json.dumps(
        {"answer": "已提供配送信息。", "citation_chunk_ids": ["c2", "invented", "c2"]}
    )
    llm = ScriptedLLM(
        LLMResponse(
            payload, usage=TokenUsage(prompt_tokens=10, completion_tokens=4, total_tokens=14)
        )
    )
    result = await generate_node(
        {
            "user_query": "配送需要多久？",
            "strategy": RetrievalStrategy.SINGLE,
            "intent": IntentResult(
                label=IntentLabel.FAQ_POLICY, confidence=0.9, prompt_version="test"
            ),
            "retrieved_chunks": chunks,
            "tool_results": [ToolResult(name="get_order", success=True, output={"status": "paid"})],
            "messages": [
                ChatMessage(role="user", content="older turn"),
                ChatMessage(role="system", content="internal notice"),
                ChatMessage(role="user", content="recent user turn"),
                ChatMessage(role="tool", content="old tool data"),
                ChatMessage(role="assistant", content="x" * 400 + "omitted suffix"),
            ],
            "summary": "s" * 600 + "omitted summary suffix",
            "token_usage": TokenUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
        },
        llm,
    )

    messages, options = llm.calls[0]
    prompt = messages[-1]["content"]
    assert "配送需要多久？" in prompt and "faq_policy" in prompt
    assert "[c6] Policy 6" in prompt and "[c7]" not in prompt
    assert '"status": "paid"' in prompt
    assert "recent user turn" in prompt and "x" * 400 in prompt and "s" * 600 in prompt
    for omitted in ("older turn", "internal notice", "old tool data", "omitted"):
        assert omitted not in prompt
    assert options["response_format"] == {"type": "json_object"}
    assert result["final_answer"] == "已提供配送信息。"
    assert [citation.chunk_id for citation in result["citations"]] == ["c2"]
    assert result["token_usage"] == TokenUsage(
        prompt_tokens=13, completion_tokens=6, total_tokens=19
    )


@pytest.mark.parametrize(
    ("state", "response", "expected", "citation_ids"),
    [
        (
            {"retrieved_chunks": [_chunk("c1", "售后", "已签收订单可以联系人工客服。")]},
            LLMResponse("invalid JSON"),
            "已签收订单可以联系人工客服。",
            ["c1"],
        ),
        (
            {
                "tool_results": [
                    ToolResult(name="get_order", success=True, output={"status": "paid"})
                ]
            },
            LLMProviderError("synthetic timeout", timeout=True),
            "已查询到系统数据",
            [],
        ),
        ({}, LLMResponse("{}"), "请补充相关信息", []),
    ],
    ids=["retrieved-evidence", "tool-data", "no-evidence"],
)
async def test_generation_falls_back_when_the_provider_cannot_answer(
    state, response, expected, citation_ids
):
    usage = TokenUsage(prompt_tokens=2, total_tokens=2)
    result = await generate_node(
        {"user_query": "这是我的订单吗？", "token_usage": usage, **state},
        ScriptedLLM(response),
    )

    assert expected in result["final_answer"]
    assert [citation.chunk_id for citation in result["citations"]] == citation_ids
    assert result["token_usage"] == usage


@pytest.mark.parametrize(
    ("tool_name", "output", "expected"),
    [
        ("create_refund_request", {"order_id": "ORD42"}, "退款申请已提交，尚不代表款项到账"),
        ("create_return_request", {"order_id": "ORD42"}, "退货申请已提交，尚不代表退款完成"),
        ("get_order", {"order_id": "ORD42", "status": "paid"}, "当前状态：已支付"),
        ("track_shipment", {"order_id": "ORD42", "events": []}, "暂无物流轨迹"),
        (
            "get_refund_status",
            {"refunds": [{"refund_id": "rf_test", "status": "requested"}]},
            "退款单 rf_test，当前状态：requested",
        ),
        ("get_refund_status", {"refunds": []}, "暂未查询到退款申请记录"),
        (
            "search_knowledge",
            {"chunks": [{"title": "说明", "content": "核对退货日期。"}]},
            "核对退货日期",
        ),
    ],
    ids=[
        "refund-request",
        "return-request",
        "paid-order",
        "no-events",
        "refund",
        "no-refund",
        "policy",
    ],
)
async def test_offline_generation_uses_only_customer_facing_tool_fields(
    tool_name, output, expected
):
    result = await generate_node(
        {
            "user_query": "订单 ORD42 的售后状态是什么？",
            "strategy": RetrievalStrategy.AGENT,
            "tool_results": [
                ToolResult(
                    name=tool_name, success=True, output={**output, "internal_audit": "hidden data"}
                ),
                ToolResult(name="get_order", success=False, error="hidden diagnostic"),
            ],
        },
        OfflineLLMProvider(),
    )
    assert expected in result["final_answer"]
    assert "hidden" not in result["final_answer"]
    assert result["citations"] == []


async def test_offline_multi_hop_answer_cites_only_chunks_covering_query_terms():
    result = await generate_node(
        {
            "user_query": "shipping returns",
            "strategy": RetrievalStrategy.MULTI,
            "retrieved_chunks": [
                _chunk("ship", "Shipping", "shipping takes three days"),
                _chunk("contact", "Contact", "business hours"),
                _chunk("return", "Returns", "returns require proof of purchase"),
            ],
        },
        OfflineLLMProvider(),
    )
    assert "shipping takes three days" in result["final_answer"]
    assert "returns require proof of purchase" in result["final_answer"]
    assert "business hours" not in result["final_answer"]
    assert [citation.chunk_id for citation in result["citations"]] == ["ship", "return"]


@pytest.fixture
def order_tools() -> ToolRegistry:
    registry = ToolRegistry()
    for tool in (
        GetOrderTool(),
        CancelOrderTool(),
        GetRefundStatusTool(),
        CreateRefundRequestTool(),
        CreateReturnRequestTool(),
    ):
        registry.register(tool)
    return registry


async def test_planner_filters_tools_outside_the_selected_skill(order_tools):
    steps = [
        PlanStep(step_id=1, goal="foreign tool", tool_name="create_refund_request"),
        PlanStep(step_id=2, goal="unknown tool", tool_name="unregistered"),
        *[
            PlanStep(step_id=index, goal="read order", tool_name="get_order")
            for index in range(3, 10)
        ],
    ]
    llm = ScriptedLLM(
        LLMResponse(AgentPlan(steps=steps, rationale="provider plan").model_dump_json())
    )
    plan = await Planner(llm, order_tools).create_plan(
        "取消订单 ORD42", "cancel_order", "用户查过配送"
    )

    messages, _ = llm.calls[0]
    assert "get_order:" in messages[0]["content"]
    assert "create_refund_request:" not in messages[0]["content"]
    assert "用户查过配送" in messages[1]["content"]
    assert [step.step_id for step in plan.steps] == list(range(3, 9))
    assert plan.rationale == "skill:order_cancellation;provider plan"


async def test_planner_uses_skill_arguments_when_provider_fails(order_tools):
    llm = ScriptedLLM(LLMProviderError("synthetic timeout", timeout=True))
    query = "帮我退款 ord42，尺码不合适"
    result = await planner_node({"user_query": query}, Planner(llm, order_tools))

    pending = result["pending_action"]
    assert result["plan"].rationale == "skill:refund_handling"
    assert pending.tool_name == "create_refund_request"
    assert pending.arguments == {"order_id": "ORD42", "reason": query}
    assert "不代表退款已到账" in pending.summary


@pytest.mark.parametrize("query", ["退款进度 ORD42", "不要帮我退款 ORD42", "帮我退款可以吗 ORD42"])
async def test_planner_omits_refund_write_for_non_action_queries(order_tools, query):
    result = await planner_node({"user_query": query}, Planner(OfflineLLMProvider(), order_tools))

    assert result["pending_action"] is None
    assert [step.tool_name for step in result["plan"].steps] == ["get_order", "get_refund_status"]


async def test_planner_requires_order_id_before_proposing_return(order_tools):
    planner = Planner(OfflineLLMProvider(), order_tools)
    missing = await planner_node({"user_query": "帮我退货"}, planner)
    assert missing["plan"].steps == []
    assert missing["extra"] == {"missing_order_id": True}
    answer = await generate_node(missing, ScriptedLLM())
    assert "请提供要处理的订单号" in answer["final_answer"]

    proposal = await planner_node({"user_query": "帮我退货 ORD42"}, planner)
    assert proposal["pending_action"].tool_name == "create_return_request"
    assert "ORD42" in proposal["pending_action"].summary


async def test_faithfulness_uses_cited_excerpt_when_checker_is_unavailable():
    llm = ScriptedLLM(LLMProviderError("synthetic timeout", timeout=True))
    chunk = _chunk("return", "退货", "定制商品不支持七天无理由。")
    result = await faithfulness_node(
        {
            "user_query": "定制商品能退吗？",
            "strategy": RetrievalStrategy.SINGLE,
            "final_answer": "随时可以退",
            "retrieved_chunks": [chunk],
        },
        FaithfulnessChecker(llm),
        llm,
    )
    assert chunk.content in result["final_answer"]
    assert "随时可以退" not in result["final_answer"]
    assert [citation.chunk_id for citation in result["citations"]] == ["return"]
    assert result["extra"]["faithfulness"]["rationale"] == "checker_unavailable"
    assert result["extra"]["self_corrected"] is False


@pytest.mark.parametrize("excerpt_supported", [True, False], ids=["excerpt-backed", "refused"])
async def test_faithfulness_checks_extractive_fallback_when_rewrite_is_unsupported(
    excerpt_supported,
):
    chunk = _chunk("return", "退货", "定制商品不支持七天无理由。")
    llm = ScriptedLLM(
        _json_response(supported=False, hallucinated_spans=["随时可以退"]),
        _json_response(answer="一律不能申请售后"),
        _json_response(supported=False, rationale="unsupported rewrite"),
        _json_response(supported=excerpt_supported, rationale="excerpt verdict"),
    )
    result = await faithfulness_node(
        {
            "user_query": "定制商品能退吗？",
            "strategy": RetrievalStrategy.SINGLE,
            "final_answer": "随时可以退",
            "retrieved_chunks": [chunk],
        },
        FaithfulnessChecker(llm),
        llm,
    )

    assert "随时可以退" in llm.calls[1][0][-1]["content"]
    assert "一律不能申请售后" in llm.calls[2][0][-1]["content"]
    assert chunk.content in llm.calls[3][0][-1]["content"]
    assert result["extra"]["faithfulness"]["supported"] is excerpt_supported
    assert result["extra"]["self_corrected"] is True
    if excerpt_supported:
        assert chunk.content in result["final_answer"]
    else:
        assert result["refused"] is True
        assert result["citations"] == []
