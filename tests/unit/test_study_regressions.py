from unittest.mock import AsyncMock

import pytest

from app.agents.nodes.generate import _offline_answer
from app.agents.nodes.memory import memory_node
from app.agents.nodes.planner import Planner, planner_node
from app.config.settings import Settings
from app.core.enums import RetrievalStrategy
from app.providers.fake import OfflineLLMProvider
from app.schemas.chat import ChatMessage
from app.schemas.retrieval import RetrievedChunk
from app.tools.factory import build_tool_registry


@pytest.mark.asyncio
async def test_summarizer_receives_displaced_turns_and_prior_summary():
    history = [ChatMessage(role="user", content=f"早期记录 {i}") for i in range(15)]
    summarizer = AsyncMock()
    summarizer.summarize.return_value = "合并后的摘要"
    result = await memory_node(
        {"messages": history, "user_query": "近期追问", "summary": "已有摘要"},
        settings=Settings(token_budget=10),
        summarizer=summarizer,
    )
    summarized = [m.content for m in summarizer.summarize.await_args.args[0]]
    assert summarized[0] == "已有摘要"
    assert "早期记录 0" in summarized
    assert "近期追问" not in summarized
    assert result["summary"] == "合并后的摘要"
    assert result["messages"][-1].content == "近期追问"


def test_offline_compound_excerpts_only_cite_displayed_chunks():
    def chunk(i, text):
        return RetrievedChunk(
            chunk_id=str(i), document_id="policy", title="运费", score=1, content=text
        )

    chunks = [chunk(1, "基础运费 8 元"), chunk(2, "基础运费 8 元"), chunk(3, "包装破损可补寄")]
    state = {
        "user_query": "基础运费怎么算，包装破损怎么办",
        "retrieved_chunks": chunks,
        "strategy": RetrievalStrategy.MULTI,
    }
    answer, ids = _offline_answer(state)
    assert ids == {"1", "3"}
    assert "包装破损可补寄" in answer


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "propose"),
    [
        ("帮我申请退款 ORD10002", True),
        ("查询 ORD10002 的申请退款进度", False),
        ("不要帮我申请退款 ORD10002", False),
    ],
)
async def test_refund_read_and_proposal_paths(message, propose):
    planner = Planner(OfflineLLMProvider(), build_tool_registry(AsyncMock()))
    result = await planner_node({"user_query": message}, planner)
    pending = result["pending_action"]
    if propose:
        assert pending.tool_name == "create_refund_request"
        assert pending.arguments["order_id"] == "ORD10002"
    else:
        assert pending is None
        steps = {step.tool_name: step for step in result["plan"].steps}
        assert "create_refund_request" not in steps
        assert steps["get_refund_status"].arguments == {"order_id": "ORD10002"}
