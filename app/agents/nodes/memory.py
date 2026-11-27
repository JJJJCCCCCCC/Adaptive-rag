"""Conversation memory compaction node."""

from app.agents.state import AgentState
from app.config.settings import Settings
from app.core.enums import CompressionLevel
from app.memory.compression import KEEP_RECENT_TURNS, apply_compression
from app.memory.summarizer import ConversationSummarizer
from app.schemas.chat import ChatMessage


async def memory_node(
    state: AgentState,
    *,
    settings: Settings,
    summarizer: ConversationSummarizer,
) -> dict:
    messages = list(state.get("messages") or [])
    messages.append(ChatMessage(role="user", content=state["user_query"]))
    final_answer = state.get("final_answer")
    if final_answer:
        messages.append(ChatMessage(role="assistant", content=final_answer))
    tool_results = list(state.get("tool_results") or [])
    summary = state.get("summary") or ""
    window, tool_results, level, summary = apply_compression(
        messages, tool_results, summary, settings
    )
    if level == CompressionLevel.AUTO_COMPACT:
        earlier = messages[: -(KEEP_RECENT_TURNS * 2)]
        if earlier:
            if summary:
                earlier.insert(0, ChatMessage(role="system", content=summary))
            summary = await summarizer.summarize(earlier) or summary
            window, tool_results, _, summary = apply_compression(
                messages, tool_results, summary, settings
            )
    return {"messages": window, "tool_results": tool_results, "summary": summary}
