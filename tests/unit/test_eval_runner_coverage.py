"""Exercise the evaluation harness with deterministic datasets and provider substitutes."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call

import pytest

from app.agents.nodes import faithfulness, intent
from app.config import settings
from app.core.enums import IntentLabel, RetrievalStrategy, ToolRisk
from app.guardrails import hallucination
from app.prompts.intent import INTENT_SYSTEM, INTENT_SYSTEM_V1, INTENT_SYSTEM_V2
from app.providers import deepseek, embeddings, reranker as rerank_providers
from app.retrieval import hybrid, ingest, reranker as reranking, vector
from app.schemas.chat import ActionProposal, ChatRequest, ChatResponse
from app.schemas.common import LatencyBreakdown
from app.services import container
from eval import runner


def dataset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, path_constant: str, rows: list[dict]
) -> None:
    path = tmp_path / f"{path_constant.lower()}.jsonl"
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8"
    )
    monkeypatch.setattr(runner, path_constant, path)


def test_intent_rules_and_overlay_score_the_same_dataset(monkeypatch, tmp_path):
    dataset(
        monkeypatch,
        tmp_path,
        "INTENT_PATH",
        [
            {"query": "你好", "intent": "greeting"},
            {"query": "礼品包装可用丝带吗", "intent": "faq_policy"},
        ],
    )

    rules = runner.eval_intent_rules()
    overlay = runner.eval_intent_overlay()

    assert rules == {
        "backend": "rules",
        "prompt_version": "intent_rules_v1",
        "n": 2,
        "accuracy": 1.0,
        "per_label": {"greeting": 1.0, "faq_policy": 1.0},
    }
    assert overlay == {
        "backend": "overlay",
        "prompt_version": "intent_overlay_v1",
        "n": 2,
        "uncovered": 1,
        "accuracy": 0.5,
        "per_label": {"greeting": 1.0, "faq_policy": 0.0},
    }


@pytest.mark.parametrize(
    ("version", "overlay", "prompt"),
    [
        ("intent_v1", False, INTENT_SYSTEM_V1),
        ("intent_v2", False, INTENT_SYSTEM_V2),
        ("intent_v3", True, INTENT_SYSTEM),
    ],
    ids=["v1", "v2", "v3-overlay"],
)
async def test_intent_llm_tracks_errors_and_wrong_predictions(
    monkeypatch, tmp_path, version, overlay, prompt
):
    dataset(
        monkeypatch,
        tmp_path,
        "INTENT_PATH",
        [
            {"query": "你好", "intent": "greeting"},
            {"query": "退款进度", "intent": "refund"},
            {"query": "怎么退货", "intent": "return_exchange"},
        ],
    )
    config = SimpleNamespace(llm_configured=True)
    provider = Mock()
    classify = AsyncMock(
        side_effect=[
            SimpleNamespace(label=IntentLabel.GREETING),
            RuntimeError("classification failed"),
            SimpleNamespace(label=IntentLabel.REFUND),
        ]
    )
    classifier_factory = Mock(return_value=SimpleNamespace(classify=classify))
    monkeypatch.setattr(settings, "get_settings", lambda: config)
    monkeypatch.setattr(deepseek, "DeepSeekProvider", provider)
    monkeypatch.setattr(intent, "IntentClassifier", classifier_factory)

    report = await runner.eval_intent_llm(version=version, overlay=overlay)

    provider.assert_called_once_with(config)
    classifier_factory.assert_called_once_with(
        provider.return_value, system_prompt=prompt, prompt_version=version, apply_overlay=overlay
    )
    assert classify.await_args_list == [call("你好"), call("退款进度"), call("怎么退货")]
    assert report == {
        "backend": "deepseek",
        "prompt_version": version,
        "overlay": overlay,
        "n": 3,
        "errors": 1,
        "accuracy": 0.3333,
        "per_label": {"greeting": 1.0, "refund": 0.0, "return_exchange": 0.0},
        "mismatches": [
            {"query": "退款进度", "gold": "refund", "pred": "ERROR"},
            {"query": "怎么退货", "gold": "return_exchange", "pred": "refund"},
        ],
    }


@pytest.mark.parametrize(
    ("evaluate", "kwargs"),
    [
        (runner.eval_intent_llm, {"version": "intent_v1", "overlay": False}),
        (runner.eval_generation, {}),
        (runner.eval_e2e, {}),
    ],
)
async def test_llm_eval_requires_configuration(monkeypatch, evaluate, kwargs):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "60")
    monkeypatch.setattr(settings, "get_settings", lambda: SimpleNamespace(llm_configured=False))
    monkeypatch.setattr(settings, "clear_settings_cache", Mock())
    provider = Mock()
    build_container = AsyncMock()
    monkeypatch.setattr(deepseek, "DeepSeekProvider", provider)
    monkeypatch.setattr(container, "build_container", build_container)

    with pytest.raises(SystemExit, match="DEEPSEEK_API_KEY missing"):
        await evaluate(**kwargs)

    provider.assert_not_called()
    build_container.assert_not_called()


@pytest.fixture
def retrieval_backends(monkeypatch):
    embedder = SimpleNamespace(backend="hash")
    bm25 = SimpleNamespace(search=Mock())
    store = object()
    retriever = AsyncMock()
    vector_search = AsyncMock()
    build_embedder = Mock(return_value=embedder)
    load_corpus = AsyncMock(return_value=(bm25, store, None))
    build_vector = Mock(return_value=SimpleNamespace(search=vector_search))
    build_hybrid = Mock(return_value=SimpleNamespace(retrieve=retriever))
    monkeypatch.setattr(embeddings, "build_embedding_provider", build_embedder)
    monkeypatch.setattr(ingest, "ingest_to_memory", load_corpus)
    monkeypatch.setattr(vector, "VectorRetriever", build_vector)
    monkeypatch.setattr(hybrid, "HybridRetriever", build_hybrid)
    return SimpleNamespace(
        bm25=bm25.search,
        vector=vector_search,
        hybrid=retriever,
        embedder=embedder,
        load_corpus=load_corpus,
        store=store,
        build_vector=build_vector,
        build_hybrid=build_hybrid,
    )


@pytest.mark.parametrize(
    ("available", "load_error", "reranker_label", "note_fragment"),
    [
        (True, None, "hybrid+cross-encoder", "tiny-reranker"),
        (False, "weights unavailable", "hybrid+lexical-fallback", "weights unavailable"),
        (False, None, "hybrid+lexical-fallback", 'pip install -e ".[rerank]"'),
    ],
)
async def test_retrieval_scores_ranked_documents_and_reranking(
    monkeypatch, tmp_path, retrieval_backends, available, load_error, reranker_label, note_fragment
):
    dataset(
        monkeypatch,
        tmp_path,
        "RAG_PATH",
        [
            {
                "id": "Q1",
                "query": "warranty",
                "relevant_doc_ids": ["warranty_policy", "warranty_faq"],
            }
        ],
    )
    config = SimpleNamespace(
        local_embedding_configured=False,
        cross_encoder_configured=True,
        cross_encoder_model="tiny-reranker",
    )
    monkeypatch.setattr(settings, "get_settings", lambda: config)
    candidates = [
        SimpleNamespace(document_id=doc_id)
        for doc_id in (
            "shipping_0",
            "shipping_1",
            "warranty_policy_0",
            "returns_0",
            "coupons_0",
            "warranty_faq_0",
        )
    ]
    retrieval_backends.bm25.return_value = candidates[:3]
    retrieval_backends.vector.return_value = []
    retrieval_backends.hybrid.side_effect = lambda query, top_k: candidates[:top_k]
    cross_encoder = SimpleNamespace(available=available, load_error=load_error)
    lexical = object()
    cross_encoder_factory = Mock(return_value=cross_encoder)
    lexical_factory = Mock(return_value=lexical)
    rerank = AsyncMock(return_value=[candidates[5], candidates[2]])
    reranker_factory = Mock(return_value=SimpleNamespace(rerank=rerank))
    monkeypatch.setattr(rerank_providers, "CrossEncoderRerankProvider", cross_encoder_factory)
    monkeypatch.setattr(rerank_providers, "LexicalRerankProvider", lexical_factory)
    monkeypatch.setattr(reranking, "DocumentReranker", reranker_factory)

    report = await runner.eval_retrieval()

    assert report["n"] == 1
    assert report["dense_backend"] == "hash-vector"
    assert report["cross_encoder_available"] is available
    assert report["cross_encoder_model"] == "tiny-reranker"
    assert note_fragment in report["note"]
    assert [system["backend"] for system in report["systems"]] == [
        "bm25",
        "hash-vector",
        "bm25+hash-vector+rrf",
        "hybrid-candidates8-no-rerank",
        reranker_label,
    ]
    bm25, dense, fused, control, reranked = report["systems"]
    assert bm25["predictions"] == [
        {"id": "Q1", "ranked_documents": ["shipping", "warranty_policy"]}
    ]
    assert bm25["mrr"] == {"mean": 0.5, "min": 0.5, "max": 0.5}
    assert dense["predictions"] == [{"id": "Q1", "ranked_documents": []}]
    assert fused["predictions"] == control["predictions"]
    assert "warranty_faq" not in control["predictions"][0]["ranked_documents"]
    assert reranked["predictions"] == [
        {"id": "Q1", "ranked_documents": ["warranty_faq", "warranty_policy"]}
    ]
    for metric in ("precision_at_1", "recall_at_3", "hit_at_1", "hit_at_3", "mrr"):
        assert reranked[metric] == {"mean": 1.0, "min": 1.0, "max": 1.0}
    retrieval_backends.load_corpus.assert_awaited_once_with(
        "knowledge", retrieval_backends.embedder
    )
    retrieval_backends.build_vector.assert_called_once_with(
        retrieval_backends.store, retrieval_backends.embedder
    )
    retrieval_backends.build_hybrid.assert_called_once_with(
        retrieval_backends.load_corpus.return_value[0],
        retrieval_backends.build_vector.return_value,
        rrf_k=60,
    )
    retrieval_backends.bm25.assert_called_once_with("warranty", top_k=5)
    retrieval_backends.vector.assert_awaited_once_with("warranty", top_k=5)
    assert retrieval_backends.hybrid.await_args_list == [
        call("warranty", top_k=5),
        call("warranty", top_k=8),
        call("warranty", top_k=8),
    ]
    cross_encoder_factory.assert_called_once_with(config)
    reranker_factory.assert_called_once_with(cross_encoder if available else lexical)
    assert lexical_factory.call_count == (0 if available else 1)
    rerank.assert_awaited_once_with("warranty", candidates, top_n=5)


async def test_generation_scores_evidence_and_only_corrects_unsupported_rag(
    monkeypatch, tmp_path, retrieval_backends
):
    dataset(
        monkeypatch,
        tmp_path,
        "RAG_PATH",
        [
            {"query": "warranty term", "gold_answer_contains": ["12 months"]},
            {"query": "shipping method", "gold_answer_contains": ["standard"]},
            {"query": "refund exception"},
        ],
    )
    config = SimpleNamespace(llm_configured=True)
    monkeypatch.setattr(settings, "get_settings", lambda: config)
    warranty_chunk = SimpleNamespace(chunk_id="w1", content="Coverage is 12 months.")
    shipping_chunk = SimpleNamespace(chunk_id="s1", content="Shipping is standard.")
    chunks_by_query = {
        "warranty term": [warranty_chunk],
        "shipping method": [shipping_chunk],
        "refund exception": [],
    }
    retrieval_backends.hybrid.side_effect = lambda query, top_k: chunks_by_query[query]

    def generate(messages, *, temperature, max_tokens, response_format):
        assert temperature == 0.0
        assert max_tokens == 500
        assert response_format == {"type": "json_object"}
        prompt = messages[1]["content"]
        if "warranty term" in prompt:
            answer = (
                "24 months"
                if "(无检索结果)" in prompt
                else "Warranty is 12 months with free repair."
            )
            if answer.startswith("Warranty"):
                assert "[w1] Coverage is 12 months." in prompt
        elif "shipping method" in prompt:
            answer = "standard"
        else:
            assert "(无检索结果)" in prompt
            answer = "Unknown"
        return SimpleNamespace(content=json.dumps({"answer": answer}))

    llm = SimpleNamespace(generate=AsyncMock(side_effect=generate))
    monkeypatch.setattr(deepseek, "DeepSeekProvider", lambda _: llm)

    def judge(*, query, answer, chunks, tool_results_text):
        assert chunks == chunks_by_query[query]
        assert tool_results_text == ""
        return hallucination.FaithfulnessResult(supported=answer == "standard")

    checker = SimpleNamespace(check=AsyncMock(side_effect=judge))
    monkeypatch.setattr(hallucination, "FaithfulnessChecker", lambda _: checker)

    def correct(**kwargs):
        assert not kwargs["first_result"].supported
        if kwargs["query"] == "warranty term":
            assert kwargs["answer"] == "Warranty is 12 months with free repair."
            return "Warranty is 12 months.", hallucination.FaithfulnessResult(supported=True), None
        return "Unknown", kwargs["first_result"], None

    correction = AsyncMock(side_effect=correct)
    monkeypatch.setattr(faithfulness, "self_correct_answer", correction)

    report = await runner.eval_generation()

    assert report["n"] == 3
    assert report["naive_no_context"] == {
        "faithfulness": 0.3333,
        "hallucination_rate": 0.6667,
        "keyword_hit": 0.6667,
    }
    assert report["rag"] == {
        "faithfulness": 0.3333,
        "hallucination_rate": 0.6667,
        "keyword_hit": 1.0,
    }
    assert report["rag_plus_self_check"] == {
        "faithfulness": 0.6667,
        "hallucination_rate": 0.3333,
        "keyword_hit": 1.0,
    }
    assert report["hallucination_relative_drop"] == 0.5
    assert llm.generate.await_count == 6
    assert checker.check.await_count == 6
    assert {request.kwargs["query"] for request in correction.await_args_list} == {
        "warranty term",
        "refund exception",
    }


@pytest.fixture
def offline_container(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "60")
    config = SimpleNamespace(llm_configured=False)
    clear_cache = Mock()
    chat = AsyncMock()
    close = AsyncMock()
    build = AsyncMock(
        return_value=SimpleNamespace(chat_service=SimpleNamespace(chat=chat), aclose=close)
    )
    monkeypatch.setattr(settings, "get_settings", lambda: config)
    monkeypatch.setattr(settings, "clear_settings_cache", clear_cache)
    monkeypatch.setattr(container, "build_container", build)
    return SimpleNamespace(
        config=config, clear_cache=clear_cache, chat=chat, close=close, build=build
    )


async def test_e2e_scores_refusals_keywords_tools_and_latency(
    monkeypatch, tmp_path, offline_container
):
    dataset(
        monkeypatch,
        tmp_path,
        "E2E_PATH",
        [
            {
                "id": "refusal",
                "query": "off-topic",
                "expect_refused": True,
                "expect_keywords": ["转人工"],
            },
            {"id": "tool", "query": "取消订单", "user_id": "buyer", "expect_keywords": ["已取消"]},
            {"id": "missing", "query": "库存", "expect_refused": True},
        ],
    )
    offline_container.chat.side_effect = [
        ChatResponse(
            answer="请转人工",
            session_id="s1",
            trace_id="eval_refusal",
            refused=True,
            strategy=RetrievalStrategy.REFUSE,
            intent=IntentLabel.OUT_OF_SCOPE,
            latency=LatencyBreakdown(total_ms=100, intent_ms=5),
        ),
        ChatResponse(
            answer="请确认",
            session_id="s2",
            trace_id="eval_tool",
            pending_action=ActionProposal(
                action_id="a1",
                tool_name="cancel_order",
                arguments={},
                risk=ToolRisk.WRITE,
                summary="请确认",
            ),
            latency=LatencyBreakdown(total_ms=300, tool_ms=200, llm_ms=60),
        ),
        ChatResponse(
            answer="暂时没找到",
            session_id="s3",
            trace_id="eval_missing",
            latency=LatencyBreakdown(retrieval_ms=20),
        ),
    ]
    timestamps = iter([0, 0, 1, 1, 2, 2.25])
    monkeypatch.setattr(runner, "time", SimpleNamespace(perf_counter=lambda: next(timestamps)))

    report = await runner.eval_e2e(allow_offline=True)

    offline_container.clear_cache.assert_called_once_with()
    offline_container.build.assert_awaited_once_with(offline_container.config, seed=True)
    assert offline_container.chat.await_args_list == [
        call(ChatRequest(message="off-topic", user_id="u_demo"), trace_id="eval_refusal"),
        call(ChatRequest(message="取消订单", user_id="buyer"), trace_id="eval_tool"),
        call(ChatRequest(message="库存", user_id="u_demo"), trace_id="eval_missing"),
    ]
    offline_container.close.assert_awaited_once_with()
    assert report["n"] == 3
    assert report["refuse_accuracy"] == report["keyword_hit"] == report["tool_path_hit"] == 0.5
    assert report["latency_ms"] == {
        "p50": 250.0,
        "p95": 300.0,
        "mean": 216.7,
        "intent_p50": 0.0,
        "retrieval_p50": 0.0,
        "llm_p50": 0.0,
        "tool_p50": 0.0,
    }
    assert report["cases"] == [
        {
            "id": "refusal",
            "refused": True,
            "strategy": "refuse",
            "intent": "out_of_scope",
            "latency_ms": 100.0,
        },
        {"id": "tool", "refused": False, "strategy": None, "intent": None, "latency_ms": 300.0},
        {"id": "missing", "refused": False, "strategy": None, "intent": None, "latency_ms": 250.0},
    ]


async def test_e2e_closes_container_when_chat_fails(monkeypatch, tmp_path, offline_container):
    dataset(monkeypatch, tmp_path, "E2E_PATH", [{"id": "failure", "query": "warranty"}])
    offline_container.chat.side_effect = RuntimeError("chat unavailable")

    with pytest.raises(RuntimeError, match="chat unavailable"):
        await runner.eval_e2e(allow_offline=True)

    offline_container.close.assert_awaited_once_with()


@pytest.mark.parametrize(
    ("task", "use_llm", "result_keys"),
    [
        (
            "all",
            False,
            {"generated_at", "intent_rules", "intent_overlay", "retrieval", "resume_bullets"},
        ),
        (
            "all",
            True,
            {
                "generated_at",
                "intent_rules",
                "intent_overlay",
                "retrieval",
                "resume_bullets",
                "intent_v1",
                "intent_v2",
                "intent_v3",
                "generation",
                "e2e",
            },
        ),
        ("intent", False, {"intent_rules", "intent_overlay", "resume_bullets"}),
        (
            "intent",
            True,
            {
                "intent_rules",
                "intent_overlay",
                "intent_v1",
                "intent_v2",
                "intent_v3",
                "resume_bullets",
            },
        ),
        ("retrieval", False, {"retrieval"}),
        ("generation", True, {"generation"}),
        ("e2e", True, {"e2e"}),
    ],
    ids=[
        "all-offline",
        "all-llm",
        "intent-offline",
        "intent-llm",
        "retrieval",
        "generation",
        "e2e",
    ],
)
def test_cli_selects_tasks_and_writes_identical_report_files(
    monkeypatch, tmp_path, capsys, task, use_llm, result_keys
):
    output = tmp_path / "reports"
    monkeypatch.setattr(runner, "REPORT_DIR", output)
    monkeypatch.setattr(sys, "argv", ["runner", "--task", task] + (["--llm"] if use_llm else []))
    monkeypatch.setattr(runner, "eval_intent_rules", Mock(return_value={"accuracy": 1.0}))
    monkeypatch.setattr(runner, "eval_intent_overlay", Mock(return_value={"accuracy": 0.75}))
    llm_intent = AsyncMock(return_value={"n": 4, "accuracy": 0.5})
    retrieval = AsyncMock(return_value={"systems": []})
    generation = AsyncMock(return_value={})
    e2e = AsyncMock(return_value={})
    monkeypatch.setattr(runner, "eval_intent_llm", llm_intent)
    monkeypatch.setattr(runner, "eval_retrieval", retrieval)
    monkeypatch.setattr(runner, "eval_generation", generation)
    monkeypatch.setattr(runner, "eval_e2e", e2e)

    runner.main()

    (stamped,) = output.glob(f"{task}_*.json")
    latest = (output / "latest.json").read_text(encoding="utf-8")
    assert latest == stamped.read_text(encoding="utf-8")
    assert set(json.loads(latest)) == result_keys
    assert capsys.readouterr().out == f"{latest}\n\nWrote {stamped}\n"
    if "intent_v3" in result_keys:
        assert llm_intent.await_args_list == [
            call(version="intent_v1", overlay=False),
            call(version="intent_v2", overlay=False),
            call(version="intent_v3", overlay=True),
        ]
    else:
        llm_intent.assert_not_awaited()
    assert retrieval.await_count == int("retrieval" in result_keys)
    assert generation.await_count == int("generation" in result_keys)
    assert e2e.await_count == int("e2e" in result_keys)
