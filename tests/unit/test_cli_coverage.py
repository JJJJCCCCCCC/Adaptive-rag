import json
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from app.cli import ChatCli, ChatFailed, client, commands, main
from app.schemas.chat import ChatRequest, ChatResponse
from app.services import container


def _cli(*, stream=False, show_tokens=False, local=False):
    return ChatCli(
        user_id="test-user",
        stream=stream,
        show_tokens=show_tokens,
        timeout=17.0,
        base_url="https://api.example.test/",
        local=local,
    )


def _mock_http_clients(monkeypatch, handler):
    client_class = httpx.AsyncClient
    transport = httpx.MockTransport(handler)
    factory = Mock(side_effect=lambda **kwargs: client_class(transport=transport, **kwargs))
    monkeypatch.setattr(client.httpx, "AsyncClient", factory)
    return factory


async def test_cli_repl_preserves_sessions_and_recovers_after_errors(monkeypatch, capsys):
    requests = []
    answers = iter(
        [
            httpx.Response(
                200,
                json={
                    "answer": "Action can be confirmed",
                    "session_id": "session-1",
                    "pending_action": {"action_id": "action-1", "summary": "Cancel order"},
                    "citations": [
                        {"section": "Returns", "title": "Policies"},
                        {"title": "Shipping"},
                        {"chunk_id": "chunk-1"},
                        {"title": "Fourth"},
                        {"title": "Hidden"},
                    ],
                },
            ),
            httpx.Response(409, json={"error_code": "STALE_ACTION", "message": "Action expired"}),
            httpx.ConnectError("connection reset"),
            httpx.Response(200, text="plain text answer"),
        ]
    )

    def handler(request):
        if request.url.path == "/api/v1/health":
            assert request.method == "GET"
            return httpx.Response(200)
        assert request.url == "https://api.example.test/api/v1/chat"
        assert request.method == "POST"
        requests.append(json.loads(request.content))
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return answer

    factory = _mock_http_clients(monkeypatch, handler)
    prompts = iter(
        ["  ", "/help", "/y", "  cancel my order ", "/y", "/n", "retry", "/new", "fresh", "/q"]
    )
    monkeypatch.setattr("builtins.input", lambda _: next(prompts))
    cli = _cli()
    try:
        await cli.loop()
    finally:
        await cli.aclose()

    assert [call.kwargs["timeout"] for call in factory.call_args_list] == [2.0, 17.0]
    assert [
        (
            request["message"],
            request["user_id"],
            request["session_id"],
            request["confirm_action_id"],
        )
        for request in requests
    ] == [
        ("cancel my order", "test-user", None, None),
        ("确认", "test-user", "session-1", "action-1"),
        ("retry", "test-user", "session-1", None),
        ("fresh", "test-user", None, None),
    ]
    output = capsys.readouterr().out
    assert "已连接 https://api.example.test" in output
    assert "命令说明" in output
    assert "当前没有待确认操作" in output
    assert "引用: Returns, Shipping, chunk-1, Fourth" in output
    assert "Hidden" not in output
    assert "待确认: Cancel order" in output
    assert "请求失败 [STALE_ACTION]: Action expired" in output
    assert "已忽略待确认" in output
    assert "出错了：connection reset" in output
    assert "已新开会话" in output
    assert "plain text answer" in output
    assert output.rstrip().endswith("再见。")


@pytest.mark.parametrize("show_tokens", [False, True])
async def test_cli_stream_renders_progress_and_returns_final_answer(
    monkeypatch, capsys, show_tokens
):
    response = "\n".join(
        [
            ": heartbeat",
            "data:",
            "data: []",
            'data: {"event": "node", "node": "classify_intent"}',
            'data: {"event": "node", "node": "custom_node"}',
            'data: {"event": "token", "text": "partial"}',
            'data: {"event": "answer", "data": null}',
            'data: {"event": "answer", "data": {"answer": "final", "session_id": "session-1"}}',
            "data: [DONE]",
        ]
    )

    def respond(request):
        if request.method == "GET":
            return httpx.Response(200)
        assert request.url.path == "/api/v1/chat/stream"
        assert json.loads(request.content)["message"] == "question"
        return httpx.Response(200, text=response, headers={"Content-Type": "text/event-stream"})

    _mock_http_clients(monkeypatch, respond)
    cli = _cli(stream=True, show_tokens=show_tokens)
    await cli.setup()
    try:
        assert await cli.chat("question") == {"answer": "final", "session_id": "session-1"}
    finally:
        await cli.aclose()
    token = "partial" if show_tokens else ""
    assert capsys.readouterr().out == f"  [意图][custom_node]{token}\n"


@pytest.mark.parametrize(
    ("stream", "response", "error"),
    [
        (True, httpx.Response(500, content=b"unavailable\xff"), r"\[500\]: unavailable�"),
        (True, httpx.Response(200, text="data: [DONE]\n"), "没有拿到完整回答"),
        (
            False,
            httpx.Response(502, text="upstream unavailable"),
            r"\[BAD_RESPONSE\]: upstream unavailable",
        ),
        (
            False,
            httpx.Response(422, json={"answer": "invalid request"}),
            r"\[422\]: invalid request",
        ),
    ],
)
async def test_cli_reports_http_and_incomplete_stream_errors(monkeypatch, stream, response, error):
    _mock_http_clients(
        monkeypatch, lambda request: httpx.Response(200) if request.method == "GET" else response
    )
    cli = _cli(stream=stream)
    await cli.setup()
    try:
        with pytest.raises(ChatFailed, match=error):
            await cli.chat("question")
    finally:
        await cli.aclose()


@pytest.mark.parametrize(("local", "stream"), [(False, False), (False, True), (True, True)])
async def test_cli_starts_local_engine_when_forced_or_health_check_fails(
    monkeypatch, capsys, local, stream
):
    def fail_health_check(request):
        assert request.url.path == "/api/v1/health"
        if stream:
            raise httpx.ConnectError("connection refused", request=request)
        return httpx.Response(503)

    factory = _mock_http_clients(monkeypatch, fail_health_check)
    answer = ChatResponse(answer="local answer", session_id="local-session", trace_id="test-trace")

    async def events():
        yield {"event": "answer", "data": answer.model_dump(mode="json")}

    service = Mock(chat=AsyncMock(return_value=answer), stream=Mock(return_value=events()))
    engine = Mock(chat_service=service, aclose=AsyncMock())
    build = AsyncMock(return_value=engine)
    monkeypatch.setattr(container, "build_container", build)
    cli = _cli(stream=stream, local=local)
    assert "引擎已就绪" in await cli.setup()
    try:
        assert await cli.chat("question", confirm="action-2") == answer.model_dump(mode="json")
    finally:
        await cli.aclose()

    assert factory.call_count == (0 if local else 1)
    assert "正在本进程启动引擎" in capsys.readouterr().out
    build.assert_awaited_once()
    engine.aclose.assert_awaited_once()
    method = service.stream if stream else service.chat
    assert method.call_args.args == (
        ChatRequest(user_id="test-user", message="question", confirm_action_id="action-2"),
    )
    assert method.call_args.kwargs["trace_id"]
    assert method.call_count == 1


@pytest.mark.parametrize("exception", [EOFError, KeyboardInterrupt])
async def test_cli_exits_on_interrupted_input(monkeypatch, capsys, exception):
    cli = _cli()
    monkeypatch.setattr(cli, "setup", AsyncMock(return_value="ready"))
    monkeypatch.setattr("builtins.input", Mock(side_effect=exception))
    await cli.loop()
    assert capsys.readouterr().out.rstrip().endswith("再见。")


@pytest.mark.parametrize("fail", [False, True])
def test_cli_entrypoint_honors_arguments_and_closes_client(monkeypatch, fail):
    cli = Mock(
        loop=AsyncMock(side_effect=RuntimeError("setup failed") if fail else None),
        aclose=AsyncMock(),
    )
    factory = Mock(return_value=cli)
    monkeypatch.setattr(commands, "ChatCli", factory)
    argv = [
        "--base-url",
        "https://api.example.test",
        "--user-id",
        "test-user",
        "--local",
        "--no-stream",
        "--tokens",
        "--timeout",
        "4.5",
    ]
    if fail:
        with pytest.raises(RuntimeError, match="setup failed"):
            main(argv)
    else:
        assert main(argv) == 0
    factory.assert_called_once_with(
        base_url="https://api.example.test",
        user_id="test-user",
        local=True,
        stream=False,
        show_tokens=True,
        timeout=4.5,
    )
    cli.loop.assert_awaited_once()
    cli.aclose.assert_awaited_once()
