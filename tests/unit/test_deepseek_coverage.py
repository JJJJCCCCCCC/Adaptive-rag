import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from app.config.settings import Settings
from app.core import retry
from app.core.error_codes import ErrorCode
from app.core.exceptions import LLMProviderError
from app.providers.deepseek import DeepSeekProvider

BASE_URL = "https://deepseek.example.test/v1"
API_KEY = "dummy-test-key"
MESSAGES = [{"role": "user", "content": "Return a JSON object"}]
RESPONSE_FORMAT = {"type": "json_object"}


def _settings(*, api_key=API_KEY, max_attempts=1):
    return Settings(
        _env_file=None,
        deepseek_base_url=BASE_URL,
        deepseek_api_key=api_key,
        deepseek_model="test-chat-model",
        llm_max_retries=max_attempts,
    )


def _mock_client(handler):
    return httpx.AsyncClient(base_url=BASE_URL, transport=httpx.MockTransport(handler))


async def test_deepseek_generate_sends_options_and_parses_usage():
    def respond(request):
        assert request.method == "POST"
        assert request.url == f"{BASE_URL}/chat/completions"
        assert request.headers["Authorization"] == f"Bearer {API_KEY}"
        assert json.loads(request.content) == {
            "model": "test-chat-model",
            "messages": MESSAGES,
            "temperature": 0.25,
            "max_tokens": 128,
            "response_format": RESPONSE_FORMAT,
        }
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"result": true}'}}],
                "usage": {"prompt_tokens": "3", "completion_tokens": 5, "total_tokens": "8"},
            },
        )

    async with _mock_client(respond) as client:
        provider = DeepSeekProvider(_settings(), client)
        assert await provider.ping()
        result = await provider.generate(
            MESSAGES, temperature=0.25, max_tokens=128, response_format=RESPONSE_FORMAT
        )
        await provider.aclose()
        assert not client.is_closed

    assert result.content == '{"result": true}'
    assert result.model == "test-chat-model"
    assert result.usage.model_dump() == {
        "prompt_tokens": 3,
        "completion_tokens": 5,
        "total_tokens": 8,
    }


async def test_deepseek_retries_transient_failures_before_returning_a_response(monkeypatch):
    sleep = AsyncMock()
    monkeypatch.setattr(retry, "asyncio", SimpleNamespace(sleep=sleep))
    requests = []
    responses = iter(
        [
            httpx.Response(429),
            httpx.Response(503),
            httpx.Response(200, json={"choices": []}),
            httpx.Response(200, json={"choices": [{"message": {"content": None}}]}),
        ]
    )

    def respond(request):
        requests.append(json.loads(request.content))
        return next(responses)

    async with _mock_client(respond) as client:
        result = await DeepSeekProvider(_settings(max_attempts=4), client).generate(MESSAGES)

    assert sleep.await_count == 3
    assert (
        requests
        == [
            {
                "model": "test-chat-model",
                "messages": MESSAGES,
                "temperature": 0.0,
                "max_tokens": 1024,
            }
        ]
        * 4
    )
    assert result.content == ""
    assert result.usage.total_tokens == 0


async def test_deepseek_stream_yields_only_content_deltas():
    def respond(request):
        assert request.method == "POST"
        assert request.url == f"{BASE_URL}/chat/completions"
        assert request.headers["Authorization"] == f"Bearer {API_KEY}"
        assert json.loads(request.content) == {
            "model": "test-chat-model",
            "messages": MESSAGES,
            "temperature": 0.3,
            "max_tokens": 64,
            "stream": True,
            "response_format": RESPONSE_FORMAT,
        }
        return httpx.Response(
            200,
            text="\n".join(
                [
                    ": heartbeat",
                    "data:  ",
                    "data: not-json",
                    'data: {"choices": []}',
                    'data: {"choices": [{"delta": {"role": "assistant"}}]}',
                    'data: {"choices": [{"delta": {"content": "first"}}]}',
                    'data: {"choices": [{"delta": {"content": " second"}}]}',
                    "data: [DONE]",
                ]
            ),
            headers={"Content-Type": "text/event-stream"},
        )

    async with _mock_client(respond) as client:
        provider = DeepSeekProvider(_settings(), client)
        chunks = [
            chunk
            async for chunk in provider.stream(
                MESSAGES, temperature=0.3, max_tokens=64, response_format=RESPONSE_FORMAT
            )
        ]

    assert chunks == ["first", " second"]


async def test_deepseek_missing_key_stops_requests_and_marks_readiness_false():
    def unexpected_request(request):
        pytest.fail(f"Unexpected request to {request.url}")

    async with _mock_client(unexpected_request) as client:
        provider = DeepSeekProvider(_settings(api_key=""), client)
        assert not await provider.ping()
        with pytest.raises(LLMProviderError, match="API key is not configured"):
            await provider.generate(MESSAGES)
        with pytest.raises(LLMProviderError, match="API key is not configured"):
            await anext(provider.stream(MESSAGES))


@pytest.mark.parametrize("stream", [False, True], ids=["generate", "stream"])
@pytest.mark.parametrize(
    ("status", "code", "retryable", "message"),
    [
        (400, ErrorCode.LLM_PROVIDER_UNAVAILABLE, False, "HTTP 400"),
        (401, ErrorCode.LLM_PROVIDER_UNAVAILABLE, False, "HTTP 401"),
        (403, ErrorCode.LLM_PROVIDER_UNAVAILABLE, False, "HTTP 403"),
        (422, ErrorCode.LLM_PROVIDER_UNAVAILABLE, False, "HTTP 422"),
        (429, ErrorCode.LLM_RATE_LIMITED, True, "rate limited"),
        (503, ErrorCode.LLM_PROVIDER_UNAVAILABLE, True, "HTTP 503"),
        (418, ErrorCode.LLM_PROVIDER_UNAVAILABLE, True, "HTTP 418"),
    ],
)
async def test_deepseek_maps_http_errors(stream, status, code, retryable, message):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(status, json={"detail": "upstream details"})

    max_attempts = 3 if not retryable else 1
    async with _mock_client(respond) as client:
        provider = DeepSeekProvider(_settings(max_attempts=max_attempts), client)
        with pytest.raises(LLMProviderError, match=message) as caught:
            if stream:
                await anext(provider.stream(MESSAGES))
            else:
                await provider.generate(MESSAGES)

    assert caught.value.error_code is code
    assert caught.value.retryable is retryable
    assert len(requests) == 1


@pytest.mark.parametrize("stream", [False, True], ids=["generate", "stream"])
@pytest.mark.parametrize(
    ("exception", "code", "message"),
    [
        (httpx.ReadTimeout, ErrorCode.LLM_TIMEOUT, "timed out"),
        (httpx.ConnectError, ErrorCode.LLM_PROVIDER_UNAVAILABLE, "transport error"),
    ],
)
async def test_deepseek_maps_transport_errors(stream, exception, code, message):
    def respond(request):
        raise exception("offline", request=request)

    async with _mock_client(respond) as client:
        provider = DeepSeekProvider(_settings(), client)
        with pytest.raises(LLMProviderError, match=message) as caught:
            if stream:
                await anext(provider.stream(MESSAGES))
            else:
                await provider.generate(MESSAGES)

    assert caught.value.error_code is code
    assert caught.value.retryable
    assert isinstance(caught.value.__cause__, exception)
