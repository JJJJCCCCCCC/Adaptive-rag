"""Verify an authenticated chat completion at the official DeepSeek endpoint."""

import argparse
import asyncio
import json
import secrets

import httpx

from app.config.settings import get_settings


def failure_message(exc):
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        hints = {
            401: "请检查本机项目配置的 DEEPSEEK_API_KEY。",
            402: "请检查 DeepSeek 开放平台账户余额。",
            407: "请检查所在网络的代理认证配置。",
        }
        return f"HTTP {status}。{hints.get(status, '请按该 HTTP 状态检查服务端或网关。')}"
    return (
        f"{type(exc).__name__}，未收到完整 HTTP 响应。"
        "请在能够通过 HTTPS 访问 api.deepseek.com:443 的环境中重试；"
        "使用受管网络时，请确认出站规则允许该域名。"
    )


async def verify(settings):
    try:
        return await _complete_challenge(settings)
    except httpx.HTTPError as exc:
        raise RuntimeError(f"DeepSeek 预检失败：{failure_message(exc)}") from None


async def _complete_challenge(settings):
    if settings.deepseek_base_url.rstrip("/") != "https://api.deepseek.com":
        raise ValueError("Expected official DeepSeek endpoint")
    key = settings.deepseek_api_key.get_secret_value()
    if not key:
        raise RuntimeError("Missing DeepSeek API credential")
    challenge = secrets.token_hex(4)
    async with httpx.AsyncClient(trust_env=True, timeout=20, follow_redirects=False) as client:
        response = await client.post(
            "https://api.deepseek.com/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": settings.deepseek_model,
                "messages": [{"role": "user", "content": f"Reply with exactly {challenge}"}],
                "temperature": 0,
                "max_tokens": 64,
            },
        )
    response.raise_for_status()
    payload = response.json()
    if (
        response.status_code != 200
        or payload["choices"][0]["message"]["content"].strip() != challenge
        or not payload.get("model")
        or not payload.get("id")
        or payload["usage"]["total_tokens"] <= 0
    ):
        raise RuntimeError("Invalid DeepSeek completion or usage receipt")
    return {k: payload[k] for k in ("id", "model", "usage")}


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    try:
        print(json.dumps(asyncio.run(verify(get_settings()))))
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None
