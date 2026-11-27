"""Standalone synthetic policy QA and judge benchmark. Sends only the fixtures below."""

import asyncio
import json
import time
from pathlib import Path

import httpx

from app.config.settings import get_settings

# Invented here solely for this benchmark; the program never reads knowledge/ or model answers from other runs.
CASES = [
    (
        "s1",
        "蓝鸥玩具城的银票将在领取后第17天失效。",
        "银票领取后第几天失效？",
        ["17"],
        "第17天。",
        "第27天。",
    ),
    (
        "s2",
        "云帆俱乐部的铜船票一次最多兑换3盒纸星星。",
        "铜船票一次最多兑换几盒？",
        ["3"],
        "3盒。",
        "13盒。",
    ),
    (
        "s3",
        "松针集市的雾灯箱押金为23枚贝币。",
        "一个雾灯箱押金多少枚贝币？",
        ["23"],
        "23枚贝币。",
        "32枚贝币。",
    ),
    (
        "s4",
        "星桥手作社的空盒按每只4枚蓝币回收，带盖的按每只9枚蓝币回收。",
        "空盒和带盖盒回收价各是多少？请按这个顺序回答。",
        ["4", "9"],
        "空盒4枚，带盖9枚。",
        "空盒9枚，带盖4枚。",
    ),
    (
        "s5",
        "池月小站的月票限住客购买。日票不限身份。",
        "月票和日票各有什么购买身份限制？",
        ["住客", "不限"],
        "月票只限住客，日票不限身份。",
        "日票只限住客，月票不限身份。",
    ),
    (
        "s6",
        "风车馆在晴日于14时送件，雨日推迟至18时送件。",
        "晴日和雨日分别几点送件？请按这个顺序回答。",
        ["14", "18"],
        "晴日14时，雨日18时。",
        "雨日14时，晴日18时。",
    ),
]
OUT = Path("eval/results/nov26/deepseek-live/synthetic-policy.json")


async def main():
    settings = get_settings()
    key = settings.deepseek_api_key.get_secret_value()
    if not key or settings.deepseek_base_url.rstrip("/") != "https://api.deepseek.com":
        raise RuntimeError("Expected configured DeepSeek API")
    gate = asyncio.Semaphore(3)
    async with httpx.AsyncClient(
        base_url="https://api.deepseek.com",
        trust_env=True,
        timeout=45,
        headers={"Authorization": "Bearer " + key},
    ) as client:

        async def call(case_id, kind, prompt):
            async with gate:
                started = time.perf_counter()
                r = await client.post(
                    "/chat/completions",
                    json={
                        "model": settings.deepseek_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0,
                        "max_tokens": 160,
                        "response_format": {"type": "json_object"},
                    },
                )
                r.raise_for_status()
                body = r.json()
                return {
                    "id": case_id,
                    "kind": kind,
                    "model": body["model"],
                    "completion_id": body["id"],
                    "usage": body["usage"],
                    "ms": round((time.perf_counter() - started) * 1000),
                    "output": json.loads(body["choices"][0]["message"]["content"]),
                }

        tasks = []
        for ident, policy, q, _facts, good, bad in CASES:
            for kind, context in [("without_evidence", "未提供规则"), ("with_evidence", policy)]:
                tasks.append(
                    call(
                        ident,
                        kind,
                        '虚构规则问答。只凭给定规则作答，缺少证据时回答“不知道”。数字使用阿拉伯数字。只输出 JSON {"answer":"..."}。\n规则：'
                        + context
                        + "\n问题："
                        + q,
                    )
                )
            for kind, candidate in [("valid", good), ("invalid", bad)]:
                tasks.append(
                    call(
                        ident,
                        kind,
                        '对虚构规则作判断。核对数值、主体和条件；答复是否有证据支持？只输出 JSON {"supported":true或false,"reason":"简短原因"}。\n规则：'
                        + policy
                        + "\n问题："
                        + q
                        + "\n答复："
                        + candidate,
                    )
                )
        values = await asyncio.gather(*tasks, return_exceptions=True)
    result = {
        "model_requested": settings.deepseek_model,
        "fixtures": CASES,
        "records": [],
        "errors": [],
    }
    for i, value in enumerate(values):
        if isinstance(value, BaseException):
            result["errors"].append(
                {"id": CASES[i // 4][0], "arm": i % 4, "error": type(value).__name__}
            )
        else:
            result["records"].append(value)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    print("Completed", len(result["records"]), "errors", result["errors"])


if __name__ == "__main__":
    asyncio.run(main())
