import asyncio
import json
import os
from pathlib import Path

import httpx
from pydantic_settings import BaseSettings, SettingsConfigDict


class Credentials(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")
    deepseek_api_key: str
    deepseek_model: str = "deepseek-chat"


async def main() -> None:
    credentials = Credentials(_env_file=Path(os.environ["RAG_EVAL_MODEL_CONFIG"]))
    async with httpx.AsyncClient(
        base_url="https://api.deepseek.com",
        trust_env=os.environ.get("RAG_EVAL_DIRECT") != "1",
        timeout=12,
        follow_redirects=False,
    ) as client:
        try:
            response = await client.post(
                "/chat/completions",
                headers={"Authorization": "Bearer " + credentials.deepseek_api_key},
                json={
                    "model": credentials.deepseek_model,
                    "messages": [
                        {"role": "user", "content": "仅输出数字 24681，不加解释。"}
                    ],
                    "temperature": 0,
                    "max_tokens": 24,
                },
            )
            if response.status_code != 200:
                print("http", response.status_code)
                raise SystemExit(1)
            body = response.json()
            assert body["choices"][0]["message"]["content"].strip() == "24681"
            receipt = {key: body[key] for key in ("id", "usage", "model")}
            print(json.dumps(receipt))
            Path(__file__).with_name("preflight.json").write_text(
                json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
            )
        except httpx.HTTPError as exc:
            print(type(exc).__name__)
            raise SystemExit(1) from None


if __name__ == "__main__":
    asyncio.run(main())
