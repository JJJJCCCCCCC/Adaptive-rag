"""Exercise the built React app and real isolated FastAPI server using a fresh Chrome profile."""

import asyncio
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
from websockets.asyncio.client import connect


def port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def wait_for(fn):
    deadline = time.monotonic() + 25
    while time.monotonic() < deadline:
        try:
            value = await fn()
            if value:
                return value
        except (httpx.HTTPError, IndexError, OSError):
            pass
        await asyncio.sleep(0.15)
    raise TimeoutError(f"Local browser test readiness timeout: {fn.__name__}")


async def main(out):
    chrome = (
        os.environ.get("CHROME_BIN")
        or shutil.which("google-chrome")
        or r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    )
    scratch = Path("scratch").resolve()
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=scratch, prefix="rag-browser-", ignore_cleanup_errors=True
    ) as tmp:
        root = Path(tmp)
        api = port()
        debug = port()
        env = {
            **os.environ,
            "APP_ENV": "testing",
            "DATABASE_URL": "sqlite+aiosqlite:///" + (root / "test.db").as_posix(),
            "DEEPSEEK_API_KEY": "",
            "EMBEDDING_API_KEY": "",
            "REDIS_URL": "",
            "QDRANT_URL": "",
        }
        procs = []
        async with httpx.AsyncClient(trust_env=False, timeout=2) as http:

            async def ready_api():
                return (await http.get(f"http://127.0.0.1:{api}/api/v1/health")).is_success

            async def page():
                return next(
                    (
                        t["webSocketDebuggerUrl"]
                        for t in (await http.get(f"http://127.0.0.1:{debug}/json/list")).json()
                        if t["type"] == "page" and t["url"].startswith(f"http://127.0.0.1:{api}/")
                    ),
                    None,
                )

            try:
                procs.append(
                    subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "uvicorn",
                            "app.main:create_app",
                            "--factory",
                            "--port",
                            str(api),
                        ],
                        env=env,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                )
                await wait_for(ready_api)
                procs.append(
                    subprocess.Popen(
                        [
                            chrome,
                            "--headless=new",
                            "--disable-gpu",
                            "--disable-background-networking",
                            "--no-first-run",
                            "--no-default-browser-check",
                            f"--user-data-dir={root/'profile'}",
                            f"--remote-debugging-port={debug}",
                            f"http://127.0.0.1:{api}/",
                        ],
                        stdout=subprocess.DEVNULL,
                    )
                )
                async with connect(await wait_for(page), proxy=None) as ws:
                    seq = 0

                    async def js(expression):
                        nonlocal seq
                        seq += 1
                        await ws.send(
                            json.dumps(
                                {
                                    "id": seq,
                                    "method": "Runtime.evaluate",
                                    "params": {"expression": expression, "returnByValue": True},
                                }
                            )
                        )
                        while True:
                            data = json.loads(await ws.recv())
                            if data.get("id") == seq:
                                return data.get("result", {}).get("result", {}).get("value")

                    await wait_for(lambda: js("!!document.querySelector('textarea')"))
                    await js(
                        """(() => {const el=document.querySelector('textarea'); Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype,'value').set.call(el,'你好');el.dispatchEvent(new Event('input',{bubbles:true}));return true})()"""
                    )
                    t = time.perf_counter()
                    await wait_for(
                        lambda: js(
                            """(() => {const b=document.querySelector('button[aria-label="发送消息"]:not([disabled])'); if(b){b.click();return true} return false})()"""
                        )
                    )
                    marker = "我可以帮你查看商品政策"
                    await wait_for(lambda: js(f"document.body.innerText.includes('{marker}')"))
                    elapsed = round((time.perf_counter() - t) * 1000, 1)
                    await ws.send(json.dumps({"id": 99999, "method": "Page.reload"}))
                    await asyncio.sleep(0.4)
                    await wait_for(lambda: js(f"document.body.innerText.includes('{marker}')"))
                    results = {
                        "built_ui_loaded": True,
                        "typed_and_sent": True,
                        "actual_sse_reply_rendered": True,
                        "reply_after_reload": True,
                        "send_to_visible_ms": elapsed,
                    }
                    Path(out).write_text(json.dumps(results, indent=2), encoding="utf8")
                    print(results)
            finally:
                for p in reversed(procs):
                    p.terminate()
                    p.wait(timeout=15)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
