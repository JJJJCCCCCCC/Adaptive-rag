"""Exercise the actual local HTTP tool proxy and SSE route with synthetic data."""

import asyncio
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.config.settings import clear_settings_cache
from app.core.exceptions import ToolExecutionError
from app.main import create_app
from app.mcp.runtime import call_mcp_tool


class Fixture(BaseHTTPRequestHandler):
    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/fail":
            self.send_error(503)
            return
        body = json.dumps({"name": payload.get("name"), "id": payload["arguments"]["id"]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


async def proxy():
    with ThreadingHTTPServer(("127.0.0.1", 0), Fixture) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        args = {"name": "echo_fixture", "arguments": {"id": "synthetic"}, "transport": "http"}
        try:
            good = await call_mcp_tool(**args, endpoint=base + "/mcp/call")
            try:
                await call_mcp_tool(**args, endpoint=base + "/fail")
            except ToolExecutionError as exc:
                failure = exc.message
            else:
                failure = "unexpected_success"
        finally:
            server.shutdown()
            thread.join()
        return {"valid_reply": good, "upstream_503": failure}


def main():
    with patch.dict(
        os.environ,
        {
            "APP_ENV": "testing",
            "DEEPSEEK_API_KEY": "",
            "EMBEDDING_API_KEY": "",
            "REDIS_URL": "",
            "QDRANT_URL": "",
            "DATABASE_URL": "sqlite+aiosqlite:///:memory:",
            "NO_PROXY": "127.0.0.1",
            "no_proxy": "127.0.0.1",
        },
    ):
        clear_settings_cache()
        result = {"http_tool": asyncio.run(proxy())}
        with TestClient(create_app()) as client:
            response = client.post(
                "/api/v1/chat/stream", json={"user_id": "demo", "message": "礼品卡可以提现吗"}
            )
            response.raise_for_status()
            events = [
                json.loads(line[6:])
                for line in response.text.splitlines()
                if line.startswith("data: ")
            ]
            first_token = next(i for i, e in enumerate(events) if e["event"] == "token")
            checked = next(i for i, e in enumerate(events) if e.get("node") == "check_faithfulness")
            full = next(e["data"]["answer"] for e in events if e["event"] == "answer")
            result["sse"] = {
                "event_types": [e["event"] for e in events],
                "first_token_after_check_node": first_token > checked,
                "joined_chunks_match_answer": "".join(
                    e["text"] for e in events if e["event"] == "token"
                )
                == full,
                "ended_with_done": events[-1]["event"] == "done",
            }
        clear_settings_cache()
    Path(sys.argv[1]).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
