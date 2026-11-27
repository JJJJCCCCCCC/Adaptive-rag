"""Record pure-rule routing or an in-process /api/v1/chat observation."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from app.agents.nodes.intent import rule_based_intent
from app.agents.nodes.intent_overlay import high_precision_intent
from app.retrieval.adaptive import rule_based_complexity

ROOT = Path(__file__).resolve().parents[1]


def observation(query, intent, routing, *, overlay_enabled):
    """Keep routing scores separate from intent scores and overlay candidates."""
    candidate = high_precision_intent(query)
    applied = bool(
        intent and any(part.startswith("overlay:") for part in intent.rationale.split(";"))
    )
    return {
        "Intent": intent.label.value if intent else None,
        "Overlay": {
            "enabled": overlay_enabled,
            "candidate": candidate.value if candidate else None,
            "applied": applied if overlay_enabled else False,
        },
        "Strategy": routing.strategy.value if routing else None,
        "Complexity": routing.complexity.value if routing else None,
        "Confidence": routing.confidence if routing else None,
        "Rationale": routing.rationale if routing else None,
        "IntentConfidence": intent.confidence if intent else None,
        "IntentRationale": intent.rationale if intent else None,
    }


def observe_rules(query):
    intent = rule_based_intent(query)
    routing = rule_based_complexity(query, intent)
    return observation(query, intent, routing, overlay_enabled=False)


def observe_chat(query, user_id):
    from fastapi.testclient import TestClient

    from app.config.settings import get_settings
    from app.main import create_app
    from app.services.chat_service import ChatService

    captured = {}
    original = ChatService._to_response

    def capture(service, result, **kwargs):
        captured.update(
            observation(query, result.get("intent"), result.get("complexity"), overlay_enabled=True)
        )
        return original(service, result, **kwargs)

    with patch.object(ChatService, "_to_response", capture):
        with TestClient(create_app()) as client:
            response = client.post(
                get_settings().api_v1_prefix + "/chat",
                json={"message": query, "user_id": user_id},
            )
            response.raise_for_status()
            body = response.json()
    if not captured:
        captured.update(observation(query, None, None, overlay_enabled=False))
        captured["ObservationNote"] = "Request ended before routing, e.g. guardrail rejection."
    captured.update(
        {
            "Intent": body["intent"],
            "Strategy": body["strategy"],
            "Complexity": body["complexity"],
            "Answer": body["answer"],
            "TraceId": body["trace_id"],
            "ErrorCode": body["error_code"],
            "LLMConfigured": get_settings().llm_configured,
        }
    )
    return captured


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=["rules", "chat"])
    parser.add_argument("--question")
    parser.add_argument("--user-id", default="u_demo")
    args = parser.parse_args()
    query = args.question or input("请输入问题：").strip()
    if not query:
        parser.error("question cannot be empty")
    record = observe_rules(query) if args.mode == "rules" else observe_chat(query, args.user_id)
    record = {
        "Mode": args.mode,
        "Question": query,
        "RecordedAt": datetime.now(UTC).isoformat(),
        **record,
    }
    print(json.dumps(record, ensure_ascii=False, indent=2))
    destination = ROOT / "output" / "routing_observations.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"记录已追加到：{destination}")


if __name__ == "__main__":
    main()
