import json

import httpx
import pytest

from eval.evidence36 import judge, parse_labels


def test_judge_formats_and_incomplete_answers():
    item = {"pass": "false", "reason": "missing value"}
    assert parse_labels(json.dumps({"labels": {"A": item}}), ["A"])["A"]["pass"] is False
    marked = json.dumps({"type": "json_object", "A": item, "B": item, "C": item})
    assert set(parse_labels(marked, "ABC")) == set("ABC")
    for extra in ({"type": "other"}, {"D": item}):
        with pytest.raises(ValueError):
            parse_labels(json.dumps({"A": item, **extra}), "A")
    with pytest.raises(ValueError):
        parse_labels(json.dumps({"type": "json_object", "A": item}), "ABC")
    for content in ("{}", "[]", '{"A":null}', '{"A":{"pass":1,"reason":"ok"}}'):
        with pytest.raises(ValueError):
            parse_labels(content, ["A"])


@pytest.mark.asyncio
async def test_judge_recovers_individual_scores_and_audits_all_calls(tmp_path):
    keys = []

    def respond(request):
        payload = json.loads(request.content)
        messages = payload["messages"]
        answers = json.loads(messages[-1]["content"])["answers"]
        assert payload["response_format"] == {"type": "json_object"}
        assert "json" in messages[0]["content"].lower()
        assert set(json.loads(messages[0]["content"].split("：")[-1])) == set(answers)
        keys.append("".join(answers))
        labels = {k: {"pass": k != "B", "reason": "reference checked"} for k in answers}
        raw = json.dumps(labels) if len(answers) == 1 else "{}"
        return httpx.Response(
            200,
            json={
                "id": str(len(keys)),
                "model": "fixture",
                "usage": {"total_tokens": 1},
                "choices": [{"message": {"content": raw}, "finish_reason": "stop"}],
            },
        )

    audit = tmp_path / "attempts.jsonl"
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond), base_url="https://fixture.invalid"
    ) as client:
        result = await judge(
            client,
            "fixture",
            {"id": "q", "query": "q", "gold": "g", "kind": "answer"},
            ["source"],
            {k: k for k in "ABC"},
            audit,
        )
    assert keys == ["ABC", "ABC", "A", "B", "C"]
    assert result["mode"] == "individual_recovery"
    assert not result["labels"]["B"]["pass"]
    attempts = [json.loads(line) for line in audit.read_text(encoding="utf8").splitlines()]
    assert sum("schema_error" in a for a in attempts) == 2
    assert len(attempts) == len(result["receipts"]) == 5


@pytest.mark.asyncio
async def test_judge_records_http_400_body(tmp_path):
    transport = httpx.MockTransport(
        lambda _: httpx.Response(400, json={"error": {"message": "invalid request"}})
    )
    audit = tmp_path / "attempts.jsonl"
    async with httpx.AsyncClient(transport=transport, base_url="https://fixture.invalid") as client:
        with pytest.raises(httpx.HTTPStatusError):
            await judge(
                client,
                "fixture",
                {"id": "H09", "query": "q", "gold": "g", "kind": "answer"},
                [],
                {"A": "answer"},
                audit,
            )
    attempt = json.loads(audit.read_text(encoding="utf8"))
    assert attempt["case"] == "H09"
    assert attempt["http_status"] == 400
    assert "invalid request" in attempt["error_body"]
