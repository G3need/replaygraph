"""Trace and fingerprint behaviour.

The determinism claim lives here: a trace re-serialised from the same
logical content must produce the same fingerprint, and timestamps must
not leak into it.
"""

from replaygraph.trace import Step, ToolCall, Trace, content_hash


def make_trace() -> Trace:
    t = Trace(graph="t", seed=7, input={"q": "hello"}, status="ok")
    s = t.add(Step(node="model", kind="model", model="m", output="hi"))
    s.calls.append(
        ToolCall(name="read", args={"id": "1"}, result={"ok": True}, allowed=True)
    )
    t.add(Step(node="end", kind="note", detail="done"))
    t.final_output = "hi"
    return t


def test_fingerprint_ignores_timing():
    a, b = make_trace(), make_trace()
    b.steps[0].started_at = 999.0
    b.steps[0].duration_ms = 12345.0
    assert a.fingerprint == b.fingerprint


def test_fingerprint_changes_on_content():
    a, b = make_trace(), make_trace()
    b.steps[0].output = "different"
    assert a.fingerprint != b.fingerprint


def test_roundtrip_preserves_fingerprint(tmp_path):
    t = make_trace()
    path = str(tmp_path / "t.json")
    t.save(path)
    assert Trace.load(path).fingerprint == t.fingerprint


def test_canonical_hash_is_order_independent():
    assert content_hash({"a": 1, "b": 2}) == content_hash({"b": 2, "a": 1})


def test_metrics():
    t = make_trace()
    assert t.token_usage["total"] == 0
    assert t.tools_used == ["read"]
    assert not t.denied_calls
    assert not t.failed_calls
    assert t.duration_ms == 0.0


def test_denied_and_failed_calls_are_visible():
    t = make_trace()
    s = t.add(Step(node="m2", kind="model"))
    s.calls.append(ToolCall(name="send_email", allowed=False, reason="forbidden"))
    s.calls.append(ToolCall(name="search", error="TimeoutError: nope"))
    assert [c.name for c in t.denied_calls] == ["send_email"]
    assert [c.name for c in t.failed_calls] == ["search"]
    assert t.tools_used == ["read", "send_email", "search"]
