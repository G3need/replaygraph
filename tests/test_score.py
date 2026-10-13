"""Evals and drift detection."""

from replaygraph.score import compare_traces, score_trace
from replaygraph.trace import Step, ToolCall, Trace


def build(status="ok", tools=(), denied=False, failed=False, steps=1, out="done") -> Trace:
    t = Trace(graph="g", seed=1, input={"q": "x"}, status=status)
    for i in range(steps):
        s = t.add(Step(node=f"n{i}", kind="model", output=out))
        for name in tools:
            s.calls.append(
                ToolCall(
                    name=name,
                    allowed=not denied,
                    result=None if denied else "r",
                    error="Boom" if failed else None,
                )
            )
    t.final_output = out
    return t


def test_clean_trace_passes():
    card = score_trace(build(tools=["read"]))
    assert card.passed
    assert card.score == 1.0
    assert not card.violations


def test_policy_violation_fails():
    card = score_trace(build(tools=["send_email"], denied=True))
    assert not card.passed
    assert any("policy" in v for v in card.violations)
    assert card.metrics["denied"] == 1


def test_tool_failure_fails():
    card = score_trace(build(tools=["read"], failed=True))
    assert not card.passed
    assert card.metrics["failed"] == 1


def test_error_status_fails():
    card = score_trace(build(status="error"))
    assert not card.passed


def test_step_budget():
    card = score_trace(build(steps=20), max_steps=5)
    assert not card.passed
    assert any("step_budget" in v for v in card.violations)


def test_runaway_loop_guard():
    t = Trace(status="ok")
    s = t.add(Step(node="n", kind="model"))
    for _ in range(10):
        s.calls.append(ToolCall(name="search", allowed=True, result="r"))
    card = score_trace(t, max_calls_per_tool=3)
    assert not card.passed
    assert any("runaway" in v for v in card.violations)


def test_require_and_forbid_tools():
    t = build(tools=["read"])
    assert score_trace(t, require_tools=["read"]).passed
    assert not score_trace(t, require_tools=["write"]).passed
    assert not score_trace(t, forbid_tools=["read"]).passed
    assert score_trace(t, forbid_tools=["write"]).passed


def test_output_substring():
    card = score_trace(build(out="Refund issued"), expect_output_substring="refund")
    assert card.passed


def test_score_is_weighted_not_binary():
    card = score_trace(build(status="error"))  # weight 3 violation
    assert 0.0 < card.score < 1.0


def test_summary_readable():
    assert "PASS" in score_trace(build()).summary()
    assert "FAIL" in score_trace(build(status="error")).summary()


def test_identical_traces_have_no_drift():
    d = compare_traces(build(tools=["read"]), build(tools=["read"]))
    assert d.is_clean
    assert "none" in d.report()


def test_new_tool_is_drift():
    d = compare_traces(build(tools=["read"]), build(tools=["read", "send_email"]))
    assert not d.is_clean
    assert d.tools_added == ["send_email"]
    assert "tools added" in d.report()


def test_status_change_is_drift():
    d = compare_traces(build(status="ok"), build(status="error"))
    assert d.status_changed and not d.is_clean


def test_token_delta_reported():
    a, b = build(), build()
    b.steps[0].completion_tokens = 50
    assert compare_traces(a, b).token_delta == 50


def test_step_added_removed():
    d = compare_traces(build(steps=1), build(steps=3))
    assert d.steps_added == ["n1", "n2"]
    assert not d.steps_removed
