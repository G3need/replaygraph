"""Replay: the determinism guarantee, and the transport contract.

These tests assert the central claim of the whole project — that a
recorded trajectory replays identically and offline — plus the
important negative cases: a replay must never invent a tool result and
must never execute a denied call.
"""

import pytest

from replaygraph.gate import Policy
from replaygraph.langgraph_support import Recorder
from replaygraph.replay import (
    ReplayMismatch,
    ReplayTransport,
    replay_trace,
)
from replaygraph.trace import ToolCall, Trace


def trace_with_calls() -> Trace:
    t = Trace(graph="g", seed=1, input={"q": "go"}, status="ok")
    s = t.add_step = None  # placeholder removed below
    s = t.add(_step("plan", output="use read"))
    s.calls.append(ToolCall(name="read", args={"id": "1"}, result="DATA", allowed=True))
    t.add(_step("finish", output="done"))
    t.final_output = "done"
    return t


def _step(node: str, output: str = ""):
    from replaygraph.trace import Step

    return Step(node=node, kind="model", output=output)


def test_replay_reproduces_same_fingerprint():
    t = trace_with_calls()
    r = replay_trace(t, times=5)
    assert r.deterministic
    assert len(set(r.fingerprints)) == 1
    assert r.steps_replayed == 2
    assert r.calls_replayed == 5


def test_replay_strips_timing():
    t = trace_with_calls()
    t.steps[0].duration_ms = 9876.0
    r = replay_trace(t, times=1)
    assert r.trace.duration_ms == 0.0
    assert r.trace.fingerprint == replay_trace(trace_with_calls()).fingerprint


def test_replay_verify_clean():
    t = trace_with_calls()
    assert replay_trace(t, times=1, verify=True).deterministic


def test_times_must_be_positive():
    with pytest.raises(ValueError):
        replay_trace(trace_with_calls(), times=0)


def test_transport_serves_recorded_result():
    t = trace_with_calls()
    tr = ReplayTransport(t)
    assert tr("read", {"id": "1"}) == "DATA"
    assert tr.exhausted


def test_transport_refuses_unrecorded_call():
    tr = ReplayTransport(trace_with_calls())
    with pytest.raises(ReplayMismatch, match="diverged"):
        tr("write", {"id": "1"})


def test_transport_refuses_denied_call():
    t = trace_with_calls()
    t.steps[0].calls[0].allowed = False
    with pytest.raises(ReplayMismatch, match="denied"):
        ReplayTransport(t)("read", {"id": "1"})


def test_transport_refuses_errored_call():
    t = trace_with_calls()
    t.steps[0].calls[0].error = "TimeoutError"
    with pytest.raises(ReplayMismatch, match="error"):
        ReplayTransport(t)("read", {"id": "1"})


def test_transport_running_past_end():
    t = trace_with_calls()
    tr = ReplayTransport(t)
    tr("read", {"id": "1"})
    with pytest.raises(ReplayMismatch, match="past the end"):
        tr("read", {"id": "1"})


# --- recorder -------------------------------------------------------


def test_recorder_records_tool_and_result():
    rec = Recorder("g")
    with rec.model("plan"):
        rec.call_tool("read", {"id": "1"}, lambda n, a: "DATA")
    tr = rec.trace
    assert tr.tools_used == ["read"]
    call = list(tr.tool_calls)[0]
    assert call.result == "DATA" and call.allowed


def test_recorder_policy_denial_blocks_execution_and_flags_trace():
    rec = Recorder("g", policy=Policy("p").allow("read"))
    with rec.model("plan"):
        with pytest.raises(PermissionError):
            rec.call_tool("send_email", {"to": "x@y"}, lambda n, a: "SENT")
    call = list(rec.trace.tool_calls)[0]
    assert not call.allowed and call.result is None
    assert rec.trace.status == "blocked"


def test_recorder_tool_never_runs_when_denied():
    calls = []

    def runner(n, a):
        calls.append(n)
        return "SENT"

    rec = Recorder("g", policy=Policy("p").deny("send_email"))
    with rec.model("plan"):
        with pytest.raises(PermissionError):
            rec.call_tool("send_email", {}, runner)
    assert calls == []


def test_recorder_tool_error_is_captured():
    def boom(n, a):
        raise TimeoutError("slow")

    rec = Recorder("g")
    with rec.model("plan"):
        with pytest.raises(TimeoutError):
            rec.call_tool("read", {}, boom)
    assert rec.trace.failed_calls


def test_recorder_model_error_marks_trace():
    rec = Recorder("g")
    with pytest.raises(RuntimeError):
        with rec.model("plan"):
            raise RuntimeError("nope")
    assert rec.trace.status == "error"
    assert rec.trace.steps[0].error
