"""Tests for the MCP tool-bridge layer.

These use a fake in-process MCP "server" (a callable stub) rather than a
real subprocess, so the tests are deterministic, need no `mcp` SDK, and
prove the property that matters: a policy-denied call never reaches the
server, and an allowed call is recorded with its result.
"""

import pytest

from replaygraph import MCPClient, MCPToolBridge, MCPToolDef, Recorder
from replaygraph.gate import Policy


class FakeMCP:
    """Stands in for an MCPClient. Records every call it receives."""

    def __init__(self, response="ok"):
        self.calls = []
        self.response = response

    def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return self.response

    def list_tools(self):
        return [MCPToolDef(name="echo", description="echo back", input_schema={})]


def make_policy():
    return Policy("test").allow("echo").allow("read_file").deny("*")


def test_allowed_call_reaches_server_and_records():
    fake = FakeMCP(response="hello")
    rec = Recorder("agent", policy=make_policy())
    bridge = MCPToolBridge(client=fake, policy=make_policy(), recorder=rec)

    out = bridge.call("echo", {"msg": "hi"})

    assert out == "hello"
    # server got exactly one call
    assert fake.calls == [("echo", {"msg": "hi"})]
    # and it landed in the trace
    calls = [c for s in rec.trace.steps for c in s.calls]
    assert len(calls) == 1
    assert calls[0].name == "echo"
    assert calls[0].allowed is True
    assert calls[0].result == "hello"


def test_denied_call_never_reaches_server():
    fake = FakeMCP()
    rec = Recorder("agent", policy=make_policy())
    bridge = MCPToolBridge(client=fake, policy=make_policy(), recorder=rec)

    with pytest.raises(PermissionError):
        bridge.call("delete_everything", {"x": 1})

    # the whole point: the server was never touched
    assert fake.calls == []
    # the denial is a recorded fact
    calls = [c for s in rec.trace.steps for c in s.calls]
    assert len(calls) == 1
    assert calls[0].allowed is False
    assert "delete_everything" in calls[0].reason or calls[0].reason != ""


def test_bridge_without_recorder_still_enforces_policy():
    fake = FakeMCP()
    bridge = MCPToolBridge(client=fake, policy=make_policy(), recorder=None)

    with pytest.raises(PermissionError):
        bridge.call("delete_everything", {})
    assert fake.calls == []

    # allowed still works with no recorder
    assert bridge.call("echo", {"a": 1}) == "ok"
    assert fake.calls == [("echo", {"a": 1})]


def test_server_error_is_captured_in_trace():
    class Boom(FakeMCP):
        def call_tool(self, name, arguments):
            raise RuntimeError("server exploded")

    rec = Recorder("agent", policy=make_policy())
    bridge = MCPToolBridge(client=Boom(), policy=make_policy(), recorder=rec)

    with pytest.raises(RuntimeError):
        bridge.call("echo", {})

    calls = [c for s in rec.trace.steps for c in s.calls]
    assert calls[0].error is not None
    assert "server exploded" in calls[0].error


def test_client_requires_no_mcp_sdk():
    # MCPClient should be constructible purely from stdlib subprocess;
    # prove the class exists and exposes the expected surface.
    assert hasattr(MCPClient, "from_command")
    assert hasattr(MCPClient, "call_tool")
    assert hasattr(MCPClient, "list_tools")


def test_mcp_tooldef_from_dict_handles_both_schema_keys():
    d = MCPToolDef.from_dict({"name": "t", "description": "d", "inputSchema": {"a": 1}})
    assert d.name == "t"
    assert d.input_schema == {"a": 1}
    d2 = MCPToolDef.from_dict({"name": "t", "input_schema": {"b": 2}})
    assert d2.input_schema == {"b": 2}
