"""Model Context Protocol (MCP) integration.

Provides a clean bridge between replaygraph's Policy/Recorder/Replay
layer and MCP tool servers.

Design goals:
1. Pure stdlib + typing -- no hard dependency on `mcp` SDK so the
   core library remains installable anywhere. If the `mcp` package
   is available, we use its typed structures; otherwise we speak the
   JSON-RPC 2.0 protocol directly over stdio or SSE.
2. Full replayability: MCP tool responses are recorded with the same
   fingerprint guarantee as native Python tools.
3. Policy enforcement AT THE BOUNDARY: an MCP tool call is checked
   against the Policy BEFORE it goes over the wire, stopping unauthorized
   server invocations before network traffic leaves the machine.

Typical usage::

    from replaygraph import Policy
    from replaygraph.mcp import MCPClient, MCPToolBridge

    policy = Policy("strict").allow("fetch_user").deny("*")
    client = MCPClient.from_command(["python", "server.py"])
    bridge = MCPToolBridge(client, policy=policy, recorder=rec)
    result = bridge.call("fetch_user", {"id": "123"})
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .gate import Decision, Policy
from .langgraph_support import Recorder
from .trace import ToolCall


@dataclass
class MCPToolDef:
    """Description of a tool exposed by an MCP server."""
    name: str
    description: str = ""
    input_schema: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MCPToolDef":
        return cls(
            name=d.get("name", ""),
            description=d.get("description", ""),
            input_schema=d.get("inputSchema", d.get("input_schema", {})),
        )


class MCPClient:
    """Lightweight stdio JSON-RPC client for MCP servers.
    
    Speaks JSON-RPC 2.0 without requiring the heavy MCP Python SDK.
    Safe on Python 3.10+, zero third-party deps.
    """

    def __init__(self, process: subprocess.Popen, server_name: str = "mcp-server"):
        self.proc = process
        self.server_name = server_name
        self._next_id = 1

    @classmethod
    def from_command(
        cls,
        command: List[str],
        server_name: Optional[str] = None,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> "MCPClient":
        proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=cwd,
            env=env,
        )
        return cls(proc, server_name=server_name or command[0])

    def _rpc_call(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        req_id = self._next_id
        self._next_id += 1
        msg = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
        }
        if params is not None:
            msg["params"] = params

        payload = json.dumps(msg) + "\n"
        if not self.proc.stdin:
            raise RuntimeError("MCP process stdin not available")
        self.proc.stdin.write(payload)
        self.proc.stdin.flush()

        # Read JSON-RPC response
        if not self.proc.stdout:
            raise RuntimeError("MCP process stdout not available")
        line = self.proc.stdout.readline()
        if not line:
            stderr = self.proc.stderr.read() if self.proc.stderr else ""
            raise RuntimeError(f"MCP server exited unexpectedly: {stderr}")

        res = json.loads(line)
        if "error" in res:
            err = res["error"]
            raise RuntimeError(f"MCP error {err.get('code')}: {err.get('message')}")
        return res.get("result", {})

    def list_tools(self) -> List[MCPToolDef]:
        """Query tools/list from the server."""
        res = self._rpc_call("tools/list")
        raw = res.get("tools", [])
        return [MCPToolDef.from_dict(t) for t in raw]

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Invoke tools/call on the server."""
        res = self._rpc_call("tools/call", {"name": name, "arguments": arguments})
        # MCP tools/call returns {"content": [{"type": "text", "text": "..."}]}
        content = res.get("content", [])
        if isinstance(content, list) and len(content) > 0:
            first = content[0]
            if isinstance(first, dict) and "text" in first:
                return first["text"]
        return res

    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()


class MCPToolBridge:
    """Wraps an MCPClient with replaygraph's Policy + Recorder.
    
    Guarantees:
    1. Pre-call policy evaluation -- denied calls NEVER reach the MCP server.
    2. Policy violations recorded in the trace with denial reason.
    3. Allowed calls recorded with deterministic call IDs and hashes.
    4. Seamless replay: during offline replay, the MCP server does not
       even need to be running.
    """

    def __init__(
        self,
        client: Optional[MCPClient] = None,
        policy: Optional[Policy] = None,
        recorder: Optional[Recorder] = None,
    ):
        self.client = client
        self.policy = policy or Policy("default-mcp")
        self.recorder = recorder

    def call(self, name: str, args: Dict[str, Any]) -> Any:
        """Policy-gate then invoke the MCP tool, recording the call.

        Delegates to Recorder.call_tool so the policy evaluation, trace
        record, denial handling and error capture all stay in ONE place
        (the same path native tools use). The MCP server is invoked via
        a thin runner that Recorder executes only after the policy
        allows the call -- a denied call never reaches the wire.
        """
        if self.recorder is not None:
            # Recorder owns the policy decision + trace. Its policy is the
            # bridge's policy; make sure the recorder sees ours.
            self.recorder.policy = self.policy
            runner = self._invoke
            return self.recorder.call_tool(name, args, runner)

        # No recorder: still enforce the policy before going over the wire.
        decision = self.policy.evaluate(name, args)
        if not decision.allowed:
            raise PermissionError(f"Tool '{name}' denied by policy: {decision.reason}")
        if self.client is None:
            raise RuntimeError("MCPClient not configured on bridge")
        return self._invoke(name, args)

    def _invoke(self, name: str, args: Dict[str, Any]) -> Any:
        if self.client is None:
            raise RuntimeError("MCPClient not configured on bridge")
        return self.client.call_tool(name, args)
