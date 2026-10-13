"""End-to-end example: gate an MCP tool server behind replaygraph.

Runs a *real* MCP server as a subprocess (a tiny echo server defined in
this file) and drives it through the bridge. Proves:

  1. an allowed MCP call executes and is recorded with its result
  2. a denied call is blocked BEFORE it reaches the server
  3. the recorded trace can be replayed offline with no server running

No API key. No `mcp` SDK. Stdlib only.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from replaygraph import MCPClient, MCPToolBridge, Policy, Recorder, replay_trace


# A minimal MCP server speaking JSON-RPC 2.0 over stdio. One tool: "echo".
SERVER_SRC = r'''
import json, sys
def send(o): sys.stdout.write(json.dumps(o)+"\n"); sys.stdout.flush()
for line in sys.stdin:
    line=line.strip()
    if not line: continue
    req=json.loads(line)
    rid=req.get("id")
    if req.get("method")=="tools/list":
        send({"jsonrpc":"2.0","id":rid,"result":{"tools":[{"name":"echo","description":"echo back","inputSchema":{}}]}})
    elif req.get("method")=="tools/call":
        args=req.get("params",{}).get("arguments",{})
        send({"jsonrpc":"2.0","id":rid,"result":{"content":[{"type":"text","text":"echo:"+json.dumps(args)}]}})
'''


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        server_path = Path(td) / "server.py"
        server_path.write_text(SERVER_SRC)

        client = MCPClient.from_command([sys.executable, str(server_path)], server_name="echo-server")

        # Echo back the tools list -- proves the RPC handshake works.
        tools = client.list_tools()
        print("server tools:", [t.name for t in tools])

        policy = Policy("mcp-demo").allow("echo").deny("*")
        rec = Recorder("mcp-agent", policy=policy)
        bridge = MCPToolBridge(client=client, policy=policy, recorder=rec)

        # 1) allowed call
        out = bridge.call("echo", {"hello": "world"})
        print("allowed echo ->", out)

        # 2) denied call -- must raise, server never sees it
        try:
            bridge.call("danger", {})
            print("!! denial FAILED")
            return 1
        except PermissionError as exc:
            print("blocked as expected:", exc)

        trace = rec.finish(output="done", status="ok")
        trace_path = Path(td) / "mcp_trace.json"
        trace.save(trace_path)

        allowed = [c for s in trace.steps for c in s.calls if c.allowed]
        denied = [c for s in trace.steps for c in s.calls if not c.allowed]
        print("trace: allowed=%d denied=%d" % (len(allowed), len(denied)))

        # 3) replay offline -- no server needed. replay_trace takes a Trace,
        # so round-trip through JSON to prove the saved file replays cleanly.
        from replaygraph import Trace
        loaded = Trace.from_dict(json.loads(trace_path.read_text()))
        replayed = replay_trace(loaded, verify=True)
        print("replay: deterministic=%s calls=%d steps=%d" % (
            replayed.deterministic, replayed.calls_replayed, replayed.steps_replayed))

        client.close()

        ok = len(allowed) == 1 and len(denied) == 1
        print("OVERALL:", "PASS" if ok else "FAIL")
        return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
