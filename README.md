# replaygraph

**Deterministic replay and evaluation for LangGraph agents.**

An agent's behaviour is a *trajectory*, not a single output. If you
cannot replay a trajectory bit-for-bit, you cannot regression-test it
— and an agent you cannot regression-test is an agent you cannot
safely ship.

replaygraph gives you three pieces:

| Piece | What it does |
|---|---|
| `gate` | Default-deny policy boundary. Authorises every tool call **before** it executes. |
| `trace` | JSON-serialisable record of a full run: every model step, tool call, policy decision, cost. |
| `replay` | Re-executes a recorded trajectory **offline, with no API key**, then fingerprints it. |

Grading is deliberately boring. A trace is scored against *observable
facts* — did it stay inside policy, did every tool resolve, did it
terminate, did it stay inside its step/token budget, did it avoid
runaway loops — rather than against a model's opinion of itself.

## Install

```bash
pip install -e .            # core
pip install -e ".[anthropic]"   # + Anthropic model support
pip install -e ".[dev]"         # + pytest
```

## Quickstart

```python
from langgraph.graph import StateGraph, END
from replaygraph import Recorder, Policy, score_trace, replay_trace

policy = (
    Policy("support")
    .allow("read_account", when_arg("account_id", lambda v: isinstance(v, str)))
    .allow("issue_refund", args_only("account_id", "amount"))
    # anything not listed is denied before it runs
)

rec = Recorder("support", policy=policy)

# ... build your graph, threading `rec` through your nodes ...

trace = rec.finish(answer, status="ok")
trace.save("traces/run.json")
```

Then, with no model call and no API key:

```bash
python -m replaygraph.cli score  traces/run.json --require-tool read_account
python -m replaygraph.cli replay traces/run.json --times 5
python -m replaygraph.cli diff   traces/baseline.json traces/candidate.json
```

Exit codes are CI-shaped: `0` = pass, `1` = invariant violated.

A complete working agent is in
[`examples/billing_support_agent.py`](examples/billing_support_agent.py):

```bash
python examples/billing_support_agent.py
```

## The gate

```python
from replaygraph import Policy
from replaygraph.gate import args_only, deny_pattern, when_arg, max_len

policy = (
    Policy("billing")
    .allow("lookup_account", when_arg("account_id", lambda v: isinstance(v, str)))
    .allow("search", args_only("q"),
           deny_pattern(r"ignore (all )?previous instructions", ["q"]))
    .allow("issue_refund", args_only("account_id", "amount"))
    .deny("send_email")      # hard block
)
```

Properties worth knowing:

- **Default-deny.** An unmatched tool is denied. Fail-closed is the
  only safe default once an agent can act.
- **Ordered rules, first match wins.** Register the specific rule
  before the broad one.
- **A broken rule fails closed**, it does not fall open.
- **Argument injection is blocked** by `args_only(...)`, which rejects
  any argument the rule did not name.
- **A denial is recorded as fact**, not raised into oblivion: the
  trace shows the call, the rule that blocked it, and why.

## Scoring

```python
card = score_trace(
    trace,
    max_steps=12,
    max_tokens=32_000,
    require_tools=["read_account"],
    forbid_tools=["send_email"],
    expect_output_substring="refund",
)
print(card.summary())
# [PASS] score=1.000 checks=8/8 violations=0
```

Weighted invariants: completion (3), policy respected (3), tools
resolved (2), no runaway loop (2), step budget, token budget, required
tools, forbidden tools, output contains.

## Drift

```python
drift = compare_traces(baseline, candidate)
print(drift.report())
```

Reports fingerprint change, status change, output change, steps and
tools added/removed, and token delta. This is the regression check:
*"is this still the run we shipped?"*

## Design notes

**No framework internals are touched.** Tracing libraries that reach
into langgraph internals break on every minor upgrade. `Recorder`
intercepts at the two places that matter — the model call and the tool
call — via a context manager you thread through your own nodes. Your
graph keeps its shape.

**Fingerprints exclude timing.** `Trace.fingerprint` hashes node, kind,
model, output, tool name/args/result/allowance. Timestamps and
durations are excluded on purpose, so a rerun of a deterministic agent
produces an identical digest.

**Traces are plain JSON.** No framework objects, no message objects, no
runnables. That is what makes them stable across versions and safe to
commit and diff with ordinary tooling.

**Replay never invents a result.** `ReplayTransport` raises
`ReplayMismatch` on any call the record cannot answer, and refuses to
execute a call the record marked as denied.

## MCP support (v0.2.0)

`replaygraph.mcp` gates an MCP tool server behind the same policy
boundary as native tools. A `MCPClient` speaks JSON-RPC 2.0 over stdio
using only the standard library (no `mcp` SDK required); `MCPToolBridge`
wraps it so that:

- **a policy-denied call never reaches the server** — the denial is
  recorded as a fact before any network traffic leaves the machine;
- allowed calls are recorded with their results through the *same*
  `Recorder` path native tools use, so traces stay uniform;
- the recorded trace replays offline with no server process running.

```python
from replaygraph import MCPClient, MCPToolBridge, Policy, Recorder

client = MCPClient.from_command(["python", "my_server.py"])
policy = Policy("strict").allow("read_file").deny("*")
recorder = Recorder("agent", policy=policy)
bridge = MCPToolBridge(client=client, policy=policy, recorder=recorder)

result = bridge.call("read_file", {"path": "notes.md"})  # policy-gated + recorded
```

Run the end-to-end demo (spins up a real subprocess MCP server):

```bash
python examples/mcp_bridge_demo.py   # OVERALL: PASS
```

## Tests

```bash
python -m pytest        # 57 passed
```

The determinism guarantee is asserted directly: `test_replay.py` replays
a recorded trace five times and requires a single unique fingerprint.

## Layout

```
src/replaygraph/
  gate.py              default-deny policy + argument predicates
  trace.py             Trace / Step / ToolCall, canonical hashing
  replay.py            offline replay, ReplayTransport
  score.py             evals + drift detection
  langgraph_support.py Recorder, run_recorded
  mcp.py               MCPClient + MCPToolBridge (stdlib JSON-RPC 2.0)
  cli.py               score / replay / diff / policy
examples/billing_support_agent.py
examples/mcp_bridge_demo.py
tests/                 57 tests, no network, no keys
```

## License

MIT
