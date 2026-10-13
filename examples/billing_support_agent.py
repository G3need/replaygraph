"""End-to-end: a real LangGraph agent, recorded, replayed, scored.

This is the demonstration that the README promises. It runs a genuine
compiled LangGraph graph with a real model call boundary and real tool
calls, wrapped by a ``Recorder``. No API key is needed: the model node
is a deterministic local function standing in for an LLM, so the whole
example is offline and CI-safe by design.
"""

from typing import TypedDict

from langgraph.graph import END, StateGraph

from replaygraph import Policy, Recorder, compare_traces, replay_trace, score_trace
from replaygraph.gate import args_only, when_arg


class State(TypedDict, total=False):
    question: str
    account_id: str
    lookup: str
    answer: str


# --- the tools the agent is allowed to use --------------------------------

LEDGER = {"A100": {"plan": "pro", "refund_due": 49.00}}


def lookup_account(account_id: str) -> dict:
    return LEDGER.get(account_id, {})


def issue_refund(account_id: str, amount: float) -> str:
    return f"refund of {amount:.2f} issued to {account_id}"


# --- the policy: what the agent may touch --------------------------------
# Default-deny. The agent may read any account by string id, and may only
# issue refunds when the amount is a sane number. Anything else is blocked
# before it executes, and the block is recorded as fact.

policy = (
    Policy("billing-support")
    .allow("lookup_account", when_arg("account_id", lambda v: isinstance(v, str)))
    .allow("issue_refund", args_only("account_id", "amount"))
)


# --- the graph ------------------------------------------------------------

rec: Recorder | None = None  # set per run


def node_triage(state: State) -> State:
    """Decide which tool the question needs. Stands in for an LLM call."""
    with rec.model("triage", model="standin-llm", prompt=state["question"]) as step:
        q = state["question"].lower()
        if "refund" in q:
            step.output = "tool:issue_refund"
            return {"lookup": "issue_refund"}
        step.output = "tool:lookup_account"
        return {"lookup": "lookup_account"}


def node_act(state: State) -> State:
    """Execute the chosen tool through the gate."""
    tool = state["lookup"]
    with rec.model("act", model="standin-llm") as step:
        if tool == "issue_refund":
            try:
                out = rec.call_tool(
                    "issue_refund",
                    {"account_id": state.get("account_id", ""), "amount": 49.00},
                    lambda n, a: issue_refund(a["account_id"], a["amount"]),
                )
            except PermissionError as exc:
                step.output = str(exc)
                return {"answer": str(exc)}
            step.output = out
            return {"answer": out}

        out = rec.call_tool(
            "lookup_account",
            {"account_id": state.get("account_id", "")},
            lambda n, a: str(lookup_account(a["account_id"])),
        )
        step.output = out
        return {"answer": out}


def node_answer(state: State) -> State:
    with rec.model("answer", model="standin-llm") as step:
        step.output = state.get("answer", "")
    return state


builder = StateGraph(State)
builder.add_node("triage", node_triage)
builder.add_node("act", node_act)
builder.add_node("answer", node_answer)
builder.set_entry_point("triage")
builder.add_edge("triage", "act")
builder.add_edge("act", "answer")
builder.add_edge("answer", END)
graph = builder.compile()


def run(question: str, account_id: str = "A100", *, recorder: Recorder | None = None) -> "tuple":
    global rec
    rec = recorder or Recorder("billing-support", policy=policy)
    rec.trace.input = {"question": question, "account_id": account_id}
    out = graph.invoke({"question": question, "account_id": account_id})
    rec.finish(out.get("answer", ""), status="ok")
    return rec.trace


def main() -> int:
    print("=" * 62)
    print("run 1 — allowed path (lookup)")
    print("=" * 62)
    t1 = run("what plan is this account on")
    print(f"answer      : {t1.final_output}")
    print(f"tools used  : {t1.tools_used}")
    print(f"fingerprint : {t1.fingerprint[:24]}")
    c1 = score_trace(t1, require_tools=["lookup_account"], forbid_tools=["issue_refund"])
    print(c1.summary())

    print()
    print("=" * 62)
    print("run 2 — allowed path (refund)")
    print("=" * 62)
    t2 = run("I want a refund please")
    print(f"answer      : {t2.final_output}")
    print(f"tools used  : {t2.tools_used}")
    c2 = score_trace(t2, require_tools=["issue_refund"])
    print(c2.summary())

    print()
    print("=" * 62)
    print("determinism — replay run 1 five times, offline, no key")
    print("=" * 62)
    r = replay_trace(t1, times=5, verify=True)
    print(f"replays={r.replay_count} deterministic={r.deterministic} "
          f"calls_replayed={r.calls_replayed}")
    print(f"fingerprint matches original: {r.fingerprint == t1.fingerprint}")

    print()
    print("=" * 62)
    print("regression — diff two runs of the same graph")
    print("=" * 62)
    drift = compare_traces(t1, t2)
    print(drift.report())

    print()
    print("=" * 62)
    print("policy — a call the agent should never make")
    print("=" * 62)
    d = policy.evaluate("drop_database", {"confirm": True})
    print(f"drop_database -> allowed={d.allowed} reason={d.reason!r}")
    d2 = policy.evaluate("issue_refund", {"amount": 49.0})
    print(f"issue_refund with an extra/unknown arg set -> allowed={d2.allowed} "
          f"reason={d2.reason!r}")

    print()
    print("=" * 62)
    print("persist — trace is plain JSON, safe to commit")
    print("=" * 62)
    t1.save("traces/billing_support_lookup.json")
    print("wrote traces/billing_support_lookup.json")

    ok = c1.passed and c2.passed and r.deterministic
    print()
    print("OVERALL:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
