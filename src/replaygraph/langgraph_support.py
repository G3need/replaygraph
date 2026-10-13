"""LangGraph integration.

This is the module that turns replaygraph from a data model into
something you can wrap around a real LangGraph agent.

The approach is deliberately non-invasive: rather than subclassing or
monkey-patching langgraph internals, we intercept at the two places
that actually matter — the model call and the tool call — via a
``Recorder`` the caller threads through their own nodes. That keeps
working across langgraph upgrades, which is the failure mode that
kills every tracing library that reaches into framework internals.

Recording an agent looks like this::

    from replaygraph import Recorder, Policy
    from replaygraph.langgraph import run_recorded

    rec = Recorder("support-agent", policy=Policy("support"))
    trace = run_recorded(my_graph, {"question": "..."}, recorder=rec)
    trace.save("traces/support.json")
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from .gate import Decision, Policy
from .trace import Step, ToolCall, Trace

ToolFn = Callable[[str, dict[str, Any]], Any]
ModelFn = Callable[[str], str]


@dataclass
class Recorder:
    """Collects a :class:`Trace` while an agent runs.

    Thread one of these through your own node functions. Nothing about
    your graph has to change shape, which is the point.
    """

    graph: str = "agent"
    policy: Policy | None = None
    trace: Trace = field(default_factory=Trace)
    _timers: dict[str, float] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.trace.graph == "agent":
            self.trace.graph = self.graph

    # ------------------------------------------------------------------
    # model steps
    @contextmanager
    def model(
        self,
        node: str,
        model: str = "unknown",
        prompt: str = "",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> Iterator[Step]:
        t0 = time.perf_counter()
        step = self.trace.add(
            Step(
                node=node,
                kind="model",
                model=model,
                prompt=prompt,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                started_at=time.time(),
            )
        )
        try:
            yield step
            step.duration_ms = (time.perf_counter() - t0) * 1000
        except Exception as exc:
            step.error = f"{type(exc).__name__}: {exc}"
            step.duration_ms = (time.perf_counter() - t0) * 1000
            self.trace.status = "error"
            raise

    def note(self, node: str, detail: str) -> Step:
        return self.trace.add(Step(node=node, kind="note", detail=detail))

    # ------------------------------------------------------------------
    # tool calls
    def call_tool(self, name: str, args: dict[str, Any] | None, runner: ToolFn) -> Any:
        """Authorise then execute one tool call, recording the decision.

        If policy denies the call the tool is never invoked and the
        denial lands in the trace. The agent sees a ToolCall with
        ``allowed=False``, so it can react instead of crashing.
        """
        args = args or {}
        decision = self._authorise(name, args)
        call = ToolCall(name=name, args=args, allowed=decision.allowed, reason=decision.reason)

        if not decision.allowed:
            self.trace.status = self.trace.status if self.trace.status == "error" else "blocked"
            self._attach(name, call)
            raise PermissionError(f"tool {name!r} denied by policy: {decision.reason}")

        t0 = time.perf_counter()
        try:
            call.result = runner(name, args)
        except Exception as exc:
            call.error = f"{type(exc).__name__}: {exc}"
            call.latency_ms = (time.perf_counter() - t0) * 1000
            self._attach(name, call)
            raise
        call.latency_ms = (time.perf_counter() - t0) * 1000
        self._attach(name, call)
        return call.result

    def _authorise(self, name: str, args: dict[str, Any]) -> Decision:
        if self.policy is None:
            return Decision(True, "no policy attached", "*")
        return self.policy.evaluate(name, args)

    def _attach(self, node: str, call: ToolCall) -> None:
        """Attach a call to the most recent open model step, or a new step."""
        for step in reversed(self.trace.steps):
            if step.kind == "model":
                step.calls.append(call)
                return
        self.trace.add(Step(node=node, kind="tool", calls=[call]))

    # ------------------------------------------------------------------
    def finish(self, output: str = "", status: str | None = None) -> Trace:
        self.trace.final_output = output
        if status:
            self.trace.status = status  # type: ignore[assignment]
        return self.trace


def run_recorded(
    graph: Any,
    state: dict[str, Any],
    *,
    recorder: Recorder | None = None,
    input_key: str = "question",
    output_key: str = "answer",
    config: Any = None,
) -> Trace:
    """Run a compiled LangGraph graph to completion, recording a trace.

    ``graph`` must be a compiled graph exposing ``.invoke(state)``.
    We record the invocation boundary rather than individual nodes,
    because a caller-supplied ``recorder`` is what instruments the
    inside. This function guarantees you always get a trace back, even
    if the graph raises.
    """
    rec = recorder or Recorder(graph=getattr(graph, "name", "agent"))
    rec.trace.input = {input_key: state.get(input_key, "")}
    if "seed" in state:
        rec.trace.seed = state["seed"]

    with rec.model("__invoke__", model=getattr(graph, "name", "graph")) as step:
        try:
            out = graph.invoke(state, config=config) if config is not None else graph.invoke(state)
        except PermissionError:
            step.output = ""
            rec.finish("", status="blocked")
            return rec.trace
        except Exception as exc:
            step.error = f"{type(exc).__name__}: {exc}"
            rec.finish("", status="error")
            return rec.trace
        text = out.get(output_key, "") if isinstance(out, dict) else str(out)
        step.output = text or ""
        rec.finish(text or "", status="ok")
        return rec.trace


def deterministic_config(seed: int = 0, tags: list[str] | None = None) -> dict[str, Any]:
    """A LangGraph/LangChain config that makes a run reproducible."""
    return {
        "configurable": {"seed": seed, "thread_id": f"replaygraph-{seed}"},
        "tags": ["replaygraph", *(tags or [])],
        "metadata": {"replaygraph": True, "seed": seed},
    }
