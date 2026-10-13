"""Evals: score a trajectory, and diff two trajectories.

The grading philosophy here is deliberately boring. A trace is scored
against *observable facts* — did it stay inside policy, did every tool
resolve, did it terminate, did it stay inside its step and token
budget — rather than against a model's opinion of itself. Those facts
are the ones that break in production, and they are the ones you can
assert in CI.

``compare_traces`` is the regression check: it answers "is this run
still the run we shipped?" and shows exactly where it drifted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from .trace import Trace

DEFAULT_MAX_STEPS = 12
DEFAULT_MAX_TOKENS = 32_000
DEFAULT_MAX_CALLS_PER_TOOL = 8


@dataclass
class Scorecard:
    """Outcome of grading one trace.

    ``passed`` means "no graded invariant was violated". It is not a
    claim of quality, and the ``checks`` list says which invariants ran.
    """

    trace_id: str = ""
    passed: bool = True
    score: float = 0.0
    checks: list[dict[str, Any]] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def add(self, name: str, ok: bool, detail: str = "", weight: float = 1.0) -> None:
        self.checks.append(
            {"check": name, "ok": bool(ok), "detail": detail, "weight": weight}
        )
        if not ok:
            self.passed = False
            self.violations.append(f"{name}: {detail}" if detail else name)
        self._rescore()

    def _rescore(self) -> None:
        total = sum(c["weight"] for c in self.checks) or 1.0
        earned = sum(c["weight"] for c in self.checks if c["ok"])
        self.score = round(earned / total, 4)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    # readable one-liner for CI logs
    def summary(self) -> str:
        flag = "PASS" if self.passed else "FAIL"
        return (
            f"[{flag}] score={self.score:.3f} "
            f"checks={sum(1 for c in self.checks if c['ok'])}/{len(self.checks)} "
            f"violations={len(self.violations)}"
        )


def score_trace(
    trace: Trace,
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_calls_per_tool: int = DEFAULT_MAX_CALLS_PER_TOOL,
    require_tools: Iterable[str] = (),
    forbid_tools: Iterable[str] = (),
    expect_output_substring: str | None = None,
) -> Scorecard:
    """Grade a trace against observable invariants."""
    card = Scorecard(trace_id=trace.trace_id)

    card.add(
        "completed",
        trace.status == "ok",
        f"status={trace.status}",
        weight=3.0,
    )

    denials = trace.denied_calls
    card.add(
        "policy_respected",
        not denials,
        f"{len(denials)} denied call(s): "
        + ", ".join(f"{c.name} ({c.reason})" for c in denials[:3]),
        weight=3.0,
    )

    failures = trace.failed_calls
    card.add(
        "tools_resolved",
        not failures,
        f"{len(failures)} failed call(s): "
        + ", ".join(f"{c.name}: {c.error}" for c in failures[:3]),
        weight=2.0,
    )

    card.add(
        "step_budget",
        len(trace.steps) <= max_steps,
        f"{len(trace.steps)} steps > max {max_steps}",
        weight=1.0,
    )

    usage = trace.token_usage
    card.add(
        "token_budget",
        usage["total"] <= max_tokens,
        f"{usage['total']} tokens > max {max_tokens}",
        weight=1.0,
    )

    used = trace.tools_used
    for name in require_tools:
        card.add(f"used_{name}", name in used, f"tool {name!r} never called", weight=1.0)

    for name in forbid_tools:
        card.add(f"avoided_{name}", name not in used, f"tool {name!r} was called", weight=2.0)

    # loop guard: the same tool called many times usually means the agent
    # is stuck retrying rather than progressing
    counts: dict[str, int] = {}
    for c in trace.tool_calls:
        counts[c.name] = counts.get(c.name, 0) + 1
    runaway = {n: n_ for n, n_ in counts.items() if n_ > max_calls_per_tool}
    card.add(
        "no_runaway_loop",
        not runaway,
        f"tools called too often: {runaway}",
        weight=2.0,
    )

    if expect_output_substring is not None:
        card.add(
            "output_contains",
            expect_output_substring.lower() in (trace.final_output or "").lower(),
            f"expected {expect_output_substring!r} in output",
            weight=2.0,
        )

    card.metrics = {
        "steps": len(trace.steps),
        "tool_calls": len(list(trace.tool_calls)),
        "tools_used": used,
        "denied": len(denials),
        "failed": len(failures),
        "tokens": usage,
        "duration_ms": round(trace.duration_ms, 2),
    }
    return card


@dataclass
class Drift:
    fingerprint_changed: bool
    old_fingerprint: str
    new_fingerprint: str
    status_changed: bool
    output_changed: bool
    steps_added: list[str] = field(default_factory=list)
    steps_removed: list[str] = field(default_factory=list)
    tools_added: list[str] = field(default_factory=list)
    tools_removed: list[str] = field(default_factory=list)
    token_delta: int = 0

    @property
    def is_clean(self) -> bool:
        return not any(
            [
                self.fingerprint_changed,
                self.status_changed,
                self.steps_added,
                self.steps_removed,
                self.tools_added,
                self.tools_removed,
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["is_clean"] = self.is_clean
        return d

    def report(self) -> str:
        if self.is_clean:
            return "drift: none (behaviour identical)"
        lines = ["drift detected:"]
        if self.fingerprint_changed:
            lines.append(f"  fingerprint {self.old_fingerprint[:12]} -> {self.new_fingerprint[:12]}")
        if self.status_changed:
            lines.append("  status changed")
        if self.output_changed:
            lines.append("  final output changed")
        if self.steps_added:
            lines.append(f"  steps added: {self.steps_added}")
        if self.steps_removed:
            lines.append(f"  steps removed: {self.steps_removed}")
        if self.tools_added:
            lines.append(f"  tools added: {self.tools_added}")
        if self.tools_removed:
            lines.append(f"  tools removed: {self.tools_removed}")
        if self.token_delta:
            lines.append(f"  token delta: {self.token_delta:+d}")
        return "\n".join(lines)


def compare_traces(old: Trace, new: Trace) -> Drift:
    """Structural + behavioural diff between two runs of the same graph."""
    old_nodes = [s.node for s in old.steps]
    new_nodes = [s.node for s in new.steps]
    old_tools, new_tools = set(old.tools_used), set(new.tools_used)

    return Drift(
        fingerprint_changed=old.fingerprint != new.fingerprint,
        old_fingerprint=old.fingerprint,
        new_fingerprint=new.fingerprint,
        status_changed=old.status != new.status,
        output_changed=(old.final_output or "") != (new.final_output or ""),
        steps_added=[n for n in new_nodes if n not in old_nodes],
        steps_removed=[n for n in old_nodes if n not in new_nodes],
        tools_added=sorted(new_tools - old_tools),
        tools_removed=sorted(old_tools - new_tools),
        token_delta=new.token_usage["total"] - old.token_usage["total"],
    )
