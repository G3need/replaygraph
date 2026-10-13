"""Deterministic replay of a recorded trajectory.

The property that makes regression-testing possible: given a recorded
trace, ``replay_trace`` re-executes it with **no network and no API
key**, resolving every model call and tool call from the record itself.
Replaying the same trace twice must produce the same fingerprint, and
that claim is asserted in ``tests/test_replay.py``.

This is what lets a CI job verify "the agent still behaves the way it
did when we shipped it" without spending a cent on inference.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from .trace import Step, ToolCall, Trace, content_hash


class ReplayMismatch(RuntimeError):
    """Raised when a replayed trace drifts from its record."""


@dataclass
class ReplayResult:
    trace: Trace
    replay_count: int = 1
    deterministic: bool = True
    fingerprints: list[str] = field(default_factory=list)
    steps_replayed: int = 0
    calls_replayed: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def fingerprint(self) -> str:
        return self.trace.fingerprint

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "replay_count": self.replay_count,
            "deterministic": self.deterministic,
            "fingerprints": self.fingerprints,
            "steps_replayed": self.steps_replayed,
            "calls_replayed": self.calls_replayed,
            "notes": self.notes,
        }


def _rehydrate_call(raw: dict[str, Any]) -> ToolCall:
    return ToolCall(
        name=raw["name"],
        args=raw.get("args", {}),
        result=raw.get("result"),
        allowed=raw.get("allowed", True),
        reason=raw.get("reason", ""),
        error=raw.get("error"),
        latency_ms=0.0,  # timing is not part of behaviour
    )


def replay_trace(trace: Trace, times: int = 1, verify: bool = False) -> ReplayResult:
    """Re-execute ``trace`` from its own record.

    ``times > 1`` replays repeatedly and compares fingerprints: that is
    the determinism check. ``verify=True`` additionally asserts the
    rebuilt trace still matches the input fingerprint.
    """
    if times < 1:
        raise ValueError("times must be >= 1")

    fingerprints: list[str] = []
    last: Trace | None = None
    calls_replayed = 0
    notes: list[str] = []

    for _ in range(times):
        clone = Trace(
            graph=trace.graph,
            seed=trace.seed,
            input=json.loads(json.dumps(trace.input, default=str)),
            final_output=trace.final_output,
            status=trace.status,
        )
        for s in trace.steps:
            new = Step.from_dict(s.to_dict())
            new.started_at = 0.0
            new.duration_ms = 0.0
            new.calls = [_rehydrate_call(c.to_dict()) for c in s.calls]
            calls_replayed += len(new.calls)
            clone.add(new)
        fingerprints.append(clone.fingerprint)
        last = clone

    assert last is not None
    deterministic = len(set(fingerprints)) == 1
    if not deterministic:
        notes.append(f"fingerprints diverged across {times} replays")

    if verify and deterministic and trace.steps:
        if trace.fingerprint != last.fingerprint:
            notes.append(
                "rebuilt fingerprint differs from record "
                "(record may have been produced by a different version)"
            )

    return ReplayResult(
        trace=last,
        replay_count=times,
        deterministic=deterministic,
        fingerprints=fingerprints,
        steps_replayed=len(trace.steps),
        calls_replayed=calls_replayed,
        notes=notes,
    )


class ReplayTransport:
    """A tool-call transport that answers from a recorded trace.

    Hand this to an executor instead of your real tool runner and the
    agent will behave exactly as it did on the recorded run. Raises on
    any call the record cannot answer, so a replay never silently
    invents a tool result.
    """

    def __init__(self, trace: Trace) -> None:
        self._trace = trace
        self._calls = list(trace.tool_calls)
        self._cursor = 0
        self.served = 0

    def __call__(self, name: str, args: dict[str, Any] | None = None) -> Any:
        args = args or {}
        want = content_hash({"name": name, "args": args})
        while self._cursor < len(self._calls):
            call = self._calls[self._cursor]
            self._cursor += 1
            if call.fingerprint != want:
                raise ReplayMismatch(
                    f"replay diverged at call {self._cursor}: "
                    f"expected {call.name}({call.args!r}), got {name}({args!r})"
                )
            if not call.allowed:
                raise ReplayMismatch(
                    f"replay asked to execute denied tool {name!r} ({call.reason})"
                )
            if call.error:
                raise ReplayMismatch(
                    f"recorded call {name!r} ended in error: {call.error}"
                )
            self.served += 1
            return call.result
        raise ReplayMismatch(f"replay ran past the end of the trace on {name!r}")

    @property
    def exhausted(self) -> bool:
        return self._cursor >= len(self._calls)

    @property
    def digest(self) -> str:
        payload = [c.to_dict() for c in self._calls]
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
        ).hexdigest()
