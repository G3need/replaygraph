"""Trace and step records.

A :class:`Trace` is the unit of truth in replaygraph. Everything the
harness does (replay, scoring, drift detection) reads a trace, and a
trace is plain JSON-friendly data so it can be committed to a repo and
diffed by ordinary tooling.

Design note: we deliberately keep the trace free of any framework
object (no messages, no runnables). That is what makes it stable
across langgraph versions and safe to store.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator, Literal

Kind = Literal["model", "tool", "policy", "note"]


def _canonical(obj: Any) -> str:
    """Canonical JSON used for hashing.

    sort_keys + no incidental whitespace means the same logical content
    always hashes to the same digest, which is the whole point.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def content_hash(obj: Any) -> str:
    return hashlib.sha256(_canonical(obj).encode("utf-8")).hexdigest()


@dataclass
class ToolCall:
    """One tool invocation: what was asked, what came back, and whether
    policy allowed it."""

    name: str
    args: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    allowed: bool = True
    reason: str = ""
    error: str | None = None
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        return content_hash({"name": self.name, "args": self.args})


@dataclass
class Step:
    """A single node execution inside the graph."""

    node: str
    kind: Kind
    order: int = 0
    started_at: float = 0.0
    duration_ms: float = 0.0
    # For kind == "model"
    model: str = ""
    prompt: str = ""
    output: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # For kind == "tool"
    calls: list[ToolCall] = field(default_factory=list)
    # For kind == "policy" / "note"
    detail: str = ""
    # Diagnostics
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Step":
        d = dict(d)
        d["calls"] = [ToolCall(**c) for c in d.get("calls", [])]
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Trace:
    """A complete agent run."""

    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    graph: str = "agent"
    seed: int = 0
    started_at: float = field(default_factory=time.time)
    input: dict[str, Any] = field(default_factory=dict)
    final_output: str = ""
    status: Literal["ok", "error", "blocked"] = "ok"
    steps: list[Step] = field(default_factory=list)

    # ------------------------------------------------------------------
    def add(self, step: Step) -> Step:
        step.order = len(self.steps)
        if not step.started_at:
            step.started_at = time.time()
        self.steps.append(step)
        return step

    # ------------------------------------------------------------------
    @property
    def tool_calls(self) -> Iterator[ToolCall]:
        for s in self.steps:
            yield from s.calls

    @property
    def denied_calls(self) -> list[ToolCall]:
        return [c for c in self.tool_calls if not c.allowed]

    @property
    def failed_calls(self) -> list[ToolCall]:
        return [c for c in self.tool_calls if c.error]

    @property
    def token_usage(self) -> dict[str, int]:
        p = sum(s.prompt_tokens for s in self.steps)
        c = sum(s.completion_tokens for s in self.steps)
        return {"prompt": p, "completion": c, "total": p + c}

    @property
    def duration_ms(self) -> float:
        return sum(s.duration_ms for s in self.steps)

    @property
    def tools_used(self) -> list[str]:
        seen: list[str] = []
        for c in self.tool_calls:
            if c.name not in seen:
                seen.append(c.name)
        return seen

    @property
    def fingerprint(self) -> str:
        """Stable digest of the run's *behaviour*.

        Deliberately excludes timestamps and durations so that a rerun
        of a deterministic agent produces an identical fingerprint.
        """
        payload = [
            {
                "node": s.node,
                "kind": s.kind,
                "model": s.model,
                "output": s.output,
                "detail": s.detail,
                "error": s.error,
                "calls": [
                    {
                        "name": c.name,
                        "args": c.args,
                        "allowed": c.allowed,
                        "result": c.result,
                        "error": c.error,
                    }
                    for c in s.calls
                ],
            }
            for s in self.steps
        ]
        return content_hash({"input": self.input, "steps": payload})

    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["fingerprint"] = self.fingerprint
        d["token_usage"] = self.token_usage
        d["tools_used"] = self.tools_used
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Trace":
        d = dict(d)
        steps = [Step.from_dict(s) for s in d.pop("steps", [])]
        d.pop("fingerprint", None)
        d.pop("token_usage", None)
        d.pop("tools_used", None)
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        t = cls(**{k: v for k, v in d.items() if k in known})
        t.steps = steps
        return t

    # ------------------------------------------------------------------
    def save(self, path: str) -> str:
        import os

        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.to_json())
        return path

    @classmethod
    def load(cls, path: str) -> "Trace":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))
