"""The gate: a default-deny policy boundary in front of every tool.

Why a gate at all: once an agent can *act*, every tool is a capability
granted to the user's data. The failure mode that matters is not the
model saying something wrong, it is the model calling something it
should not have. So authorisation happens before execution, and a
denial is recorded as a first-class fact rather than an exception.

Every decision is a :class:`Decision` with a machine-readable reason,
which is what makes an audit trail useful instead of decorative.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

Rule = Callable[[str, dict[str, Any]], "Decision | bool | str | None"]


@dataclass
class Decision:
    allowed: bool
    reason: str = ""
    rule: str = ""

    def __bool__(self) -> bool:  # so `if decision:` reads naturally
        return self.allowed

    def to_dict(self) -> dict[str, Any]:
        return {"allowed": self.allowed, "reason": self.reason, "rule": self.rule}


ALLOW_ALL = "allow"
DENY_ALL = "deny"


@dataclass
class Policy:
    """Ordered, default-deny policy.

    Rules are evaluated in order and the first one that returns a
    non-``None`` answer wins. If nothing answers, access is denied.
    Default-deny is the only safe default for an agent that can act.

    Example::

        policy = Policy("support-agent")
        policy.allow("read_account", when_arg("account_id", is_str))
        policy.deny("send_email")               # hard block
        policy.allow("*")                        # opening the door
    """

    name: str = "default"
    default: str = DENY_ALL
    _rules: list[tuple[str, Rule, bool]] = field(default_factory=list, repr=False)

    # ------------------------------------------------------------------
    def rule(self, pattern: str, fn: Rule) -> "Policy":
        """Attach a callable rule. Register order == evaluation order."""
        self._rules.append((pattern, fn, False))
        return self

    def allow(self, pattern: str, fn: Rule | None = None) -> "Policy":
        self._rules.append((pattern, fn, True))
        return self

    def deny(self, pattern: str, fn: Rule | None = None) -> "Policy":
        self._rules.append((pattern, fn, False))
        return self

    # ------------------------------------------------------------------
    def matches(self, pattern: str, tool: str) -> bool:
        if pattern in (ALLOW_ALL, DENY_ALL):
            return pattern == ALLOW_ALL
        if pattern == tool:
            return True
        # fnmatch-style, but only for names that actually contain a wildcard
        if any(ch in pattern for ch in "*?[]"):
            return re.fullmatch(_glob_to_regex(pattern), tool) is not None
        return False

    # ------------------------------------------------------------------
    def evaluate(self, tool: str, args: dict[str, Any] | None = None) -> Decision:
        args = args or {}
        for pattern, fn, is_allow in self._rules:
            if not self.matches(pattern, tool):
                continue
            if fn is None:
                return Decision(is_allow, f"rule {pattern!r}", pattern)
            try:
                out = fn(tool, args)
            except Exception as exc:  # a broken rule must fail closed
                return Decision(False, f"rule {pattern!r} raised {exc!r}", pattern)
            if out is None:
                continue
            if isinstance(out, Decision):
                return out
            if isinstance(out, bool):
                return Decision(out, f"rule {pattern!r}", pattern)
            return Decision(True if out else False, str(out), pattern)
        if self.default == ALLOW_ALL:
            return Decision(True, "default allow", "*")
        return Decision(False, f"no rule matched {tool!r}; default deny", "*")

    __call__ = evaluate

    # ------------------------------------------------------------------
    @property
    def rules(self) -> list[dict[str, Any]]:
        return [
            {"pattern": p, "kind": "allow" if a else "deny", "fn": fn is not None}
            for p, fn, a in self._rules
        ]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "default": self.default, "rules": self.rules}


def _glob_to_regex(pat: str) -> str:
    out = []
    for ch in pat:
        if ch == "*":
            out.append(".*")
        elif ch == "?":
            out.append(".")
        else:
            out.append(re.escape(ch))
    return "".join(out)


# ----------------------------------------------------------------------
# Small argument predicates so rules read like prose.

def is_str(v: Any) -> bool:
    return isinstance(v, str)


def is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def max_len(n: int) -> Callable[[Any], bool]:
    return lambda v: isinstance(v, str) and len(v) <= n


def one_of(*values: Any) -> Callable[[Any], bool]:
    return lambda v: v in values


def less_than(n: float) -> Callable[[Any], bool]:
    return lambda v: is_number(v) and v < n


def when_arg(key: str, pred: Callable[[Any], bool]) -> Callable[[str, dict], bool]:
    """Predicate on a single argument being present and valid."""

    def _check(tool: str, args: dict[str, Any]) -> bool:
        if key not in args:
            return False
        return bool(pred(args[key]))

    return _check


def args_only(*keys: str) -> Callable[[str, dict], bool]:
    """Reject unknown arguments — blocks argument injection."""

    def _check(tool: str, args: dict[str, Any]) -> bool:
        return set(args) <= set(keys)

    return _check


def all_of(*preds: Callable[[str, dict], bool]) -> Callable[[str, dict], bool]:
    def _check(tool: str, args: dict[str, Any]) -> bool:
        return all(p(tool, args) for p in preds)

    return _check


def any_of(*preds: Callable[[str, dict], bool]) -> Callable[[str, dict], bool]:
    def _check(tool: str, args: dict[str, Any]) -> bool:
        return any(p(tool, args) for p in preds)

    return _check


def not_(pred: Callable[[str, dict], bool]) -> Callable[[str, dict], bool]:
    return lambda tool, args: not pred(tool, args)


def deny_pattern(regex: str, on_args: Iterable[str] = ()) -> Callable[[str, dict], bool]:
    """Deny when any listed argument value matches ``regex``.

    Used to block obvious injection payloads (e.g. ``ignore previous``)
    without pretending to be a security boundary.
    """

    rx = re.compile(regex, re.IGNORECASE | re.DOTALL)

    def _check(tool: str, args: dict[str, Any]) -> bool:
        for k in on_args:
            v = args.get(k)
            if isinstance(v, str) and rx.search(v):
                return False
        return True

    return _check
