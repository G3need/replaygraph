"""replaygraph — deterministic replay and evaluation for LangGraph agents.

The core idea: an agent's behaviour is a *trajectory*, not a single
output. If you cannot replay a trajectory bit-for-bit you cannot
regression-test it, and an agent you cannot regression-test is an
agent you cannot safely ship.

This package gives you three things:

1. ``gate``      - a policy boundary that authorises every tool call
                   before it executes (default-deny).
2. ``trace``     - a JSON-serialisable record of a full run: every model
                   step, every tool call, every policy decision, cost.
3. ``replay``    - re-execute a recorded trajectory with no network and
                   no API key, then diff it against the original.

Replay is offline and deterministic by construction, which makes the
determinism check runnable in CI with zero secrets.
"""

from replaygraph.gate import Decision, Policy
from replaygraph.trace import Step, ToolCall, Trace
from replaygraph.replay import ReplayTransport, replay_trace
from replaygraph.score import Scorecard, score_trace, compare_traces
from replaygraph.langgraph_support import (
    Recorder,
    deterministic_config,
    run_recorded,
)

__version__ = "0.1.0"

__all__ = [
    "Decision",
    "Policy",
    "Step",
    "ToolCall",
    "Trace",
    "Recorder",
    "deterministic_config",
    "run_recorded",
    "ReplayTransport",
    "replay_trace",
    "Scorecard",
    "score_trace",
    "compare_traces",
    "__version__",
]
