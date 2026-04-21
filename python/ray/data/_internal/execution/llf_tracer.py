"""JSONL tracer for LLF scheduling decisions.

Emits one record per scheduling tick (rate-limited by llf_trace_min_interval)
capturing which operators were eligible, their deadlines and queue depths,
and which one was selected.
"""

import json
import threading
import time
from typing import Any, Dict, List, Optional


class LLFTracer:
    """Append-only JSONL writer."""

    def __init__(self, path: str, min_interval: float = 0.05):
        self._path = path
        self._min_interval = min_interval
        self._lock = threading.Lock()
        self._last_write = 0.0
        self._start = time.perf_counter()
        self._f = open(path, "a", buffering=1)

    def _ready(self) -> bool:
        now = time.perf_counter()
        if now - self._last_write < self._min_interval:
            return False
        self._last_write = now
        return True

    def log_decision(
        self,
        policy: str,
        latency_target: float,
        inter_arrival_time: float,
        ops_info: List[Dict[str, Any]],
        selected: Optional[str],
    ) -> None:
        with self._lock:
            if not self._ready():
                return
            rec = {
                "t": time.perf_counter() - self._start,
                "policy": policy,
                "L": latency_target,
                "T": inter_arrival_time,
                "ops": ops_info,
                "selected": selected,
            }
            self._f.write(json.dumps(rec) + "\n")

    def close(self) -> None:
        with self._lock:
            try:
                self._f.flush()
                self._f.close()
            except Exception:
                pass


_tracer_singleton: Optional[LLFTracer] = None
_tracer_lock = threading.Lock()


def get_tracer(path: Optional[str], min_interval: float) -> Optional[LLFTracer]:
    """Return the process-wide tracer, constructing it on first call.

    Returns None if path is falsy. Reuses the same tracer across executions
    within a process so a long-running job appends to one file.
    """
    global _tracer_singleton
    if not path:
        return None
    with _tracer_lock:
        if _tracer_singleton is None or _tracer_singleton._path != path:
            if _tracer_singleton is not None:
                _tracer_singleton.close()
            _tracer_singleton = LLFTracer(path, min_interval)
    return _tracer_singleton
