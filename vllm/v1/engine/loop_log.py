# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experiment-only engine loop timeline (aris/layout-ctrlpath).

Set VLLM_LOOP_LOG=<prefix> to append one JSON line per event to <prefix>.<pid>
(the API server and the engine core write separate files). Wall-clock seconds.
Events: in_add (input thread decode + preprocess), req (main thread handles a
client request), sched / wait / upd (step_with_batch_queue phases), send (output
thread encode + send), fe (frontend output processing), lag (asyncio event loop
woke up late by more than 5 ms).
"""

import json
import os
import threading

_PATH = os.environ.get("VLLM_LOOP_LOG") or None
_lock = threading.Lock()
_f = None


def enabled() -> bool:
    return _PATH is not None


def log(ev: str, t0: float, t1: float | None = None, **kw) -> None:
    global _f
    if _PATH is None:
        return
    rec = {"ev": ev, "t0": t0, "t1": t0 if t1 is None else t1, **kw}
    with _lock:
        if _f is None:
            _f = open(f"{_PATH}.{os.getpid()}", "a", buffering=1)
        _f.write(json.dumps(rec) + "\n")
