# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experiment-only engine loop timeline (aris/layout-ctrlpath).

Set VLLM_LOOP_LOG=<prefix> to append one JSON line per event to <prefix>.<pid>
(the API server and the engine core write separate files). Wall-clock seconds.
Events: in_add (input thread decode + preprocess), req (main thread handles a
client request), sched / wait / upd (step_with_batch_queue phases), send (output
thread encode + send), fe (frontend output processing), lag (asyncio event loop
woke up late by more than 5 ms), adm (the offloading scheduler created a CPU ->
GPU load job).

Experiments 201-208 (2026-11-10): every schedule() call gets a number sid. The
sched event carries it with the batch shape (nreq requests, npf of them with more
than one scheduled token, new / resumed / preempted ids); the upd event that
processes that batch's output carries the same sid and the finished ids; an adm
event carries the sid of the schedule() call that admitted the load.
"""

import json
import os
import threading

_PATH = os.environ.get("VLLM_LOOP_LOG") or None
_lock = threading.Lock()
_f = None
SID = 0


def enabled() -> bool:
    return _PATH is not None


def next_sid() -> int:
    global SID
    SID += 1
    return SID


def sched_fields(so) -> dict:
    nst = so.num_scheduled_tokens
    d = {"nreq": len(nst), "npf": sum(1 for v in nst.values() if v > 1)}
    if so.scheduled_new_reqs:
        d["new_ids"] = [r.req_id for r in so.scheduled_new_reqs]
    resumed = getattr(so.scheduled_cached_reqs, "resumed_req_ids", None)
    if resumed:
        d["res_ids"] = sorted(resumed)
    preempted = getattr(so, "preempted_req_ids", None)
    if preempted:
        d["pre_ids"] = sorted(preempted)
    return d


def log(ev: str, t0: float, t1: float | None = None, **kw) -> None:
    global _f
    if _PATH is None:
        return
    rec = {"ev": ev, "t0": t0, "t1": t0 if t1 is None else t1, **kw}
    with _lock:
        if _f is None:
            _f = open(f"{_PATH}.{os.getpid()}", "a", buffering=1)
        _f.write(json.dumps(rec) + "\n")
