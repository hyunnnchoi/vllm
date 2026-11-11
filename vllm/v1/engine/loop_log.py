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
# Experiment 212: in a schedule replay (VLLM_ARIS_PLAN_GO set) the engine spins on empty schedules while it holds requests,
# so sched / wait / upd events of empty schedules are dropped; VLLM_ARIS_UPD_RIDS=1 adds the ids of the requests that got
# tokens to each upd event (engine-side token times), and sched / upd carry fsid (the scheduler's forward-step count).
QUIET = bool(os.environ.get("VLLM_ARIS_PLAN_GO"))
RIDS = bool(os.environ.get("VLLM_ARIS_UPD_RIDS"))
_lock = threading.Lock()
_f = None
SID = 0


def enabled() -> bool:
    return _PATH is not None


def next_sid() -> int:
    global SID
    SID += 1
    return SID


def keep(so) -> bool:
    return not QUIET or so.total_num_scheduled_tokens > 0


def fsid(scheduler):
    return getattr(getattr(scheduler, "_aris", None), "fsid", None)


def upd_fields(so, scheduler, outs) -> dict:
    d = {"fsid": fsid(scheduler)}
    if RIDS:
        d["rids"] = [o.request_id for o in outs if o.new_token_ids]
    return d


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
