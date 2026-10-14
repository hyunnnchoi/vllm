# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""aris/layout-ctrlpath: experiment switches shared by the KV offload worker and the scheduler (same process with the
uniprocess executor). Flags are the 4th field of a "ctrlq:<cap>:<chunk>:<flags>" string in VLLM_KV_LOAD_PACE_FILE:
  t  a thread refills the load queue every 0.5 ms (instead of once per engine step)
  s  shortest remaining load first at chunk boundaries
  e  early completion: the worker's monitor thread marks a request's load done as soon as its end event completes and
     the scheduler promotes the request at its next schedule() instead of after the step's output comes back
  y  yield: when an earlier-arrived request's load is done, later-arrived running prefill chunks leave it enough of the
     step's token budget (decodes are not affected)
  w  loads do not wait for the compute stream (UNSAFE upper bound: only for models without KV zeroing, measurement)
"""
import os
import threading
import time

_FILE = os.environ.get("VLLM_KV_LOAD_PACE_FILE") or None
# Baton-style layer-wise offload: requests are promoted when layer 0 of their load arrived
LW_ASYNC = os.environ.get("VLLM_KV_OFFLOAD_LAYERWISE", "0") == "2"
_st = {"t": 0.0, "mtime": -1.0, "flags": ""}
LOCK = threading.Lock()
EARLY_DONE: set[str] = set()      # request ids whose loads finished (seen by the monitor thread)
JOB_REQ: dict[int, str] = {}      # load job id -> request id
REQ_PENDING: dict[str, int] = {}  # request id -> load jobs not yet finished


def flags() -> str:
    if _FILE is None:
        return ""
    now = time.monotonic()
    if now - _st["t"] > 0.05:
        _st["t"] = now
        try:
            m = os.stat(_FILE).st_mtime
            if m != _st["mtime"]:
                _st["mtime"] = m
                with open(_FILE) as f:
                    raw = f.read().strip()
                parts = raw.split(":")
                _st["flags"] = parts[3] if raw.startswith("ctrlq:") and len(parts) > 3 else ""
        except OSError:
            pass
    return _st["flags"]


def job_submitted(job_id: int, req_id: str) -> None:
    with LOCK:
        JOB_REQ[job_id] = req_id
        REQ_PENDING[req_id] = REQ_PENDING.get(req_id, 0) + 1


def job_done(job_id: int) -> None:
    with LOCK:
        req = JOB_REQ.pop(job_id, None)
        if req is None:
            return
        n = REQ_PENDING.get(req, 1) - 1
        if n <= 0:
            REQ_PENDING.pop(req, None)
            EARLY_DONE.add(req)
        else:
            REQ_PENDING[req] = n


def take_early_done() -> set[str]:
    with LOCK:
        out = set(EARLY_DONE)
        EARLY_DONE.clear()
    return out
