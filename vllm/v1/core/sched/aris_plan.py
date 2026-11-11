# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experiment 212 (aris/layout-ctrlpath): replay a recorded engine schedule.

fsid counts the non-empty schedule() calls (forward steps). A request whose
kv_transfer_params carry "aris_key" is logged to the loop log when it is added
(padd), when its CPU -> GPU load is admitted (pload) and when it finishes
(pfin), each with the fsid at that moment. A record run (plain replay) gives the
plan; follow runs replay it.

A request that also carries "aris_plan" = {"ord", "arr", "prev", "off"[, "join"]}
is held outside the waiting queue until the go file (VLLM_ARIS_PLAN_GO) exists,
then until it is due: rel >= arr and, when prev is set, prev has finished and
rel >= fin(prev) + off, rel being fsid minus the fsid at go (fin(prev) is
counted the same way). Due requests enter the waiting queue in "ord" order. When
nothing runs or waits and no held request is due, the held requests with the
smallest due step are released (event ff), as an idle engine takes no steps.

Optional "arr0"/"off0": the call's arrival step in the record run (its admission
"arr"/"off" can be later when it waited for KV space); when the arrival step is
reached the event parr is logged once (TTFT is counted from there).

"join" (variant A): a request restored from the CPU tier leaves
WAITING_FOR_REMOTE_KVS only "join" forward steps after its load was admitted;
when that step is reached and its load has not finished, schedule() schedules
nothing (barrier) until it has.
"""

import os
import time

from vllm.v1.engine import loop_log

GO = os.environ.get("VLLM_ARIS_PLAN_GO") or None


def _params(request) -> dict:
    return request.kv_transfer_params or {}


class PlanState:
    def __init__(self) -> None:
        self.fsid = 0
        self.go_fsid: int | None = None
        self.held: list = []
        self.fin: dict[str, int] = {}
        self.n_ff = 0

    # bookkeeping ------------------------------------------------------------
    def rel(self) -> int:
        return self.fsid - (self.go_fsid or 0)

    def on_add(self, request) -> bool:
        """Log the arrival; return True when the request is held (not queued)."""
        p = _params(request)
        key = p.get("aris_key")
        if key is not None:
            loop_log.log("padd", time.time(), key=key, fsid=self.fsid,
                         req=request.request_id, plan="aris_plan" in p)
        if "aris_plan" in p:
            self.held.append(request)
            return True
        return False

    def on_load(self, request) -> None:
        request._aris_adm = self.fsid
        key = _params(request).get("aris_key")
        if key is not None:
            loop_log.log("pload", time.time(), key=key, fsid=self.fsid)

    def on_finish(self, request) -> None:
        key = _params(request).get("aris_key")
        if key is None:
            return
        self.fin[key] = self.rel() if self.go_fsid is not None else self.fsid
        loop_log.log("pfin", time.time(), key=key, fsid=self.fsid)

    def drop(self, request) -> None:
        if request in self.held:
            self.held.remove(request)

    # gate -------------------------------------------------------------------
    def _due_step(self, request, arr="arr", off="off") -> int | None:
        """Relative step from which the request may run, None while its prev runs."""
        pl = _params(request)["aris_plan"]
        if arr not in pl:
            return None
        due = int(pl[arr])
        prev = pl.get("prev")
        if prev is not None:
            if prev not in self.fin:
                return None
            due = max(due, self.fin[prev] + int(pl.get(off, 0)))
        return due

    def release(self, sched) -> None:
        """Move due held requests into the waiting queue in plan order."""
        if not self.held:
            return
        if self.go_fsid is None:
            if GO is None or not os.path.exists(GO):
                time.sleep(0.002)   # before go the engine only holds requests
                return
            self.go_fsid = self.fsid
            loop_log.log("go", time.time(), fsid=self.fsid, held=len(self.held))
        rel = self.rel()
        for r in self.held:   # arrival (record run) reached: log once
            if not getattr(r, "_aris_arrived", False):
                a0 = self._due_step(r, "arr0", "off0")
                if a0 is not None and a0 <= rel:
                    r._aris_arrived = True
                    loop_log.log("parr", time.time(), key=_params(r)["aris_key"], fsid=self.fsid, rel=rel)
        steps = {id(r): self._due_step(r) for r in self.held}
        due = [r for r in self.held if steps[id(r)] is not None and steps[id(r)] <= rel]
        idle = not (sched.running or sched.waiting or sched.skipped_waiting)
        if not due and idle:
            cand = [steps[id(r)] for r in self.held if steps[id(r)] is not None]
            if cand:
                m = min(cand)
                due = [r for r in self.held if steps[id(r)] == m]
                self.n_ff += 1
                loop_log.log("ff", time.time(), fsid=self.fsid, rel=rel, to=m,
                             keys=[_params(r)["aris_key"] for r in due])
        if not due:
            if idle:
                time.sleep(0.0005)
            return
        due.sort(key=lambda r: int(_params(r)["aris_plan"]["ord"]))
        for r in due:
            self.held.remove(r)
            sched._enqueue_waiting_request(r)
            loop_log.log("rel", time.time(), key=_params(r)["aris_key"], fsid=self.fsid,
                         rel=rel)

    def join_ok(self, request) -> bool:
        """Variant A: may this loaded request leave WAITING_FOR_REMOTE_KVS now?"""
        pl = _params(request).get("aris_plan")
        if not pl or pl.get("join") is None:
            return True
        return self.fsid >= getattr(request, "_aris_adm", 0) + int(pl["join"])

    def barrier(self, sched) -> bool:
        """Variant A: a restored request is due to join but its load is not done."""
        for r in sched.skipped_waiting:
            pl = _params(r).get("aris_plan")
            if (pl and pl.get("join") is not None
                    and self.fsid >= getattr(r, "_aris_adm", 0) + int(pl["join"])
                    and r.request_id not in sched.finished_recving_kv_req_ids
                    and r.status.name == "WAITING_FOR_REMOTE_KVS"):
                time.sleep(0.0002)
                return True
        return False
