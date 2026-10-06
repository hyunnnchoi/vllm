# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""aris/layout-ctrlpath: keep the model runner's control path off the copy engine.

A CPU->GPU KV load (or GPU->CPU store) submitted to the copy engine starves every
other copy in the same direction until it drains, including the model runner's
per-step input copies (input ids, positions, block tables, ...) and the
sampled-token copy back to the host, so decode stops for the whole transfer.
When enabled, small non-blocking copies between pinned host memory and the GPU
that go through ``Tensor.copy_`` or ``Tensor.to`` are done by a tiny Triton
kernel that reads/writes the pinned memory directly (UVA), while KV transfers
stay on the copy engine at full rate.

Enable with VLLM_CTRL_UVA=1, or at runtime by writing a string that starts with
"ctrl" into the file named by VLLM_KV_LOAD_PACE_FILE (re-read every 50 ms).

VLLM_CTRL_TRACE=<path> (measurement): every host<->GPU copy that goes through
Tensor.copy_ / Tensor.to / Tensor.cpu / UvaBufferPool.copy_to_gpu is bracketed
by two timing events on the current stream and logged as one JSON line: wall
time of the call, GPU time when the stream reached the copy (g0) and when the
copy finished (g1), size, direction, call site, stream and whether the SM path
was taken. g0 - t is the wait for earlier work on the same stream; g1 - g0 is
the copy itself (including waiting for the copy engine).
"""
import collections
import json
import os
import sys
import threading
import time

import torch

from vllm.logger import init_logger
from vllm.triton_utils import HAS_TRITON, tl, triton

logger = init_logger(__name__)

MAX_BYTES = 4 << 20   # only control-sized copies
BLOCK = 4096

_ENV_ON = os.environ.get("VLLM_CTRL_UVA", "0") == "1"
_FILE = os.environ.get("VLLM_KV_LOAD_PACE_FILE") or None
_state = {"on": False, "checked": 0.0, "mtime": -1.0, "n": 0}
_orig_copy = None
_orig_to = None
# The pinned caching host allocator only knows a block is in use by the GPU when
# the copy goes through Tensor.copy_ (it records an event); our kernel bypasses
# that, so hold both tensors until the kernel has run, or a freed temporary
# (e.g. async_tensor_h2d's pinned staging tensor) is reused before the GPU reads it.
_pending: collections.deque = collections.deque()

_TRACE = os.environ.get("VLLM_CTRL_TRACE") or None
_tr: dict = {"f": None, "ref": None, "t_ref": 0.0, "q": collections.deque(), "tl": threading.local()}


def _trace_start() -> None:
    if _tr["f"] is not None:
        return
    _tr["f"] = open(_TRACE, "a", buffering=1)
    ref = torch.cuda.Event(enable_timing=True)
    ref.record()
    ref.synchronize()
    _tr["ref"], _tr["t_ref"] = ref, time.time()
    threading.Thread(target=_trace_loop, args=(torch.cuda.current_device(),), daemon=True,
                     name="ctrl-uva-trace").start()
    logger.info("ctrl_uva trace -> %s", _TRACE)


def _trace_loop(dev: int) -> None:
    torch.cuda.set_device(dev)
    q, f, ref, t_ref = _tr["q"], _tr["f"], _tr["ref"], _tr["t_ref"]
    while True:
        try:
            while q and q[0][1].query():
                e0, e1, t, t_ret, op, d, nb, site, sid, sm = q.popleft()
                g0 = t_ref + ref.elapsed_time(e0) / 1e3
                g1 = t_ref + ref.elapsed_time(e1) / 1e3
                f.write(json.dumps(dict(t=round(t, 5), t_ret=round(t_ret, 5), g0=round(g0, 5), g1=round(g1, 5), op=op,
                                        dir=d, bytes=nb, site=site, stream=sid, sm=int(sm))) + "\n")
        except Exception as e:  # never break the engine
            logger.warning_once("ctrl_uva trace: %s", e)
        time.sleep(0.002)


def _site() -> str:
    f = sys._getframe(2)
    for _ in range(16):
        if f is None:
            break
        fn = f.f_code.co_filename
        if not fn.endswith("ctrl_uva.py") and "/torch/" not in fn:
            return f"{fn.split('/vllm/')[-1]}:{f.f_lineno}:{f.f_code.co_name}"
        f = f.f_back
    return "?"


def _traced(op: str, nbytes: int, d: str, call):
    if not torch.cuda.is_available() or torch.cuda.is_current_stream_capturing():
        return call()
    _trace_start()
    s = torch.cuda.current_stream()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    tl = _tr["tl"]
    tl.sm = False
    site = _site()
    t = time.time()
    e0.record(s)
    out = call()
    e1.record(s)
    _tr["q"].append((e0, e1, t, time.time(), op, d, nbytes, site, s.stream_id, getattr(tl, "sm", False)))
    return out


def _target_dev(args, kwargs):
    dev = kwargs.get("device")
    if dev is None and args:
        a0 = args[0]
        if isinstance(a0, torch.Tensor):
            return a0.device
        if isinstance(a0, (str, torch.device, int)):
            dev = a0
    if dev is None:
        return None
    try:
        return torch.device("cuda", dev) if isinstance(dev, int) else torch.device(dev)
    except Exception:
        return None


if HAS_TRITON:

    @triton.jit(do_not_specialize=["src_addr", "dst_addr", "nbytes"])
    def _uva_copy(src_addr, dst_addr, nbytes, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        src = src_addr.to(tl.pointer_type(tl.uint8))
        dst = dst_addr.to(tl.pointer_type(tl.uint8))
        m = offs < nbytes
        tl.store(dst + offs, tl.load(src + offs, mask=m), mask=m)


def _enabled() -> bool:
    if _ENV_ON:
        return True
    if _FILE is None:
        return False
    now = time.monotonic()
    if now - _state["checked"] > 0.05:
        _state["checked"] = now
        try:
            mtime = os.stat(_FILE).st_mtime
            if mtime != _state["mtime"]:
                _state["mtime"] = mtime
                with open(_FILE) as f:
                    raw = f.read().strip()
                parts = raw.split(":")
                # 'n' in a ctrlq flags field turns the SM control path off
                _state["on"] = raw.startswith("ctrl") and not (
                    raw.startswith("ctrlq:") and len(parts) > 3 and "n" in parts[3]
                )
        except OSError:
            pass
    return _state["on"]


def _small(t: torch.Tensor) -> bool:
    nbytes = t.numel() * t.element_size()
    return 0 < nbytes <= MAX_BYTES and t.is_contiguous()


def _launch(dst: torch.Tensor, src: torch.Tensor) -> None:
    _tr["tl"].sm = True
    nbytes = dst.numel() * dst.element_size()
    _uva_copy[(triton.cdiv(nbytes, BLOCK),)](src.data_ptr(), dst.data_ptr(), nbytes, BLOCK=BLOCK)
    ev = torch.cuda.Event()
    ev.record()
    _pending.append((ev, src, dst))
    while _pending and _pending[0][0].query():
        _pending.popleft()
    _state["n"] += 1
    if _state["n"] in (1, 1000) or _state["n"] % 100000 == 0:
        logger.info("ctrl_uva: %d control copies through the SM kernel", _state["n"])


def _copy_(self, src, non_blocking=False, *args, **kwargs):
    if (
        non_blocking
        and not args
        and not kwargs
        and isinstance(src, torch.Tensor)
        and _enabled()
        and not torch.cuda.is_current_stream_capturing()
    ):
        try:
            if (
                self.dtype == src.dtype
                and self.numel() == src.numel()
                and _small(self)
                and _small(src)
            ):
                if self.is_cuda and src.device.type == "cpu" and src.is_pinned():
                    with torch.cuda.device(self.device):
                        _launch(self, src)
                    return self
                if src.is_cuda and self.device.type == "cpu" and self.is_pinned():
                    with torch.cuda.device(src.device):
                        _launch(self, src)
                    return self
        except Exception as e:  # fall back to the copy engine
            logger.warning_once("ctrl_uva copy_ fallback: %s", e)
    return _orig_copy(self, src, non_blocking, *args, **kwargs)


def _to(self, *args, **kwargs):
    if (
        kwargs.get("non_blocking")
        and set(kwargs) <= {"non_blocking", "device", "dtype"}
        and kwargs.get("dtype") in (None, self.dtype)
        and len(args) + ("device" in kwargs) == 1
        and _enabled()
        and not torch.cuda.is_current_stream_capturing()
    ):
        try:
            dev = args[0] if args else kwargs["device"]
            if isinstance(dev, (str, torch.device, int)):
                dev = torch.device(dev) if not isinstance(dev, int) else torch.device("cuda", dev)
                if dev.type == "cuda" and self.device.type == "cpu" and self.is_pinned() and _small(self):
                    if dev.index is None:
                        dev = torch.device("cuda", torch.cuda.current_device())
                    out = torch.empty(self.shape, dtype=self.dtype, device=dev)
                    with torch.cuda.device(dev):
                        _launch(out, self)
                    return out
                if dev.type == "cpu" and self.is_cuda and _small(self):
                    out = torch.empty(self.shape, dtype=self.dtype, device="cpu", pin_memory=True)
                    with torch.cuda.device(self.device):
                        _launch(out, self)
                    return out
        except Exception as e:
            logger.warning_once("ctrl_uva to fallback: %s", e)
    return _orig_to(self, *args, **kwargs)


def _copy_traced(self, src, non_blocking=False, *args, **kwargs):
    if isinstance(src, torch.Tensor) and self.device.type != src.device.type:
        d = "h2d" if self.device.type == "cuda" else "d2h"
        return _traced("copy_", self.numel() * self.element_size(), d,
                       lambda: _copy_(self, src, non_blocking, *args, **kwargs))
    return _copy_(self, src, non_blocking, *args, **kwargs)


def _to_traced(self, *args, **kwargs):
    dev = _target_dev(args, kwargs)
    if dev is not None and dev.type != self.device.type and {dev.type, self.device.type} == {"cpu", "cuda"}:
        d = "h2d" if dev.type == "cuda" else "d2h"
        return _traced(".to", self.numel() * self.element_size(), d, lambda: _to(self, *args, **kwargs))
    return _to(self, *args, **kwargs)


_orig_cpu = None


def _cpu_traced(self, *args, **kwargs):
    if self.is_cuda:
        return _traced(".cpu", self.numel() * self.element_size(), "d2h", lambda: _orig_cpu(self, *args, **kwargs))
    return _orig_cpu(self, *args, **kwargs)


def _nbytes(x) -> int:
    if isinstance(x, torch.Tensor):
        return x.numel() * x.element_size()
    return int(getattr(x, "nbytes", 0) or (len(x) * 4 if isinstance(x, (list, tuple)) else 0))


def _patch_v2_buffers() -> None:
    """Model runner V2 stages step inputs in pinned UVA buffers and copies them
    with ``out.copy_(uva_view)`` / ``uva_view.clone()``: a GPU-side view of host
    memory, which the copy engine executes as an H2D copy. Route it through the
    SM kernel too (the UVA view's address is the host address)."""
    try:
        from vllm.v1.worker.gpu import buffer_utils as bu
    except Exception:
        return
    orig = bu.UvaBufferPool.copy_to_gpu

    def copy_to_gpu(self, x, out=None):
        if not _enabled() or torch.cuda.is_current_stream_capturing():
            return orig(self, x, out)
        uva = self.copy_to_uva(x)
        try:
            if _small(uva) and (out is None or (out.dtype == uva.dtype and out.numel() == uva.numel() and _small(out))):
                dst = torch.empty(uva.shape, dtype=uva.dtype, device=uva.device) if out is None else out
                _launch(dst, uva)
                return dst
        except Exception as e:
            logger.warning_once("ctrl_uva v2 buffer fallback: %s", e)
        return uva.clone() if out is None else _orig_copy(out, uva, True)

    if _TRACE is not None:
        inner = copy_to_gpu

        def copy_to_gpu_traced(self, x, out=None):
            return _traced("uva_pool", _nbytes(x), "h2d", lambda: inner(self, x, out))

        bu.UvaBufferPool.copy_to_gpu = copy_to_gpu_traced
    else:
        bu.UvaBufferPool.copy_to_gpu = copy_to_gpu


def install() -> None:
    """Patch Tensor.copy_ / Tensor.to (and the V2 UVA buffer pool) in this process (idempotent)."""
    global _orig_copy, _orig_to, _orig_cpu
    if _orig_copy is not None or not HAS_TRITON:
        return
    if not (_ENV_ON or _FILE):
        return
    _orig_copy = torch.Tensor.copy_
    _orig_to = torch.Tensor.to
    if _TRACE is not None:
        _orig_cpu = torch.Tensor.cpu
        torch.Tensor.copy_ = _copy_traced
        torch.Tensor.to = _to_traced
        torch.Tensor.cpu = _cpu_traced
    else:
        torch.Tensor.copy_ = _copy_
        torch.Tensor.to = _to
    _patch_v2_buffers()
    logger.info("ctrl_uva installed (env=%s, control file=%s)", _ENV_ON, _FILE)


def count() -> int:
    return _state["n"]
