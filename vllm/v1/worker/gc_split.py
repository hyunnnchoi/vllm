# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""aris/layout-ctrlpath experiment 109: engine-wide SM partition with CUDA green
contexts (VLLM_GC_SPLIT=<N>).

The device SMs are split once into an N-SM group and the remainder
(cuDevSmResourceSplitByCount; two separate torch GreenContext.create calls
overlap, see exp/108). A green context is made from each. The remainder context
becomes current in the worker process (and in every thread started afterwards),
so all streams, graphs and kernels created later run on the remaining SMs. The
N-SM context is used only by launch_on_part() for the KV copy kernel.
"""

import ctypes
import os
import threading
from ctypes import POINTER, byref, c_char_p, c_int, c_uint, c_void_p

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


class _Sm(ctypes.Structure):
    _fields_ = [("smCount", c_uint)]


class _U(ctypes.Union):
    _fields_ = [("sm", _Sm), ("_oversize", ctypes.c_ubyte * 48)]


class _DevRes(ctypes.Structure):  # CUdevResource_v1 (cuda.h 12.9)
    _fields_ = [("type", c_int), ("_pad", ctypes.c_ubyte * 92), ("u", _U)]


_SIG = {
    "cuGetErrorName": [c_int, POINTER(c_char_p)],
    "cuDeviceGet": [POINTER(c_int), c_int],
    "cuDeviceGetDevResource": [c_int, POINTER(_DevRes), c_int],
    "cuDevSmResourceSplitByCount": [
        POINTER(_DevRes), POINTER(c_uint), POINTER(_DevRes), POINTER(_DevRes),
        c_uint, c_uint,
    ],
    "cuDevResourceGenerateDesc": [POINTER(c_void_p), POINTER(_DevRes), c_uint],
    "cuGreenCtxCreate": [POINTER(c_void_p), c_void_p, c_int, c_uint],
    "cuCtxFromGreenCtx": [POINTER(c_void_p), c_void_p],
    "cuGreenCtxGetDevResource": [c_void_p, POINTER(_DevRes), c_int],
    "cuGreenCtxStreamCreate": [POINTER(c_void_p), c_void_p, c_uint, c_int],
    "cuCtxPushCurrent_v2": [c_void_p],
    "cuCtxPopCurrent_v2": [POINTER(c_void_p)],
    "cuCtxSetCurrent": [c_void_p],
}
_cu = None


def _ck(r: int, what: str) -> None:
    if r != 0:
        s = c_char_p()
        _cu.cuGetErrorName(r, byref(s))
        raise RuntimeError(f"{what}: {s.value}")


class _Part:
    def __init__(self, res: _DevRes, dev: int):
        desc, g, c, s, got = c_void_p(), c_void_p(), c_void_p(), c_void_p(), _DevRes()
        _ck(_cu.cuDevResourceGenerateDesc(byref(desc), byref(res), 1), "GenerateDesc")
        _ck(_cu.cuGreenCtxCreate(byref(g), desc, dev, 0x1), "cuGreenCtxCreate")
        _ck(_cu.cuCtxFromGreenCtx(byref(c), g), "cuCtxFromGreenCtx")
        _ck(_cu.cuGreenCtxGetDevResource(g, byref(got), 1), "GetDevResource")
        _ck(_cu.cuGreenCtxStreamCreate(byref(s), g, 0x1, 0), "GreenCtxStreamCreate")
        self.ctx, self.sm = c.value, got.u.sm.smCount
        self.stream = torch.cuda.ExternalStream(s.value)


_part: _Part | None = None
_rest: _Part | None = None


def enabled() -> bool:
    return _part is not None


def install() -> None:
    global _cu, _part, _rest
    n = int(os.environ.get("VLLM_GC_SPLIT", "0") or 0)
    if n <= 0 or _part is not None:
        return
    torch.cuda.init()
    torch.zeros(1, device="cuda")
    _cu = ctypes.CDLL("libcuda.so.1")
    for k, v in _SIG.items():
        getattr(_cu, k).restype, getattr(_cu, k).argtypes = c_int, v
    dev = c_int()
    _ck(_cu.cuDeviceGet(byref(dev), torch.cuda.current_device()), "cuDeviceGet")
    full, res, rem, nb = _DevRes(), (_DevRes * 1)(), _DevRes(), c_uint(1)
    _ck(_cu.cuDeviceGetDevResource(dev, byref(full), 1), "GetDevResource")
    _ck(
        _cu.cuDevSmResourceSplitByCount(res, byref(nb), byref(full), byref(rem), 0, n),
        "cuDevSmResourceSplitByCount",
    )
    _part, _rest = _Part(res[0], dev.value), _Part(rem, dev.value)
    _ck(_cu.cuCtxSetCurrent(_rest.ctx), "cuCtxSetCurrent")
    rest_ctx = _rest.ctx
    run = threading.Thread.run

    def _run(self):  # threads started later also run in the remainder context
        _cu.cuCtxSetCurrent(rest_ctx)
        run(self)

    threading.Thread.run = _run
    logger.info(
        "gc_split installed: KV copy partition %d SMs, engine %d SMs",
        _part.sm, _rest.sm,
    )


def launch_on_part(launch) -> None:
    """Run launch() on the N-SM stream, ordered after and before the current
    stream (events across the two green contexts)."""
    assert _part is not None and _rest is not None
    cur = torch.cuda.current_stream()
    before = torch.cuda.Event()
    before.record(cur)
    _ck(_cu.cuCtxPushCurrent_v2(_part.ctx), "cuCtxPushCurrent")
    try:
        _part.stream.wait_event(before)
        with torch.cuda.stream(_part.stream):
            launch()
        after = torch.cuda.Event()
        after.record(_part.stream)
    finally:
        p = c_void_p()
        _ck(_cu.cuCtxPopCurrent_v2(byref(p)), "cuCtxPopCurrent")
    cur.wait_event(after)
