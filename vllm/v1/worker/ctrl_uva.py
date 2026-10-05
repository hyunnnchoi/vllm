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
"""
import os
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
                    _state["on"] = f.read().strip().startswith("ctrl")
        except OSError:
            pass
    return _state["on"]


def _small(t: torch.Tensor) -> bool:
    nbytes = t.numel() * t.element_size()
    return 0 < nbytes <= MAX_BYTES and t.is_contiguous()


def _launch(dst: torch.Tensor, src: torch.Tensor) -> None:
    nbytes = dst.numel() * dst.element_size()
    _uva_copy[(triton.cdiv(nbytes, BLOCK),)](src.data_ptr(), dst.data_ptr(), nbytes, BLOCK=BLOCK)
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

    bu.UvaBufferPool.copy_to_gpu = copy_to_gpu


def install() -> None:
    """Patch Tensor.copy_ / Tensor.to (and the V2 UVA buffer pool) in this process (idempotent)."""
    global _orig_copy, _orig_to
    if _orig_copy is not None or not HAS_TRITON:
        return
    if not (_ENV_ON or _FILE):
        return
    _orig_copy = torch.Tensor.copy_
    _orig_to = torch.Tensor.to
    torch.Tensor.copy_ = _copy_
    torch.Tensor.to = _to
    _patch_v2_buffers()
    logger.info("ctrl_uva installed (env=%s, control file=%s)", _ENV_ON, _FILE)


def count() -> int:
    return _state["n"]
