# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep the model runner's step-critical copies off the copy engine.

With async scheduling, step k+1's input copies (input ids, positions, block
tables, ...) are issued right after forward k is launched and wait for it on
the same stream. A CPU-tier KV load issued on the same process' copy engine
becomes ready at about the same time; the copy engine then serves the whole
load before the small input copy, so decode stops for the duration of the
load. The same happens to the sampled-token copy behind a GPU->CPU store.

When VLLM_CTRL_UVA=1, small non-blocking copies between pinned host memory and
the GPU that go through ``Tensor.copy_``, ``Tensor.to`` or the V2 model
runner's ``UvaBufferPool.copy_to_gpu`` run as a tiny Triton kernel that reads
or writes the pinned memory directly (UVA). KV transfers are not touched and
stay on the copy engine at full rate. Copies during CUDA graph capture, large
or non-contiguous copies and pageable memory keep the normal path.
"""

import collections

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.triton_utils import HAS_TRITON, tl, triton

logger = init_logger(__name__)

MAX_BYTES = 4 << 20  # only control-sized copies
BLOCK = 4096

_state = {"n": 0}
_orig_copy = None
_orig_to = None
_orig_copy_to_gpu = None
# The pinned caching host allocator only knows a block is in use by the GPU
# when the copy goes through Tensor.copy_ (it records an event); the kernel
# bypasses that, so hold both tensors until the kernel has run, or a freed
# temporary (e.g. async_tensor_h2d's pinned staging tensor) is reused before
# the GPU reads it.
_pending: collections.deque = collections.deque()


if HAS_TRITON:

    @triton.jit(do_not_specialize=["src_addr", "dst_addr", "nbytes"])
    def _uva_copy(src_addr, dst_addr, nbytes, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        src = src_addr.to(tl.pointer_type(tl.uint8))
        dst = dst_addr.to(tl.pointer_type(tl.uint8))
        m = offs < nbytes
        tl.store(dst + offs, tl.load(src + offs, mask=m), mask=m)


def _small(t: torch.Tensor) -> bool:
    nbytes = t.numel() * t.element_size()
    return 0 < nbytes <= MAX_BYTES and t.is_contiguous()


def _launch(dst: torch.Tensor, src: torch.Tensor) -> None:
    nbytes = dst.numel() * dst.element_size()
    _uva_copy[(triton.cdiv(nbytes, BLOCK),)](
        src.data_ptr(), dst.data_ptr(), nbytes, BLOCK=BLOCK
    )
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
        and not torch.cuda.is_current_stream_capturing()
    ):
        try:
            dev = args[0] if args else kwargs["device"]
            if isinstance(dev, (str, torch.device, int)):
                dev = (
                    torch.device("cuda", dev)
                    if isinstance(dev, int)
                    else torch.device(dev)
                )
                if (
                    dev.type == "cuda"
                    and self.device.type == "cpu"
                    and self.is_pinned()
                    and _small(self)
                ):
                    if dev.index is None:
                        dev = torch.device("cuda", torch.cuda.current_device())
                    out = torch.empty(self.shape, dtype=self.dtype, device=dev)
                    with torch.cuda.device(dev):
                        _launch(out, self)
                    return out
                if dev.type == "cpu" and self.is_cuda and _small(self):
                    out = torch.empty(
                        self.shape, dtype=self.dtype, device="cpu", pin_memory=True
                    )
                    with torch.cuda.device(self.device):
                        _launch(out, self)
                    return out
        except Exception as e:
            logger.warning_once("ctrl_uva to fallback: %s", e)
    return _orig_to(self, *args, **kwargs)


def _patch_v2_buffers() -> None:
    """Model runner V2 stages step inputs in pinned UVA buffers and copies them
    with ``out.copy_(uva_view)`` / ``uva_view.clone()``: a GPU-side view of
    host memory, which the copy engine executes as an H2D copy. Route it
    through the SM kernel too (the UVA view's address is the host address)."""
    global _orig_copy_to_gpu
    try:
        from vllm.v1.worker.gpu import buffer_utils as bu
    except Exception:
        return
    orig = _orig_copy_to_gpu = bu.UvaBufferPool.copy_to_gpu

    def copy_to_gpu(self, x, out=None):
        if torch.cuda.is_current_stream_capturing():
            return orig(self, x, out)
        uva = self.copy_to_uva(x)
        try:
            if _small(uva) and (
                out is None
                or (
                    out.dtype == uva.dtype
                    and out.numel() == uva.numel()
                    and _small(out)
                )
            ):
                dst = (
                    torch.empty(uva.shape, dtype=uva.dtype, device=uva.device)
                    if out is None
                    else out
                )
                _launch(dst, uva)
                return dst
        except Exception as e:
            logger.warning_once("ctrl_uva v2 buffer fallback: %s", e)
        return uva.clone() if out is None else _orig_copy(out, uva, True)

    bu.UvaBufferPool.copy_to_gpu = copy_to_gpu


def install(force: bool = False) -> None:
    """Patch Tensor.copy_ / Tensor.to and the V2 UVA buffer pool in this
    process when VLLM_CTRL_UVA=1 (or force). Idempotent."""
    global _orig_copy, _orig_to
    if _orig_copy is not None or not HAS_TRITON:
        return
    if not (force or envs.VLLM_CTRL_UVA):
        return
    _orig_copy = torch.Tensor.copy_
    _orig_to = torch.Tensor.to
    torch.Tensor.copy_ = _copy_
    torch.Tensor.to = _to
    _patch_v2_buffers()
    logger.info("ctrl_uva installed: step-critical host<->GPU copies run on SMs")


def uninstall() -> None:
    """Undo install() (tests)."""
    global _orig_copy, _orig_to, _orig_copy_to_gpu
    if _orig_copy is None:
        return
    torch.cuda.synchronize()
    torch.Tensor.copy_ = _orig_copy
    torch.Tensor.to = _orig_to
    if _orig_copy_to_gpu is not None:
        from vllm.v1.worker.gpu import buffer_utils as bu

        bu.UvaBufferPool.copy_to_gpu = _orig_copy_to_gpu
    _orig_copy = _orig_to = _orig_copy_to_gpu = None
    _pending.clear()


def count() -> int:
    """Number of copies that took the SM path in this process."""
    return _state["n"]
