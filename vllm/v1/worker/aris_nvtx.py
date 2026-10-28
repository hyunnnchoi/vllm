# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""aris/layout-ctrlpath: NVTX ranges for Nsight Systems timelines (VLLM_ARIS_NVTX=1).

kv_prepare / kv_submit  descriptor build and copy call of one KV transfer (offload handler)
decode_step / sample    model runner execute_model / sample_tokens, with the batch shape
critical_copy           small host<->GPU copies (Tensor.copy_ / .to / .cpu, V2 UVA buffer pool)
token_ready             host wait for the sampled-token copy (AsyncOutput.get_output)

Ranges cover host execution only; a range end is not the GPU completion time.
"""
import contextlib
import itertools
import os
import sys

import torch

ON = os.environ.get("VLLM_ARIS_NVTX", "0") == "1"
SMALL = 4 << 20
_step = itertools.count()
_cur = {"step": -1}


@contextlib.contextmanager
def rng(msg: str):
    torch.cuda.nvtx.range_push(msg)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def _caller() -> str:
    f = sys._getframe(2)
    for _ in range(8):
        if f is None:
            break
        fn = f.f_code.co_filename
        if not fn.endswith(("aris_nvtx.py", "ctrl_uva.py", "buffer_utils.py")) and "/torch/" not in fn:
            return f.f_code.co_name
        f = f.f_back
    return "?"


def _wrap_copies() -> None:
    copy_, to, cpu = torch.Tensor.copy_, torch.Tensor.to, torch.Tensor.cpu

    def small_xfer(dst_cuda: bool, src_cuda: bool, nbytes: int) -> bool:
        return dst_cuda != src_cuda and 0 < nbytes <= SMALL and not torch.cuda.is_current_stream_capturing()

    def copy_nvtx(self, src, *args, **kwargs):
        if isinstance(src, torch.Tensor) and small_xfer(self.is_cuda, src.is_cuda, self.nbytes):
            with rng(f"critical_copy {'h2d' if self.is_cuda else 'd2h'} {self.nbytes} {_caller()} step={_cur['step']}"):
                return copy_(self, src, *args, **kwargs)
        return copy_(self, src, *args, **kwargs)

    def to_nvtx(self, *args, **kwargs):
        dev = kwargs.get("device", args[0] if args else None)
        if isinstance(dev, (str, torch.device)) and self.nbytes <= SMALL:
            d_cuda = torch.device(dev).type == "cuda"
            if small_xfer(d_cuda, self.is_cuda, self.nbytes):
                with rng(f"critical_copy {'h2d' if d_cuda else 'd2h'} {self.nbytes} {_caller()} step={_cur['step']}"):
                    return to(self, *args, **kwargs)
        return to(self, *args, **kwargs)

    def cpu_nvtx(self, *args, **kwargs):
        if small_xfer(False, self.is_cuda, self.nbytes):
            with rng(f"critical_copy d2h {self.nbytes} {_caller()} step={_cur['step']}"):
                return cpu(self, *args, **kwargs)
        return cpu(self, *args, **kwargs)

    torch.Tensor.copy_, torch.Tensor.to, torch.Tensor.cpu = copy_nvtx, to_nvtx, cpu_nvtx
    try:
        from vllm.v1.worker.gpu import buffer_utils as bu
    except Exception:
        return
    pool_copy = bu.UvaBufferPool.copy_to_gpu

    def copy_to_gpu_nvtx(self, x, out=None):
        if torch.cuda.is_current_stream_capturing():
            return pool_copy(self, x, out)
        n = x.nbytes if hasattr(x, "nbytes") else len(x) * 4
        with rng(f"critical_copy h2d {n} uva_pool:{_caller()} step={_cur['step']}"):
            return pool_copy(self, x, out)

    bu.UvaBufferPool.copy_to_gpu = copy_to_gpu_nvtx


def install(runner) -> None:
    """Wrap the model runner's step entry points, the copy paths and AsyncOutput.get_output."""
    if not ON:
        return
    ex, st = runner.execute_model, runner.sample_tokens

    def execute_model(scheduler_output, *args, **kwargs):
        n = getattr(scheduler_output, "num_scheduled_tokens", None)
        if kwargs.get("dummy_run") or not n or torch.cuda.is_current_stream_capturing():
            return ex(scheduler_output, *args, **kwargs)
        step = _cur["step"] = next(_step)
        npf = sum(1 for v in n.values() if v > 1)
        with rng(f"decode_step {step} reqs={len(n)} prefill={npf} "
                 f"tokens={scheduler_output.total_num_scheduled_tokens}"):
            return ex(scheduler_output, *args, **kwargs)

    def sample_tokens(*args, **kwargs):
        step = _cur["step"]
        with rng(f"sample {step}"):
            out = st(*args, **kwargs)
        if out is not None and hasattr(out, "get_output"):
            try:
                out._aris_step = step
            except Exception:
                pass
        return out

    runner.execute_model, runner.sample_tokens = execute_model, sample_tokens
    _wrap_copies()
    try:
        from vllm.v1.worker.gpu import async_utils as au
    except Exception:
        return
    get_output = au.AsyncOutput.get_output

    def get_output_nvtx(self):
        with rng(f"token_ready {getattr(self, '_aris_step', -1)}"):
            return get_output(self)

    au.AsyncOutput.get_output = get_output_nvtx
