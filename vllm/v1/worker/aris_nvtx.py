# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""aris/layout-ctrlpath: NVTX ranges for Nsight Systems timelines (VLLM_ARIS_NVTX=1).

kv_prepare / kv_submit  descriptor build and copy call of one KV transfer (offload handler)
decode_step / sample    model runner execute_model / sample_tokens, with the batch shape
critical_copy           small host<->GPU copies (Tensor.copy_ / .to / .cpu, V2 UVA buffer pool)
token_ready             host wait for the sampled-token copy (AsyncOutput.get_output)

VLLM_ARIS_TRACE=1 (async restore timing spec, 2026-11-11) turns these on and adds the engine-loop ranges with one
batch id B per scheduler.schedule() call (the loop log's sid when VLLM_LOOP_LOG is set): schedule, dispatch,
sample_dispatch, output_wait/scheduler_update (owner_B), worker ranges carrying B, post_forward/start_load_kv, and
markers for restore_plan/fence/complete_report/reflect/promote and the new requests of each batch.

Ranges cover host execution only; a range end is not the GPU completion time.
"""
import contextlib
import itertools
import os
import sys

import torch

TRACE = os.environ.get("VLLM_ARIS_TRACE", "0") == "1"
ON = os.environ.get("VLLM_ARIS_NVTX", "0") == "1" or TRACE
SMALL = 4 << 20
_step = itertools.count()
_bid = itertools.count()
_cur = {"step": -1, "bid": -1}
_NULL = contextlib.nullcontext()
_compiling = getattr(getattr(torch, "compiler", None), "is_compiling", lambda: False)   # dynamo tracing: no wrappers


def next_bid() -> int:
    return next(_bid)


def cur_bid() -> int:
    return _cur["bid"]


def r(name: str, b: int):
    """Range "<name> B=<b>" when tracing, else a no-op (no string formatting on the off path)."""
    return rng(f"{name} B={b}") if TRACE else _NULL


def mark(msg: str) -> None:
    if TRACE:
        torch.cuda.nvtx.mark(msg)


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
        if _compiling():
            return copy_(self, src, *args, **kwargs)
        if isinstance(src, torch.Tensor) and small_xfer(self.is_cuda, src.is_cuda, self.nbytes):
            with rng(f"critical_copy {'h2d' if self.is_cuda else 'd2h'} {self.nbytes} {_caller()} step={_cur['step']} "
                     f"B={_cur['bid']}"):
                return copy_(self, src, *args, **kwargs)
        return copy_(self, src, *args, **kwargs)

    def to_nvtx(self, *args, **kwargs):
        if _compiling():
            return to(self, *args, **kwargs)
        dev = kwargs.get("device", args[0] if args else None)
        if isinstance(dev, (str, torch.device)) and self.nbytes <= SMALL:
            d_cuda = torch.device(dev).type == "cuda"
            if small_xfer(d_cuda, self.is_cuda, self.nbytes):
                with rng(f"critical_copy {'h2d' if d_cuda else 'd2h'} {self.nbytes} {_caller()} step={_cur['step']} "
                         f"B={_cur['bid']}"):
                    return to(self, *args, **kwargs)
        return to(self, *args, **kwargs)

    def cpu_nvtx(self, *args, **kwargs):
        if _compiling():
            return cpu(self, *args, **kwargs)
        if small_xfer(False, self.is_cuda, self.nbytes):
            with rng(f"critical_copy d2h {self.nbytes} {_caller()} step={_cur['step']} B={_cur['bid']}"):
                return cpu(self, *args, **kwargs)
        return cpu(self, *args, **kwargs)

    torch.Tensor.copy_, torch.Tensor.to, torch.Tensor.cpu = copy_nvtx, to_nvtx, cpu_nvtx
    try:
        from vllm.v1.worker.gpu import buffer_utils as bu
    except Exception:
        return
    pool_copy = bu.UvaBufferPool.copy_to_gpu

    def copy_to_gpu_nvtx(self, x, out=None):
        if _compiling() or torch.cuda.is_current_stream_capturing():
            return pool_copy(self, x, out)
        n = x.nbytes if hasattr(x, "nbytes") else len(x) * 4
        with rng(f"critical_copy h2d {n} uva_pool:{_caller()} step={_cur['step']} B={_cur['bid']}"):
            return pool_copy(self, x, out)

    bu.UvaBufferPool.copy_to_gpu = copy_to_gpu_nvtx


def install(runner) -> None:
    """Wrap the model runner's step entry points, the copy paths and AsyncOutput.get_output."""
    if not ON:
        return
    if TRACE:
        import threading
        print(f"aris_trace runner={type(runner).__module__}.{type(runner).__name__} pid={os.getpid()} "
              f"tid={threading.get_native_id()} thread={threading.current_thread().name}", file=sys.stderr, flush=True)
    ex, st = runner.execute_model, runner.sample_tokens

    def execute_model(scheduler_output, *args, **kwargs):
        n = getattr(scheduler_output, "num_scheduled_tokens", None)
        if kwargs.get("dummy_run") or torch.cuda.is_current_stream_capturing():
            return ex(scheduler_output, *args, **kwargs)
        b = _cur["bid"] = getattr(scheduler_output, "_aris_bid", -1)
        if not n:
            if not TRACE:
                return ex(scheduler_output, *args, **kwargs)
            with rng(f"worker_noforward B={b}"):
                return ex(scheduler_output, *args, **kwargs)
        step = _cur["step"] = next(_step)
        npf = sum(1 for v in n.values() if v > 1)
        with rng(f"decode_step {step} reqs={len(n)} prefill={npf} "
                 f"tokens={scheduler_output.total_num_scheduled_tokens} B={b}"):
            return ex(scheduler_output, *args, **kwargs)

    def sample_tokens(*args, **kwargs):
        step, b = _cur["step"], _cur["bid"]
        with rng(f"sample {step} B={b}"):
            out = st(*args, **kwargs)
        if out is not None and hasattr(out, "get_output"):
            try:
                out._aris_step = step
                out._aris_bid = b
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
        with rng(f"token_ready {getattr(self, '_aris_step', -1)} owner_B={getattr(self, '_aris_bid', -1)}"):
            return get_output(self)

    au.AsyncOutput.get_output = get_output_nvtx
