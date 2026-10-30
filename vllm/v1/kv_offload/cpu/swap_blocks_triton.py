# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernel + tuned constants for the ``swap_blocks_batch`` fast path."""

from __future__ import annotations

import os

import torch

from vllm import _custom_ops as ops
from vllm.triton_utils import tl, triton

# Constants tuned empirically on H100 (PCIe Gen5):
#   NUM_SMS         - smallest SM slice within 5% of peak bandwidth at the
#                     8-32 KB block sizes that matter in practice
#   THRESHOLD_BYTES - max payload per descriptor where Triton beats DMA; above
#                     this the C++ cuMemcpyBatchAsync path takes the lead
#   MIN_N           - minimum batch size where Triton's per-launch cost is
#                     amortized; below this DMA wins
NUM_SMS = 12
THRESHOLD_BYTES = 28 * 1024
MIN_N = 16


@triton.jit
def _swap_blocks_kernel(
    src_addrs,
    dst_addrs,
    sizes,
    n_jobs,  # type: ignore[name-defined]
    BYTES_PER_CHUNK: tl.constexpr,  # type: ignore[name-defined]
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    WORDS_PER_CHUNK: tl.constexpr = BYTES_PER_CHUNK // 8
    offsets = tl.arange(0, WORDS_PER_CHUNK)
    job = pid
    while job < n_jobs:
        src = tl.load(src_addrs + job).to(tl.pointer_type(tl.int64))
        dst = tl.load(dst_addrs + job).to(tl.pointer_type(tl.int64))
        words = tl.load(sizes + job) // 8
        for start in range(0, words, WORDS_PER_CHUNK):
            idx = start + offsets
            mask = idx < words
            data = tl.load(src + idx, mask=mask, other=0)
            tl.store(dst + idx, data, mask=mask)
        job += num_progs


def swap_blocks_batch(
    src_addrs: torch.Tensor,
    dst_addrs: torch.Tensor,
    sizes: torch.Tensor,
    is_src_access_order_any: bool = False,
    *,
    bytes_per_chunk: int,
) -> None:
    """Triton implementation of ``swap_blocks_batch`` for small CPU->GPU batches."""
    n = src_addrs.numel()
    # Too few descriptors to amortize Triton's launch cost.
    if n < MIN_N:
        ops.swap_blocks_batch(
            src_addrs,
            dst_addrs,
            sizes,
            is_src_access_order_any=is_src_access_order_any,
        )
        return
    _swap_blocks_kernel[(min(NUM_SMS, n),)](
        src_addrs.to("cuda", non_blocking=True),
        dst_addrs.to("cuda", non_blocking=True),
        sizes.to("cuda", non_blocking=True),
        n,
        BYTES_PER_CHUNK=bytes_per_chunk,
    )


# aris/layout-ctrlpath experiment 109: the kernel above with J descriptors per
# program step ([J, WORDS] block) and a free grid size P / warp count W
# (exp/108_sm-parallelism). "triton:<P>:<W>:<J>" / "tritongc:<P>:<W>:<J>".
@triton.jit
def _swap_blocks_kernel_j(
    src_addrs,
    dst_addrs,
    sizes,
    n_jobs,  # type: ignore[name-defined]
    J: tl.constexpr,  # type: ignore[name-defined]
    WORDS: tl.constexpr,  # type: ignore[name-defined]
    TAG: tl.constexpr,  # type: ignore[name-defined]
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    rows = tl.arange(0, J)
    cols = tl.arange(0, WORDS)
    base = pid * J
    while base < n_jobs:
        jobs = base + rows
        jm = jobs < n_jobs
        src = tl.load(src_addrs + jobs, mask=jm, other=0).to(tl.pointer_type(tl.int64))
        dst = tl.load(dst_addrs + jobs, mask=jm, other=0).to(tl.pointer_type(tl.int64))
        words = tl.load(sizes + jobs, mask=jm, other=0) // 8
        maxw = tl.max(words, axis=0)
        for start in range(0, maxw, WORDS):
            idx = start + cols
            m = idx[None, :] < words[:, None]
            d = tl.load(src[:, None] + idx[None, :], mask=m, other=0)
            tl.store(dst[:, None] + idx[None, :], d, mask=m)
        base += nprog * J


_UVA_DESC = os.environ.get("VLLM_KV_UVA_DESC", "0") == "1"


def _dev_view(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda:
        return t
    if t.is_pinned() and t.is_contiguous():
        from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

        return get_accelerator_view_from_cpu_tensor(t)
    return t.to("cuda", non_blocking=True)


def swap_blocks_batch_j(
    src_addrs: torch.Tensor,
    dst_addrs: torch.Tensor,
    sizes: torch.Tensor,
    is_src_access_order_any: bool = False,
    *,
    bytes_per_chunk: int,
    P: int,
    W: int,
    J: int,
    green: bool = False,
) -> None:
    n = src_addrs.numel()
    if n == 0:
        return
    if _UVA_DESC:
        # read the descriptor arrays from pinned host memory through UVA, so that their upload does not go through
        # the copy engine and wait behind an in-flight bulk DMA load of the same process
        s, d, z = (_dev_view(t) for t in (src_addrs, dst_addrs, sizes))
    else:
        s = src_addrs.to("cuda", non_blocking=True)
        d = dst_addrs.to("cuda", non_blocking=True)
        z = sizes.to("cuda", non_blocking=True)

    def launch() -> None:
        _swap_blocks_kernel_j[(min(P, n),)](
            s, d, z, n, J=J, WORDS=bytes_per_chunk // 8, TAG=int(green), num_warps=W
        )

    if not green:
        launch()
        return
    from vllm.v1.worker import gc_split

    gc_split.launch_on_part(launch)
