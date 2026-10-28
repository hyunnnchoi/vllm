# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
import os
import threading
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, triton
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.kv_offload.base import (
    BlockIDsLoadStoreSpec,
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalPageMapping,
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingWorker,
    TransferResult,
)
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion
from vllm.v1.kv_offload.cpu import io_flags as _io_flags
from vllm.v1.kv_offload.cpu import swap_blocks_triton as _sbt
from vllm.v1.worker import aris_nvtx as _nvtx
from vllm.v1.kv_offload.cpu.swap_blocks_triton import (
    THRESHOLD_BYTES,
    swap_blocks_batch,
)

logger = init_logger(__name__)


def _select_swap_blocks_fn(
    layer_refs_per_group: list[list[CanonicalKVCacheRef]],
    gpu_to_cpu: bool,
):
    """Resolve the swap_blocks function for a handler at init time."""
    # GPU->CPU is bandwidth-bound; the dedicated copy engine beats Triton.
    if gpu_to_cpu:
        return ops.swap_blocks_batch
    # Fall back to the C++ DMA path on platforms where Triton isn't usable
    # (e.g. ROCm host mappings) or where GPU kernels cannot directly
    # dereference CPU pointers (XPU lacks CUDA's unified virtual address space,
    # so the Triton kernel's tl.load(cpu_ptr) is invalid on XPU).
    if not HAS_TRITON or current_platform.is_xpu() or current_platform.is_rocm():
        return ops.swap_blocks_batch
    page_sizes = [r.page_size_bytes for g in layer_refs_per_group for r in g]
    # Triton wins only on small, 8-byte-aligned payloads.
    if (
        not page_sizes
        or max(page_sizes) >= THRESHOLD_BYTES
        or any(s % 8 for s in page_sizes)
    ):
        return ops.swap_blocks_batch
    chunk = min(triton.next_power_of_2(max(page_sizes)), 8192)
    return functools.partial(swap_blocks_batch, bytes_per_chunk=chunk)


# aris layer-wise offload (VLLM_KV_OFFLOAD_LAYERWISE=1): a CPU->GPU load is issued inside the step that schedules the
# request, one copy batch per KV-cache layer, with an event per layer; the attention layer waits on its event
# (OffloadingConnector.wait_for_layer_load), so transfer of layer l+1 overlaps the compute of layer l.
LAYERWISE = os.environ.get("VLLM_KV_OFFLOAD_LAYERWISE", "0") == "1"
# layer name of every CanonicalKVCacheRef, per KV-cache group (set by the offloading worker at registration)
REF_LAYER_NAMES: list[list[str]] = []
# layer name -> events of this step's layer-wise loads (consumed by wait_for_layer_load)
LAYER_EVENTS: dict[str, list] = {}

# aris Baton-style layer-wise offload (VLLM_KV_OFFLOAD_LAYERWISE=2), after the layer-wise
# H2D of /home/work/yhgo/baton-incremental-v023 (load path only; no LAS scheduler, no
# prediction or speculative prefetch). A CPU-tier load stays asynchronous (submitted after
# the forward is launched), but the background thread submits it one layer at a time in
# layer order with at most LW_WINDOW layers in flight, publishing an event per layer. When
# layer 0 has arrived the request is reported to the scheduler (io_flags early completion)
# and runs in the next step; that step's forward waits per layer only for the loads of the
# requests it computes (wait_layer), so the rest of the transfer overlaps the forward.
LAYERWISE_ASYNC = os.environ.get("VLLM_KV_OFFLOAD_LAYERWISE", "0") == "2"
LW_WINDOW = int(os.environ.get("VLLM_KV_OFFLOAD_LW_WINDOW", "2"))
LW_JOBS: dict[int, "Transfer"] = {}
LW_COND = threading.Condition()


def wait_layer(job_id: int, layer_name: str, stream) -> None:
    """Make `stream` wait for `layer_name` of layer-wise load `job_id` (host waits only
    until the background thread has submitted that layer)."""
    t = LW_JOBS.get(job_id)
    if t is None or t.layer_events is None:
        return
    deadline = time.monotonic() + 60.0
    with LW_COND:
        while layer_name not in t.layer_events and not t.submitted:
            if not LW_COND.wait(timeout=0.05) and time.monotonic() > deadline:
                raise RuntimeError(f"layer-wise load {job_id}: {layer_name} never published")
        ev = t.layer_events.get(layer_name)
    stream.wait_event(ev if ev is not None else t.end_event)


@dataclass
class Transfer:
    job_id: int
    stream: torch.cuda.Stream
    start_event: torch.Event
    end_event: torch.Event
    num_bytes: int
    batch_src: torch.Tensor
    batch_dst: torch.Tensor
    batch_sizes: torch.Tensor
    # Paced CPU->GPU submission (aris/layout-pacing): descriptors not yet
    # submitted are src/dst/sizes[next_op:num_ops]; end_event is recorded only
    # after the last chunk, so `submitted` guards queries on it.
    num_ops: int = 0
    next_op: int = 0
    submitted: bool = True
    is_src_access_order_any: bool = False
    t_submit: float = 0.0
    # wall-clock times when the monitor thread first saw start/end complete
    t_start_seen: float = 0.0
    t_end_seen: float = 0.0
    # Baton-style layer-wise load: [layer name, first op, end op] per layer, the
    # published per-layer events, the next layer to submit, layer 0 seen complete
    layer_groups: list | None = None
    layer_events: dict | None = None
    layer_next: int = 0
    first_ready: bool = False


def compute_sub_block_ptrs(
    block_ids: np.ndarray,
    blocks_per_chunk: int,
    output: np.ndarray,
    tensor: torch.Tensor,
    skip_count: int = 0,
):
    """
    Compute byte pointers for sub-blocks of the given block IDs.

    Each block in block_ids contains blocks_per_chunk sub-blocks.
    The pointer for sub-block j of block b is:
        base_ptr + b * row_stride + j * block_page_size

    where block_page_size = tensor.shape[1] // blocks_per_chunk (gpu page size).

    This handles tensors where row_stride != blocks_per_chunk * block_page_size
    (e.g. non-contiguous CPU tensors).

    Args:
        block_ids: array of block IDs at the tensor's native granularity.
        blocks_per_chunk: number of sub-blocks per block.
        output: pre-allocated pointer array to write pointers into.
        tensor: the source or destination tensor.
        skip_count: sub-blocks to skip in the first block.
    """
    assert skip_count < blocks_per_chunk

    num_sub_blocks = len(output)
    base_ptr = tensor.data_ptr()
    row_stride = tensor.stride(0)

    if blocks_per_chunk == 1:
        # Fast path: 1:1 mapping, no sub-block expansion needed.
        output[:] = base_ptr + block_ids.astype(np.uint64)[:num_sub_blocks] * row_stride
        return

    # Vectorized expansion for blocks_per_chunk > 1.
    assert tensor.shape[1] % blocks_per_chunk == 0
    block_page_size = tensor.shape[1] // blocks_per_chunk
    sub_offsets = np.arange(blocks_per_chunk, dtype=np.uint64) * block_page_size
    # (num_blocks, 1) + (1, blocks_per_chunk) -> (num_blocks, blocks_per_chunk)
    all_ptrs = (
        base_ptr + block_ids.astype(np.uint64)[:, np.newaxis] * row_stride
    ) + sub_offsets[np.newaxis, :]
    # Flatten and apply skip_count / truncation
    flat = all_ptrs.ravel()
    output[:] = flat[skip_count : skip_count + num_sub_blocks]


class CopyPlan(NamedTuple):
    """Precomputed fragment-copy template for one data ref under the canonical
    CPU layout, unrolled from the ref's mapped runs. Offsets are relative to
    the per-block base pointers on each side."""

    frag_offsets_src: np.ndarray
    frag_offsets_dst: np.ndarray
    frag_sizes: np.ndarray
    total_bytes: int

    @property
    def num_frags(self) -> int:
        return len(self.frag_sizes)


def _build_copy_plan(ref: CanonicalKVCacheRef, gpu_to_cpu: bool) -> CopyPlan:
    """Unroll one data ref's mapped runs into a per-fragment CopyPlan."""
    mapping = ref.mapping
    assert mapping is not None
    local: list[int] = []
    canonical: list[int] = []
    sizes: list[int] = []
    for run in mapping.runs:
        for i in range(run.num_fragments):
            local.append(run.local_offset + i * run.local_stride)
            canonical.append(run.canonical_offset + i * run.canonical_stride)
            sizes.append(run.fragment_size)
    src, dst = (local, canonical) if gpu_to_cpu else (canonical, local)
    return CopyPlan(
        frag_offsets_src=np.asarray(src, dtype=np.uint64),
        frag_offsets_dst=np.asarray(dst, dtype=np.uint64),
        frag_sizes=np.asarray(sizes, dtype=np.int64),
        total_bytes=sum(sizes),
    )


def _canonical_page_ids(
    block_ids: np.ndarray, blocks_per_chunk: int, count: int, skip_count: int
) -> np.ndarray:
    """Global canonical page ids matching compute_sub_block_ptrs' enumeration.
    These identify canonical pages consistently across ranks, so they key
    CanonicalPageMapping.is_writer rotation."""
    if blocks_per_chunk == 1:
        return block_ids[:count]
    flat = (
        block_ids[:, np.newaxis] * blocks_per_chunk + np.arange(blocks_per_chunk)
    ).ravel()
    return flat[skip_count : skip_count + count]


def _canonical_block_sizes(
    layer_refs_per_group: list[list[CanonicalKVCacheRef]], num_tensors: int
) -> list[int]:
    """Canonical CPU bytes per GPU block for each tensor, taken from the refs'
    mappings. Requires every ref to carry a mapping."""
    canonical_bytes_per_block = [0] * num_tensors
    for layer_refs in layer_refs_per_group:
        for ref in layer_refs:
            assert ref.mapping is not None
            canonical_bytes_per_block[ref.tensor_idx] = max(
                canonical_bytes_per_block[ref.tensor_idx],
                ref.mapping.canonical_page_size_bytes,
            )
    assert all(size > 0 for size in canonical_bytes_per_block)
    return canonical_bytes_per_block


def pin_mmap_region(region: SharedOffloadRegion) -> None:
    """Register the entire mmap as CUDA pinned memory via cudaHostRegister."""
    if not current_platform.is_cuda_alike():
        logger.info(
            "Skipping mmap host registration on %s; cudaHostRegister is only "
            "available on CUDA/ROCm.",
            current_platform.device_name,
        )
        return

    rank = region.rank

    base_ptr = region._base.data_ptr()
    result = torch.cuda.cudart().cudaHostRegister(base_ptr, region.total_size_bytes, 0)
    if result.value != 0:
        logger.warning(
            "cudaHostRegister failed for rank=%d (code=%d) — "
            "transfers will still work but may be slower (unpinned DMA)",
            rank,
            result,
        )
    else:
        logger.debug(
            "cudaHostRegister rank=%d %.2f GB",
            rank,
            region.total_size_bytes / 1e9,
        )
        region.is_pinned = True


def _new_descriptor_buffers(
    num_copy_ops: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pin = PIN_MEMORY
    # CUDA cache_kernels.cu requires int64; XPU DMA engine requires uint64.
    ptr_dtype = torch.uint64 if current_platform.is_xpu() else torch.int64
    return (
        torch.empty(num_copy_ops, dtype=ptr_dtype, pin_memory=pin),
        torch.empty(num_copy_ops, dtype=ptr_dtype, pin_memory=pin),
        torch.empty(num_copy_ops, dtype=ptr_dtype, pin_memory=pin),
    )


class SingleDirectionOffloadingHandler:
    """
    Handles transfers for a single direction, either CPU->GPU or GPU->CPU.
    Transfers are guaranteed to be executed in order of their submission.
    Each transfer uses a unique CUDA stream, and its stream will start
    executing only after the streams of previous transfers have finished.
    """

    def __init__(
        self,
        gpu_tensors: list[torch.Tensor],
        cpu_tensors: list[torch.Tensor],
        blocks_per_chunk: int,
        layer_refs_per_group: list[list[CanonicalKVCacheRef]],
        gpu_to_cpu: bool,
        canonical_layout: bool = False,
    ):
        """
        Initialize a SingleDirectionOffloadingHandler.

        Args:
            gpu_tensors: list of GPU KV cache tensors.
                Each of shape (num_gpu_blocks, gpu_page_size_bytes) with dtype int8.
            cpu_tensors: list of CPU KV cache tensors.
                Each of shape (num_cpu_chunks, cpu_page_size_bytes) with dtype int8.
                Order should match gpu_tensors.
            layer_refs_per_group: list of CanonicalKVCacheRef per group.
            gpu_to_cpu: if True, transfer from GPU to CPU; otherwise CPU to GPU.
            canonical_layout: if True, CPU pages use the canonical layout
                described by the refs' mappings.
        """
        assert len(gpu_tensors) == len(cpu_tensors)
        assert len(gpu_tensors) > 0

        canonical_bytes_per_block = (
            _canonical_block_sizes(layer_refs_per_group, len(gpu_tensors))
            if canonical_layout
            else None
        )

        # assert input tensors are as expected
        for t_idx, (gpu_tensor, cpu_tensor) in enumerate(zip(gpu_tensors, cpu_tensors)):
            assert gpu_tensor.dtype == torch.int8
            assert gpu_tensor.ndim == 2
            assert gpu_tensor.is_cuda or gpu_tensor.is_xpu
            assert cpu_tensor.dtype == torch.int8
            assert cpu_tensor.ndim == 2
            assert cpu_tensor.device.type == "cpu"
            _, gpu_page_size = gpu_tensor.shape
            _, cpu_page_size = cpu_tensor.shape
            if canonical_bytes_per_block is not None:
                assert (
                    cpu_page_size == canonical_bytes_per_block[t_idx] * blocks_per_chunk
                )
            else:
                assert cpu_page_size == gpu_page_size * blocks_per_chunk

        self.src_tensors: list[torch.Tensor] = (
            gpu_tensors if gpu_to_cpu else cpu_tensors
        )
        self.dst_tensors: list[torch.Tensor] = (
            cpu_tensors if gpu_to_cpu else gpu_tensors
        )
        self.gpu_to_cpu: bool = gpu_to_cpu
        self.layer_refs_per_group = layer_refs_per_group
        self._ref_ends: list[tuple[int, int, int]] | None = None
        self._swap_blocks_batch = _select_swap_blocks_fn(
            layer_refs_per_group, gpu_to_cpu
        )
        # aris/layout-pacing: Triton UVA copy usable for any page size when the
        # control setting asks for it ("triton<k>", k = number of programs/SMs)
        _pages = [r.page_size_bytes for g in layer_refs_per_group for r in g]
        self._triton_fn = (
            functools.partial(
                swap_blocks_batch,
                bytes_per_chunk=min(triton.next_power_of_2(max(_pages)), 8192),
            )
            if HAS_TRITON and not gpu_to_cpu and _pages and all(x % 8 == 0 for x in _pages)
            and not current_platform.is_rocm() and not current_platform.is_xpu()
            else None
        )

        # GPU blocks may be smaller
        # cpu_page_size = gpu_page_size * blocks_per_chunk.
        self.src_blocks_per_chunk = 1 if self.gpu_to_cpu else blocks_per_chunk
        self.dst_blocks_per_chunk = blocks_per_chunk if self.gpu_to_cpu else 1

        # Per (group, ref) static copy plans for the canonical layout
        self._canonical_copy_plans: list[list[CopyPlan]] | None = (
            [
                [_build_copy_plan(ref, gpu_to_cpu) for ref in layer_refs]
                for layer_refs in layer_refs_per_group
            ]
            if canonical_layout
            else None
        )
        self._fill_group_ops = (
            self._fill_canonical_ops if canonical_layout else self._fill_direct_ops
        )
        # Reusable per-block base-pointer scratch for the canonical fill,
        # sized to the largest possible group (grown on demand)
        num_scratch_blocks = gpu_tensors[0].shape[0] if canonical_layout else 0
        self._scratch_bases_src = np.empty(num_scratch_blocks, dtype=np.uint64)
        self._scratch_bases_dst = np.empty(num_scratch_blocks, dtype=np.uint64)

        # Step-synchronous pacing of CPU->GPU loads (aris/layout-pacing).
        # A DMA load that keeps the H2D copy engine busy starves the model
        # runner's small per-step H2D input copies until it drains, so decode
        # stalls for the whole load. With pacing, at most one chunk of at most
        # VLLM_KV_LOAD_PACE_BYTES is in flight; the next chunk is submitted
        # from get_finished(), i.e. once per engine step after the forward
        # launch. VLLM_KV_LOAD_PACE_FILE (optional) holds the byte budget and
        # is re-read when it changes. 0 disables pacing.
        self._pace_bytes = int(os.environ.get("VLLM_KV_LOAD_PACE_BYTES", "0"))
        self._pace_file = os.environ.get("VLLM_KV_LOAD_PACE_FILE") or None
        self._pace_file_mtime = -1.0
        self._pace_last_event: torch.Event | None = None
        # queue mode ("ctrlq:<cap>:<chunk>"): keep at most <cap> descriptors
        # submitted-but-unfinished, submitting chunks of <chunk> descriptors,
        # so cuMemcpyBatchAsync never waits for queue space (returns quickly)
        # while the copy engine always has work.
        self._q_chunks: deque = deque()
        # adaptive mode (budget < 0, e.g. -80): chunk = 0.80 x link bytes/s x
        # recent engine step time, step time = EWMA of the interval between
        # get_finished() calls (one per engine step); intervals under 5 ms are
        # idle spins (no forward) and count as 50 ms (load at full speed).
        self._pace_bw = float(os.environ.get("VLLM_KV_LOAD_PACE_BW", "12e9"))
        self._pace_step_s = 0.05
        self._pace_last_call: float | None = None
        # VLLM_KV_XFER_LOG (optional): append one JSON line per finished
        # transfer (wall-clock submit/detect time, bytes, GPU time) and one per
        # gap of > 100 ms between get_finished() calls of the load handler
        # (an engine step that took that long), to line up client-side
        # inter-token gaps with KV loads.
        xfer_log = os.environ.get("VLLM_KV_XFER_LOG")
        self._xfer_log = open(xfer_log, "a", buffering=1) if xfer_log else None
        self._xfer_dir = "store" if gpu_to_cpu else "load"
        self._xfer_last_call: float | None = None
        # queue-mode flags ("ctrlq:<cap>:<chunk>:<flags>"): 't' = a pump thread
        # refills the queue every 0.5 ms instead of once per engine step;
        # 's' = shortest remaining load first (preemptive at chunk
        # boundaries), loads no longer wait for the previous load, and
        # finished loads are reported out of order.
        self._q_thread = False
        self._q_srpt = False
        self._lock = threading.RLock()
        self._bg: threading.Thread | None = None
        self._bg_device = torch.cuda.current_device() if torch.cuda.is_available() else None
        if self._xfer_log is not None:
            self._start_bg()

        # job_id -> event
        self._transfer_events: dict[int, torch.Event] = {}
        # queue of transfers (job_id, stream, event)
        self._transfers: deque[Transfer] = deque()
        self._lw_inflight: deque = deque()
        # list of CUDA streams available for re-use
        self._stream_pool: list[torch.cuda.Stream] = []
        # list of CUDA events available for re-use
        self._event_pool: list[torch.Event] = []
        # list of pinned descriptor buffer sets available for re-use
        self._buffer_pool: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def _estimate_max_copy_ops(self, group_sizes: Sequence[int]) -> int:
        """Upper bound on the number of copy descriptors for a transfer.

        Exact for the direct layout. The canonical path may fill fewer:
        writer rotation later drops the blocks this rank does not write."""
        num_copy_ops = 0
        for g_idx, (group_size, layer_refs) in enumerate(
            zip(group_sizes, self.layer_refs_per_group)
        ):
            if self._canonical_copy_plans is None:
                num_copy_ops += group_size * len(layer_refs)
            else:
                num_copy_ops += group_size * sum(
                    plan.num_frags for plan in self._canonical_copy_plans[g_idx]
                )
        return num_copy_ops

    def _fill_direct_ops(
        self,
        g_idx: int,
        group_src: np.ndarray,
        group_dst: np.ndarray,
        group_size: int,
        src_skip_count: int,
        dst_skip_count: int,
        all_src: np.ndarray,
        all_dst: np.ndarray,
        all_sizes: np.ndarray,
        op_idx: int,
    ) -> tuple[int, int]:
        """Fill one group's copy descriptors for the direct (worker-private)
        layout: one whole-page copy per (block, ref).

        Returns (op_idx past the filled descriptors, bytes added)."""
        num_bytes = 0
        for r_idx, data_ref in enumerate(self.layer_refs_per_group[g_idx]):
            t_idx = data_ref.tensor_idx
            end_idx = op_idx + group_size

            compute_sub_block_ptrs(
                group_src,
                self.src_blocks_per_chunk,
                all_src[op_idx:end_idx],
                self.src_tensors[t_idx],
                skip_count=src_skip_count,
            )
            compute_sub_block_ptrs(
                group_dst,
                self.dst_blocks_per_chunk,
                all_dst[op_idx:end_idx],
                self.dst_tensors[t_idx],
                skip_count=dst_skip_count,
            )

            all_sizes[op_idx:end_idx] = data_ref.page_size_bytes
            num_bytes += group_size * data_ref.page_size_bytes
            op_idx = end_idx
            if self._ref_ends is not None:
                self._ref_ends.append((g_idx, r_idx, op_idx))
        return op_idx, num_bytes

    def _fill_canonical_ops(
        self,
        g_idx: int,
        group_src: np.ndarray,
        group_dst: np.ndarray,
        group_size: int,
        src_skip_count: int,
        dst_skip_count: int,
        all_src: np.ndarray,
        all_dst: np.ndarray,
        all_sizes: np.ndarray,
        op_idx: int,
    ) -> tuple[int, int]:
        """Fill one group's copy descriptors for the canonical layout:
        scatter each block through the ref's precomputed CopyPlan, keeping
        only the blocks this rank writes.

        Returns (op_idx past the filled descriptors, bytes added)."""
        assert self._canonical_copy_plans is not None
        # Zero-copy reinterpretation for pointer arithmetic: uint64 and the
        # buffers' int64 are bit-equivalent for addresses
        all_src_u64 = all_src.view(np.uint64)
        all_dst_u64 = all_dst.view(np.uint64)
        if group_size > len(self._scratch_bases_src):
            self._scratch_bases_src = np.empty(group_size, dtype=np.uint64)
            self._scratch_bases_dst = np.empty(group_size, dtype=np.uint64)

        num_bytes = 0
        for r_idx, (plan, data_ref) in enumerate(
            zip(self._canonical_copy_plans[g_idx], self.layer_refs_per_group[g_idx])
        ):
            if plan.num_frags == 0:
                if self._ref_ends is not None:
                    self._ref_ends.append((g_idx, r_idx, op_idx))
                continue
            t_idx = data_ref.tensor_idx

            # 1. Base byte pointer of every block on each side
            block_bases_src = self._scratch_bases_src[:group_size]
            block_bases_dst = self._scratch_bases_dst[:group_size]
            compute_sub_block_ptrs(
                group_src,
                self.src_blocks_per_chunk,
                block_bases_src,
                self.src_tensors[t_idx],
                skip_count=src_skip_count,
            )
            compute_sub_block_ptrs(
                group_dst,
                self.dst_blocks_per_chunk,
                block_bases_dst,
                self.dst_tensors[t_idx],
                skip_count=dst_skip_count,
            )

            # 2. On store, keep only the blocks this rank is elected to write
            mapping = data_ref.mapping
            assert mapping is not None
            if self.gpu_to_cpu and mapping.num_writers > 1:
                block_bases_src, block_bases_dst = self._filter_writer_blocks(
                    block_bases_src,
                    block_bases_dst,
                    mapping,
                    group_dst,
                    group_size,
                    dst_skip_count,
                )
            num_active_blocks = len(block_bases_src)

            # 3. Expand (block base + fragment offset) into one descriptor
            #    per (block, fragment), writing straight into the descriptor
            #    buffers: reshaping a contiguous 1D slice is a view, so the
            #    broadcasts below allocate nothing
            end_idx = op_idx + num_active_blocks * plan.num_frags
            np.add(
                block_bases_src[:, None],
                plan.frag_offsets_src[None, :],
                out=all_src_u64[op_idx:end_idx].reshape(
                    num_active_blocks, plan.num_frags
                ),
            )
            np.add(
                block_bases_dst[:, None],
                plan.frag_offsets_dst[None, :],
                out=all_dst_u64[op_idx:end_idx].reshape(
                    num_active_blocks, plan.num_frags
                ),
            )
            all_sizes[op_idx:end_idx].reshape(num_active_blocks, plan.num_frags)[:] = (
                plan.frag_sizes
            )
            num_bytes += num_active_blocks * plan.total_bytes
            op_idx = end_idx
            if self._ref_ends is not None:
                self._ref_ends.append((g_idx, r_idx, op_idx))
        return op_idx, num_bytes

    def _filter_writer_blocks(
        self,
        block_bases_src: np.ndarray,
        block_bases_dst: np.ndarray,
        mapping: CanonicalPageMapping,
        group_dst: np.ndarray,
        group_size: int,
        dst_skip_count: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Keep only the blocks this rank writes: replicated ranks take turns
        writing shared canonical pages, keyed by the rank-consistent CPU-side
        canonical page id."""
        cpu_page_ids = _canonical_page_ids(
            group_dst,
            self.dst_blocks_per_chunk,
            group_size,
            dst_skip_count,
        )
        writer_mask = cpu_page_ids % mapping.num_writers == mapping.writer_index
        return block_bases_src[writer_mask], block_bases_dst[writer_mask]

    def transfer_async(
        self,
        job_id: int,
        src_spec: LoadStoreSpec,
        dst_spec: LoadStoreSpec,
        layerwise: bool = False,
    ) -> bool:
        assert isinstance(src_spec, BlockIDsLoadStoreSpec)
        layerwise = layerwise and not self.gpu_to_cpu
        lw_async = layerwise == "async"
        self._ref_ends = [] if layerwise else None
        assert isinstance(dst_spec, BlockIDsLoadStoreSpec)

        src_blocks = src_spec.block_ids
        dst_blocks = dst_spec.block_ids
        assert src_blocks.ndim == 1
        assert dst_blocks.ndim == 1

        num_src_blocks = len(src_blocks)
        num_dst_blocks = len(dst_blocks)
        xdir = "d2h" if self.gpu_to_cpu else "h2d"
        if _nvtx.ON:
            torch.cuda.nvtx.range_push(f"kv_prepare {xdir} job={job_id} blocks={num_dst_blocks}")

        # There are 2 types of transfers:
        # 1. GPU -> CPU
        # 2. CPU -> GPU
        #
        # transfers are also to CPU chunks, EXCEPT MAYBE for the first and last chunk.
        # i.e. the first and last CPU chunks in src_blocks can match against
        # a smaller (byte-wise) set of GPU blocks in dst_blocks.
        # In such cases, we may need to skip some gpu-sized sub-blocks,
        # and start reading/writing from the middle of the first CPU chunk.
        # If we have multiple KV cache groups (when using HMA with hybrid models),
        # we may have a partial first/last CPU chunk per each group.
        # The group_sizes parameter encodes the size of each group of blocks
        # in the GPU dst_blocks.
        # If group_sizes is None, we assume all blocks belong to a single group.
        # The logical_offset parameter maps each group of blocks to its logical
        # offset inside the request, counting in GPU blocks.
        # This allows us to find the correct starting position
        # in the matching first CPU chunk.

        # extract group_sizes from the GPU spec
        gpu_spec = src_spec if self.gpu_to_cpu else dst_spec
        assert isinstance(gpu_spec, GPULoadStoreSpec)
        group_sizes = gpu_spec.group_sizes
        assert len(group_sizes) == len(self.layer_refs_per_group)

        # extract block indices from the GPU spec
        block_indices = gpu_spec.block_indices
        assert len(block_indices) == len(self.layer_refs_per_group)

        num_copy_ops = self._estimate_max_copy_ops(group_sizes)

        # reuse a pooled buffer set, growing it if this transfer needs more room
        batch_src, batch_dst, batch_sizes = (
            self._buffer_pool.pop()
            if self._buffer_pool
            else _new_descriptor_buffers(num_copy_ops)
        )
        if batch_src.numel() < num_copy_ops:
            batch_src, batch_dst, batch_sizes = _new_descriptor_buffers(num_copy_ops)

        src = batch_src[:num_copy_ops]
        dst = batch_dst[:num_copy_ops]
        sizes = batch_sizes[:num_copy_ops]
        all_src = src.numpy()
        all_dst = dst.numpy()
        all_sizes = sizes.numpy()

        src_offset = 0
        dst_offset = 0
        op_idx = 0
        # count total number of bytes copied
        num_transfer_bytes = 0
        for g_idx, (group_size, block_idx) in enumerate(
            zip(group_sizes, block_indices)
        ):
            if group_size == 0:
                continue

            src_logical_blocks_to_skip = block_idx % self.src_blocks_per_chunk
            dst_logical_blocks_to_skip = block_idx % self.dst_blocks_per_chunk
            src_logical_blocks_count = group_size + src_logical_blocks_to_skip
            dst_logical_blocks_count = group_size + dst_logical_blocks_to_skip

            dst_blocks_count = cdiv(dst_logical_blocks_count, self.dst_blocks_per_chunk)
            dst_end_offset = dst_offset + dst_blocks_count
            assert dst_end_offset <= num_dst_blocks

            src_blocks_count = cdiv(src_logical_blocks_count, self.src_blocks_per_chunk)
            src_end_offset = src_offset + src_blocks_count
            assert src_end_offset <= num_src_blocks

            op_idx, group_bytes = self._fill_group_ops(
                g_idx,
                group_src=src_blocks[src_offset:src_end_offset],
                group_dst=dst_blocks[dst_offset:dst_end_offset],
                group_size=group_size,
                src_skip_count=src_logical_blocks_to_skip,
                dst_skip_count=dst_logical_blocks_to_skip,
                all_src=all_src,
                all_dst=all_dst,
                all_sizes=all_sizes,
                op_idx=op_idx,
            )
            num_transfer_bytes += group_bytes

            src_offset = src_end_offset
            dst_offset = dst_end_offset

        assert src_offset == num_src_blocks
        assert dst_offset == num_dst_blocks
        # Writer rotation may skip non-writer blocks, leaving op_idx below
        # the sized upper bound
        assert op_idx <= num_copy_ops
        src = src[:op_idx]
        dst = dst[:op_idx]
        sizes = sizes[:op_idx]
        if _nvtx.ON:
            torch.cuda.nvtx.range_pop()

        stream = (
            self._stream_pool.pop() if self._stream_pool else current_platform.Stream()
        )
        start_event = (
            self._event_pool.pop()
            if self._event_pool
            else torch.Event(enable_timing=True)
        )
        end_event = (
            self._event_pool.pop()
            if self._event_pool
            else torch.Event(enable_timing=True)
        )

        # Stores must wait for the model to finish writing the KV they read.
        # Loads must wait for pending writes (including zeroing) to their
        # destination blocks: the scheduler's _skip_zero_block_ids only edits
        # the step being scheduled and cannot retract zeroing shipped in an
        # earlier step for a since-reallocated block; with async scheduling
        # nothing else orders that zeroing against this copy.
        mode, val = self._mode() if not self.gpu_to_cpu else ("stock", 0)
        if self.gpu_to_cpu or "w" not in _io_flags.flags():
            stream.wait_stream(current_platform.current_stream())
        srpt = mode == "queue" and self._q_srpt
        if self._transfers and not srpt:
            last_transfer: Transfer = self._transfers[-1]
            last_event = last_transfer.end_event
            # assure job will start only after the previous one completes
            stream.wait_event(last_event)
        # CPU->GPU reads from host pinned memory, which is never written
        # by a concurrent GPU stream, so CU_MEMCPY_SRC_ACCESS_ORDER_ANY is
        # safe and lets the driver pipeline source reads. GPU->CPU reads
        # from the live GPU KV cache, which the compute stream keeps
        # writing; we must keep STREAM ordering so source reads are gated
        # by the transfer stream's wait_stream(compute) barrier.
        is_src_access_order_any = not self.gpu_to_cpu
        paced = mode in ("pace", "pace_in", "adapt", "queue") and op_idx > 0 and not layerwise
        layer_groups = None
        if lw_async:
            layer_groups, prev = [], 0
            for g_idx, r_idx, end in self._ref_ends:
                name = REF_LAYER_NAMES[g_idx][r_idx]
                if layer_groups and layer_groups[-1][0] == name and layer_groups[-1][2] == prev:
                    layer_groups[-1][2] = end
                else:
                    layer_groups.append([name, prev, end])
                prev = end
        fn = self._swap_blocks_batch
        if mode == "triton" and self._triton_fn is not None and not layerwise:
            _sbt.NUM_SMS = val
            fn = self._triton_fn
        elif mode in ("tritonx", "tritongc") and self._triton_fn is not None and not layerwise:
            fn = functools.partial(
                _sbt.swap_blocks_batch_j,
                bytes_per_chunk=self._triton_fn.keywords["bytes_per_chunk"],
                P=val[0], W=val[1], J=val[2], green=mode == "tritongc",
            )
        t_call = time.time()   # before the copy call (a large batch call blocks)
        with current_platform.stream(stream):
            start_event.record(stream)
            if lw_async:
                pass  # layers are submitted by the background thread (_pump_layers)
            elif layerwise:
                # one copy batch per ref (= per KV-cache layer), an event after each
                prev = 0
                for g_idx, r_idx, end in self._ref_ends:
                    if end > prev:
                        fn(
                            src[prev:end],
                            dst[prev:end],
                            sizes[prev:end],
                            is_src_access_order_any=is_src_access_order_any,
                        )
                    ev = torch.Event()
                    ev.record(stream)
                    LAYER_EVENTS.setdefault(REF_LAYER_NAMES[g_idx][r_idx], []).append(ev)
                    prev = end
            elif op_idx > 0 and not paced:
                if _nvtx.ON:
                    torch.cuda.nvtx.range_push(
                        f"kv_submit {xdir} job={job_id} ops={op_idx} bytes={num_transfer_bytes} mode={mode}")
                fn(
                    src,
                    dst,
                    sizes,
                    is_src_access_order_any=is_src_access_order_any,
                )
                if _nvtx.ON:
                    torch.cuda.nvtx.range_pop()
            if not paced and not (lw_async and layer_groups):
                end_event.record(stream)
        self._ref_ends = None

        self._transfer_events[job_id] = end_event
        self._lock.acquire()
        self._transfers.append(
            Transfer(
                job_id=job_id,
                stream=stream,
                start_event=start_event,
                end_event=end_event,
                num_bytes=num_transfer_bytes,
                batch_src=batch_src,
                batch_dst=batch_dst,
                batch_sizes=batch_sizes,
                num_ops=op_idx,
                next_op=0,
                submitted=not paced and not (lw_async and layer_groups),
                is_src_access_order_any=is_src_access_order_any,
                t_submit=t_call,
                layer_groups=layer_groups if lw_async and layer_groups else None,
                layer_events={} if lw_async and layer_groups else None,
            )
        )
        if lw_async:
            LW_JOBS[job_id] = self._transfers[-1]
            if not layer_groups:   # nothing to copy: report at once
                _io_flags.job_done(job_id)
            self._start_bg()
        try:
            if paced:
                self._pump()
        finally:
            self._lock.release()

        # success
        return True

    def _start_bg(self) -> None:
        if self._bg is not None:
            return
        self._bg = threading.Thread(target=self._bg_loop, daemon=True, name="kv-xfer-bg")
        self._bg.start()

    def _bg_loop(self) -> None:
        if self._bg_device is not None:
            torch.cuda.set_device(self._bg_device)
        while True:
            try:
                if not self.gpu_to_cpu:
                    mode = self._mode()[0]
                    if mode == "queue" and self._q_thread:
                        with self._lock:
                            self._pump_queue()
                    if LAYERWISE_ASYNC:
                        with self._lock:
                            self._pump_layers()
                early = not self.gpu_to_cpu and "e" in _io_flags.flags()
                if self._xfer_log is not None or early:
                    with self._lock:
                        snap = list(self._transfers)
                    now = time.time()
                    for t in snap:
                        if t.t_start_seen == 0.0 and t.start_event.query():
                            t.t_start_seen = now
                        if t.submitted and t.t_end_seen == 0.0 and t.end_event.query():
                            t.t_end_seen = now
                            if not self.gpu_to_cpu:
                                _io_flags.job_done(t.job_id)
            except Exception as e:  # never kill the engine from here
                logger.warning_once("kv-xfer-bg: %s", e)
            time.sleep(0.0005)

    def _pump_layers(self) -> None:
        """Baton-style layer-wise load: submit layers in order, <= LW_WINDOW in flight,
        publish one event per layer, report layer-0 readiness to the scheduler."""
        lw = [t for t in self._transfers if t.layer_groups is not None]
        if not lw:
            return
        for t in lw:
            if not t.first_ready and t.layer_events:
                ev0 = t.layer_events.get(t.layer_groups[0][0])
                if ev0 is not None and ev0.query():
                    t.first_ready = True
                    _io_flags.job_done(t.job_id)
        while self._lw_inflight and self._lw_inflight[0].query():
            self._lw_inflight.popleft()
        published = False
        while len(self._lw_inflight) < LW_WINDOW:
            t = next((x for x in lw if x.layer_next < len(x.layer_groups)), None)
            if t is None:
                break
            name, a, b = t.layer_groups[t.layer_next]
            with current_platform.stream(t.stream):
                if b > a:
                    self._swap_blocks_batch(
                        t.batch_src[a:b],
                        t.batch_dst[a:b],
                        t.batch_sizes[a:b],
                        is_src_access_order_any=t.is_src_access_order_any,
                    )
                ev = torch.Event()
                ev.record(t.stream)
                t.layer_next += 1
                if t.layer_next == len(t.layer_groups):
                    t.end_event.record(t.stream)
                    t.submitted = True
            with LW_COND:
                t.layer_events[name] = ev
            self._lw_inflight.append(ev)
            published = True
        if published:
            with LW_COND:
                LW_COND.notify_all()

    def _pacing_budget(self) -> int:
        mode, val = self._mode()
        if mode in ("pace", "pace_in"):
            return val
        if mode == "adapt":
            nbytes = (-val / 100.0) * self._pace_bw * self._pace_step_s
            return int(min(max(nbytes, 16 << 20), 1 << 30))
        return 0

    def _observe_step(self) -> None:
        now = time.perf_counter()
        if self._pace_last_call is not None:
            dt = now - self._pace_last_call
            dt = 0.05 if dt < 0.005 else min(dt, 1.0)
            self._pace_step_s = 0.5 * self._pace_step_s + 0.5 * dt
        self._pace_last_call = now

    def _mode(self) -> tuple[str, int]:
        """Control setting: '0' stock, '<n>' pace n bytes per chunk from
        get_finished(), 'in:<n>' also pump right before the forward (after the
        step's input copies), '-<p>' adaptive, 'triton<k>' Triton UVA copy with
        k programs for every page size (no pacing)."""
        if self._pace_file is not None:
            try:
                mtime = os.stat(self._pace_file).st_mtime
                if mtime != self._pace_file_mtime:
                    with open(self._pace_file) as f:
                        self._pace_raw = f.read().strip() or "0"
                    self._pace_file_mtime = mtime
            except OSError:
                pass
        raw = getattr(self, "_pace_raw", str(self._pace_bytes))
        try:
            if raw.startswith("ctrlq:"):
                parts = raw[6:].split(":")
                cap, chunk = int(parts[0]), int(parts[1])
                flags = parts[2] if len(parts) > 2 else ""
                self._q_cap, self._q_chunk = cap, chunk
                self._q_thread = "t" in flags
                self._q_srpt = "s" in flags
                if (self._q_thread or "e" in flags) and not self.gpu_to_cpu:
                    self._start_bg()
                return "queue", cap
            if raw.startswith(("triton:", "tritongc:")):
                head, *pwj = raw.split(":")
                return ("tritonx" if head == "triton" else "tritongc"), tuple(int(x) for x in pwj)
            if raw.startswith("triton"):
                return "triton", int(raw[6:])
            if raw.startswith("in:"):
                return "pace_in", int(raw[3:])
            n = int(raw)
        except ValueError:
            return "stock", 0
        if n > 0:
            return "pace", n
        if n < 0:
            return "adapt", n
        return "stock", 0

    def _pacing_setting(self) -> int:
        mode, _ = self._mode()
        return 0 if mode in ("stock", "triton", "tritonx", "tritongc") else 1

    def pump_at_inputs(self) -> None:
        """Called right before the forward, after the step's input copies."""
        if not self.gpu_to_cpu and self._mode()[0] == "pace_in":
            self._pump()

    def _pump(self, force: bool = False) -> None:
        """Submit the next chunk of the oldest partially submitted load.

        Without force: only if the previous chunk has finished (one chunk in
        flight), and at most the pacing budget in bytes. With force: submit
        everything that is left (used by wait() and shutdown())."""
        if not force and self._mode()[0] == "queue":
            self._pump_queue()
            return
        pending = next(
            (t for t in self._transfers
             if not t.submitted and (force or t.layer_groups is None)),
            None,
        )
        while pending is not None:
            if pending.layer_groups is not None:
                # forced drain of a layer-wise load: submit what is left at once
                lg, k = pending.layer_groups, pending.layer_next
                pending.next_op = lg[k][1] if k < len(lg) else pending.num_ops
                pending.layer_next = len(lg)
            if not force and self._pace_last_event is not None:
                if not self._pace_last_event.query():
                    return
            budget = self._pacing_budget()
            sizes = pending.batch_sizes
            start = pending.next_op
            if force or self._pacing_setting() == 0:
                end = pending.num_ops
            else:
                csum = np.cumsum(sizes[start:pending.num_ops].numpy())
                end = start + max(1, int(np.searchsorted(csum, budget, side="right")))
            with current_platform.stream(pending.stream):
                if start == 0:
                    # keep FIFO order with the previous load (its end_event was
                    # not recorded when this transfer was created)
                    prev = None
                    for t in self._transfers:
                        if t is pending:
                            break
                        prev = t
                    if prev is not None:
                        pending.stream.wait_event(prev.end_event)
                self._swap_blocks_batch(
                    pending.batch_src[start:end],
                    pending.batch_dst[start:end],
                    sizes[start:end],
                    is_src_access_order_any=pending.is_src_access_order_any,
                )
                pending.next_op = end
                if end >= pending.num_ops:
                    pending.end_event.record(pending.stream)
                    pending.submitted = True
                    self._pace_last_event = pending.end_event
                else:
                    ev = torch.Event()
                    ev.record(pending.stream)
                    self._pace_last_event = ev
            if not force:
                return
            pending = next((t for t in self._transfers if not t.submitted), None)

    def _pump_queue(self) -> None:
        while self._q_chunks and self._q_chunks[0][0].query():
            self._q_chunks.popleft()
        in_flight = sum(n for _, n in self._q_chunks)
        while True:
            if self._q_srpt:
                cands = [t for t in self._transfers if not t.submitted]
                pending = min(cands, key=lambda t: t.num_ops - t.next_op) if cands else None
            else:
                pending = next((t for t in self._transfers if not t.submitted), None)
            if pending is None:
                return
            start = pending.next_op
            n = min(self._q_chunk, pending.num_ops - start)
            if in_flight + n > self._q_cap and in_flight > 0:
                return
            end = start + n
            with current_platform.stream(pending.stream):
                if start == 0 and not self._q_srpt:
                    prev = None
                    for t in self._transfers:
                        if t is pending:
                            break
                        prev = t
                    if prev is not None:
                        pending.stream.wait_event(prev.end_event)
                if _nvtx.ON:
                    torch.cuda.nvtx.range_push(f"kv_submit_chunk job={pending.job_id} ops={start}:{end} queue")
                self._swap_blocks_batch(
                    pending.batch_src[start:end],
                    pending.batch_dst[start:end],
                    pending.batch_sizes[start:end],
                    is_src_access_order_any=pending.is_src_access_order_any,
                )
                if _nvtx.ON:
                    torch.cuda.nvtx.range_pop()
                pending.next_op = end
                if end >= pending.num_ops:
                    pending.end_event.record(pending.stream)
                    pending.submitted = True
                    ev = pending.end_event
                else:
                    ev = torch.Event()
                    ev.record(pending.stream)
            self._q_chunks.append((ev, n))
            in_flight += n

    def get_finished(self) -> list[TransferResult]:
        with self._lock:
            return self._get_finished_locked()

    def _get_finished_locked(self) -> list[TransferResult]:
        if not self.gpu_to_cpu:
            self._observe_step()
            self._pump()
        if self._xfer_log is not None and not self.gpu_to_cpu:
            now = time.time()
            if self._xfer_last_call is not None and now - self._xfer_last_call > 0.1:
                self._xfer_log.write(
                    '{"ev":"gap","dir":"%s","t0":%.4f,"t1":%.4f,"inflight":%d,'
                    '"inflight_bytes":%d,"pid":%d}\n'
                    % (self._xfer_dir, self._xfer_last_call, now,
                       len(self._transfers),
                       sum(t.num_bytes for t in self._transfers), os.getpid())
                )
            self._xfer_last_call = now
        results: list[TransferResult] = []
        if not self.gpu_to_cpu and self._mode()[0] == "queue" and self._q_srpt:
            done = [t for t in self._transfers if t.submitted and t.end_event.query()]
            if done:
                self._transfers = deque(
                    t for t in self._transfers if not any(t is d for d in done)
                )
        else:
            done = []
            while (
                self._transfers
                and self._transfers[0].submitted
                and self._transfers[0].end_event.query()
            ):
                done.append(self._transfers.popleft())
        for transfer in done:
            LW_JOBS.pop(transfer.job_id, None)
            transfer_time = (
                transfer.start_event.elapsed_time(transfer.end_event) * 1e-3
            )  # elapsed_time is in milliseconds
            result = TransferResult(
                job_id=transfer.job_id,
                success=True,
                transfer_size=transfer.num_bytes,
                transfer_time=transfer_time,
            )

            results.append(result)
            if self._xfer_log is not None:
                self._xfer_log.write(
                    '{"ev":"xfer","dir":"%s","job":%d,"bytes":%d,"ops":%d,'
                    '"t_submit":%.4f,"t_detect":%.4f,"gpu_s":%.5f,'
                    '"t_start_seen":%.4f,"t_end_seen":%.4f,"pid":%d}\n'
                    % (self._xfer_dir, transfer.job_id, transfer.num_bytes,
                       transfer.num_ops, transfer.t_submit, time.time(),
                       transfer_time, transfer.t_start_seen, transfer.t_end_seen,
                       os.getpid())
                )
            self._stream_pool.append(transfer.stream)
            self._event_pool.append(transfer.end_event)
            self._event_pool.append(transfer.start_event)
            self._buffer_pool.append(
                (transfer.batch_src, transfer.batch_dst, transfer.batch_sizes)
            )
            del self._transfer_events[transfer.job_id]
        return results

    def wait(self, job_ids: set[int]):
        with self._lock:
            if any(not t.submitted for t in self._transfers):
                self._pump(force=True)
        for job_id in job_ids:
            event = self._transfer_events.get(job_id)
            if event is not None:
                event.synchronize()

    def shutdown(self) -> None:
        """Drain this direction and release its transfer-side resources."""
        with self._lock:
            if any(not t.submitted for t in self._transfers):
                self._pump(force=True)
        sync_error: Exception | None = None
        while self._transfers:
            transfer = self._transfers[0]
            try:
                transfer.end_event.synchronize()
            except Exception as e:
                logger.exception(
                    "Failed to synchronize transfer end event; "
                    "skipping %d remaining transfers",
                    len(self._transfers) - 1,
                )
                self._transfers.clear()
                sync_error = e
                break
            self._transfers.popleft()

        self._transfer_events.clear()
        self._stream_pool.clear()
        self._event_pool.clear()
        self._buffer_pool.clear()
        self.src_tensors.clear()
        self.dst_tensors.clear()
        if sync_error is not None:
            raise sync_error


class CPUOffloadingWorker(OffloadingWorker):
    """OffloadingWorker for CPU offloading.

    Composes two SingleDirectionOffloadingHandler instances (one for each
    direction) and exposes them through the explicit submit_store /
    submit_load API.
    """

    def __init__(
        self,
        kv_caches: CanonicalKVCaches,
        blocks_per_chunk: int,
        num_cpu_chunks: int,
        mmap_region: SharedOffloadRegion | None = None,
        canonical_layout: bool = False,
    ):
        assert not canonical_layout or mmap_region is not None
        # The caller owns mmap_region until this constructor returns. After a
        # successful construction, the worker is the sole owner and releases
        # it after both transfer directions have stopped.
        self._mmap_region = mmap_region
        pin_memory = PIN_MEMORY
        logger.info("Allocating %d CPU tensors...", len(kv_caches.tensors))
        if mmap_region is not None and pin_memory:
            pin_mmap_region(mmap_region)

        canonical_bytes_per_block = (
            _canonical_block_sizes(kv_caches.group_data_refs, len(kv_caches.tensors))
            if canonical_layout
            else None
        )

        gpu_tensors: list[torch.Tensor] = []
        cpu_tensors: list[torch.Tensor] = []
        for t_idx, kv_cache_tensor in enumerate(kv_caches.tensors):
            gpu_page_size_bytes = kv_cache_tensor.page_size_bytes
            gpu_tensor = kv_cache_tensor.tensor.view(torch.int8).view(
                (-1, gpu_page_size_bytes)
            )
            cpu_page_size_bytes = gpu_page_size_bytes * blocks_per_chunk

            if canonical_bytes_per_block is not None:
                assert mmap_region is not None
                cpu_tensor = mmap_region.create_next_canonical_view(
                    canonical_bytes_per_block[t_idx] * blocks_per_chunk
                )
            elif mmap_region is not None:
                cpu_tensor = mmap_region.create_next_worker_view(cpu_page_size_bytes)
            else:
                t0 = time.monotonic()
                cpu_tensor = torch.zeros(
                    (num_cpu_chunks, cpu_page_size_bytes),
                    dtype=torch.int8,
                    device="cpu",
                    pin_memory=pin_memory,
                )
                logger.debug(
                    "torch.zeros pinned tensor %d×%d (%.2f GB): %.3f s",
                    num_cpu_chunks,
                    cpu_page_size_bytes,
                    num_cpu_chunks * cpu_page_size_bytes / 1e9,
                    time.monotonic() - t0,
                )

            gpu_tensors.append(gpu_tensor)
            cpu_tensors.append(cpu_tensor)

        self._store_handler = SingleDirectionOffloadingHandler(
            gpu_tensors=gpu_tensors,
            cpu_tensors=cpu_tensors,
            blocks_per_chunk=blocks_per_chunk,
            layer_refs_per_group=kv_caches.group_data_refs,
            gpu_to_cpu=True,
            canonical_layout=canonical_layout,
        )

        self._load_handler = SingleDirectionOffloadingHandler(
            gpu_tensors=gpu_tensors,
            cpu_tensors=cpu_tensors,
            blocks_per_chunk=blocks_per_chunk,
            layer_refs_per_group=kv_caches.group_data_refs,
            gpu_to_cpu=False,
            canonical_layout=canonical_layout,
        )

    def submit_store(
        self, job_id: int, src_spec: GPULoadStoreSpec, dst_spec: LoadStoreSpec
    ) -> bool:
        """Async GPU -> CPU."""
        return self._store_handler.transfer_async(job_id, src_spec, dst_spec)

    def submit_load(
        self, job_id: int, src_spec: LoadStoreSpec, dst_spec: GPULoadStoreSpec
    ) -> bool:
        """Async CPU -> GPU."""
        return self._load_handler.transfer_async(job_id, src_spec, dst_spec)

    def submit_load_layerwise_async(
        self, job_id: int, src_spec: LoadStoreSpec, dst_spec: GPULoadStoreSpec
    ) -> bool:
        """Baton-style CPU -> GPU, layers submitted by the background thread."""
        return self._load_handler.transfer_async(
            job_id, src_spec, dst_spec, layerwise="async"
        )

    def submit_load_layerwise(
        self, job_id: int, src_spec: LoadStoreSpec, dst_spec: GPULoadStoreSpec
    ) -> bool:
        """In-step CPU -> GPU, one batch and one event per layer (LAYER_EVENTS)."""
        return self._load_handler.transfer_async(
            job_id, src_spec, dst_spec, layerwise=True
        )

    def get_finished(self) -> list[TransferResult]:
        return self._store_handler.get_finished() + self._load_handler.get_finished()

    def pump_at_inputs(self) -> None:
        self._load_handler.pump_at_inputs()

    def wait(self, job_ids: set[int]) -> None:
        self._store_handler.wait(job_ids)
        self._load_handler.wait(job_ids)

    def shutdown(self) -> None:
        handler_failed = False
        try:
            self._store_handler.shutdown()
        except Exception:
            logger.exception("Failed to shut down store offloading handler")
            handler_failed = True

        try:
            self._load_handler.shutdown()
        except Exception:
            logger.exception("Failed to shut down load offloading handler")
            handler_failed = True

        if self._mmap_region is not None:
            if handler_failed:
                try:
                    torch.accelerator.synchronize()
                except Exception:
                    logger.warning(
                        "Device sync before mmap cleanup failed; "
                        "proceeding with cleanup anyway",
                        exc_info=True,
                    )
            self._mmap_region.cleanup()
            self._mmap_region = None
