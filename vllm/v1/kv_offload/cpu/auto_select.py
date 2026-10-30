# SPDX-License-Identifier: Apache-2.0
"""aris "ctrlauto": startup calibration of the CPU->GPU copy paths and per-load path lookup.

At worker start (VLLM_KV_AUTO_CALIB=1, idle GPU, no decode), copies of the model's real fragment size between a
pinned host buffer and a GPU buffer are timed (CUDA events, median of `reps` after one warm-up):
  - SM copy kernel shapes P x J (W = 4) at N_LARGE descriptors; the shape is the smallest P (then J) whose bandwidth
    is >= KEEP x the best shape's;
  - protected DMA (cuMemcpyBatchAsync in calls of 4,096 descriptors) and the chosen SM shape at each N in N_SMALL and
    N_LARGE.
For each N the path is SM if its bandwidth is > FASTER x DMA's, else DMA (ties -> DMA). A load of n descriptors uses
the decision of the nearest measured N on a log scale.
"""
import functools
import time

import numpy as np
import torch

from vllm import _custom_ops as ops
from vllm.v1.kv_offload.cpu import swap_blocks_triton as _sbt

SM_P = (1, 2, 4, 8, 12, 16)
SM_J = (1, 2, 4, 8)
SM_W = 4
N_SMALL = (16, 32, 64, 128, 256, 1024)
N_LARGE = 8192
DMA_CHUNK = 4096
KEEP, FASTER = 0.98, 1.05


def _dma_chunked(s, d, z, is_src_access_order_any=False):
    for i in range(0, s.numel(), DMA_CHUNK):
        ops.swap_blocks_batch(s[i:i + DMA_CHUNK], d[i:i + DMA_CHUNK], z[i:i + DMA_CHUNK])


class AutoTable:
    def __init__(self, frag: int, bytes_per_chunk: int, reps: int = 3):
        t0 = time.time()
        size = frag * N_LARGE
        cpu = torch.empty(size, dtype=torch.int8, pin_memory=True)
        gpu = torch.empty(size, dtype=torch.int8, device="cuda")
        perm = np.random.default_rng(0).permutation(N_LARGE)
        src = torch.from_numpy((cpu.data_ptr() + np.arange(N_LARGE, dtype=np.int64) * frag).astype(np.int64))
        dst = torch.from_numpy((gpu.data_ptr() + perm.astype(np.int64) * frag).astype(np.int64))
        sz = torch.full((N_LARGE,), frag, dtype=torch.int64)
        stream = torch.cuda.Stream()

        def bw(fn, n: int, r: int = reps) -> float:
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            ms = []
            with torch.cuda.stream(stream):
                for i in range(r + 1):
                    e0.record(stream)
                    fn(src[:n], dst[:n], sz[:n])
                    e1.record(stream)
                    e1.synchronize()
                    if i:
                        ms.append(e0.elapsed_time(e1))
            return n * frag / (float(np.median(ms)) * 1e-3)

        def sm(p: int, j: int):
            return functools.partial(_sbt.swap_blocks_batch_j, bytes_per_chunk=bytes_per_chunk, P=p, W=SM_W, J=j)

        self.frag = frag
        self.bw_sm = {(p, j): bw(sm(p, j), N_LARGE) for p in SM_P for j in SM_J}
        best = max(self.bw_sm.values())
        self.shape = min((pj for pj, b in self.bw_sm.items() if b >= KEEP * best), key=lambda pj: (pj[0], pj[1]))
        fn_sm = sm(*self.shape)
        self.table = {n: (bw(_dma_chunked, n, 5), bw(fn_sm, n, 5)) for n in N_SMALL}
        self.table[N_LARGE] = (bw(_dma_chunked, N_LARGE), self.bw_sm[self.shape])
        self.path = {n: ("sm" if s > FASTER * d else "dma") for n, (d, s) in self.table.items()}
        self.scratch_bytes = 2 * size
        del cpu, gpu
        torch.cuda.empty_cache()
        self.seconds = time.time() - t0

    def choose(self, n: int) -> str:
        k = min(self.path, key=lambda m: abs(np.log2(max(n, 1)) - np.log2(m)))
        return self.path[k]

    def summary(self) -> dict:
        return dict(frag=self.frag, shape=list(self.shape), seconds=round(self.seconds, 3),
                    scratch_mib=self.scratch_bytes / 2**20,
                    bw_sm={f"{p}x{j}": round(b / 1e9, 3) for (p, j), b in self.bw_sm.items()},
                    table={n: [round(d / 1e9, 3), round(s / 1e9, 3), self.path[n]] for n, (d, s) in self.table.items()})
