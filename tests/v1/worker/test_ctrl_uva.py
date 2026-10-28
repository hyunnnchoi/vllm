# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ctrl_uva: small pinned host<->GPU copies take the SM path and keep their
values; everything else keeps the normal path."""

import numpy as np
import pytest
import torch

from vllm.v1.worker import ctrl_uva

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)


@pytest.fixture
def installed():
    ctrl_uva.install(force=True)
    yield
    ctrl_uva.uninstall()


def test_h2d_and_d2h(installed):
    n0 = ctrl_uva.count()
    host = torch.arange(1024, dtype=torch.int32).pin_memory()
    gpu = torch.empty(1024, dtype=torch.int32, device="cuda")
    gpu.copy_(host, non_blocking=True)
    back = torch.empty(1024, dtype=torch.int32).pin_memory()
    back.copy_(gpu, non_blocking=True)
    moved = host.to("cuda", non_blocking=True)
    moved_back = gpu.to("cpu", non_blocking=True)
    torch.cuda.synchronize()
    assert ctrl_uva.count() - n0 == 4
    assert torch.equal(back, host)
    assert torch.equal(moved.cpu(), host)
    assert torch.equal(moved_back, host)


def test_odd_sizes(installed):
    for n in (1, 3, 4095, 4097, 70000):
        host = torch.randint(0, 255, (n,), dtype=torch.uint8).pin_memory()
        gpu = torch.zeros(n, dtype=torch.uint8, device="cuda")
        gpu.copy_(host, non_blocking=True)
        torch.cuda.synchronize()
        assert torch.equal(gpu.cpu(), host)


def test_other_copies_keep_normal_path(installed):
    n0 = ctrl_uva.count()
    big = torch.ones(ctrl_uva.MAX_BYTES // 4 + 1, dtype=torch.int32).pin_memory()
    torch.empty_like(big, device="cuda").copy_(big, non_blocking=True)
    pageable = torch.ones(16, dtype=torch.int32)
    torch.empty(16, dtype=torch.int32, device="cuda").copy_(pageable, non_blocking=True)
    pinned = torch.ones(16, dtype=torch.int32).pin_memory()
    torch.empty(16, dtype=torch.int32, device="cuda").copy_(pinned)  # blocking
    torch.cuda.synchronize()
    assert ctrl_uva.count() == n0


def test_v2_uva_buffer_pool(installed):
    from vllm.v1.worker.gpu.buffer_utils import UvaBufferPool

    pool = UvaBufferPool(256, torch.int32)
    n0 = ctrl_uva.count()
    x = pool.copy_to_gpu(np.arange(100, dtype=np.int32))
    torch.cuda.synchronize()
    assert ctrl_uva.count() - n0 == 1
    assert torch.equal(x.cpu(), torch.arange(100, dtype=torch.int32))
