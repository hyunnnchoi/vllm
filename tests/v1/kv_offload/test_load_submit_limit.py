# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""wait() on the CPU->GPU handler must keep the load submit-depth limit.

A store flush (jobs_to_flush) calls wait() on both directions. The load handler
used to force-submit every pending load, past the limit. CUDA events, streams
and the batch copy are replaced by CPU fakes; a fake event completes only when
it or a later event is synchronized.
"""

import contextlib
import threading
from collections import deque
from types import SimpleNamespace

import pytest

from vllm.v1.kv_offload.cpu import gpu_worker

CAP, CHUNK, OPS = 16384, 4096, 424320


class _Gpu:
    def __init__(self):
        self.order: list = []
        self.submits: list[int] = []
        self.peak = 0

    def event(self):
        gpu = self

        class Event:
            done = False

            def record(self, stream=None):
                gpu.order.append(self)

            def query(self):
                return self.done

            def synchronize(self):
                for e in gpu.order[: gpu.order.index(self) + 1]:
                    e.done = True

        return Event()

    def copy(self, src, dst, sizes, **kwargs):
        pending = zip(self.submits, self.order)
        outstanding = sum(n for n, e in pending if not e.done) + len(src)
        self.peak = max(self.peak, outstanding)
        self.submits.append(len(src))


@pytest.fixture
def handler_with_load(monkeypatch):
    gpu = _Gpu()
    monkeypatch.setattr(gpu_worker.torch, "Event", gpu.event)
    monkeypatch.setattr(
        gpu_worker,
        "current_platform",
        SimpleNamespace(stream=lambda s: contextlib.nullcontext()),
    )
    handler = object.__new__(gpu_worker.SingleDirectionOffloadingHandler)
    transfer = SimpleNamespace(
        job_id=7,
        num_ops=OPS,
        next_op=0,
        submitted=False,
        stream=None,
        end_event=gpu.event(),
        batch_src=list(range(OPS)),
        batch_dst=list(range(OPS)),
        batch_sizes=[65536] * OPS,
        is_src_access_order_any=False,
    )
    handler._lock = threading.RLock()
    handler._transfers = deque([transfer])
    handler._inflight_chunks = deque()
    handler._submit_cap, handler._submit_chunk = CAP, CHUNK
    handler._transfer_events = {}
    handler._swap_blocks_batch = gpu.copy
    handler._submit_pending()
    assert sum(gpu.submits) == CAP
    return gpu, handler, transfer


def test_store_flush_wait_does_not_submit_pending_loads(handler_with_load):
    gpu, handler, transfer = handler_with_load
    handler.wait({999})
    assert sum(gpu.submits) == CAP
    assert gpu.peak <= CAP


def test_wait_on_load_submits_it_within_the_limit(handler_with_load):
    gpu, handler, transfer = handler_with_load
    handler.wait({7})
    assert transfer.submitted
    assert sum(gpu.submits) == OPS
    assert gpu.peak <= CAP
