# SPDX-License-Identifier: Apache-2.0
"""aris experiment baseline ("ipc" pace setting): CPU->GPU KV loads issued by a separate
long-lived process (its own CUDA context, so its own copy-engine time slice) that writes
the engine's paged KV cache through CUDA IPC.

Worker side (IpcLoaderClient): starts `python -m vllm.v1.kv_offload.cpu.ipc_loader` once,
sends the GPU KV tensors (torch CUDA IPC reductions) and the CPU-tier shared region (path
of an open file descriptor of this process), then per load: an interprocess "ready" event
recorded on the compute stream (the load must not start before pending writes to its
destination blocks) and an interprocess "done" event, plus the descriptors as offsets
(CPU region offset, KV tensor index and offset, size). The loader waits on "ready", runs
one cuMemcpyBatchAsync on its stream, records "done", and replies with the GPU start/end
time on the wall clock (its own timing events).
"""
import mmap
import os
import secrets
import subprocess
import sys
import tempfile
import threading
import time
from multiprocessing.connection import Client, Listener

import numpy as np
import torch


class _IpcDone:
    """Completion handle of one IPC load. The interprocess event alone is not enough:
    it is recorded by the loader only after the loader receives the message, and an
    event that was never recorded reports complete on query()/synchronize(). So a load
    is complete when the loader's reply (sent after its copy finished) has arrived."""

    def __init__(self, client: "IpcLoaderClient", job_id: int, event: torch.cuda.Event):
        self._client, self._job, self.event = client, job_id, event

    def query(self) -> bool:
        return self._job in self._client.times and self.event.query()

    def synchronize(self) -> None:
        while self._job not in self._client.times:
            if self._client._proc.poll() is not None:
                raise RuntimeError("ipc loader process exited")
            time.sleep(0.0005)
        self.event.synchronize()

    def wait(self, stream=None) -> None:
        # GPU-side ordering after an IPC load: only another IPC load chains on it, and
        # those run in order on the loader's single stream
        self.event.wait(stream)


class IpcLoaderClient:
    def __init__(self, gpu_tensors: list[torch.Tensor], region_base: int, region_size: int,
                 region_fd: int, device: int):
        from torch.multiprocessing.reductions import reduce_tensor

        self._gpu_bases = np.array([t.data_ptr() for t in gpu_tensors], dtype=np.int64)
        self._gpu_ends = self._gpu_bases + np.array(
            [t.numel() * t.element_size() for t in gpu_tensors], dtype=np.int64)
        self._order = np.argsort(self._gpu_bases)
        self._region_base, self._region_size = region_base, region_size
        self._device = device
        addr = os.path.join(tempfile.gettempdir(), f"aris_ipc_loader_{os.getpid()}_{secrets.token_hex(4)}")
        key = secrets.token_bytes(16)
        listener = Listener(addr, family="AF_UNIX", authkey=key)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", ""))
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "vllm.v1.kv_offload.cpu.ipc_loader", addr, key.hex(), str(device)],
            env=env)
        self._conn = listener.accept()
        listener.close()
        self._conn.send(dict(
            kv=[reduce_tensor(t) for t in gpu_tensors],
            region_path=f"/proc/{os.getpid()}/fd/{region_fd}", region_size=region_size))
        assert self._conn.recv() == "ready"
        self.times: dict[int, tuple[float, float]] = {}
        self._send_lock = threading.Lock()
        self._reader = threading.Thread(target=self._read, daemon=True, name="ipc-loader-reply")
        self._reader.start()

    def _read(self) -> None:
        while True:
            try:
                job, g0, g1 = self._conn.recv()
            except (EOFError, OSError):
                return
            self.times[job] = (g0, g1)

    def submit(self, job_id: int, src: np.ndarray, dst: np.ndarray, sizes: np.ndarray,
               stream: torch.cuda.Stream) -> _IpcDone:
        """Queue one load; returns its completion handle (loader reply + interprocess event)."""
        k = np.searchsorted(self._gpu_bases[self._order], dst, side="right") - 1
        tidx = self._order[k]
        dst_off = dst - self._gpu_bases[tidx]
        assert (dst + sizes <= self._gpu_ends[tidx]).all() and (dst_off >= 0).all()
        src_off = src - self._region_base
        assert (src_off >= 0).all() and (src_off + sizes <= self._region_size).all()
        ready = torch.cuda.Event(interprocess=True)
        ready.record(stream)
        done = torch.cuda.Event(interprocess=True)
        with self._send_lock:
            self._conn.send((job_id, src_off.astype(np.int64), tidx.astype(np.int32),
                             dst_off.astype(np.int64), sizes.astype(np.int64),
                             ready.ipc_handle(), done.ipc_handle()))
        return _IpcDone(self, job_id, done)

    def close(self) -> None:
        try:
            with self._send_lock:
                self._conn.send(None)
            self._proc.wait(timeout=30)
        except Exception:
            self._proc.kill()


def _main(addr: str, key_hex: str, device: int) -> None:
    from vllm import _custom_ops as ops

    conn = Client(addr, family="AF_UNIX", authkey=bytes.fromhex(key_hex))
    torch.cuda.set_device(device)
    init = conn.recv()
    kv = [fn(*args) for fn, args in init["kv"]]
    bases = np.array([t.data_ptr() for t in kv], dtype=np.int64)
    fd = os.open(init["region_path"], os.O_RDWR)
    mm = mmap.mmap(fd, init["region_size"])
    host = torch.frombuffer(memoryview(mm), dtype=torch.int8)
    rc = torch.cuda.cudart().cudaHostRegister(host.data_ptr(), init["region_size"], 0)
    assert rc.value == 0, f"cudaHostRegister failed: {rc}"
    hbase = host.data_ptr()
    stream = torch.cuda.Stream()
    ref = torch.cuda.Event(enable_timing=True)
    ref.record(stream)
    ref.synchronize()
    t_ref = time.time()
    lock = threading.Lock()
    pending: list = []
    cv = threading.Condition()

    def reply() -> None:
        while True:
            with cv:
                while not pending:
                    cv.wait()
                item = pending.pop(0)
            if item is None:
                return
            job, s, e = item
            e.synchronize()
            with lock:
                conn.send((job, t_ref + ref.elapsed_time(s) / 1e3, t_ref + ref.elapsed_time(e) / 1e3))

    th = threading.Thread(target=reply, daemon=True)
    th.start()
    conn.send("ready")
    while True:
        msg = conn.recv()
        if msg is None:
            break
        job, src_off, tidx, dst_off, sizes, ready_h, done_h = msg
        ready = torch.cuda.Event.from_ipc_handle(device, ready_h)
        done = torch.cuda.Event.from_ipc_handle(device, done_h)
        src = torch.from_numpy(hbase + src_off)
        dst = torch.from_numpy(bases[tidx] + dst_off)
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            stream.wait_event(ready)
            s.record(stream)
            ops.swap_blocks_batch(src, dst, torch.from_numpy(sizes))
            e.record(stream)
            done.record(stream)
        with cv:
            pending.append((job, s, e))
            cv.notify()
    with cv:
        pending.append(None)
        cv.notify()
    th.join(timeout=30)


if __name__ == "__main__":
    _main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
