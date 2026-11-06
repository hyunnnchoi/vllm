// SPDX-License-Identifier: Apache-2.0
// aris experiment 133 (control "strata:<blocks>[:<warps>]"): Strata's GPU-assisted I/O kernel (SGLang HiCache), for a
// baseline restore in vLLM's own engine. The device code below is copied unchanged (NVIDIA path) from SGLang
// python/sglang/kernels/aot/csrc/kvcacheio/transfer.cu at commit 0163f8ff74c3d8f32e364e2b812d24dc715af039
// (https://github.com/sgl-project/sglang, Apache-2.0): transfer_item_warp, get_global_offset_lf, get_global_offset_pf
// and transfer_kernel_impl. Only the launcher is new: it takes raw pointers and a stream instead of torch tensors, so the
// library builds with nvcc alone (no torch ABI) and is called from Python through ctypes.
// Launch geometry is SGLang's transfer_kv_launcher: items_per_warp = ceil(items / (block_quota x warps_per_block)),
// blocks = ceil(items / (items_per_warp x warps_per_block)), 32 x warps_per_block threads. SGLang's default is
// block_quota 2, warps_per_block 32 (two blocks of 1024 threads), the configuration of Strata Fig. 5.
// Build: nvcc -O3 -arch=sm_90 -Xcompiler -fPIC -shared -o libstrata_transfer.so strata_transfer.cu
#include <cuda_runtime.h>

#include <cstdint>

#define WARP_SIZE 32

__device__ __forceinline__ void
transfer_item_warp(int32_t lane_id, const void* src_addr, void* dst_addr, int64_t item_size_bytes) {
  const uint64_t* __restrict__ src = static_cast<const uint64_t*>(src_addr);
  uint64_t* __restrict__ dst = static_cast<uint64_t*>(dst_addr);
  const int total_chunks = item_size_bytes / sizeof(uint64_t);

#pragma unroll
  for (int j = lane_id; j < total_chunks; j += WARP_SIZE) {
    uint64_t tmp;
    asm volatile("ld.global.nc.b64 %0,[%1];" : "=l"(tmp) : "l"(src + j) : "memory");
    asm volatile("st.global.cg.b64 [%0],%1;" ::"l"(dst + j), "l"(tmp) : "memory");
  }
}

template <typename T>
__device__ __forceinline__ T* get_global_offset_lf(
    T* base,
    const uintptr_t* __restrict__ /*unused*/,
    int64_t layer_id,
    int64_t layer_dim,
    int64_t page_id,
    int64_t item_size_bytes) {
  // layer first
  return base + layer_id * layer_dim + page_id * item_size_bytes;
}

template <typename T>
__device__ __forceinline__ T* get_global_offset_pf(
    T* base,
    const uintptr_t* __restrict__ /*unused*/,
    int64_t layer_id,
    int64_t page_dim,
    int64_t page_id,
    int64_t item_size_bytes) {
  // page first
  return base + page_id * page_dim + layer_id * item_size_bytes;
}

template <auto SrcOffsetFn, auto DstOffsetFn, bool IsMLA>
__global__ void transfer_kernel_impl(
    const void* __restrict__ src_k,
    void* __restrict__ dst_k,
    const void* __restrict__ src_v,
    void* __restrict__ dst_v,
    const int64_t* __restrict__ src_indices,
    const int64_t* __restrict__ dst_indices,
    int64_t start_layer_id,
    int64_t num_layers_to_process,
    int64_t num_items,
    int64_t items_per_warp,
    int64_t item_size_bytes,
    int64_t src_layout_dim,
    int64_t dst_layout_dim,
    const uintptr_t* __restrict__ src_k_layer_tbl,
    const uintptr_t* __restrict__ dst_k_layer_tbl,
    const uintptr_t* __restrict__ src_v_layer_tbl,
    const uintptr_t* __restrict__ dst_v_layer_tbl) {
  int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t lane_id = tid % WARP_SIZE;
  int32_t warp_id = tid / WARP_SIZE;

  for (int i = 0; i < items_per_warp; ++i) {
    int64_t item_id = warp_id * items_per_warp + i;
    if (item_id >= num_items) {
      break;
    }
    const int64_t src_page_id = src_indices[item_id];
    const int64_t dst_page_id = dst_indices[item_id];

    // Loop over layers if necessary
    for (int64_t layer_id = start_layer_id; layer_id < start_layer_id + num_layers_to_process; ++layer_id) {
      const char* src_ptr = SrcOffsetFn(
          static_cast<const char*>(src_k), src_k_layer_tbl, layer_id, src_layout_dim, src_page_id, item_size_bytes);
      char* dst_ptr = DstOffsetFn(
          static_cast<char*>(dst_k), dst_k_layer_tbl, layer_id, dst_layout_dim, dst_page_id, item_size_bytes);
      transfer_item_warp(lane_id, src_ptr, dst_ptr, item_size_bytes);

      if constexpr (!IsMLA) {
        const char* src_v_ptr = SrcOffsetFn(
            static_cast<const char*>(src_v), src_v_layer_tbl, layer_id, src_layout_dim, src_page_id, item_size_bytes);
        char* dst_v_ptr = DstOffsetFn(
            static_cast<char*>(dst_v), dst_v_layer_tbl, layer_id, dst_layout_dim, dst_page_id, item_size_bytes);
        transfer_item_warp(lane_id, src_v_ptr, dst_v_ptr, item_size_bytes);
      }
    }
  }
}

// vLLM restore: one item = one KV block, i.e. the 64 KiB K+V page of every layer, so the kernel runs in its MLA form
// (one buffer, no separate V). Source = the CPU tier, page first (a block's layers sit in one row of src_page_dim bytes);
// destination = the GPU KV cache, layer first (layer l of every block at dst_base + l x dst_layer_dim). The indices are
// row / block numbers relative to src_base / dst_base, int64 arrays in GPU memory. Returns the cudaError_t of the launch.
extern "C" int strata_restore_pf_lf(const void* src_base, void* dst_base, const int64_t* src_indices,
                                    const int64_t* dst_indices, int64_t num_items, int64_t num_layers,
                                    int64_t item_size, int64_t src_page_dim, int64_t dst_layer_dim,
                                    int64_t block_quota, int64_t num_warps_per_block, void* stream) {
  if (num_items <= 0) return 0;
  auto div_up = [](int64_t x, int64_t y) { return (x + y - 1) / y; };
  const int64_t items_per_warp = div_up(num_items, block_quota * num_warps_per_block);
  const int32_t num_blocks = div_up(num_items, items_per_warp * num_warps_per_block);
  const int32_t threads_per_block = num_warps_per_block * WARP_SIZE;
  transfer_kernel_impl<get_global_offset_pf<const char>, get_global_offset_lf<char>, true>
      <<<num_blocks, threads_per_block, 0, static_cast<cudaStream_t>(stream)>>>(
          src_base, dst_base, nullptr, nullptr, src_indices, dst_indices, 0, num_layers, num_items, items_per_warp,
          item_size, src_page_dim, dst_layer_dim, nullptr, nullptr, nullptr, nullptr);
  return static_cast<int>(cudaGetLastError());
}
