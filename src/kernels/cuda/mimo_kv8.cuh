#pragma once
#include <cuda_fp16.h>
#include <cstdint>

namespace strata::kernels {
// FP32 scale per [position][KV head]. No zero-scale division, no FP16 scale underflow.
// MMA/shared-half readers round the reconstructed value once to FP16.
template<bool KV8>
__device__ __forceinline__ float mimo_cache_load(const void* cache, const float* scales,
                                                size_t row, int dim, int d) {
    if constexpr (KV8) return float(static_cast<const int8_t*>(cache)[row * dim + d]) * scales[row];
    else return __half2float(static_cast<const __half*>(cache)[row * dim + d]);
}
}
