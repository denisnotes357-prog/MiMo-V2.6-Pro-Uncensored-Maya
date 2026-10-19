#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace strata::kernels {
// Row-major fused Q|K|V, FP32; caches are [ring, kv_heads, head_dim], FP16.
// rope_kv requires T <= ring (unique destinations). For attention, the caller
// must additionally keep T <= ring-window+1 on SWA layers, to retain old keys.
struct MimoPrefillArgs {
    int heads, kv_heads, head_k, head_v, rope_dim, ring, window;
    float rope_base, v_scale;
};
void mimo_prefill_rope_kv(float* qkv, uint16_t* kc, uint16_t* vc, int p0, int T,
                          const MimoPrefillArgs& a, cudaStream_t stream);
// Causal tiled attention; optional sink is a zero-valued extra softmax entry.
// CUDA 192/128 geometry uses tensor cores. Other supported head sizes use FP32.
void mimo_prefill_attention(const float* qkv, const uint16_t* kc, const uint16_t* vc,
                            const float* sinks, float* out, int p0, int T,
                            const MimoPrefillArgs& a, cudaStream_t stream, bool scalar = false);
// Store already-rotated FP32 K/V. Input strides are floats; one scale per position/head.
void mimo_kv8_store(const float* k, int k_ld, const float* v, int v_ld,
                    int8_t* kc, int8_t* vc, float* ks, float* vs,
                    int p0, int T, int ring, int kv_heads, int head_k, int head_v, cudaStream_t stream);
void mimo_prefill_rope_kv8(float* qkv, int8_t* kc, int8_t* vc, float* ks, float* vs,
                          int p0, int T, const MimoPrefillArgs& a, cudaStream_t stream);
void mimo_prefill_attention_kv8(const float* qkv, const int8_t* kc, const int8_t* vc,
                               const float* ks, const float* vs, const float* sinks,
                               float* out, int p0, int T, const MimoPrefillArgs& a,
                               cudaStream_t stream, bool scalar = false);
}
