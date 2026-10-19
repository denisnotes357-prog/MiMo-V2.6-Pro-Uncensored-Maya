#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace strata::kernels {
// MiMo-V2 (mimo2) attention for the token path and the MTP verify window over a long cache: split-K flash decoding
// that shares each KV head's keys and values among its query heads (GQA) and among the window's rows.
//
// The per-(row, head) kernel (glm_fast_path.cu mimo_attn_flash) reads a KV head's whole cache once for every one of
// its n_head / n_kv query heads and once more for every window row - 16 x 8 reads of ~60 MB a full layer at 12K
// context - and walks the keys serially in one block per head.  Here a block takes one KV head, a slice of the keys
// and up to kMimoDecRows query rows (row = t * group + g: window row t, query head kvh * group + g); it loads each key
// tile once into shared memory, scores every row against it, and keeps per-row online-softmax state.  A second kernel
// merges the slices (and the sink logit, a zero-valued extra entry) into the output.
//
// q: rows x [n_head x head_k] FP32 (row stride q_ld), row t at position p0 + t; kc / vc: FP16 [ring][n_kv][head_k|v],
// position j at row j % ring; window > 0: row t sees keys max(0, p0+t-window+1) .. p0+t, else 0 .. p0+t.
// out: rows x [n_head x head_v] FP32 (row stride out_ld), times v_scale.  head_k == 192, head_v == 128.
struct MimoDecAttnArgs {
    int n_head, n_kv, head_k, head_v, ring, window, p0, T;
    float scale, v_scale;
};
constexpr int kMimoDecRows = 64;   // query rows a block scores (group x window rows; 16 x 8 = 128 rows -> 2 row blocks)

// Scratch floats the call needs for these arguments and n_split slices.
size_t mimo_decode_attn_scratch(const MimoDecAttnArgs& a, int n_split);
// The slice count it would use for this many keys (1 .. 32).
int mimo_decode_attn_splits(const MimoDecAttnArgs& a);
// false: geometry not covered (head_k != 192 / head_v != 128 / group * T > 512), nothing launched.
bool mimo_decode_attn(const float* q, int q_ld, const uint16_t* kc, const uint16_t* vc, const float* sinks, float* out,
                      int out_ld, const MimoDecAttnArgs& a, float* scratch, size_t scratch_floats, cudaStream_t stream);
// Same algorithm, INT8 [ring][n_kv][dim], FP32 scales [ring][n_kv].
// Reconstructed K/V are rounded to FP16 in shared memory, as in the prefill MMA path.
bool mimo_decode_attn_kv8(const float* q, int q_ld, const int8_t* kc, const int8_t* vc,
                          const float* ks, const float* vs, const float* sinks, float* out,
                          int out_ld, const MimoDecAttnArgs& a, float* scratch,
                          size_t scratch_floats, cudaStream_t stream);
}  // namespace strata::kernels
