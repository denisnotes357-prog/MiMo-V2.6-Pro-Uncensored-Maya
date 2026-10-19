// src/kernels/cuda/mimo_decode_attn.cu - MiMo-V2 (mimo2) split-K, GQA-sharing attention for the token path and the
// MTP verify window (strata/kernels/mimo_decode_attn.hpp).
#include "strata/kernels/mimo_decode_attn.hpp"

#include <cuda_fp16.h>
#include "mimo_kv8.cuh"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace strata::kernels {
namespace {
constexpr int HK = 192, HV = 128;          // MiMo-V2's head dims (q.k 192, v 128)
constexpr int KT = 32;                     // keys a tile
constexpr int RB = kMimoDecRows;           // query rows a block
constexpr int NT = 256;                    // threads a block
constexpr int KS2 = HK / 2 + 1;            // a key row in shared memory, in half2 (odd: no bank conflicts across keys)
constexpr int VS2 = HV / 2;                // a value row, in half2
constexpr int kMaxSplit = 64;

constexpr size_t smem_bytes() {
    return sizeof(float) * (size_t) RB * HK                 // sQ (scaled q rows)
         + sizeof(__half2) * (size_t) KT * KS2              // sK
         + sizeof(__half2) * (size_t) KT * VS2              // sV
         + sizeof(float) * (size_t) RB * KT                 // sP (scores, then probabilities)
         + sizeof(float) * (size_t) RB * 3;                 // sM, sL, sC
}

// one KV head (blockIdx.x), one key slice (blockIdx.y), one block of RB query rows (blockIdx.z)
template<bool KV8>
__global__ void __launch_bounds__(NT) dec_partial(const float* __restrict__ q, int q_ld, const __half2* __restrict__ kc,
                                                 const __half2* __restrict__ vc, const float* ks, const float* vs, MimoDecAttnArgs a, int n_split,
                                                 float* __restrict__ part_o, float* __restrict__ part_m,
                                                 float* __restrict__ part_l) {
    extern __shared__ __align__(16) unsigned char smem[];
    float* sQ = (float*) smem;
    __half2* sK = (__half2*) (sQ + RB * HK);
    __half2* sV = sK + KT * KS2;
    float* sP = (float*) (sV + KT * VS2);
    float* sM = sP + RB * KT;
    float* sL = sM + RB;
    float* sC = sL + RB;

    const int kvh = blockIdx.x, split = blockIdx.y, rb = blockIdx.z;
    const int G = a.n_head / a.n_kv, R = G * a.T;
    const int r0 = rb * RB, nr = min(RB, R - r0);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    if (nr <= 0) return;

    // the block's keys: the union of its rows' ranges, cut into n_split slices
    const int t_first = r0 / G, t_last = (r0 + nr - 1) / G;
    const int lo_min = a.window > 0 ? max(0, a.p0 + t_first - a.window + 1) : 0;
    const int hi = a.p0 + t_last;
    const int total = hi - lo_min + 1;
    const int per = (total + n_split - 1) / n_split;
    const int k0 = lo_min + split * per, k1 = min(hi + 1, k0 + per);

    for (int idx = tid; idx < nr * HK; idx += NT) {
        const int i = idx / HK, d = idx % HK, gr = r0 + i;
        const int t = gr / G, h = kvh * G + gr % G;
        sQ[idx] = q[(size_t) t * q_ld + (size_t) h * HK + d] * a.scale;
    }
    for (int i = tid; i < RB; i += NT) {
        sM[i] = -INFINITY;
        sL[i] = 0.0f;
        sC[i] = 1.0f;
    }
    float acc[RB / 2];
#pragma unroll
    for (int k = 0; k < RB / 2; ++k) acc[k] = 0.0f;
    const int d_own = tid % HV, i_own = tid / HV;   // this thread's output dim; rows i_own, i_own + 2, ...
    __syncthreads();

    for (int kt = k0; kt < k1; kt += KT) {
        const int nk = min(KT, k1 - kt);
        for (int idx = tid; idx < nk * (HK / 2); idx += NT) {
            const int c = idx / (HK / 2), d2 = idx % (HK / 2);
            const size_t row = (size_t) ((kt + c) % a.ring) * a.n_kv + kvh;
            if constexpr (KV8) sK[c * KS2 + d2] = __floats2half2_rn(
                mimo_cache_load<true>(kc, ks, row, HK, d2 * 2),
                mimo_cache_load<true>(kc, ks, row, HK, d2 * 2 + 1));
            else sK[c * KS2 + d2] = kc[row * (HK / 2) + d2];
        }
        for (int idx = tid; idx < nk * VS2; idx += NT) {
            const int c = idx / VS2, d2 = idx % VS2;
            const size_t row = (size_t) ((kt + c) % a.ring) * a.n_kv + kvh;
            if constexpr (KV8) sV[c * VS2 + d2] = __floats2half2_rn(
                mimo_cache_load<true>(vc, vs, row, HV, d2 * 2),
                mimo_cache_load<true>(vc, vs, row, HV, d2 * 2 + 1));
            else sV[c * VS2 + d2] = vc[row * VS2 + d2];
        }
        __syncthreads();
        // scores: row i against key c (masked to the row's own range)
        for (int idx = tid; idx < nr * KT; idx += NT) {
            const int i = idx / KT, c = idx % KT, gr = r0 + i;
            const int pos = a.p0 + gr / G;
            const int lo = a.window > 0 ? max(0, pos - a.window + 1) : 0;
            const int j = kt + c;
            float s = -INFINITY;
            if (c < nk && j <= pos && j >= lo) {
                const float* qi = sQ + i * HK;
                const __half2* kr = sK + c * KS2;
                float dot = 0.0f;
#pragma unroll 8
                for (int d2 = 0; d2 < HK / 2; ++d2) {
                    const float2 kf = __half22float2(kr[d2]);
                    dot += qi[2 * d2] * kf.x + qi[2 * d2 + 1] * kf.y;
                }
                s = dot;
            }
            sP[i * KT + c] = s;
        }
        __syncthreads();
        // the rows' online softmax: a warp a row, a lane a key
        for (int i = warp; i < nr; i += NT / 32) {
            const float s = sP[i * KT + lane];
            float mx = s;
            for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            const float m_old = sM[i], m_new = fmaxf(m_old, mx);
            float p = 0.0f, corr = 1.0f;
            if (m_new != -INFINITY) {
                p = s == -INFINITY ? 0.0f : __expf(s - m_new);
                corr = m_old == -INFINITY ? 0.0f : __expf(m_old - m_new);
            }
            float sum = p;
            for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
            sP[i * KT + lane] = p;
            if (lane == 0) {
                sL[i] = sL[i] * corr + sum;
                sM[i] = m_new;
                sC[i] = corr;
            }
        }
        __syncthreads();
        // the outputs: a thread one dim of every other row
        const __half* sVh = (const __half*) sV;
#pragma unroll 4
        for (int k = 0; k < RB / 2; ++k) {
            const int i = i_own + 2 * k;
            if (i < nr) {
                float o = acc[k] * sC[i];
                const float* pi = sP + i * KT;
                for (int c = 0; c < nk; ++c) o += pi[c] * __half2float(sVh[c * HV + d_own]);
                acc[k] = o;
            }
        }
        __syncthreads();
    }
    const size_t base = ((size_t) split * a.n_kv + kvh) * R;
#pragma unroll 4
    for (int k = 0; k < RB / 2; ++k) {
        const int i = i_own + 2 * k;
        if (i < nr) part_o[(base + r0 + i) * HV + d_own] = acc[k];
    }
    for (int i = tid; i < nr; i += NT) {
        part_m[base + r0 + i] = sM[i];
        part_l[base + r0 + i] = sL[i];
    }
}

// ---- the tensor-core path for one window row (T = 1: the token path and every row of the pipelined MTP window) with
// 16 query heads a KV head (MiMo-V2's full layers, 128 / 8): the group's 16 rows are one m16 tile.  A block = one KV head x
// one key slice, MW warps; the slice's keys come in tiles of MK through shared memory (cp.async, double-buffered) and warp
// w takes keys [16w, 16w + 16) of each tile: S = Q K^T (mma m16n8k16, f16 in, f32 out), the online softmax on the
// fragments, O += P V with P kept in registers (FlashAttention-2's reuse).  The warps' (m, l, O) merge in shared memory
// into the same slice partial the scalar kernel writes, so dec_combine (the slices, the sink, v_scale) is unchanged.
constexpr int MK = 64;                     // keys a tile
constexpr int MW = 4;                      // warps a block
constexpr int MKS = HK + 8;                // a key row in shared memory, in halves (400 B: 16-byte aligned, ldmatrix
constexpr int MVS = HV + 8;                //   rows spread over the banks); a value row (272 B)
constexpr size_t mma_smem_bytes() {
    return sizeof(__half) * (16 * (size_t) MKS + 2 * (size_t) MK * MKS + 2 * (size_t) MK * MVS);
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    const unsigned s = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void ldsm_x4(unsigned (&r)[4], const void* p) {
    const unsigned s = (unsigned) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
__device__ __forceinline__ void ldsm_x4_t(unsigned (&r)[4], const void* p) {
    const unsigned s = (unsigned) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
__device__ __forceinline__ void mma16816(float (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ unsigned pack_h2(float lo, float hi) {
    __half2 h = __floats2half2_rn(lo, hi);
    return *(unsigned*) &h;
}

// one tile of keys for warp `warp`'s 16 keys: S, the masked online softmax, O += P V (tK / tV: the warp's rows)
struct MmaRow {
    float o[HV / 8][4];
    float m0, m1, l0, l1;   // rows g and g + 8
};
__device__ __forceinline__ void mma_tile(MmaRow& st, const unsigned (&qf)[HK / 16][4], const __half* tK, const __half* tV,
                                         int jb, int k1, int lane) {
    const int tq = lane & 3;
    float sc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#pragma unroll
    for (int s = 0; s < HK / 16; ++s) {
        unsigned b[4];
        ldsm_x4(b, tK + ((lane & 7) + (lane >> 4) * 8) * MKS + s * 16 + ((lane >> 3) & 1) * 8);
        mma16816(sc[0], qf[s], b[0], b[1]);
        mma16816(sc[1], qf[s], b[2], b[3]);
    }
    jb += 2 * tq;
    float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
    for (int n = 0; n < 2; ++n)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            if (jb + 8 * n + e >= k1) sc[n][e] = sc[n][2 + e] = -INFINITY;
            mx0 = fmaxf(mx0, sc[n][e]);
            mx1 = fmaxf(mx1, sc[n][2 + e]);
        }
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
    const float nm0 = fmaxf(st.m0, mx0), nm1 = fmaxf(st.m1, mx1);
    const float c0 = nm0 == -INFINITY ? 1.0f : __expf(st.m0 - nm0), c1 = nm1 == -INFINITY ? 1.0f : __expf(st.m1 - nm1);
    float sum0 = 0.0f, sum1 = 0.0f;
#pragma unroll
    for (int n = 0; n < 2; ++n)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            sc[n][e] = nm0 == -INFINITY ? 0.0f : __expf(sc[n][e] - nm0);
            sc[n][2 + e] = nm1 == -INFINITY ? 0.0f : __expf(sc[n][2 + e] - nm1);
            sum0 += sc[n][e];
            sum1 += sc[n][2 + e];
        }
    sum0 += __shfl_xor_sync(0xffffffffu, sum0, 1);
    sum0 += __shfl_xor_sync(0xffffffffu, sum0, 2);
    sum1 += __shfl_xor_sync(0xffffffffu, sum1, 1);
    sum1 += __shfl_xor_sync(0xffffffffu, sum1, 2);
    st.l0 = st.l0 * c0 + sum0;
    st.l1 = st.l1 * c1 + sum1;
    st.m0 = nm0;
    st.m1 = nm1;
    const unsigned pa[4] = {pack_h2(sc[0][0], sc[0][1]), pack_h2(sc[0][2], sc[0][3]), pack_h2(sc[1][0], sc[1][1]),
                            pack_h2(sc[1][2], sc[1][3])};
#pragma unroll
    for (int u = 0; u < HV / 16; ++u) {
        unsigned b[4];
        ldsm_x4_t(b, tV + ((lane & 7) + ((lane >> 3) & 1) * 8) * MVS + u * 16 + (lane >> 4) * 8);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            float(&on)[4] = st.o[2 * u + h];
            on[0] *= c0;
            on[1] *= c0;
            on[2] *= c1;
            on[3] *= c1;
            mma16816(on, pa, b[2 * h], b[2 * h + 1]);
        }
    }
}

// the warps' (m, l, O) -> shared memory at `scratch` (free tile space), merged into the slice partial
__device__ __forceinline__ void mma_merge(const MmaRow& st, unsigned char* scratch, size_t base, float* part_o,
                                          float* part_m, float* part_l, int tid, int warp, int lane) {
    const int g = lane >> 2, tq = lane & 3;
    float* sm = (float*) scratch;   // [MW][16] m, [MW][16] l, [MW][16][HV] o
    float* sl = sm + MW * 16;
    float* so = sl + MW * 16;
    if (tq == 0) {
        sm[warp * 16 + g] = st.m0;
        sm[warp * 16 + g + 8] = st.m1;
        sl[warp * 16 + g] = st.l0;
        sl[warp * 16 + g + 8] = st.l1;
    }
#pragma unroll
    for (int n = 0; n < HV / 8; ++n) {
        float* r0 = so + ((size_t) warp * 16 + g) * HV + 8 * n + 2 * tq;
        float* r1 = so + ((size_t) warp * 16 + g + 8) * HV + 8 * n + 2 * tq;
        r0[0] = st.o[n][0];
        r0[1] = st.o[n][1];
        r1[0] = st.o[n][2];
        r1[1] = st.o[n][3];
    }
    __syncthreads();
    for (int idx = tid; idx < 16 * HV; idx += MW * 32) {
        const int r = idx / HV, d = idx % HV;
        float M = -INFINITY;
#pragma unroll
        for (int w = 0; w < MW; ++w) M = fmaxf(M, sm[w * 16 + r]);
        float L = 0.0f, O = 0.0f;
        if (M != -INFINITY)
#pragma unroll
            for (int w = 0; w < MW; ++w) {
                const float mw = sm[w * 16 + r];
                if (mw == -INFINITY) continue;
                const float f = __expf(mw - M);
                L += sl[w * 16 + r] * f;
                O += so[((size_t) w * 16 + r) * HV + d] * f;
            }
        part_o[(base + r) * HV + d] = O;
        if (d == 0) {
            part_m[base + r] = M;
            part_l[base + r] = L;
        }
    }
}

__device__ __forceinline__ void cp_async4(void* smem, const void* gmem) {
    const unsigned s = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(s), "l"(gmem));
}

// KV8 (STRATA_MIMO_KV8): the cache holds int8 rows (k: 192 bytes, v: 128) and an FP32 scale per (position, KV head);
// the tiles come in as int8 (half the bytes - the kernel is memory-bound), are dequantized once into the FP16 tiles
// ldmatrix reads, and the rest is the FP16 kernel's
template <bool KV8>
constexpr size_t mma_smem_bytes_t() {
    return KV8 ? sizeof(__half) * (16 * (size_t) MKS + (size_t) MK * MKS + (size_t) MK * MVS) +
                     2 * ((size_t) MK * (HK + HV) + 2 * sizeof(float) * MK)
               : mma_smem_bytes();
}

template <bool KV8>
__global__ void __launch_bounds__(MW * 32) dec_partial_mma(const float* __restrict__ q, const void* __restrict__ kc,
                                                           const void* __restrict__ vc, const float* __restrict__ ks,
                                                           const float* __restrict__ vs, MimoDecAttnArgs a, int n_split,
                                                           float* __restrict__ part_o, float* __restrict__ part_m,
                                                           float* __restrict__ part_l) {
    extern __shared__ __align__(16) unsigned char smem[];
    __half* sQ = (__half*) smem;
    __half* sK = sQ + 16 * MKS;   // FP16: [2][MK][MKS]; KV8: [MK][MKS] (the dequantized tile)
    __half* sV = sK + (KV8 ? 1 : 2) * MK * MKS;
    unsigned char* s8 = (unsigned char*) (sV + (KV8 ? 1 : 2) * MK * MVS);   // KV8: [2] x {k8 [MK][HK], v8 [MK][HV],
                                                                            //        ksc [MK], vsc [MK]}
    constexpr size_t S8 = (size_t) MK * (HK + HV) + 2 * sizeof(float) * MK;
    const int kvh = blockIdx.x, split = blockIdx.y;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int lo = a.window > 0 ? max(0, a.p0 - a.window + 1) : 0;
    const int total = a.p0 - lo + 1;
    const int per = (total + n_split - 1) / n_split;
    const int k0 = lo + split * per, k1 = min(a.p0 + 1, k0 + per);
    const size_t base = ((size_t) split * a.n_kv + kvh) * 16;
    if (k0 >= k1) {   // (an empty slice: nothing to merge)
        for (int i = tid; i < 16; i += MW * 32) {
            part_m[base + i] = -INFINITY;
            part_l[base + i] = 0.0f;
        }
        for (int i = tid; i < 16 * HV; i += MW * 32) part_o[base * HV + i] = 0.0f;
        return;
    }
    for (int idx = tid; idx < 16 * HK; idx += MW * 32) {
        const int r = idx / HK, d = idx % HK;
        sQ[r * MKS + d] = __float2half_rn(q[(size_t) (kvh * 16 + r) * HK + d] * a.scale);
    }
    const int n_tiles = (k1 - k0 + MK - 1) / MK;
    const auto load_tile = [&](int it, int buf) {
        const int kt = k0 + it * MK;
        if constexpr (KV8) {
            unsigned char* b8 = s8 + buf * S8;
            constexpr int KC = HK / 16, VC = HV / 16;   // 16-byte chunks an int8 row
            for (int idx = tid; idx < MK * (KC + VC); idx += MW * 32) {
                const int c = idx / (KC + VC), x = idx % (KC + VC);
                const int j = min(kt + c, k1 - 1);
                const size_t row = (size_t) (j % a.ring) * a.n_kv + kvh;
                if (x < KC) cp_async16(b8 + c * HK + x * 16, (const int8_t*) kc + row * HK + x * 16);
                else cp_async16(b8 + MK * HK + c * HV + (x - KC) * 16, (const int8_t*) vc + row * HV + (x - KC) * 16);
            }
            float* sc8 = (float*) (b8 + MK * (HK + HV));
            for (int c = tid; c < 2 * MK; c += MW * 32) {
                const int cc = c % MK;
                const int j = min(kt + cc, k1 - 1);
                const size_t row = (size_t) (j % a.ring) * a.n_kv + kvh;
                cp_async4(sc8 + c, (c < MK ? ks : vs) + row);
            }
        } else {
            __half* dK = sK + buf * MK * MKS;
            __half* dV = sV + buf * MK * MVS;
            constexpr int KC = HK / 8, VC = HV / 8;   // 16-byte chunks an FP16 row
            for (int idx = tid; idx < MK * (KC + VC); idx += MW * 32) {
                const int c = idx / (KC + VC), x = idx % (KC + VC);
                const int j = min(kt + c, k1 - 1);   // (past the slice: a valid row, masked below - never garbage)
                const size_t row = (size_t) (j % a.ring) * a.n_kv + kvh;
                if (x < KC) cp_async16(dK + c * MKS + x * 8, (const __half*) kc + row * HK + x * 8);
                else cp_async16(dV + c * MVS + (x - KC) * 8, (const __half*) vc + row * HV + (x - KC) * 8);
            }
        }
    };
    load_tile(0, 0);
    cp_async_commit();
    __syncthreads();   // (sQ)
    unsigned qf[HK / 16][4];
#pragma unroll
    for (int s = 0; s < HK / 16; ++s)
        ldsm_x4(qf[s], sQ + ((lane & 7) + ((lane >> 3) & 1) * 8) * MKS + s * 16 + (lane >> 4) * 8);
    MmaRow st;
#pragma unroll
    for (int n = 0; n < HV / 8; ++n) st.o[n][0] = st.o[n][1] = st.o[n][2] = st.o[n][3] = 0.0f;
    st.m0 = st.m1 = -INFINITY;
    st.l0 = st.l1 = 0.0f;
    for (int it = 0; it < n_tiles; ++it) {
        const int buf = it & 1;
        if (it + 1 < n_tiles) {
            load_tile(it + 1, buf ^ 1);
            cp_async_commit();
            cp_async_wait<1>();
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();
        const __half* tK;
        const __half* tV;
        if constexpr (KV8) {
            // dequantize the int8 tile (x scale) into the FP16 tile: 4 values a step
            const unsigned char* b8 = s8 + buf * S8;
            const float* sc8 = (const float*) (b8 + MK * (HK + HV));
            for (int idx = tid; idx < MK * (HK / 4); idx += MW * 32) {
                const int c = idx / (HK / 4), d4 = idx % (HK / 4);
                const char4 v = *(const char4*) (b8 + c * HK + d4 * 4);
                const float f = sc8[c];
                __half2* dst = (__half2*) (sK + c * MKS + d4 * 4);
                dst[0] = __floats2half2_rn((float) v.x * f, (float) v.y * f);
                dst[1] = __floats2half2_rn((float) v.z * f, (float) v.w * f);
            }
            for (int idx = tid; idx < MK * (HV / 4); idx += MW * 32) {
                const int c = idx / (HV / 4), d4 = idx % (HV / 4);
                const char4 v = *(const char4*) (b8 + MK * HK + c * HV + d4 * 4);
                const float f = sc8[MK + c];
                __half2* dst = (__half2*) (sV + c * MVS + d4 * 4);
                dst[0] = __floats2half2_rn((float) v.x * f, (float) v.y * f);
                dst[1] = __floats2half2_rn((float) v.z * f, (float) v.w * f);
            }
            __syncthreads();
            tK = sK + warp * 16 * MKS;
            tV = sV + warp * 16 * MVS;
        } else {
            tK = sK + buf * MK * MKS + warp * 16 * MKS;
            tV = sV + buf * MK * MVS + warp * 16 * MVS;
        }
        mma_tile(st, qf, tK, tV, k0 + it * MK + warp * 16, k1, lane);
        __syncthreads();   // (the buffers are the next tiles')
    }
    mma_merge(st, (unsigned char*) sK, base, part_o, part_m, part_l, tid, warp, lane);
}

// the slices of one (window row, query head) merged with the sink logit (a zero-valued entry), times v_scale
__global__ void __launch_bounds__(HV) dec_combine(const float* __restrict__ part_o, const float* __restrict__ part_m,
                                                 const float* __restrict__ part_l, const float* __restrict__ sinks,
                                                 float* __restrict__ out, int out_ld, MimoDecAttnArgs a, int n_split) {
    const int h = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    const int G = a.n_head / a.n_kv, R = G * a.T, kvh = h / G;
    const int gr = t * G + h % G;
    float M = sinks != nullptr ? sinks[h] : -INFINITY;
    for (int s = 0; s < n_split; ++s) M = fmaxf(M, part_m[((size_t) s * a.n_kv + kvh) * R + gr]);
    float L = sinks != nullptr ? __expf(sinks[h] - M) : 0.0f, o = 0.0f;
    for (int s = 0; s < n_split; ++s) {
        const size_t r = ((size_t) s * a.n_kv + kvh) * R + gr;
        const float m = part_m[r];
        if (m == -INFINITY) continue;
        const float w = __expf(m - M);
        L += part_l[r] * w;
        o += part_o[r * HV + d] * w;
    }
    out[(size_t) t * out_ld + (size_t) h * HV + d] = L > 0.0f ? o * a.v_scale / L : 0.0f;
}
}  // namespace

int mimo_decode_attn_splits(const MimoDecAttnArgs& a) {
    const int keys = a.window > 0 ? std::min(a.window, a.p0 + a.T) : a.p0 + a.T;
    return std::max(1, std::min(kMaxSplit, (keys + 383) / 384));
}

size_t mimo_decode_attn_scratch(const MimoDecAttnArgs& a, int n_split) {
    const size_t rows = (size_t) n_split * a.n_kv * (size_t) (a.n_head / std::max(1, a.n_kv)) * a.T;
    return rows * (HV + 2);
}

template<bool KV8>
bool decode_impl(const float* q, int q_ld, const void* kc, const void* vc, const float* ks, const float* vs, const float* sinks, float* out,
                      int out_ld, const MimoDecAttnArgs& a, float* scratch, size_t scratch_floats, cudaStream_t stream) {
    if (a.head_k != HK || a.head_v != HV || a.n_kv <= 0 || a.n_head % a.n_kv != 0 || a.T <= 0) return false;
    const int G = a.n_head / a.n_kv, R = G * a.T;
    if (R > 8 * RB) return false;
    const int n_split = mimo_decode_attn_splits(a);
    const size_t need = mimo_decode_attn_scratch(a, n_split);
    if (scratch == nullptr || scratch_floats < need) return false;
    int dev = 0;
    cudaGetDevice(&dev);
    static bool attr_set[64] = {};
    if (dev >= 0 && dev < 64 && !attr_set[dev]) {
        if (cudaFuncSetAttribute(dec_partial<KV8>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem_bytes()) !=
            cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        attr_set[dev] = true;
    }
    const size_t rows = (size_t) n_split * a.n_kv * R;
    float* part_o = scratch;
    float* part_m = part_o + rows * HV;
    float* part_l = part_m + rows;
    // one window row and 16 heads a KV head: the tensor-core kernel (STRATA_MIMO_DEC_MMA=0: the scalar one, for A/B);
    // (T = 1: the one q row, q_ld unused)
    static const bool mma_env = [] {
        const char* e = getenv("STRATA_MIMO_DEC_MMA");
        return e == nullptr || std::atoi(e) != 0;
    }();
    if (mma_env && a.T == 1 && G == 16) {
        static bool mma_set[64] = {};
        if (dev >= 0 && dev < 64 && !mma_set[dev]) {
            if (cudaFuncSetAttribute(dec_partial_mma<KV8>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     (int) mma_smem_bytes_t<KV8>()) != cudaSuccess) {
                cudaGetLastError();
                return false;
            }
            mma_set[dev] = true;
        }
        dec_partial_mma<KV8><<<dim3(a.n_kv, n_split), MW * 32, mma_smem_bytes_t<KV8>(), stream>>>(
            q, kc, vc, ks, vs, a, n_split, part_o, part_m, part_l);
        dec_combine<<<dim3(a.n_head, a.T), HV, 0, stream>>>(part_o, part_m, part_l, sinks, out, out_ld, a, n_split);
        return true;
    }
    dec_partial<KV8><<<dim3(a.n_kv, n_split, (R + RB - 1) / RB), NT, smem_bytes(), stream>>>(
        q, q_ld, (const __half2*) kc, (const __half2*) vc, ks, vs, a, n_split, part_o, part_m, part_l);
    dec_combine<<<dim3(a.n_head, a.T), HV, 0, stream>>>(part_o, part_m, part_l, sinks, out, out_ld, a, n_split);
    return true;
}
bool mimo_decode_attn(const float* q, int q_ld, const uint16_t* kc, const uint16_t* vc, const float* sinks, float* out,
                      int out_ld, const MimoDecAttnArgs& a, float* scratch, size_t sf, cudaStream_t stream) {
    return decode_impl<false>(q, q_ld, kc, vc, nullptr, nullptr, sinks, out, out_ld, a, scratch, sf, stream);
}
bool mimo_decode_attn_kv8(const float* q, int q_ld, const int8_t* kc, const int8_t* vc,
                          const float* ks, const float* vs, const float* sinks, float* out,
                          int out_ld, const MimoDecAttnArgs& a, float* scratch, size_t sf, cudaStream_t stream) {
    if (!ks || !vs) return false;
    return decode_impl<true>(q, q_ld, kc, vc, ks, vs, sinks, out, out_ld, a, scratch, sf, stream);
}
}  // namespace strata::kernels
