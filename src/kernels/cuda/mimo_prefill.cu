#include "strata/kernels/mimo_prefill.hpp"
#include <cuda_fp16.h>
#include "mimo_kv8.cuh"
#include <cmath>
#include <cstdlib>
#if !defined(STRATA_USE_HIP)
#include <mma.h>
#endif

namespace strata::kernels {
namespace {
__global__ void rope_rows(float* qkv, int p0, int T, MimoPrefillArgs a) {
    const int half = a.rope_dim / 2, nh = a.heads + a.kv_heads;
    const int ld = nh * a.head_k + a.kv_heads * a.head_v;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * nh * half) return;
    const int j = i % half, h = (i / half) % nh, t = i / (nh * half);
    float* q = qkv + (size_t) t * ld + h * a.head_k + j;
    const float angle = (float) (p0 + t) * powf(a.rope_base, -2.0f * j / a.rope_dim);
    float sn, cs; sincosf(angle, &sn, &cs);
    const float x = q[0], y = q[half];
    q[0] = x * cs - y * sn; q[half] = y * cs + x * sn;
}
__global__ void store_rows(const float* qkv, __half* kc, __half* vc, int p0, int T, MimoPrefillArgs a) {
    const int rk = a.kv_heads * a.head_k, rv = a.kv_heads * a.head_v;
    const int nq = a.heads * a.head_k, ld = nq + rk + rv;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * (rk + rv)) return;
    const int t = i / (rk + rv), d = i % (rk + rv), row = (p0 + t) % a.ring;
    const float v = qkv[(size_t) t * ld + nq + d];
    if (d < rk) kc[(size_t) row * rk + d] = __float2half(v);
    else vc[(size_t) row * rv + d - rk] = __float2half(v);
}
__device__ float warp_sum(float x) {
    for (int d = 16; d; d >>= 1) x += __shfl_xor_sync(0xffffffff, x, d);
    return x;
}
// Reference/fallback: one warp per query head, bounded registers for Dk <=256, Dv <=256.
template<bool KV8>
__global__ void attn_scalar(const float* qkv, const __half* kc, const __half* vc,
                             const float* ks, const float* vs, const float* sinks, float* out, int p0, MimoPrefillArgs a) {
    const int h = blockIdx.x, t = blockIdx.y, lane = threadIdx.x, kh = h / (a.heads / a.kv_heads);
    const int ld = (a.heads + a.kv_heads) * a.head_k + a.kv_heads * a.head_v, p = p0 + t;
    const int lo = a.window ? max(0, p - a.window + 1) : 0;
    float qr[8] = {}, o[8] = {};
    for (int d = lane; d < a.head_k; d += 32) qr[d / 32] = qkv[(size_t) t * ld + h * a.head_k + d];
    float m = sinks ? sinks[h] : -INFINITY, l = sinks ? 1.0f : 0.0f;
    for (int j = lo; j <= p; ++j) {
        const size_t r = (size_t) (j % a.ring) * a.kv_heads + kh;
        float dot = 0.0f;
        for (int d = lane; d < a.head_k; d += 32) dot += qr[d / 32] * mimo_cache_load<KV8>(kc, ks, r, a.head_k, d);
        dot = warp_sum(dot) * rsqrtf((float) a.head_k);
        const float mn = fmaxf(m, dot), c = expf(m - mn), w = expf(dot - mn);
        l = l * c + w; m = mn;
        for (int d = lane; d < a.head_v; d += 32) o[d / 32] = o[d / 32] * c + w * mimo_cache_load<KV8>(vc, vs, r, a.head_v, d);
    }
    for (int d = lane; d < a.head_v; d += 32) out[((size_t) t * a.heads + h) * a.head_v + d] = o[d / 32] * (a.v_scale / l);
}
#if !defined(STRATA_USE_HIP)
// Each warp owns 16 queries and keeps the FP32 output in registers. Four
// warps reuse each K/V tile, with padded shared strides. This reduces K/V
// traffic and avoids spilling the running output through shared memory.
__device__ __forceinline__ void mimo_mma(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ uint32_t mimo_h2(float x, float y) {
    const __half2 h = __floats2half2_rn(x, y);
    return *reinterpret_cast<const uint32_t*>(&h);
}
template<int N, bool KV8>
__global__ void __launch_bounds__(128) attn_mma_reuse(const float* __restrict__ qkv,
    const __half* __restrict__ kc, const __half* __restrict__ vc,
    const float* ks, const float* vs, const float* sinks, float* __restrict__ out, int p0, int T, MimoPrefillArgs a) {
    constexpr int Q=64, DK=192, DV=128, KS=200, NS=N+8;
    extern __shared__ __align__(16) __half shared[];
    auto* sk=reinterpret_cast<__half (*)[KS]>(shared);
    auto* vt=reinterpret_cast<__half (*)[NS]>(shared+N*KS);
    auto* sp=reinterpret_cast<__half (*)[NS]>(shared+N*KS+DV*NS);
    const int tid=threadIdx.x, warp=tid/32, lane=tid%32, gid=lane/4, tig=lane%4;
    const int h=blockIdx.x, t0=blockIdx.y*Q, qr=t0+warp*16, kh=h/(a.heads/a.kv_heads);
    const int ld=(a.heads+a.kv_heads)*DK+a.kv_heads*DV;
    const int end=p0+min(T,t0+Q), begin=a.window?max(0,p0+t0-a.window+1):0;
    uint32_t q[12][4];
    #pragma unroll
    for(int d=0;d<12;++d) {
        const int x=d*16+2*tig;
        const float* u=qkv+(size_t)(qr+gid)*ld+h*DK+x;
        const float* v=qkv+(size_t)(qr+gid+8)*ld+h*DK+x;
        q[d][0]=qr+gid<T ? mimo_h2(u[0],u[1]):0;
        q[d][1]=qr+gid+8<T ? mimo_h2(v[0],v[1]):0;
        q[d][2]=qr+gid<T ? mimo_h2(u[8],u[9]):0;
        q[d][3]=qr+gid+8<T ? mimo_h2(v[8],v[9]):0;
    }
    float acc[16][4]={};
    float m0=sinks?sinks[h]:-INFINITY, m1=m0, l0=sinks?1.0f:0.0f, l1=l0;
    for(int j0=begin;j0<end;j0+=N) {
        for(int i=tid;i<N*DK/8;i+=128) {
            const int r=i/(DK/8), d=(i%(DK/8))*8, j=j0+r;
            uint4 v=make_uint4(0,0,0,0);
            if constexpr (KV8) {
                if(j<end) {
                    __half vals[8];
                    #pragma unroll
                    for(int z=0;z<8;++z) vals[z]=__float2half(mimo_cache_load<true>(kc,ks,(size_t)(j%a.ring)*a.kv_heads+kh,DK,d+z));
                    v=*reinterpret_cast<const uint4*>(vals);
                }
            } else if(j<end) v=*reinterpret_cast<const uint4*>(kc+((size_t)(j%a.ring)*a.kv_heads+kh)*DK+d);
            *reinterpret_cast<uint4*>(&sk[r][d])=v;
        }
        for(int i=tid;i<N*DV;i+=128) {
            const int r=i/DV,d=i%DV,j=j0+r;
            if constexpr (KV8) vt[d][r]=j<end?__float2half(mimo_cache_load<true>(vc,vs,(size_t)(j%a.ring)*a.kv_heads+kh,DV,d)):__float2half(0.0f);
            else vt[d][r]=j<end?vc[((size_t)(j%a.ring)*a.kv_heads+kh)*DV+d]:__float2half(0.0f);
        }
        __syncthreads();
        float scores[N/8][4]={};
        #pragma unroll
        for(int d=0;d<12;++d) {
            #pragma unroll
            for(int n=0;n<N/8;++n) {
                uint32_t b[2];
                b[0]=*reinterpret_cast<const uint32_t*>(&sk[n*8+gid][d*16+2*tig]);
                b[1]=*reinterpret_cast<const uint32_t*>(&sk[n*8+gid][d*16+2*tig+8]);
                mimo_mma(scores[n],q[d],b);
            }
        }
        float mx0=m0,mx1=m1;
        #pragma unroll
        for(int n=0;n<N/8;++n) {
            #pragma unroll
            for(int z=0;z<4;++z) {
                const int p=p0+qr+gid+(z/2)*8, j=j0+n*8+2*tig+z%2;
                const bool valid=qr+gid+(z/2)*8<T && j<=p && j<end && (!a.window || j>p-a.window);
                scores[n][z]=valid?scores[n][z]*0.07216878364870322f:-INFINITY;
                if(z<2) mx0=fmaxf(mx0,scores[n][z]); else mx1=fmaxf(mx1,scores[n][z]);
            }
        }
        #pragma unroll
        for(int d=1;d<4;d*=2) {
            mx0=fmaxf(mx0,__shfl_xor_sync(0xffffffff,mx0,d));
            mx1=fmaxf(mx1,__shfl_xor_sync(0xffffffff,mx1,d));
        }
        const float c0=m0==-INFINITY?0.0f:__expf(m0-mx0), c1=m1==-INFINITY?0.0f:__expf(m1-mx1);
        float sum0=0,sum1=0;
        #pragma unroll
        for(int n=0;n<N/8;++n) {
            float e[4];
            #pragma unroll
            for(int z=0;z<4;++z) e[z]=scores[n][z]==-INFINITY?0.0f:__expf(scores[n][z]-(z<2?mx0:mx1));
            sum0+=e[0]+e[1];sum1+=e[2]+e[3];
            *reinterpret_cast<uint32_t*>(&sp[warp*16+gid][n*8+2*tig])=mimo_h2(e[0],e[1]);
            *reinterpret_cast<uint32_t*>(&sp[warp*16+gid+8][n*8+2*tig])=mimo_h2(e[2],e[3]);
        }
        #pragma unroll
        for(int d=1;d<4;d*=2) {
            sum0+=__shfl_xor_sync(0xffffffff,sum0,d);sum1+=__shfl_xor_sync(0xffffffff,sum1,d);
        }
        l0=l0*c0+sum0;l1=l1*c1+sum1;m0=mx0;m1=mx1;
        #pragma unroll
        for(int d=0;d<16;++d) {acc[d][0]*=c0;acc[d][1]*=c0;acc[d][2]*=c1;acc[d][3]*=c1;}
        __syncwarp();
        #pragma unroll
        for(int k=0;k<N;k+=16) {
            uint32_t p[4];
            p[0]=*reinterpret_cast<const uint32_t*>(&sp[warp*16+gid][k+2*tig]);
            p[1]=*reinterpret_cast<const uint32_t*>(&sp[warp*16+gid+8][k+2*tig]);
            p[2]=*reinterpret_cast<const uint32_t*>(&sp[warp*16+gid][k+2*tig+8]);
            p[3]=*reinterpret_cast<const uint32_t*>(&sp[warp*16+gid+8][k+2*tig+8]);
            #pragma unroll
            for(int d=0;d<16;++d) {
                uint32_t b[2];
                b[0]=*reinterpret_cast<const uint32_t*>(&vt[d*8+gid][k+2*tig]);
                b[1]=*reinterpret_cast<const uint32_t*>(&vt[d*8+gid][k+2*tig+8]);
                mimo_mma(acc[d],p,b);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int d=0;d<16;++d) {
        const int x=d*8+2*tig;
        if(qr+gid<T) *reinterpret_cast<float2*>(out+((size_t)(qr+gid)*a.heads+h)*DV+x)=make_float2(acc[d][0]*(a.v_scale/l0),acc[d][1]*(a.v_scale/l0));
        if(qr+gid+8<T) *reinterpret_cast<float2*>(out+((size_t)(qr+gid+8)*a.heads+h)*DV+x)=make_float2(acc[d][2]*(a.v_scale/l1),acc[d][3]*(a.v_scale/l1));
    }
}

// 16 causal queries x 32 keys, GQA 192/128. FP16 products, FP32 scores,
// normalization and accumulators. No quadratic score allocation; ~36 KiB/block.
template<bool KV8>
__global__ void __launch_bounds__(128) attn_mma(const float* qkv, const __half* kc, const __half* vc,
                                               const float* ks, const float* vs, const float* sinks, float* out, int p0, int T, MimoPrefillArgs a) {
    using namespace nvcuda;
    constexpr int Q = 16, N = 32, DK = 192, DV = 128;
    __shared__ __align__(32) __half sq[Q * DK], sk[N * DK], sv[N * DV], sp[Q * N];
    __shared__ __align__(32) float score[Q * N], so[Q * DV];
    __shared__ float sm[Q], sl[Q], sc[Q];
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int h = blockIdx.x, t0 = blockIdx.y * Q, kh = h / (a.heads / a.kv_heads);
    const int ld = (a.heads + a.kv_heads) * DK + a.kv_heads * DV;
    const int end = p0 + min(T, t0 + Q), begin = a.window ? max(0, p0 + t0 - a.window + 1) : 0;
    for (int i = tid; i < Q * DK; i += 128) {
        const int t = t0 + i / DK;
        sq[i] = __float2half(t < T ? qkv[(size_t) t * ld + h * DK + i % DK] : 0.0f);
    }
    for (int i = tid; i < Q * DV; i += 128) so[i] = 0.0f;
    if (tid < Q) { sm[tid] = sinks ? sinks[h] : -INFINITY; sl[tid] = sinks ? 1.0f : 0.0f; }
    __syncthreads();
    for (int j0 = begin; j0 < end; j0 += N) {
        for (int i = tid; i < N * DK; i += 128) {
            const int j = j0 + i / DK;
            if constexpr (KV8) sk[i] = j < end ? __float2half(mimo_cache_load<true>(kc,ks,(size_t)(j%a.ring)*a.kv_heads+kh,DK,i%DK)) : __float2half(0.0f);
            else sk[i] = j < end ? kc[((size_t) (j % a.ring) * a.kv_heads + kh) * DK + i % DK] : __float2half(0.0f);
        }
        for (int i = tid; i < N * DV; i += 128) {
            const int j = j0 + i / DV;
            if constexpr (KV8) sv[i] = j < end ? __float2half(mimo_cache_load<true>(vc,vs,(size_t)(j%a.ring)*a.kv_heads+kh,DV,i%DV)) : __float2half(0.0f);
            else sv[i] = j < end ? vc[((size_t) (j % a.ring) * a.kv_heads + kh) * DV + i % DV] : __float2half(0.0f);
        }
        __syncthreads();
        if (warp < 2) {
            wmma::fragment<wmma::accumulator,16,16,16,float> acc;
            wmma::fill_fragment(acc, 0.0f);
            for (int d = 0; d < DK; d += 16) {
                wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> q;
                wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> k;
                wmma::load_matrix_sync(q, sq + d, DK);
                wmma::load_matrix_sync(k, sk + warp * 16 * DK + d, DK);
                wmma::mma_sync(acc, q, k, acc);
            }
            wmma::store_matrix_sync(score + warp * 16, acc, N, wmma::mem_row_major);
        }
        __syncthreads();
        for (int r = warp; r < Q; r += 4) {
            const int p = p0 + t0 + r, j = j0 + lane;
            const bool valid = t0 + r < T && j <= p && j < end && (!a.window || j > p - a.window);
            float x = valid ? score[r * N + lane] * rsqrtf((float) DK) : -INFINITY;
            float mx = x;
            for (int d = 16; d; d >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, d));
            mx = fmaxf(mx, sm[r]);
            // Tail queries may be wholly masked. Keep zero probabilities, no NaN.
            const float e = valid ? expf(x - mx) : 0.0f;
            const float corr = sm[r] == -INFINITY ? 0.0f : expf(sm[r] - mx);
            sp[r * N + lane] = __float2half(e);
            const float sum = warp_sum(e);
            if (!lane) { sl[r] = sl[r] * corr + sum; sm[r] = mx; sc[r] = corr; }
        }
        __syncthreads();
        for (int i = tid; i < Q * DV; i += 128) so[i] *= sc[i / DV];
        __syncthreads();
        for (int tile = warp; tile < DV / 16; tile += 4) {
            wmma::fragment<wmma::accumulator,16,16,16,float> acc;
            wmma::load_matrix_sync(acc, so + tile * 16, DV, wmma::mem_row_major);
            for (int n = 0; n < N; n += 16) {
                wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> p;
                wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::row_major> v;
                wmma::load_matrix_sync(p, sp + n, N);
                wmma::load_matrix_sync(v, sv + n * DV + tile * 16, DV);
                wmma::mma_sync(acc, p, v, acc);
            }
            wmma::store_matrix_sync(so + tile * 16, acc, DV, wmma::mem_row_major);
        }
        __syncthreads();
    }
    for (int i = tid; i < Q * DV; i += 128) {
        const int t = t0 + i / DV;
        if (t < T) out[((size_t) t * a.heads + h) * DV + i % DV] = so[i] * (a.v_scale / sl[i / DV]);
    }
}
#endif
// One warp per (token, KV head), all reductions confined to the warp.
__global__ void store_kv8(const float* k, int kld, const float* v, int vld, int8_t* kc, int8_t* vc,
                          float* ks, float* vs, int p0, int T, int ring, int nh, int dk, int dv) {
    const int lane=threadIdx.x%32, i=(blockIdx.x*blockDim.x+threadIdx.x)/32;
    if(i>=T*nh) return;
    const int t=i/nh, h=i%nh;
    const float* kp=k+(size_t)t*kld+h*dk;
    const float* vp=v+(size_t)t*vld+h*dv;
    float km=0,vm=0;
    for(int d=lane;d<dk;d+=32) km=fmaxf(km,fabsf(kp[d]));
    for(int d=lane;d<dv;d+=32) vm=fmaxf(vm,fabsf(vp[d]));
    for(int o=16;o;o>>=1) {
        km=fmaxf(km,__shfl_xor_sync(0xffffffff,km,o));
        vm=fmaxf(vm,__shfl_xor_sync(0xffffffff,vm,o));
    }
    const float sk=km>0?km/127.0f:1.0f, sv=vm>0?vm/127.0f:1.0f;
    const size_t row=(size_t)((p0+t)%ring)*nh+h;
    if(!lane) {ks[row]=sk;vs[row]=sv;}
    for(int d=lane;d<dk;d+=32) kc[row*dk+d]=(int8_t)max(-127,min(127,__float2int_rn(kp[d]/sk)));
    for(int d=lane;d<dv;d+=32) vc[row*dv+d]=(int8_t)max(-127,min(127,__float2int_rn(vp[d]/sv)));
}
}
void mimo_kv8_store(const float* k, int kld, const float* v, int vld, int8_t* kc, int8_t* vc,
                    float* ks, float* vs, int p0, int T, int ring, int nh, int dk, int dv, cudaStream_t s) {
    store_kv8<<<(T*nh+3)/4,128,0,s>>>(k,kld,v,vld,kc,vc,ks,vs,p0,T,ring,nh,dk,dv);
}
void mimo_prefill_rope_kv8(float* qkv, int8_t* kc, int8_t* vc, float* ks, float* vs,
                          int p0, int T, const MimoPrefillArgs& a, cudaStream_t s) {
    rope_rows<<<(T*(a.heads+a.kv_heads)*(a.rope_dim/2)+255)/256,256,0,s>>>(qkv,p0,T,a);
    const int nq=a.heads*a.head_k, ld=nq+a.kv_heads*(a.head_k+a.head_v);
    mimo_kv8_store(qkv+nq,ld,qkv+nq+a.kv_heads*a.head_k,ld,kc,vc,ks,vs,p0,T,a.ring,a.kv_heads,a.head_k,a.head_v,s);
}
void mimo_prefill_attention_kv8(const float* qkv, const int8_t* kc, const int8_t* vc,
                               const float* ks, const float* vs, const float* sinks,
                               float* out, int p0, int T, const MimoPrefillArgs& a, cudaStream_t s, bool scalar) {
#if !defined(STRATA_USE_HIP)
    if (!scalar && a.head_k==192 && a.head_v==128) {
        if (!a.window && getenv("STRATA_MIMO_PREFILL_LEGACY")==nullptr) {
            attn_mma_reuse<32,true><<<dim3(a.heads,(T+63)/64),128,28160,s>>>(qkv,(const __half*)kc,(const __half*)vc,ks,vs,sinks,out,p0,T,a);
            return;
        }
        attn_mma<true><<<dim3(a.heads,(T+15)/16),128,0,s>>>(qkv,(const __half*)kc,(const __half*)vc,ks,vs,sinks,out,p0,T,a);
        return;
    }
#endif
    attn_scalar<true><<<dim3(a.heads,T),32,0,s>>>(qkv,(const __half*)kc,(const __half*)vc,ks,vs,sinks,out,p0,a);
}
void mimo_prefill_rope_kv(float* qkv, uint16_t* kc, uint16_t* vc, int p0, int T,
                          const MimoPrefillArgs& a, cudaStream_t s) {
    rope_rows<<<(T * (a.heads + a.kv_heads) * (a.rope_dim / 2) + 255) / 256,256,0,s>>>(qkv,p0,T,a);
    store_rows<<<(T * a.kv_heads * (a.head_k + a.head_v) + 255) / 256,256,0,s>>>(qkv,(__half*)kc,(__half*)vc,p0,T,a);
}
void mimo_prefill_attention(const float* qkv, const uint16_t* kc, const uint16_t* vc,
                            const float* sinks, float* out, int p0, int T,
                            const MimoPrefillArgs& a, cudaStream_t s, bool scalar) {
#if !defined(STRATA_USE_HIP)
    if (!scalar && a.head_k == 192 && a.head_v == 128) {
        if (!a.window && getenv("STRATA_MIMO_PREFILL_LEGACY") == nullptr) {
            attn_mma_reuse<32,false><<<dim3(a.heads,(T+63)/64),128,28160,s>>>(qkv,(const __half*)kc,(const __half*)vc,nullptr,nullptr,sinks,out,p0,T,a);
            return;
        }
        attn_mma<false><<<dim3(a.heads,(T+15)/16),128,0,s>>>(qkv,(const __half*)kc,(const __half*)vc,nullptr,nullptr,sinks,out,p0,T,a);
        return;
    }
#endif
    attn_scalar<false><<<dim3(a.heads,T),32,0,s>>>(qkv,(const __half*)kc,(const __half*)vc,nullptr,nullptr,sinks,out,p0,a);
}
}
