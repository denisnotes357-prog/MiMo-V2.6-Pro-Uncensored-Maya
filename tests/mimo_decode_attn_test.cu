// tests/mimo_decode_attn_test.cu - the split-K GQA decode attention (strata/kernels/mimo_decode_attn.hpp) against an
// independent FP64 CPU reference on random data: full (window 0) and windowed layers, a sink, window rows T = 1..8,
// short and long caches, a ring that wraps.  Also times it against a per-(row, head) reference kernel shape.
#include "strata/kernels/mimo_decode_attn.hpp"
#include "strata/kernels/mimo_prefill.hpp"
#include "strata/core/mimo_kv8.hpp"
#include <algorithm>
#include <cstring>
#include <cstdlib>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace strata::kernels;


// the per-(row, head) kernel the token path used (glm_fast_path.cu mimo_attn_flash), for the timing comparison only
constexpr int kFW = 8, kFV = 8;
__global__ void __launch_bounds__(kFW * 32)
old_flash(const float* __restrict__ q, int q_ld, const __half* __restrict__ kc, const __half* __restrict__ vc, int ring,
          int n_head, int n_kv, int head_k, int head_v, int window, int p0, const float* __restrict__ sinks, float scale,
          float v_scale, float* __restrict__ out, int out_ld) {
    const int h = blockIdx.x, row = blockIdx.y, p = p0 + row;
    const int lo = window > 0 ? max(0, p - window + 1) : 0, kvh = h / (n_head / n_kv);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const float* qh = q + (size_t) row * q_ld + (size_t) h * head_k;
    const int nk = head_k / 32, nv = head_v / 32;
    float qr[8];
    for (int i = 0; i < nk; ++i) qr[i] = qh[lane + 32 * i] * scale;
    float m = -INFINITY, l = 0.0f, acc[kFV];
    for (int i = 0; i < kFV; ++i) acc[i] = 0.0f;
    for (int j = lo + warp; j <= p; j += kFW) {
        const size_t r = (size_t) (j % ring) * n_kv + kvh;
        const __half* kj = kc + r * head_k;
        float sdot = 0.0f;
        for (int i = 0; i < nk; ++i) sdot += qr[i] * __half2float(kj[lane + 32 * i]);
        for (int o = 16; o > 0; o >>= 1) sdot += __shfl_xor_sync(0xffffffffu, sdot, o);
        const float mn = fmaxf(m, sdot), corr = expf(m - mn), e = expf(sdot - mn);
        l = l * corr + e;
        const __half* vj = vc + r * head_v;
        for (int i = 0; i < nv; ++i) acc[i] = acc[i] * corr + e * __half2float(vj[lane + 32 * i]);
        m = mn;
    }
    __shared__ float sm[kFW], sl[kFW], sacc[kFW][32 * kFV];
    if (lane == 0) { sm[warp] = m; sl[warp] = l; }
    for (int i = 0; i < nv; ++i) sacc[warp][lane + 32 * i] = acc[i];
    __syncthreads();
    float M = sinks != nullptr ? sinks[h] : -INFINITY;
    for (int w = 0; w < kFW; ++w) M = fmaxf(M, sm[w]);
    float L = sinks != nullptr ? expf(sinks[h] - M) : 0.0f;
    for (int w = 0; w < kFW; ++w) L += sm[w] == -INFINITY ? 0.0f : sl[w] * expf(sm[w] - M);
    for (int d = threadIdx.x; d < head_v; d += blockDim.x) {
        float o = 0.0f;
        for (int w = 0; w < kFW; ++w) if (sm[w] != -INFINITY) o += sacc[w][d] * expf(sm[w] - M);
        out[(size_t) row * out_ld + (size_t) h * head_v + d] = o * v_scale / L;
    }
}

static float h2f(uint16_t h) { __half x; *(uint16_t*) &x = h; return __half2float(x); }
static uint16_t f2h(float f) { __half x = __float2half(f); return *(uint16_t*) &x; }

static bool run(int n_head, int n_kv, int ring, int window, int p0, int T, bool sink, bool timing) {
    const int HK = 192, HV = 128;
    std::mt19937 rng(1234 + p0 + T * 7 + window);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const int q_ld = n_head * HK + 77, out_ld = n_head * HV + 5;
    std::vector<float> q((size_t) T * q_ld), sinks(n_head);
    for (auto& x : q) x = nd(rng);
    for (auto& x : sinks) x = nd(rng) * 2.0f;
    std::vector<uint16_t> kc((size_t) ring * n_kv * HK), vc((size_t) ring * n_kv * HV);
    for (auto& x : kc) x = f2h(nd(rng) * 0.5f);
    for (auto& x : vc) x = f2h(nd(rng));
    MimoDecAttnArgs a{n_head, n_kv, HK, HV, ring, window, p0, T, 1.0f / std::sqrt((float) HK), 0.612f};
    const int ns = mimo_decode_attn_splits(a);
    const size_t sf = mimo_decode_attn_scratch(a, ns);
    float *dq, *dout, *dscr, *dsink;
    uint16_t *dk, *dv;
    cudaMalloc(&dq, q.size() * 4);
    cudaMalloc(&dout, (size_t) T * out_ld * 4);
    cudaMalloc(&dscr, sf * 4);
    cudaMalloc(&dsink, n_head * 4);
    cudaMalloc(&dk, kc.size() * 2);
    cudaMalloc(&dv, vc.size() * 2);
    cudaMemcpy(dq, q.data(), q.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dsink, sinks.data(), n_head * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dk, kc.data(), kc.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dv, vc.data(), vc.size() * 2, cudaMemcpyHostToDevice);
    cudaMemset(dout, 0, (size_t) T * out_ld * 4);
    if (!mimo_decode_attn(dq, q_ld, dk, dv, sink ? dsink : nullptr, dout, out_ld, a, dscr, sf, 0)) {
        std::printf("FAIL: not launched\n");
        return false;
    }
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        std::printf("FAIL: %s\n", cudaGetErrorString(e));
        return false;
    }
    std::vector<float> out((size_t) T * out_ld);
    cudaMemcpy(out.data(), dout, out.size() * 4, cudaMemcpyDeviceToHost);
    double max_abs = 0.0, ref_norm = 0.0, err_norm = 0.0;
    const int G = n_head / n_kv;
    // (timing past 65536 keys: no FP64 reference - it takes minutes on the CPU)
    const int T_ref = timing && p0 > 65536 ? 0 : T;
    for (int t = 0; t < T_ref; ++t)
        for (int h = 0; h < n_head; ++h) {
            const int kvh = h / G, pos = p0 + t, lo = window > 0 ? std::max(0, pos - window + 1) : 0;
            std::vector<double> sc;
            double M = sink ? (double) sinks[h] : -INFINITY;
            for (int j = lo; j <= pos; ++j) {
                double dot = 0.0;
                for (int d = 0; d < HK; ++d)
                    dot += (double) q[(size_t) t * q_ld + (size_t) h * HK + d] *
                           h2f(kc[((size_t) (j % ring) * n_kv + kvh) * HK + d]);
                dot /= std::sqrt((double) HK);
                sc.push_back(dot);
                M = std::max(M, dot);
            }
            double L = sink ? std::exp((double) sinks[h] - M) : 0.0;
            for (double s : sc) L += std::exp(s - M);
            for (int d = 0; d < HV; ++d) {
                double o = 0.0;
                for (size_t c = 0; c < sc.size(); ++c)
                    o += std::exp(sc[c] - M) * h2f(vc[((size_t) ((lo + (int) c) % ring) * n_kv + kvh) * HV + d]);
                o = o / L * 0.612;
                const double got = out[(size_t) t * out_ld + (size_t) h * HV + d];
                max_abs = std::max(max_abs, std::fabs(got - o));
                ref_norm += o * o;
                err_norm += (got - o) * (got - o);
            }
        }
    const double rel = std::sqrt(err_norm / std::max(ref_norm, 1e-30));
    double ms = 0.0;
    if (timing) {
        for (int w = 0; w < 3; ++w) mimo_decode_attn(dq, q_ld, dk, dv, nullptr, dout, out_ld, a, dscr, sf, 0);
        cudaDeviceSynchronize();
        const auto t0 = std::chrono::steady_clock::now();
        for (int w = 0; w < 20; ++w) mimo_decode_attn(dq, q_ld, dk, dv, nullptr, dout, out_ld, a, dscr, sf, 0);
        cudaDeviceSynchronize();
        ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 20.0;
        const auto t1 = std::chrono::steady_clock::now();
        for (int w = 0; w < 5; ++w)
            old_flash<<<dim3(n_head, T), kFW * 32>>>(dq, q_ld, (const __half*) dk, (const __half*) dv, ring, n_head, n_kv,
                                                    HK, HV, window, p0, nullptr, a.scale, a.v_scale, dout, out_ld);
        cudaDeviceSynchronize();
        const double old_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count() / 5.0;
        std::printf("     timing ctx %d T %d: new %.3f ms, old per-head kernel %.3f ms (x%.1f)\n", p0, T, ms, old_ms,
                    old_ms / ms);
    }
    const bool ok = T_ref == 0 || (rel < 2e-3 && max_abs < 5e-3);
    std::printf("%s heads %d/%d ring %d window %d p0 %d T %d sink %d splits %d: max_abs %.2e rel_l2 %.2e%s\n",
                ok ? "ok  " : "FAIL", n_head, n_kv, ring, window, p0, T, sink ? 1 : 0, ns, max_abs, rel,
                timing ? (" | " + std::to_string(ms) + " ms").c_str() : "");
    cudaFree(dq); cudaFree(dout); cudaFree(dscr); cudaFree(dsink); cudaFree(dk); cudaFree(dv);
    return ok;
}


static void ck(cudaError_t e) {
    if(e!=cudaSuccess) { std::fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e)); std::exit(1); }
}
static std::vector<double> oracle(const std::vector<float>& q, int qld, const std::vector<float>& k,
                                  const std::vector<float>& v, const std::vector<float>& sinks,
                                  const MimoDecAttnArgs& a, int old) {
    std::vector<double> out((size_t)a.T*a.n_head*a.head_v);
    for(int t=0;t<a.T;++t) for(int h=0;h<a.n_head;++h) {
        int pos=a.p0+t,lo=a.window?std::max(0,pos-a.window+1):0,kh=h/(a.n_head/a.n_kv);
        std::vector<double> scores(pos-lo+1);
        double m=sinks.empty()?-INFINITY:sinks[h];
        for(int j=lo;j<=pos;++j) {
            double dot=0;
            for(int d=0;d<a.head_k;++d) dot+=double(q[(size_t)t*qld+h*a.head_k+d])*k[((size_t)(j%a.ring)*a.n_kv+kh)*a.head_k+d];
            // Use the exact FP32 caller scale, isolating arithmetic from input rounding.
            scores[j-lo]=dot*a.scale;m=std::max(m,scores[j-lo]);
        }
        double l=sinks.empty()?0:std::exp(sinks[h]-m);
        for(auto& x:scores) {x=std::exp(x-m);l+=x;}
        for(int d=0;d<a.head_v;++d) {
            double o=0;
            for(int j=lo;j<=pos;++j) o+=scores[j-lo]*v[((size_t)(j%a.ring)*a.n_kv+kh)*a.head_v+d];
            out[((size_t)t*a.n_head+h)*a.head_v+d]=o*a.v_scale/l;
        }
    }
    return out;
}
static bool run_kv8(int p0, int T, bool sink=false, bool wrap=false) {
    constexpr int H=128,NH=8,DK=192,DV=128;
    const int ring=wrap?256:p0+T+7;
    MimoDecAttnArgs a{H,NH,DK,DV,ring,wrap?128:0,p0,T,1.0f/std::sqrt(float(DK)),0.612f};
    const int qld=H*DK+19,old=H*DV+13;
    std::mt19937 rng(789+p0+T);std::normal_distribution<float> nd(0,0.5f);
    std::vector<float> q((size_t)T*qld),k((size_t)ring*NH*DK),v((size_t)ring*NH*DV),sinks(sink?H:0);
    for(auto& x:q)x=nd(rng);for(auto& x:k)x=h2f(f2h(nd(rng)));for(auto& x:v)x=h2f(f2h(nd(rng)*2));
    for(auto& x:sinks)x=nd(rng)*2;
    // Zero heads and differently scaled heads exercise zero scales and dynamic range.
    for(size_t row=0;row<(size_t)ring*NH;++row) {
        const float scale=row%29==0?0.0f:(row%7==0?0.00001f:1.0f);
        for(int d=0;d<DK;++d)k[row*DK+d]*=scale;
        for(int d=0;d<DV;++d)v[row*DV+d]*=scale;
    }
    auto kref=k,vref=v;
    std::vector<int8_t> ki(k.size()),vi(v.size());std::vector<float> ks((size_t)ring*NH),vs(ks.size());
    auto quant=[](const std::vector<float>& src,int dim,std::vector<int8_t>& dst,std::vector<float>& scales,std::vector<float>& deq) {
        for(size_t row=0;row<scales.size();++row) {
            float m=0;for(int d=0;d<dim;++d)m=std::max(m,std::fabs(src[row*dim+d]));
            float sc=m>0?m/127.0f:1.0f;scales[row]=sc;
            for(int d=0;d<dim;++d) {int z=int(std::nearbyint(src[row*dim+d]/sc));dst[row*dim+d]=z;deq[row*dim+d]=z*sc;}
        }
    };
    std::vector<float> kd(k.size()),vd(v.size());quant(k,DK,ki,ks,kd);quant(v,DV,vi,vs,vd);
    float *dq,*dkraw,*dvraw,*dks,*dvs,*dsink,*dout,*scr;
    int8_t *dk,*dv;
    ck(cudaMalloc(&dq,q.size()*4));ck(cudaMalloc(&dkraw,k.size()*4));ck(cudaMalloc(&dvraw,v.size()*4));
    ck(cudaMalloc(&dk,k.size()));ck(cudaMalloc(&dv,v.size()));ck(cudaMalloc(&dks,ks.size()*4));ck(cudaMalloc(&dvs,vs.size()*4));
    ck(cudaMalloc(&dsink,H*4));ck(cudaMalloc(&dout,(size_t)T*old*4));
    const size_t sf=mimo_decode_attn_scratch(a,mimo_decode_attn_splits(a));ck(cudaMalloc(&scr,sf*4));
    ck(cudaMemcpy(dq,q.data(),q.size()*4,cudaMemcpyHostToDevice));ck(cudaMemcpy(dkraw,k.data(),k.size()*4,cudaMemcpyHostToDevice));
    ck(cudaMemcpy(dvraw,v.data(),v.size()*4,cudaMemcpyHostToDevice));
    if(sink)ck(cudaMemcpy(dsink,sinks.data(),H*4,cudaMemcpyHostToDevice));
    // Exercise the production token/window writer, not just a CPU packer.
    mimo_kv8_store(dkraw,NH*DK,dvraw,NH*DV,dk,dv,dks,dvs,0,ring,ring,NH,DK,DV,0);
    std::vector<int8_t> kg(ki.size()),vg(vi.size());std::vector<float> ksg(ks.size()),vsg(vs.size());
    ck(cudaMemcpy(kg.data(),dk,kg.size(),cudaMemcpyDeviceToHost));ck(cudaMemcpy(vg.data(),dv,vg.size(),cudaMemcpyDeviceToHost));
    ck(cudaMemcpy(ksg.data(),dks,ks.size()*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(vsg.data(),dvs,vs.size()*4,cudaMemcpyDeviceToHost));
    bool ok=kg==ki&&vg==vi&&ksg==ks&&vsg==vs;
    // Nonfinite padding detects row-stride/cross-row writes.
    ck(cudaMemset(dout,0xff,(size_t)T*old*4));
    ok&=mimo_decode_attn_kv8(dq,qld,dk,dv,dks,dvs,sink?dsink:nullptr,dout,old,a,scr,sf,0);
    ck(cudaDeviceSynchronize());std::vector<float> got((size_t)T*old);ck(cudaMemcpy(got.data(),dout,got.size()*4,cudaMemcpyDeviceToHost));
    auto ref=oracle(q,qld,kd,vd,sinks,a,old),fp=oracle(q,qld,kref,vref,sinks,a,old);
    double mx=0,es=0,rs=0,qs=0,fs=0;
    for(int t=0;t<T;++t)for(int d=0;d<H*DV;++d) {
        size_t i=(size_t)t*H*DV+d;double z=got[(size_t)t*old+d],e=z-ref[i];
        ok&=std::isfinite(z);mx=std::max(mx,std::abs(e));es+=e*e;rs+=ref[i]*ref[i];
        qs+=(ref[i]-fp[i])*(ref[i]-fp[i]);fs+=fp[i]*fp[i];
    }
    for(int t=0;t<T;++t)for(int d=H*DV;d<old;++d)ok&=std::isnan(got[(size_t)t*old+d]);
    double rel=std::sqrt(es/std::max(rs,1e-30)),qrel=std::sqrt(qs/std::max(fs,1e-30));
    ok&=mx<0.002&&rel<0.002&&qrel<0.025;
    std::printf("%s KV8 p0=%d T=%d sink=%d wrap=%d writer=%s FP64(dequant) max_abs=%.3e rel_l2=%.3e quant_vs_FP16=%.3e\n",
        ok?"ok":"FAIL",p0,T,sink,wrap,(kg==ki&&vg==vi&&ksg==ks&&vsg==vs)?"exact":"FAIL",mx,rel,qrel);
    for(auto x:{dq,dkraw,dvraw,dks,dvs,dsink,dout,scr})ck(cudaFree(x));ck(cudaFree(dk));ck(cudaFree(dv));
    return ok;
}
// Deterministic exactly-representable data. One variant resident at a time; 4 KV
// heads (GQA=16) limit the largest allocation to 500 MiB rather than >1 GiB.
template<bool KV8>
__global__ void bench_init(void* k,void* v,float* ks,float* vs,size_t rows) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<rows*192) {
        int z=int((i*17+53)%255)-127;
        if constexpr(KV8) ((int8_t*)k)[i]=z;else ((__half*)k)[i]=__float2half(z/256.0f);
    }
    if(i<rows*128) {
        int z=int((i*23+97)%255)-127;
        if constexpr(KV8) ((int8_t*)v)[i]=z;else ((__half*)v)[i]=__float2half(z/128.0f);
    }
    if constexpr(KV8) if(i<rows){ks[i]=1.0f/256;vs[i]=1.0f/128;}
}
static bool bench_kv8(int keys,int T) {
    const int H=64,NH=4;
    MimoDecAttnArgs a{H,NH,192,128,keys,0,keys-T,T,1.0f/std::sqrt(192.0f),0.612f};
    size_t sf=mimo_decode_attn_scratch(a,mimo_decode_attn_splits(a)),rows=(size_t)keys*NH;
    std::vector<float> q((size_t)T*H*192,0.1f),fpout((size_t)T*H*128),io(fpout.size());
    float *dq,*out,*scr;ck(cudaMalloc(&dq,q.size()*4));ck(cudaMalloc(&out,fpout.size()*4));ck(cudaMalloc(&scr,sf*4));
    ck(cudaMemcpy(dq,q.data(),q.size()*4,cudaMemcpyHostToDevice));
    double ms[2];
    for(int mode=0;mode<2;++mode) {
        void *k,*v;float *ks=nullptr,*vs=nullptr;
        ck(cudaMalloc(&k,rows*192*(mode?1:2)));ck(cudaMalloc(&v,rows*128*(mode?1:2)));
        if(mode){ck(cudaMalloc(&ks,rows*4));ck(cudaMalloc(&vs,rows*4));bench_init<true><<<(rows*192+255)/256,256>>>(k,v,ks,vs,rows);}
        else bench_init<false><<<(rows*192+255)/256,256>>>(k,v,ks,vs,rows);
        auto launch=[&] {
            bool ok=mode?mimo_decode_attn_kv8(dq,H*192,(const int8_t*)k,(const int8_t*)v,ks,vs,nullptr,out,H*128,a,scr,sf,0):
                mimo_decode_attn(dq,H*192,(const uint16_t*)k,(const uint16_t*)v,nullptr,out,H*128,a,scr,sf,0);
            if(!ok){std::puts("FAIL: benchmark not launched");std::exit(1);}
        };
        launch();ck(cudaDeviceSynchronize());cudaEvent_t start,end;ck(cudaEventCreate(&start));ck(cudaEventCreate(&end));
        ck(cudaEventRecord(start));for(int i=0;i<5;++i)launch();ck(cudaEventRecord(end));ck(cudaEventSynchronize(end));
        float elapsed;ck(cudaEventElapsedTime(&elapsed,start,end));ms[mode]=elapsed/5;
        ck(cudaMemcpy(mode?io.data():fpout.data(),out,fpout.size()*4,cudaMemcpyDeviceToHost));
        ck(cudaEventDestroy(start));ck(cudaEventDestroy(end));ck(cudaFree(k));ck(cudaFree(v));if(mode){ck(cudaFree(ks));ck(cudaFree(vs));}
    }
    double mx=0;bool ok=true;for(size_t i=0;i<io.size();++i){ok&=std::isfinite(io[i])&&std::isfinite(fpout[i]);mx=std::max(mx,double(std::abs(io[i]-fpout[i])));}
    ok&=mx<0.002;
    std::printf("%s BENCH keys=%d T=%d heads=64/4 FP16=%.3f ms INT8=%.3f ms speedup=%.3fx max_abs=%.3e\n",ok?"ok":"FAIL",keys,T,ms[0],ms[1],ms[0]/ms[1],mx);
    ck(cudaFree(dq));ck(cudaFree(out));ck(cudaFree(scr));return ok;
}
int main(int argc, char** argv) {
    const bool timing = argc > 1 && std::strcmp(argv[1],"--kv8-bench")!=0;
    if(argc>1&&std::strcmp(argv[1],"--kv8-bench")==0) {
        bool ok=true;for(int keys:{32768,131072,204800})for(int T:{1,8})ok&=bench_kv8(keys,T);
        std::puts(ok?"PASS: KV8 bounded-memory kernel benchmark":"FAILED");return ok?0:1;
    }
    bool ok = true;
    // MiMo-V2.6-Pro geometry: 128 query heads, 8 KV heads; full layers (window 0) and SWA (window 128, ring 256)
    ok &= run(128, 8, 4096, 0, 0, 1, false, false);
    ok &= run(128, 8, 4096, 0, 37, 1, false, false);
    ok &= run(128, 8, 4096, 0, 1000, 1, false, false);
    ok &= run(128, 8, 4096, 0, 3000, 8, false, false);
    ok &= run(128, 8, 4096, 0, 2500, 5, true, false);
    ok &= run(128, 8, 256, 128, 1000, 1, true, false);    // ring wraps, window
    ok &= run(128, 8, 256, 128, 5000, 8, true, false);
    ok &= run(16, 2, 512, 0, 400, 3, false, false);       // another group / head count
    for(int p:{0,37,769}) ok &= run_kv8(p,1);
    ok &= run_kv8(777,8,true);
    ok &= run_kv8(2500,5,false);
    ok &= run_kv8(1000,8,true,true);
    const strata::core::MimoKv8Layout layout(12,524288,8,192,128);
    ok &= (layout.end-12)*4 == int64_t(524288)*8*(192+128+8);
    if (timing) {
        for (int ctx : {2048, 12288, 32768, 65536, 131072, 204800})
            for (int T : {1, 8}) run(128, 8, ctx + 16, 0, ctx, T, false, true);
    }
    std::printf(ok ? "PASS: mimo decode FP16 + KV8 attention vs FP64 CPU reference\n" : "FAILED\n");
    return ok ? 0 : 1;
}
