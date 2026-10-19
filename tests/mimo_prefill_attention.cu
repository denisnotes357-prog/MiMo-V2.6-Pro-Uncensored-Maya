#include "strata/kernels/mimo_prefill.hpp"
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
using strata::kernels::MimoPrefillArgs;
static void ck(cudaError_t e) { if (e != cudaSuccess) { std::fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e)); std::exit(1); } }
static void check_case(int window, int extra, int chunk, bool scalar) {
    const int nt=289, H=4, HK=2, DK=192, DV=128, R=64;
    const int ring=window ? window+extra : nt, nq=H*DK, rk=HK*DK, rv=HK*DV, ld=nq+rk+rv;
    MimoPrefillArgs a{H,HK,DK,DV,R,ring,window,window ? 10000.0f : 10000000.0f,0.612f};
    std::vector<float> raw(nt*ld), rot, ref(nt*H*DV), got(ref.size()), sinks{0.2f,-3.0f,5.0f,-0.4f};
    for (size_t i=0;i<raw.size();++i) raw[i]=0.7f*std::sin(float((i*17+113)%4093)*0.13f);
    rot=raw;
    // Independent CPU oracle: complete dense causal scores, FP16 cache rounding.
    for (int t=0;t<nt;++t) for(int h=0;h<H+HK;++h) for(int d=0;d<R/2;++d) {
        const double ang=t*std::pow(double(a.rope_base),-2.0*d/R);
        size_t i=(size_t)t*ld+h*DK+d;
        const float x=raw[i],y=raw[i+R/2];
        rot[i]=float(x*std::cos(ang)-y*std::sin(ang));
        rot[i+R/2]=float(y*std::cos(ang)+x*std::sin(ang));
    }
    for(int t=0;t<nt;++t) for(int h=0;h<H;++h) {
        const int lo=window ? std::max(0,t-window+1) : 0, kh=h/(H/HK);
        std::vector<double> w(t-lo+1); double m=window ? sinks[h] : -INFINITY;
        for(int j=lo;j<=t;++j) {
            double z=0;
            for(int d=0;d<DK;++d) z+=rot[(size_t)t*ld+h*DK+d]*double(__half2float(__float2half(rot[(size_t)j*ld+nq+kh*DK+d])));
            w[j-lo]=z/std::sqrt(double(DK)); m=std::max(m,w[j-lo]);
        }
        double sum=window ? std::exp(sinks[h]-m) : 0;
        for(double& wj:w) { wj=std::exp(wj-m);sum+=wj; }
        for(int d=0;d<DV;++d) {
            double z=0;
            for(int j=lo;j<=t;++j) z+=w[j-lo]*__half2float(__float2half(raw[(size_t)j*ld+nq+rk+kh*DV+d]));
            ref[((size_t)t*H+h)*DV+d]=float(z*a.v_scale/sum);
        }
    }
    float *q,*o,*si; uint16_t *kc,*vc;
    ck(cudaMalloc(&q,raw.size()*4));ck(cudaMalloc(&o,got.size()*4));ck(cudaMalloc(&si,H*4));
    ck(cudaMalloc(&kc,(size_t)ring*rk*2));ck(cudaMalloc(&vc,(size_t)ring*rv*2));
    ck(cudaMemcpy(q,raw.data(),raw.size()*4,cudaMemcpyHostToDevice));
    ck(cudaMemcpy(si,sinks.data(),H*4,cudaMemcpyHostToDevice));
    ck(cudaMemset(kc,0xff,(size_t)ring*rk*2));ck(cudaMemset(vc,0xff,(size_t)ring*rv*2));
    cudaStream_t s;ck(cudaStreamCreate(&s));
    const int step=window ? std::min(chunk,ring-window+1):chunk;
    for(int t=0;t<nt;t+=step) {
        const int n=std::min(step,nt-t);
        strata::kernels::mimo_prefill_rope_kv(q+(size_t)t*ld,kc,vc,t,n,a,s);
        strata::kernels::mimo_prefill_attention(q+(size_t)t*ld,kc,vc,window?si:nullptr,o+(size_t)t*H*DV,t,n,a,s,scalar);
    }
    ck(cudaStreamSynchronize(s)); ck(cudaMemcpy(got.data(),o,got.size()*4,cudaMemcpyDeviceToHost));
    double worst=0,ds=0,rs=0;
    for(size_t i=0;i<got.size();++i) {
        if(!std::isfinite(got[i])) {std::fprintf(stderr,"nonfinite %zu\n",i);std::exit(1);}
        const double d=got[i]-ref[i];worst=std::max(worst,std::abs(d));ds+=d*d;rs+=double(ref[i])*ref[i];
    }
    std::printf("window=%d ring=%d chunk=%d %s max_abs=%.6g rel_l2=%.6g\n",window,ring,chunk,scalar?"scalar":"mma",worst,std::sqrt(ds/rs));
    if(worst>0.002 || std::sqrt(ds/rs)>0.003) std::exit(1);
    ck(cudaFree(q));ck(cudaFree(o));ck(cudaFree(si));ck(cudaFree(kc));ck(cudaFree(vc));ck(cudaStreamDestroy(s));
}

// A long, already-populated prefix checks register-tiled attention against the
// independent FP32 scalar path, including the second 64-query tile and sinks.
static void check_long_prefix() {
    constexpr int P=65536,T=67,H=4,HK=2,DK=192,DV=128,LD=(H+HK)*DK+HK*DV;
    MimoPrefillArgs a{H,HK,DK,DV,64,P+T,0,10000000.0f,0.612f};
    std::vector<float> raw((size_t)T*LD),ref((size_t)T*H*DV),got(ref.size());
    std::vector<__half> cache((size_t)(P+T)*HK*DK);
    for(size_t i=0;i<raw.size();++i) raw[i]=0.7f*std::sin(float(i%4093)*0.13f);
    for(size_t i=0;i<cache.size();++i) cache[i]=__float2half(0.7f*std::sin(float(i%4091)*0.17f));
    float *q,*o,*si;uint16_t *k,*v;
    ck(cudaMalloc(&q,raw.size()*4));ck(cudaMalloc(&o,ref.size()*4));ck(cudaMalloc(&si,H*4));
    ck(cudaMalloc(&k,cache.size()*2));ck(cudaMalloc(&v,(size_t)(P+T)*HK*DV*2));
    ck(cudaMemcpy(q,raw.data(),raw.size()*4,cudaMemcpyHostToDevice));
    ck(cudaMemcpy(k,cache.data(),cache.size()*2,cudaMemcpyHostToDevice));
    ck(cudaMemcpy(v,cache.data(),(size_t)(P+T)*HK*DV*2,cudaMemcpyHostToDevice));
    const float sink[H]={0.2f,-3.0f,5.0f,-0.4f};ck(cudaMemcpy(si,sink,sizeof(sink),cudaMemcpyHostToDevice));
    cudaStream_t stream;ck(cudaStreamCreate(&stream));
    strata::kernels::mimo_prefill_attention(q,k,v,si,o,P,T,a,stream,true);
    ck(cudaStreamSynchronize(stream));ck(cudaMemcpy(ref.data(),o,ref.size()*4,cudaMemcpyDeviceToHost));
    strata::kernels::mimo_prefill_attention(q,k,v,si,o,P,T,a,stream,false);
    ck(cudaStreamSynchronize(stream));ck(cudaMemcpy(got.data(),o,got.size()*4,cudaMemcpyDeviceToHost));
    double worst=0,ds=0,rs=0;
    for(size_t i=0;i<ref.size();++i) {
        if(!std::isfinite(got[i]))std::exit(1);
        double d=ref[i]-got[i];worst=std::max(worst,std::abs(d));ds+=d*d;rs+=double(ref[i])*ref[i];
    }
    std::printf("prefix=%d queries=%d register MMA vs FP32 scalar max_abs=%.6g rel_l2=%.6g\n",P,T,worst,std::sqrt(ds/rs));
    if(worst>0.002 || std::sqrt(ds/rs)>0.003)std::exit(1);
    ck(cudaFree(q));ck(cudaFree(o));ck(cudaFree(si));ck(cudaFree(k));ck(cudaFree(v));ck(cudaStreamDestroy(stream));
}

int main() {
    for(bool scalar:{true,false}) {
        check_case(0,0,53,scalar); // causal tails and nonzero prefix
        check_case(16,7,71,scalar); // ring wrap many times, sink, partial tiles
        check_case(128,128,289,scalar); // real SWA geometry and 129-row write-ahead limit
    }
    check_long_prefix();
    std::puts("PASS: MiMo prefill attention vs independent CPU oracle");
}
