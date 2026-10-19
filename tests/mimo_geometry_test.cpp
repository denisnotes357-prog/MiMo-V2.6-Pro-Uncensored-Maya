// tests/mimo_geometry_test.cpp - MiMo-V2.6-Pro port, step M1 (docs/MIMO2-ENGINE-PORT.md): Glm5Geometry::from_gguf on
// an arch "mimo2" GGUF.  Parsing only: no weights are uploaded.
//
//   mimo_geometry_test MODEL.gguf [--expect-tiny | --expect-pro]
//
//   --expect-tiny  a small synthetic mimo2 GGUF (convert_hf_to_gguf of a tiny HF model)
//   --expect-pro   the real MiMo-V2.6-Pro (first shard of an M23A quant)
// The last line is always "mimo_geometry_test: PASS" or "mimo_geometry_test: FAIL: ...".
#include "strata/core/glm_model.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

using strata::core::Glm5Geometry;
std::vector<std::string> problems;

void want(bool ok, const std::string& what) {
    if (!ok) problems.push_back(what);
}

void want_eq(long long got, long long want_v, const std::string& what) {
    if (got != want_v) {
        char b[160];
        std::snprintf(b, sizeof(b), "%s: got %lld want %lld", what.c_str(), got, want_v);
        problems.push_back(b);
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: mimo_geometry_test MODEL.gguf [--expect-tiny | --expect-pro]\n");
        return 1;
    }
    const std::string path = argv[1];
    const std::string mode = argc > 2 ? argv[2] : "";
    Glm5Geometry g;
    std::string err;
    try {
        strata::GgufFile f(path);
        if (!Glm5Geometry::from_gguf(f, g, err)) {
            std::printf("mimo_geometry_test: FAIL: %s\n", err.c_str());
            return 1;
        }
    } catch (const std::exception& e) {
        std::printf("mimo_geometry_test: FAIL: cannot read %s: %s\n", path.c_str(), e.what());
        return 1;
    }
    std::string swa;
    for (size_t i = 0; i < g.swa_layer.size(); ++i) swa += (i ? "," : "") + std::to_string(g.swa_layer[i]);
    std::printf("geometry: %s n_embd=%d n_layers=%d nextn=%d n_head=%d n_head_kv=%d head_k=%d head_v=%d "
                "rope_first=%d rope_base=%g rope_base_swa=%g swa_window=%d v_scale=%g n_expert=%d n_exp_used=%d "
                "n_ff_exp=%d n_ff_dense=%d dense_lead=%d n_shared=%d w_scale=%g norm_w=%d n_vocab=%d "
                "norm_eps=%g swa=[%s]\n",
                g.arch == Glm5Geometry::Glm5Arch::Mimo2 ? "Mimo2" : "other", g.n_embd, g.n_layers, g.nextn,
                g.n_head, g.n_head_kv_gqa, g.head_k, g.head_v, g.rope_first, (double) g.rope_base,
                (double) g.rope_base_swa, g.swa_window, (double) g.v_scale, g.n_expert, g.n_exp_used, g.n_ff_exp,
                g.n_ff_dense, g.dense_lead, g.n_shared, (double) g.w_scale, g.norm_w, g.n_vocab,
                (double) g.norm_eps, swa.c_str());

    if (mode == "--expect-tiny" || mode == "--expect-pro") {
        const bool tiny = mode == "--expect-tiny";
        want(g.arch == Glm5Geometry::Glm5Arch::Mimo2, "arch == Mimo2");
        want_eq(g.n_embd, tiny ? 512 : 6144, "n_embd");
        want_eq(g.n_layers, tiny ? 6 : 70, "n_layers");
        want_eq(g.nextn, 3, "nextn (the converter always writes 3 MTP slots)");
        want_eq(g.n_head, tiny ? 4 : 128, "n_head");
        want_eq(g.n_head_kv_gqa, tiny ? 2 : 8, "n_head_kv");
        want_eq(g.head_k, 192, "head_k");
        want_eq(g.head_v, 128, "head_v");
        want_eq(g.rope_first, 64, "rope_first");
        want(std::fabs((double) g.rope_base - 1e7) < 1.0, "rope_base == 1e7");
        want(std::fabs((double) g.rope_base_swa - 1e4) < 1e-3, "rope_base_swa == 1e4");
        want_eq(g.swa_window, tiny ? 16 : 128, "swa_window");
        want(std::fabs((double) g.v_scale - 0.612) < 1e-6, "v_scale == 0.612");
        want_eq(g.n_expert, tiny ? 8 : 384, "n_expert");
        want_eq(g.n_exp_used, tiny ? 2 : 8, "n_exp_used");
        want_eq(g.n_ff_exp, tiny ? 256 : 2048, "n_ff_exp");
        want_eq(g.dense_lead, 1, "dense_lead (layer 0 dense)");
        want_eq(g.n_shared, 0, "n_shared");
        if (tiny) {
            want(g.swa_layer == std::vector<int>({0, 1, 1, 0, 1, 1}), "swa == 0,1,1,0,1,1");
            want_eq(g.n_vocab, 2048, "n_vocab");
        } else {
            std::vector<int> want_swa(70, 1);
            for (int il : {0, 7, 15, 23, 31, 39, 47, 55, 62, 69}) want_swa[(size_t) il] = 0;
            want(g.swa_layer == want_swa, "swa == hybrid_layer_pattern (full at 0,7,15,23,31,39,47,55,62,69)");
            want_eq(g.n_vocab, 152576, "n_vocab");
        }
    }
    if (!problems.empty()) {
        for (const auto& p : problems) std::printf("  FAIL: %s\n", p.c_str());
        std::printf("mimo_geometry_test: FAIL: %zu check(s): first: %s\n", problems.size(), problems[0].c_str());
        return 1;
    }
    std::printf("mimo_geometry_test: PASS\n");
    return 0;
}
