#pragma once
#include <cstdint>

namespace strata::core {
// All arena offsets are in float units, including the INT8 arrays.
// K bytes (rounded up to 4), K scales, V bytes (rounded up to 4), V scales.
struct MimoKv8Layout {
    int64_t k, ks, v, vs, end;
    MimoKv8Layout(int64_t start, int64_t rows, int nh, int dk, int dv) {
        k=start; ks=k+(rows*nh*dk+3)/4;
        v=ks+rows*nh; vs=v+(rows*nh*dv+3)/4;
        end=vs+rows*nh;
    }
};
}
