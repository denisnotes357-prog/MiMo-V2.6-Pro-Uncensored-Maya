"""tools/mimo/derive.py - refusal directions from capture.py's residual streams.

For every layer l (0 = embedding output, l = output of decoder layer l-1): d_l = mean(H) - mean(B) at the last prompt
position, and a separation score: (mean_H.proj - mean_B.proj) / pooled std of the projections on the unit d_l, with
the score measured by leave-half-out (directions from even prompts, scored on odd ones) so a layer cannot win by
memorising its own prompts.  Writes dirs.npy [71, hidden] (unit vectors) and prints the table; --pick L also writes
dir.npy (the chosen unit direction) for bake.py.

python derive.py [--caps caps-thinkon.npy] [--pick L]
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/data/mimo/refusal")
    ap.add_argument("--caps", default="caps-thinkon.npy")
    ap.add_argument("--pick", type=int, default=None)
    a = ap.parse_args()
    d = pathlib.Path(a.dir)
    caps = np.load(d / a.caps).astype(np.float64)                 # [n, 71, H]
    tags = np.array(json.load(open(d / "tags.json"))["tags"])
    h, b = tags == "H", tags != "H"
    idx = np.arange(len(tags))
    tr, te = idx % 2 == 0, idx % 2 == 1
    dirs = np.zeros(caps.shape[1:], dtype=np.float32)
    rows = []
    for l in range(caps.shape[1]):
        x = caps[:, l]
        v = x[h].mean(0) - x[b].mean(0)
        dirs[l] = (v / (np.linalg.norm(v) + 1e-12)).astype(np.float32)
        vt = x[h & tr].mean(0) - x[b & tr].mean(0)
        u = vt / (np.linalg.norm(vt) + 1e-12)
        p = x[te] @ u
        ph, pb = p[h[te]], p[b[te]]
        sep = (ph.mean() - pb.mean()) / np.sqrt(0.5 * (ph.var() + pb.var()) + 1e-12)
        rel = np.linalg.norm(v) / np.linalg.norm(x.mean(0))
        rows.append((l, sep, rel))
    np.save(d / "dirs.npy", dirs)
    print(" l   sep(held-out)  |dmean|/|mean|")
    for l, sep, rel in rows:
        print(f"{l:2d}  {sep:8.2f}       {rel:6.3f}")
    best = max(rows[1:], key=lambda r: r[1])
    print(f"best held-out separation: layer {best[0]} ({best[1]:.2f})")
    if a.pick is not None:
        np.save(d / "dir.npy", dirs[a.pick])
        print(f"dir.npy <- layer {a.pick}")


if __name__ == "__main__":
    main()
