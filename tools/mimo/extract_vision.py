#!/usr/bin/env python3
"""tools/mimo/extract_vision.py - MiMo-V2.6-Pro's own vision tower as a small standalone folder for mimo_vision.py.

The checkpoint keeps the vision tower (364 `visual.*` tensors, ~1.48 GB BF16) inside its ~1 TB of shards.  This copies
those tensors, byte for byte, into one `visual.safetensors` beside the checkpoint's config.json and
preprocessor_config.json and a `model.safetensors.index.json` that maps them - the layout mimo_vision.py reads with
`--mmproj <folder>` - so the image path needs 1.5 GB instead of the whole checkpoint.

    python tools/mimo/extract_vision.py --model /data/models/MiMo-V2.6-Pro-MOPD-heretic --out models/<name>/vision
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="the MiMo-V2.6-Pro checkpoint (HF safetensors)")
    ap.add_argument("--out", required=True, help="the vision folder to write")
    a = ap.parse_args()
    from safetensors import safe_open             # noqa: E402
    from safetensors.torch import save_file       # noqa: E402

    src, out = pathlib.Path(a.model), pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    wmap = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    names = sorted(k for k in wmap if k.startswith("visual."))
    if not names:
        raise SystemExit("no visual.* tensors in " + str(src))
    tensors = {}
    for shard in sorted({wmap[k] for k in names}):
        with safe_open(str(src / shard), framework="pt") as f:
            for k in names:
                if wmap[k] == shard:
                    tensors[k] = f.get_tensor(k)
    save_file(tensors, str(out / "visual.safetensors"), metadata={"format": "pt"})
    total = sum(t.numel() * t.element_size() for t in tensors.values())
    index = {"metadata": {"total_size": total}, "weight_map": {k: "visual.safetensors" for k in names}}
    (out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
    for f in ("config.json", "preprocessor_config.json"):
        shutil.copy2(src / f, out / f)
    print(f"{len(names)} tensors, {total / 1e9:.2f} GB -> {out}")


if __name__ == "__main__":
    main()
