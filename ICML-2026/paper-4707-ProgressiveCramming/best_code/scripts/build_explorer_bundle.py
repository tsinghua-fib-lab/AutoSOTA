"""Package the per-token explorer data into the inlined bundle the page loads.

Reads the faithful per-model JSON produced by ``export_token_explorer_data.py``
(``page/data/<model_key>.json`` + ``manifest.json``) and writes
``page/data/trajectories.js`` -- a single file that assigns
``window.__TRAJECTORIES__`` in the schema ``page/js/explorer.js`` consumes:

    { models: [ { model, variants: [ { variant,
        samples: [ { sample_id, horizon, tokens: [ {t, s, k, L, ok}, ... ] } ] } ] } ] }

Per token: t=display text, s=surprisal (bits, null for the first token),
k=steps-to-converge (null past the last recorded stage), L=1-based prefix length,
ok=that stage reconstructed exactly. Shipping an inlined ``.js`` (rather than a
fetched ``.json``) keeps the page openable straight from ``file://``.

Run after the export, from the public repo root::

    python scripts/build_explorer_bundle.py --data_dir page/data
"""

from __future__ import annotations

import argparse
import json
import os


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_dir", default="page/data", help="Dir with manifest.json + <model_key>.json")
    args = ap.parse_args()

    with open(os.path.join(args.data_dir, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)

    models_out = []
    for m in manifest["models"]:
        with open(os.path.join(args.data_dir, f"{m['key']}.json"), encoding="utf-8") as f:
            md = json.load(f)
        variant_specs = [("base", "Progressive"), ("lowdim", f"+ Low-dim ({md['lowdim_label']})")]
        variants_out = []
        for vk, vlabel in variant_specs:
            samples_out = []
            for sid in sorted(md["samples"], key=int):
                s = md["samples"][sid]
                if vk not in s["variants"]:
                    continue
                var = s["variants"][vk]
                toks = s["tokens"]
                surp = s["surprisal"]
                steps = var["steps"]
                conv = var["converged"]
                tok_objs = [
                    {"t": toks[i], "s": surp[i], "k": steps[i], "L": i + 1, "ok": bool(conv[i])}
                    for i in range(len(toks))
                ]
                samples_out.append(
                    {
                        "sample_id": int(sid),
                        "horizon": var["horizon"],
                        "capped": False,
                        "n_tokens_shown": len(tok_objs),
                        "total_tokens": len(tok_objs),
                        "tokens": tok_objs,
                    }
                )
            variants_out.append({"variant": vlabel, "samples": samples_out})
        models_out.append({"model": md["name"], "variants": variants_out})

    payload = {
        "models": models_out,
        "generated_with": "scripts/export_token_explorer_data.py + scripts/build_explorer_bundle.py",
    }
    out_js = os.path.join(args.data_dir, "trajectories.js")
    with open(out_js, "w", encoding="utf-8") as f:
        f.write("window.__TRAJECTORIES__ = ")
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
        f.write(";\n")
    kb = os.path.getsize(out_js) / 1024
    n_traj = sum(len(v["samples"]) for m in models_out for v in m["variants"])
    print(f"wrote {out_js} ({kb:.0f} KB) — {len(models_out)} models, {n_traj} trajectories")


if __name__ == "__main__":
    main()
