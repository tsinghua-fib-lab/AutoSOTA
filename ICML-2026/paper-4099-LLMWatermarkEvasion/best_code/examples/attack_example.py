# Copyright 2024 THU-BPM MarkLLM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ==========================================
# Description: Minimal example — run a watermark-removal (paraphrasing) attack
#   on a single piece of text with the attack of your choice.
#
#   Usage (run from the repo root):
#     python examples/attack_example.py --attack BIRA --backend hf --text "..."
#     python examples/attack_example.py --attack vanilla_paraphrasing --text "..."
#     python examples/attack_example.py --attack dipper-2 --text "..."
#     python examples/attack_example.py --attack SIRA --text "..."
#
#   It builds the chosen Attack from attack_utils.attacks, loads its model(s),
#   and calls `rewrite()` — the exact per-sample attack the full pipeline runs —
#   then prints the original vs. attacked text. (No watermark detection here;
#   this demo focuses on the rewriting step.)
# ==========================================

import argparse
import sys
from types import SimpleNamespace

sys.path.append(".")
from attack_utils.attacks import ATTACK_REGISTRY


# A sample paragraph to attack. Replace with --text "..." (e.g. your watermarked text).
EXAMPLE_TEXT = (
    "The discovery of the new exoplanet has excited astronomers around the world. "
    "Located roughly forty light-years away, the planet orbits within its star's "
    "habitable zone, where liquid water could plausibly exist on the surface. "
    "Researchers plan to study its atmosphere with next-generation telescopes in "
    "the hope of detecting chemical signatures associated with life."
)


def build_attack_args(cli):
    """Assemble the argument namespace the Attack classes expect.

    Only the fields relevant to the chosen attack are actually read; the rest
    are harmless defaults (mirroring pipeline/run_attack.py).
    """
    return SimpleNamespace(
        attack_algorithms=cli.attack,
        # --- BIRA ---
        backend=cli.backend,
        model_cfg_path=cli.model_cfg_path,
        api_model_name=cli.api_model_name,
        beta=cli.beta,
        percentile=cli.percentile,
        learning_rate=0.125,
        use_sampling=True,
        max_iter=cli.max_iter,
        # --- dipper / SIRA: path label ---
        model_name=cli.model_name,
        # --- SIRA ---
        paraphrase_model_path=cli.model_path,
        attack_model_path=cli.model_path,
        threshold=cli.threshold,
        max_new_tokens=cli.max_new_tokens,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run a single-text watermark-removal (paraphrasing) attack."
    )
    parser.add_argument("--attack", choices=list(ATTACK_REGISTRY), default="BIRA",
                        help="Which attack to demonstrate.")
    parser.add_argument("--text", nargs="+", default=[EXAMPLE_TEXT],
                        help="One or more texts to attack (each quoted). The attack runs over all of them.")
    parser.add_argument("--input_file", default=None,
                        help="Optional: a file with one text per line (overrides --text).")

    # BIRA / vanilla_paraphrasing
    parser.add_argument("--backend", choices=["hf", "api"], default="hf")
    parser.add_argument("--model_cfg_path", default="./model_config/llama3.1-8b.yaml",
                        help="Rewriter model config (hf backend / SIRA aux model).")
    parser.add_argument("--api_model_name", default="gpt4o-mini",
                        help="API model name (api backend).")
    parser.add_argument("--beta", type=float, default=-4.0,
                        help="BIRA logit bias (recommended: -4.0 for Llama-3.1-8B/70B, -11.0 for gpt-4o-mini).")
    parser.add_argument("--percentile", type=int, default=50,
                        help="BIRA self-information percentile / SIRA uses --threshold.")
    parser.add_argument("--max_iter", type=int, default=20)

    # dipper / SIRA
    parser.add_argument("--model_name", default="demo",
                        help="Label used for output paths (not needed for this demo).")
    parser.add_argument("--model_path", default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                        help="SIRA paraphrase/attack model.")
    parser.add_argument("--threshold", type=int, default=30,
                        help="SIRA self-information blanking percentile.")
    parser.add_argument("--max_new_tokens", type=int, default=1500, help="SIRA max new tokens.")

    cli = parser.parse_args()

    # Input text(s): a single string or a whole list — the attack runs over all.
    if cli.input_file:
        with open(cli.input_file) as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        texts = cli.text

    args = build_attack_args(cli)
    attack = ATTACK_REGISTRY[args.attack_algorithms](args)
    attack.load_models()  # load once, reuse across all texts

    backend_note = f" (backend={args.backend})" if args.attack_algorithms in ("BIRA", "vanilla_paraphrasing") else ""
    print("=" * 80)
    print(f"attack: {args.attack_algorithms}{backend_note}  |  {len(texts)} text(s)")
    print("=" * 80)
    for i, text in enumerate(texts):
        attacked_text, meta = attack.rewrite(text)
        print(f"\n--- [{i + 1}/{len(texts)}] ---")
        print("[ original ]\n" + text)
        print("\n[ attacked ]\n" + attacked_text)
        if meta:
            print(f"[ meta ] {meta}")


if __name__ == "__main__":
    main()
