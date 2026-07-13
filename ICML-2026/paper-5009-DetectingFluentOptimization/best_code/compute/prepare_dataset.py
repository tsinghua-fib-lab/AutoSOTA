#!/usr/bin/env python3
import argparse
import os
import yaml
import pandas as pd


def load_model_configs(path="config/models.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)["model_configs"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--global_csv",
        default="data/tydiqa_pp_matched_700_lang.csv",
        help="CSV used as the source of benign prompts to merge with other attacks.",
    )
    parser.add_argument(
        "--autodan_csv",
        default="data/full_prompt_dataset.csv",
        help="Optional CSV used to source AutoDAN prompts (if not present in global_csv). Use an empty string to disable.",
    )
    parser.add_argument("--out_csv", default=None)
    parser.add_argument(
        "--normal-label",
        default="normal",
        help="Algorithm label to assign to normal prompts from global_csv.",
    )
    args = parser.parse_args()

    out = args.out_csv or f"data/{args.model}_dataset.csv"

    configs = load_model_configs()
    mcfg = configs.get(args.model)
    if not mcfg:
        raise ValueError(f"Model {args.model} not found in configs")
    print(f"Loading model config for {args.model}: {mcfg}")

    if str(args.global_csv).strip() == "":
        df = pd.DataFrame({"full_prompt": [], "suffix": [], "is_adversarial": [], "algorithm": []})
    else:
        df = pd.read_csv(args.global_csv)

    # ---------------- AutoDAN (if present in global_csv) ----------------
    df_auto = pd.DataFrame({"full_prompt": [], "suffix": [], "is_adversarial": [], "algorithm": []})
    if "is_adversarial" in df.columns:
        auto_raw = df[df["is_adversarial"].fillna(0).astype(int) == 1]
        auto_raw = auto_raw[auto_raw["algorithm"].fillna("").str.lower() == "autodan"].copy()
        if not auto_raw.empty:
            if "suffix" not in auto_raw.columns:
                auto_raw["suffix"] = auto_raw["full_prompt"]
            auto_raw["is_adversarial"] = 1
            auto_raw["algorithm"] = "autodan"
            df_auto = auto_raw[["full_prompt", "suffix", "is_adversarial", "algorithm"]]
    if df_auto.empty:
        autodan_csv = (args.autodan_csv or "").strip()
        if autodan_csv:
            if os.path.exists(autodan_csv):
                df_autodan_src = pd.read_csv(autodan_csv)
                if "is_adversarial" in df_autodan_src.columns and "algorithm" in df_autodan_src.columns:
                    auto_raw = df_autodan_src[df_autodan_src["is_adversarial"].fillna(0).astype(int) == 1]
                    auto_raw = auto_raw[auto_raw["algorithm"].fillna("").str.lower() == "autodan"].copy()
                    if not auto_raw.empty:
                        if "suffix" not in auto_raw.columns:
                            auto_raw["suffix"] = auto_raw["full_prompt"]
                        auto_raw["is_adversarial"] = 1
                        auto_raw["algorithm"] = "autodan"
                        df_auto = auto_raw[["full_prompt", "suffix", "is_adversarial", "algorithm"]]
                        print(f"Loaded {len(df_auto)} AutoDAN prompts from {autodan_csv}")
            else:
                print(f"Warning: AutoDAN CSV not found at {autodan_csv}. Skipping AutoDAN prompts.")

    # ---------------- Normals ----------------
    normals = df.copy()
    if "is_adversarial" in normals.columns:
        normals = normals[normals["is_adversarial"].fillna(0).astype(int) == 0]
    normals = normals[~normals["algorithm"].fillna("").str.lower().isin({"advprompter", "gcg", "autodan"})]
    normals = normals.copy()
    normals["algorithm"] = args.normal_label
    normals["is_adversarial"] = 0

    # ---------------- AdvPrompter ----------------
    adv_path = mcfg.get("advprompter_path", "")
    if adv_path and os.path.exists(adv_path):
        df_adv_raw = pd.read_csv(adv_path)
        df_adv = pd.DataFrame(
            {
                "full_prompt": df_adv_raw["full_instruct"],
                "suffix": df_adv_raw["suffix"],
                "is_adversarial": 1,
                "algorithm": "advprompter",
            }
        )
    else:
        print(f"Warning: No advprompter data found at {adv_path}. Skipping adversarial prompts.")
        df_adv = pd.DataFrame({"full_prompt": [], "suffix": [], "is_adversarial": [], "algorithm": []})

    # ---------------- GCG ----------------
    gcg_path = os.path.join("data", "gcg_llamaguard_bypass.csv")
    if os.path.exists(gcg_path):
        df_gcg_raw = pd.read_csv(gcg_path)
        df_gcg = pd.DataFrame(
            {
                "full_prompt": df_gcg_raw["prompt"] + df_gcg_raw["trigger"],
                "suffix": df_gcg_raw["trigger"],
                "is_adversarial": 1,
                "algorithm": "gcg",
            }
        )
    else:
        print(f"Warning: No GCG data found at {gcg_path}. Skipping GCG prompts.")
        df_gcg = pd.DataFrame({"full_prompt": [], "suffix": [], "is_adversarial": [], "algorithm": []})

    # Combine and deduplicate
    df_all = pd.concat([normals, df_adv, df_gcg, df_auto], axis=0, ignore_index=True)
    df_all = df_all.drop_duplicates(subset=["full_prompt", "suffix", "algorithm"])

    # Keep only relevant columns
    keep_cols = [c for c in ["full_prompt", "suffix", "is_adversarial", "algorithm"] if c in df_all.columns]
    df_all = df_all[keep_cols]

    os.makedirs(os.path.dirname(out), exist_ok=True)
    df_all.to_csv(out, index=False)
    print(f"Wrote {len(df_all)} rows to {out}")


if __name__ == "__main__":
    main()
