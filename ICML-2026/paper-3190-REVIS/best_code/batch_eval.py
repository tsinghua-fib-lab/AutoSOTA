#!/usr/bin/env python3
"""Batch evaluation wrapper for CHAIR benchmark with different steering configs."""
import subprocess, sys, json, os, time

CONFIGS = [
    # (alpha_vis, tau_low, risk_gamma, label)
    (1.6, 0.2, 1.0, "Vis27_aV1.6_Gamma1.0_TauL0.2_recomputed"),
    (2.5, 0.2, 1.0, "Vis27_aV2.5_Gamma1.0_TauL0.2_recomputed"),
    (1.2, 0.2, 1.0, "Vis27_aV1.2_Gamma1.0_TauL0.2_recomputed"),
    (2.0, 0.2, 1.0, "Vis27_aV2.0_Gamma1.0_TauL0.2_recomputed"),
    (3.0, 0.2, 1.0, "Vis27_aV3.0_Gamma1.0_TauL0.2_recomputed"),
    (4.0, 0.2, 1.0, "Vis27_aV4.0_Gamma1.0_TauL0.2_recomputed"),
]

VECTOR_FILE = sys.argv[1] if len(sys.argv) > 1 else "./vector/qwen2.5vl_none_image_recomputed.pt"
BASE_DIR = sys.argv[2] if len(sys.argv) > 2 else "./results"

for alpha, tau, gamma, label in CONFIGS:
    print(f"\n{=*60}")
    print(f"Running: alpha={alpha}, tau_low={tau}, gamma={gamma}")
    print(f"{=*60}")
    t0 = time.time()
    
    cmd = [
        "python3", "main_all_visual_only.py", "run",
        "--model_path", "/models/Qwen2.5-VL-7B-Instruct",
        "--vector_file", VECTOR_FILE,
        "--benchmark", "chair",
        "--results_dir", BASE_DIR,
        "--question_dir", "./data/chair_val2014_100.jsonl",
        "--image_folder", "/datasets/coco/val2014",
        "--gt_dir", "/datasets/coco",
        "--vis_layers", "27",
        "--alpha_visual", str(alpha),
        "--tau_low", str(tau),
        "--risk_gamma", str(gamma),
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, cwd="/repo")
    elapsed = time.time() - t0
    
    print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[-500:])
    print(f"Elapsed: {elapsed:.1f}s")
    
    # Run CHAIR eval on generated captions
    caption_file = f"{BASE_DIR}/chair/{label}/chair_captions.jsonl"
    if os.path.exists(caption_file):
        eval_cmd = [
            "python3", "compute_chair_metrics.py",
            "--response_file", caption_file,
            "--coco_path", "/datasets/coco"
        ]
        eval_result = subprocess.run(eval_cmd, capture_output=True, text=True, timeout=300, cwd="/repo")
        print(f"EVAL: {eval_result.stdout.strip()}")
    
    print(f"Done: {label} ({elapsed:.0f}s)")
