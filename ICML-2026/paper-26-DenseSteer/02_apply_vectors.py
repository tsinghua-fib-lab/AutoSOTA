import os
import subprocess

# === CONFIGURATION ===
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
VECTOR_PATH = "./vectors/steering_vector.pt"
TASKS = "gsm8k_cot_zeroshot_unified"
LAYERS = [8, 13, 14]     # Specific layers to test
LAMBDAS = [1.0, -1.0]    # Steering strengths (Positive and Negative)
LIMIT = 1000             # Max samples to evaluate

def run_experiment(layer, lam):
    """Constructs and executes the lm-eval command."""
    tag = f"L{layer}_lam{str(lam).replace('.', 'p')}"
    out_path = f"./eval_results/{tag}"
    
    model_args = (
        f"pretrained={MODEL_ID},dtype=float16,"
        f"steer_layer={layer},steer_lambda={lam},"
        f"steer_vec_path={VECTOR_PATH}"
    )

    cmd = [
        "lm_eval", "--model", "steer_hf",
        "--model_args", model_args,
        "--tasks", TASKS,
        "--batch_size", "16",
        "--output_path", out_path,
        "--limit", str(LIMIT),
        "--apply_chat_template",
        "--log_samples"
    ]
    
    print(f"\n[RUNNING] Layer: {layer}, Alpha: {lam}")
    subprocess.run(cmd)

# === EXECUTION ===
os.makedirs("./eval_results", exist_ok=True)
for l in LAYERS:
    for lam in LAMBDAS:
        run_experiment(l, lam)

print("\nAll evaluation runs completed.")