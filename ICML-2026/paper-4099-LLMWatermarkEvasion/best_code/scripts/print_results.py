"""Print BIRA attack results."""
import json

result_path = "/repo/experimental_results/BIRA/Llama-3.1-8B-Instruct/SIR/BIRA_beta_-4.0_percentile_50_num_data_500.json"

with open(result_path) as f:
    data = json.load(f)

detect = data[-1].get("detectability", {})
print("=== BIRA Attack Results (SIR watermark, Llama-3.1-8B-Instruct) ===")
print("TPR@FPR=1%:  {:.3f} (lower is better, paper: 0.012)".format(detect["tpr_target_fpr_0.01"]["TPR"]))
print("TPR@FPR=10%: {:.3f} (lower is better, paper: 0.114)".format(detect["tpr_target_fpr_0.1"]["TPR"]))
print("Best F1:     {:.3f} (lower is better, paper: 0.667)".format(detect["f1_best"]["F1"]))

results = [d for d in data if "is_watermarked" in d]
asr = sum(1 for d in results if not d["is_watermarked"]) / len(results)
print("ASR:         {:.1f}% (higher is better, paper: 99.6%)".format(asr * 100))
