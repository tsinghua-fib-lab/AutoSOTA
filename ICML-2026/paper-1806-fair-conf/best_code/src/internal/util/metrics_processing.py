from typing import Any

from internal.util.writer import Writer


def process_marginal_metric(
    metrics_marg: dict[str, Any], writer: Writer, cfg: dict
) -> int:
    metrics_per_label_group = {
        k: v
        for k, v in metrics_marg.items()
        if k in ["top1_acc_per_label", "top1_acc_per_group", "disparate_impact_acc"]
    }
    metrics_marg = {
        k: v
        for k, v in metrics_marg.items()
        if k not in ["top1_acc_per_label", "top1_acc_per_group", "disparate_impact_acc"]
    }

    metrics_marg["alpha"] = round(cfg["alpha"], 3)
    writer.write_json("metrics_marginal", metrics_marg)

    print(" ")
    print("=============================")
    print(" ")
    print("Marginal metrics:")
    print(f"alpha = {metrics_marg['alpha']}")
    print(f"Cvg@1 = {round(metrics_marg['top1'], 3)}")
    print(f"Cvg@k = {round(metrics_marg['topk'], 3)}")
    print(f"Cvg   = {round(metrics_marg['coverage'], 3)}")
    print(f"Size  = {round(metrics_marg['size'], 3)}")
    print(f"ECE@1  = {round(metrics_marg['ece'], 3)}")
    print(f"TPR@1  = {round(metrics_marg['tpr'], 3)}")
    print(f"FPR@1  = {round(metrics_marg['fpr'], 3)}")

    if metrics_per_label_group:
        writer.write_json("metrics_per_label_group", metrics_per_label_group)

        if "top1_acc_per_label" in metrics_per_label_group:
            print("\nCvg@1 per label:")
            for label, acc in metrics_per_label_group["top1_acc_per_label"].items():
                print(f"{label}: {round(acc, 3)}")
        if "top1_acc_per_group" in metrics_per_label_group:
            print("\nCvg@1 per group:")
            for group, acc in metrics_per_label_group["top1_acc_per_group"].items():
                print(f"{group}: {round(acc, 3)}")
        if "disparate_impact" in metrics_per_label_group:
            print(f"\nDisparate Impact: {metrics_per_label_group['disparate_impact']}")

    return int(metrics_marg["size"])


def process_conditional_metric(metrics_cond: dict[str, Any], writer: Writer, cfg: dict):
    metrics_cond["alpha"] = round(cfg["alpha"], 3)
    writer.write_json("metrics_conditional", metrics_cond)

    print(" ")
    print("=============================")
    print(" ")
    print("Conditional metrics:")
    print(f"alpha = {metrics_cond['alpha']}")
    print(f"Cvg@1 = {round(metrics_cond['top1'], 3)}")
    print(f"Cvg@k = {round(metrics_cond['topk'], 3)}")
    print(f"Cvg   = {round(metrics_cond['coverage'], 3)}")
    print(f"Size  = {round(metrics_cond['size'], 3)}")
    print(f"ECE@1  = {round(metrics_cond['ece'], 3)}")
    print(f"TPR@1  = {round(metrics_cond['tpr'], 3)}")
    print(f"FPR@1  = {round(metrics_cond['fpr'], 3)}")


def process_avg_k_metric(
    metrics_avgk: dict[str, Any], writer: Writer, cfg: dict, k_avgk
):
    metrics_avgk["alpha"] = round(cfg["alpha"], 3)
    writer.write_json("metrics_avgk", metrics_avgk)

    print(" ")
    print("=============================")
    print(" ")
    print("Average-K metrics:")
    print(f"k_avgk = {k_avgk}")
    print(f"Cvg@1 = {round(metrics_avgk['top1'], 3)}")
    print(f"Cvg@k = {round(metrics_avgk['topk'], 3)}")
    print(f"Cvg   = {round(metrics_avgk['coverage'], 3)}")
    print(f"Size  = {round(metrics_avgk['size'], 3)}")
    print(f"ECE@1  = {round(metrics_avgk['ece'], 3)}")
    print(f"TPR@1  = {round(metrics_avgk['tpr'], 3)}")
    print(f"FPR@1  = {round(metrics_avgk['fpr'], 3)}")

def process_backward_metric(metrics_back: dict[str, Any], writer: Writer, cfg: dict):
    metrics_back["alpha"] = round(cfg["alpha"], 3)
    writer.write_json("metrics_back", metrics_back)

    print(" ")
    print("=============================")
    print(" ")
    print("Backward metrics:")
    print(f"Cvg@1        = {round(metrics_back['top1'], 3)}")
    print(f"Cvg@k        = {round(metrics_back['topk'], 3)}")
    print(f"Cvg          = {round(metrics_back['coverage'], 3)}")
    # print(f"Calib Cvg    = {round(metrics_back['empirical_calib_coverage'], 3)}")
    # print(f"Est LOO Cvg  = {round(metrics_back['loo_coverage'], 3)}")
    # print(f"Est test Cvg = {round(metrics_back['avg_test_cvg'], 3)}")
    print(f"Size         = {round(metrics_back['size'], 3)}")
    print(f"ECE@1        = {round(metrics_back['ece'], 3)}")
    print(f"TPR@1        = {round(metrics_back['tpr'], 3)}")
    print(f"FPR@1        = {round(metrics_back['fpr'], 3)}")

def process_clustered_metric(metrics_clustered: dict[str, Any], writer: Writer, cfg: dict, method: str):
    metrics_clustered["alpha"] = round(cfg["alpha"], 3)
    writer.write_json(f"metrics_{method}", metrics_clustered)
    print(f"{method.replace('_', ' ').title()} metrics:")
    print(f"alpha = {metrics_clustered['alpha']}")
    print(f"Cvg@1 = {round(metrics_clustered['top1'], 3)}")
    print(f"Cvg@k = {round(metrics_clustered['topk'], 3)}")
    print(f"Cvg   = {round(metrics_clustered['coverage'], 3)}")
    print(f"Size  = {round(metrics_clustered['size'], 3)}")
    print(f"ECE@1  = {round(metrics_clustered['ece'], 3)}")
    print(f"TPR@1  = {round(metrics_clustered['tpr'], 3)}")
    print(f"FPR@1  = {round(metrics_clustered['fpr'], 3)}")

