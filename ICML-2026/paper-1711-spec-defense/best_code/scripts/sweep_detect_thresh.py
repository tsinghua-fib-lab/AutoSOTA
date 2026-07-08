"""
Sensitivity sweep for CSR detection thresholds on adversarial samples.

Runs CLIP zero-shot evaluation with CSR defense enabled across a threshold
range and saves structured CSV / JSON / Markdown summaries.

By default this script uses one worker subprocess per GPU and lets each worker
evaluate a subset of thresholds sequentially, reusing the loaded model and
text features for better throughput.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights
from transformers import CLIPModel, CLIPProcessor

from csr.config import CSRConfig, HF_MODEL_CONFIGS, PathConfig
from csr.data.collate import default_collate_fn
from csr.data.dataset import CLIPDataset
from csr.defense import CSRDefense, FastCSRDefense
from csr.evaluation.evaluator import ZeroShotEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep CSR detect_thresh on adversarial samples")
    parser.add_argument("--model", type=str, default="CLIP-B-16")
    parser.add_argument("--defense", type=str, default="csr", choices=["none", "csr", "fast_csr"])
    parser.add_argument("--dataset", type=str, default="General/ImageNet")
    parser.add_argument("--attack", type=str, default="APGD")
    parser.add_argument("--adv_root", type=str, default="./outputs/adv_samples/APGD")
    parser.add_argument("--epsilon", type=str, default="4_255")
    parser.add_argument("--input_mode", type=str, default="adv", choices=["adv", "clean"])
    parser.add_argument("--sample_n", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lpf_radius", type=int, default=40)
    parser.add_argument("--purify_steps", type=int, default=3)
    parser.add_argument("--filter_type", type=str, default="gaussian")
    parser.add_argument("--threshold_start", type=float, default=0.75)
    parser.add_argument("--threshold_end", type=float, default=0.90)
    parser.add_argument("--threshold_step", type=float, default=0.01)
    parser.add_argument("--thresholds", type=float, nargs="+", default=None)
    parser.add_argument("--gpus", type=int, nargs="+", default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--worker_output", type=str, default=None)
    parser.add_argument("--worker_thresholds", type=float, nargs="+", default=None)
    parser.add_argument("--physical_gpu", type=int, default=None)
    return parser.parse_args()


def build_thresholds(args):
    if args.thresholds:
        return [round(float(x), 4) for x in args.thresholds]

    thresholds = []
    current = args.threshold_start
    while current <= args.threshold_end + 1e-9:
        thresholds.append(round(current, 4))
        current += args.threshold_step
    return thresholds


def default_output_dir(args):
    if args.output_dir:
        return args.output_dir

    return os.path.join(
        "./results",
        "threshold_sweeps",
    )


def format_float_tag(value):
    return f"{value:.2f}".replace(".", "p")


def build_output_prefix(args, thresholds):
    out_dir = default_output_dir(args)
    os.makedirs(out_dir, exist_ok=True)
    start = min(thresholds)
    end = max(thresholds)
    step = args.threshold_step if args.thresholds is None else 0.0
    dataset_tag = args.dataset.replace("/", "_").lower()
    attack_tag = args.attack.lower()
    defense_tag = args.defense.lower()
    if args.input_mode == "clean":
        run_tag = "clean"
    else:
        run_tag = f"{attack_tag}_{args.epsilon}"
    if args.defense == "none":
        defense_variant_tag = "none"
    else:
        defense_variant_tag = f"{defense_tag}_p{args.purify_steps}"
    return os.path.join(
        out_dir,
        (
            f"{args.model.lower().replace('-', '_')}_{dataset_tag}_{run_tag}_"
            f"{defense_variant_tag}_thresh_{format_float_tag(start)}_{format_float_tag(end)}_"
            f"step_{format_float_tag(step)}"
        ),
    )


def get_dataset_and_classes(args):
    paths = PathConfig()
    dataset_root = os.path.join(paths.data_root, args.dataset)
    csv_path = os.path.join(dataset_root, "labels.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Label file not found: {csv_path}")

    if args.input_mode == "clean":
        input_dir = os.path.join(dataset_root, "images")
    else:
        input_dir = os.path.join(args.adv_root, args.model, args.dataset, args.epsilon)
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Adversarial directory not found: {input_dir}")

    if "ImageNet" in args.dataset:
        all_classes = ResNet50_Weights.DEFAULT.meta["categories"]
    else:
        df = pd.read_csv(csv_path)
        all_classes = sorted(list(set(df.iloc[:, 1].values)))

    dataset = CLIPDataset(
        csv_path=csv_path,
        img_root=input_dir,
        dataset_name=args.dataset,
        sample_n=args.sample_n,
        all_classes=all_classes,
    )
    return dataset, all_classes, csv_path, input_dir


def load_defense(args, device):
    model_path = HF_MODEL_CONFIGS[args.model]
    if args.defense == "none":
        model = CLIPModel.from_pretrained(model_path, local_files_only=True).to(device).eval()
        processor = CLIPProcessor.from_pretrained(model_path, local_files_only=True)
        for param in model.parameters():
            param.requires_grad = False
        return model, processor

    csr_cfg = CSRConfig(
        filter_type=args.filter_type,
        lpf_radius=args.lpf_radius,
        detect_thresh=0.85,
        purify_steps=args.purify_steps,
    )
    DefClass = FastCSRDefense if args.defense == "fast_csr" else CSRDefense
    model = DefClass(model_path, config=csr_cfg, device=device)
    processor = model.processor

    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    return model, processor


@torch.no_grad()
def evaluate_threshold(model, processor, dataloader, text_features, threshold, device, physical_gpu=None):
    model.eval()
    if hasattr(model, "config") and hasattr(model.config, "detect_thresh"):
        model.config.detect_thresh = float(threshold)

    correct = 0
    total = 0
    detected = 0
    start_time = time.time()

    for batch_imgs, batch_labels, _ in dataloader:
        if batch_imgs is None:
            continue

        batch_labels = batch_labels.to(device)
        use_raw_pixel_values = hasattr(model, "process_images") or hasattr(model, "process_images_and_feats")
        processor_kwargs = {"images": batch_imgs, "return_tensors": "pt"}
        if use_raw_pixel_values:
            processor_kwargs["do_normalize"] = False
        inputs = processor(**processor_kwargs).to(device)
        image_features = model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

        similarity = 100.0 * image_features @ text_features.T
        predictions = similarity.argmax(dim=-1)

        correct += (predictions == batch_labels).sum().item()
        total += batch_labels.shape[0]

        detection_mask = getattr(model, "last_detection_mask", None)
        if detection_mask is not None:
            detected += detection_mask.sum().item()

    elapsed_sec = time.time() - start_time
    accuracy = correct / total if total else 0.0
    detection_rate = detected / total if total else 0.0

    return {
        "threshold": float(threshold),
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100.0,
        "correct": int(correct),
        "detected": int(detected),
        "detection_rate": detection_rate,
        "detection_rate_percent": detection_rate * 100.0,
        "elapsed_sec": round(elapsed_sec, 3),
        "samples": int(total),
        "device": str(device),
        "physical_gpu": physical_gpu,
    }


def run_worker(args):
    thresholds = [round(float(x), 4) for x in (args.worker_thresholds or [])]
    if not thresholds:
        raise ValueError("Worker received no thresholds")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    physical_gpu = args.physical_gpu
    dataset, all_classes, _, input_dir = get_dataset_and_classes(args)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=default_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    model, processor = load_defense(args, device)
    evaluator = ZeroShotEvaluator(model, processor, device=device, batch_size=args.batch_size)
    text_features = evaluator.compute_text_features(all_classes)

    rows = []
    worker_start = time.time()

    for threshold in thresholds:
        print(
            f"[Worker pid={os.getpid()} gpu={physical_gpu} device={device}] threshold={threshold:.2f} "
            f"attack={args.attack} epsilon={args.epsilon}"
        )
        row = evaluate_threshold(
            model,
            processor,
            dataloader,
            text_features,
            threshold,
            device,
            physical_gpu=physical_gpu,
        )
        row.update(
            {
                "model": args.model,
                "defense": args.defense,
                "dataset": args.dataset,
                "attack": args.attack if args.input_mode == "adv" else "Clean",
                "epsilon": args.epsilon if args.input_mode == "adv" else "-",
                "adv_root": input_dir if args.input_mode == "adv" else "",
                "input_mode": args.input_mode,
                "input_root": input_dir,
                "lpf_radius": args.lpf_radius,
                "purify_steps": args.purify_steps,
                "filter_type": args.filter_type,
            }
        )
        rows.append(row)
        print(
            f"[Worker pid={os.getpid()} gpu={physical_gpu} device={device}] threshold={threshold:.2f} "
            f"acc={row['accuracy_percent']:.2f}% det={row['detection_rate_percent']:.2f}% "
            f"elapsed={row['elapsed_sec']:.2f}s"
        )

    payload = {
        "worker_pid": os.getpid(),
        "device": device,
        "physical_gpu": physical_gpu,
        "thresholds": thresholds,
        "elapsed_sec": round(time.time() - worker_start, 3),
        "results": rows,
    }

    if not args.worker_output:
        raise ValueError("Worker output path is required")

    with open(args.worker_output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def choose_gpus(args, thresholds):
    if args.gpus:
        return args.gpus

    if not torch.cuda.is_available():
        return []

    gpu_count = torch.cuda.device_count()
    return list(range(min(gpu_count, len(thresholds))))


def partition_thresholds(thresholds, gpus):
    chunks = {gpu: [] for gpu in gpus}
    for idx, threshold in enumerate(thresholds):
        gpu = gpus[idx % len(gpus)]
        chunks[gpu].append(threshold)
    return chunks


def launch_workers(args, thresholds, output_prefix):
    gpus = choose_gpus(args, thresholds)
    if not gpus:
        raise RuntimeError("No GPU available for threshold sweep")

    assignments = partition_thresholds(thresholds, gpus)
    temp_dir = tempfile.mkdtemp(prefix="csr_thresh_", dir=os.path.dirname(output_prefix))

    processes = []
    partial_files = []

    for gpu, chunk in assignments.items():
        if not chunk:
            continue

        worker_output = os.path.join(temp_dir, f"gpu{gpu}.json")
        partial_files.append(worker_output)

        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--worker",
            "--model", args.model,
            "--defense", args.defense,
            "--dataset", args.dataset,
            "--attack", args.attack,
            "--adv_root", args.adv_root,
            "--epsilon", args.epsilon,
            "--input_mode", args.input_mode,
            "--sample_n", str(args.sample_n),
            "--batch_size", str(args.batch_size),
            "--num_workers", str(args.num_workers),
            "--lpf_radius", str(args.lpf_radius),
            "--purify_steps", str(args.purify_steps),
            "--filter_type", args.filter_type,
            "--physical_gpu", str(gpu),
            "--worker_output", worker_output,
            "--worker_thresholds",
        ] + [f"{x:.4f}" for x in chunk]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        log_path = f"{worker_output}.log"
        log_file = open(log_path, "w", encoding="utf-8")
        process = subprocess.Popen(
            cmd,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        processes.append((gpu, chunk, process, log_file, log_path))

    failures = []
    for gpu, chunk, process, log_file, log_path in processes:
        return_code = process.wait()
        log_file.close()
        if return_code != 0:
            failures.append((gpu, chunk, log_path, return_code))

    if failures:
        details = []
        for gpu, chunk, log_path, code in failures:
            details.append(
                f"gpu={gpu}, thresholds={chunk}, exit_code={code}, log={log_path}"
            )
        raise RuntimeError("Worker failed: " + " | ".join(details))

    return partial_files


def load_partial_results(partial_files):
    rows = []
    for path in partial_files:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        rows.extend(payload.get("results", []))
    rows.sort(key=lambda x: x["threshold"])
    return rows


def save_outputs(args, thresholds, rows, output_prefix):
    df = pd.DataFrame(rows)
    df = df.sort_values("threshold").reset_index(drop=True)

    summary = {
        "model": args.model,
        "defense": args.defense,
        "dataset": args.dataset,
        "attack": args.attack if args.input_mode == "adv" else "Clean",
        "epsilon": args.epsilon if args.input_mode == "adv" else "-",
        "input_mode": args.input_mode,
        "sample_n": args.sample_n,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "lpf_radius": args.lpf_radius,
        "purify_steps": args.purify_steps,
        "filter_type": args.filter_type,
        "thresholds": thresholds,
        "gpus_used": sorted({row["physical_gpu"] for row in rows if row.get("physical_gpu") is not None}),
        "best_accuracy_threshold": None,
        "best_detection_threshold": None,
        "results": rows,
    }

    if not df.empty:
        best_acc_row = df.sort_values(["accuracy", "detection_rate"], ascending=[False, False]).iloc[0]
        best_det_row = df.sort_values(["detection_rate", "accuracy"], ascending=[False, False]).iloc[0]
        summary["best_accuracy_threshold"] = float(best_acc_row["threshold"])
        summary["best_detection_threshold"] = float(best_det_row["threshold"])

    csv_path = f"{output_prefix}.csv"
    json_path = f"{output_prefix}.json"
    md_path = f"{output_prefix}.md"

    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    display_df = df[
        [
            "threshold",
            "accuracy",
            "accuracy_percent",
            "detected",
            "detection_rate",
            "detection_rate_percent",
            "correct",
            "samples",
            "elapsed_sec",
            "device",
            "physical_gpu",
        ]
    ].copy()
    display_df["threshold"] = display_df["threshold"].map(lambda x: f"{x:.2f}")
    display_df["accuracy"] = display_df["accuracy"].map(lambda x: f"{x:.4f}")
    display_df["accuracy_percent"] = display_df["accuracy_percent"].map(lambda x: f"{x:.2f}")
    display_df["detection_rate"] = display_df["detection_rate"].map(lambda x: f"{x:.4f}")
    display_df["detection_rate_percent"] = display_df["detection_rate_percent"].map(lambda x: f"{x:.2f}")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# CSR Threshold Sensitivity Sweep\n\n")
        f.write(f"- Model: `{args.model}`\n")
        f.write(f"- Defense: `{args.defense}`\n")
        f.write(f"- Dataset: `{args.dataset}`\n")
        f.write(f"- Attack: `{args.attack}`\n")
        f.write(f"- Epsilon: `{args.epsilon}`\n")
        f.write(f"- Samples: `{args.sample_n}`\n")
        f.write(f"- LPF radius: `{args.lpf_radius}`\n")
        f.write(f"- Purify steps: `{args.purify_steps}`\n\n")
        f.write(display_df.to_markdown(index=False))
        f.write("\n")

    return csv_path, json_path, md_path


def main():
    args = parse_args()

    if args.worker:
        run_worker(args)
        return

    thresholds = build_thresholds(args)
    if not thresholds:
        raise ValueError("No thresholds to evaluate")

    output_prefix = build_output_prefix(args, thresholds)
    overall_start = time.time()
    partial_files = launch_workers(args, thresholds, output_prefix)
    rows = load_partial_results(partial_files)
    csv_path, json_path, md_path = save_outputs(args, thresholds, rows, output_prefix)

    print(f"[Sweep] Finished in {time.time() - overall_start:.2f}s")
    print(f"[Sweep] CSV  : {csv_path}")
    print(f"[Sweep] JSON : {json_path}")
    print(f"[Sweep] MD   : {md_path}")

    if rows:
        best_acc = max(rows, key=lambda x: (x["accuracy"], x["detection_rate"]))
        best_det = max(rows, key=lambda x: (x["detection_rate"], x["accuracy"]))
        print(
            f"[Sweep] Best accuracy threshold={best_acc['threshold']:.2f}, "
            f"acc={best_acc['accuracy_percent']:.2f}%, det={best_acc['detection_rate_percent']:.2f}%"
        )
        print(
            f"[Sweep] Best detection threshold={best_det['threshold']:.2f}, "
            f"acc={best_det['accuracy_percent']:.2f}%, det={best_det['detection_rate_percent']:.2f}%"
        )


if __name__ == "__main__":
    main()
