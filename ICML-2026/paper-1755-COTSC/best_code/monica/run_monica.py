import argparse
import logging
import sys
from pathlib import Path
from monica import DynamicMonitorSteerProcessor, generate_monica_answers
from monica_tools import *
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))



def main() -> int:
    parser = argparse.ArgumentParser(description="Run Monica in camera_ready")
    parser.add_argument("--file_tag", type=str, default="deepseek_llama8b_latest")
    parser.add_argument("--model_tag", type=str, default="deepseek_llama8b")
    parser.add_argument("--vec_tag", type=str, default="latest")
    parser.add_argument("--model_name", type=str, default="unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit")
    parser.add_argument("--steer_layers", type=int, nargs="+", default=[21, 22, 23, 24, 25, 26])
    parser.add_argument("--monitor_layers", type=int, nargs="+", default=[21, 22, 23])
    parser.add_argument("--steer_max", type=float, default=-4)
    parser.add_argument("--steer_min", type=float, default=-2)
    parser.add_argument("--hs_tokens", type=int, default=5)
    parser.add_argument("--artifact_file", type=str, default="data/artifacts/deepseek_llama8b.monica")
    parser.add_argument("--datasets", type=str, nargs="+", default=["mmlu_moral_scenarios"], choices=["gpqa_main", "mmlu_moral_scenarios", "aime_2024_multichoice", "aime_2025_multichoice", "all"])
    parser.add_argument("--cue_types", type=str, nargs="+", default=["all"], choices=["user_suggestion", "tick_mark", "prefilled_wrong_answer", "metadata", "validation_function", "unauthorized_access", "all"])
    parser.add_argument("--steer_method", type=str, default="monica", choices=["monica", "default_steer"])
    parser.add_argument("--debug_topk", type=int, default=1)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=16800)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--no_monitor_log", action="store_true")
    args = parser.parse_args()

    root = ROOT
    run_cfg = {"file_tag": args.file_tag,"model_tag": args.model_tag, "vec_tag": args.vec_tag, "model_name": args.model_name,"steer_layers": args.steer_layers, "monitor_layers": args.monitor_layers, "steer_max": args.steer_max, "steer_min": args.steer_min,"hs_tokens": int(args.hs_tokens),"artifact_file": args.artifact_file,}
    artifact_path = root / run_cfg["artifact_file"]
    if not artifact_path.exists():
        raise FileNotFoundError(f"Missing artifact file: {artifact_path}")

    calibrator, monitor, _ = load_monica_artifact(artifact_path)
    calibrator_vec = {i: vec for i, vec in enumerate(calibrator)}

    monitor_layers = [int(x) for x in run_cfg["monitor_layers"]]
    monitor_vec = {}
    for layer in monitor_layers:
        key = layer if layer in monitor else str(layer)
        if key in monitor:
            monitor_vec[layer] = load_probe(monitor[key])

    missing = [layer for layer in monitor_layers if layer not in monitor_vec]
    if missing:
        raise KeyError(f"Probe missing for layers: {missing}")

    tokenizer = AutoTokenizer.from_pretrained(run_cfg["model_name"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    logging.info("Loading model %s on %s (%s)", run_cfg["model_name"], device, dtype)

    base = AutoModelForCausalLM.from_pretrained(run_cfg["model_name"], torch_dtype=dtype)
    base.to(device).eval()

    model = steerLRM(base, [int(x) for x in run_cfg["steer_layers"]])
    punctuation_ids = get_punctuation_token_ids(tokenizer)

    processor = DynamicMonitorSteerProcessor(
        model_wrapper=model,
        monitor_vec=monitor_vec,
        calibrator_vec=calibrator_vec,
        lrm_config=run_cfg,
        monitor_layers=monitor_layers,
        prompt_len=0,
        punctuation_token_ids=punctuation_ids,
        hs_tokens=run_cfg["hs_tokens"],
        tokenizer=tokenizer,
    )

    dataset_dir = root / "data/expData"
    output_dir = root / "outputs" / run_cfg["file_tag"]

    selected_datasets = args.datasets
    if "all" in selected_datasets:
        selected_datasets = [ "gpqa_main", "mmlu_moral_scenarios","aime_2024_multichoice","aime_2025_multichoice",]

    generate_monica_answers(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        calibrator_vec=calibrator_vec,
        lrm_config=run_cfg,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        datasets=selected_datasets,
        cue_types=args.cue_types,
        steer_method=args.steer_method,
        max_new_tokens=args.max_tokens,
        debug_topk=args.debug_topk,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        no_monitor_log=args.no_monitor_log,
        start_idx=args.start_idx,
    )

    print(f"Done. Outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
