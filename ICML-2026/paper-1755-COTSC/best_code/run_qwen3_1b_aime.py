"""Run MONICA evaluation on Qwen3-1.7B with AIME 2024 metadata leakage."""
import argparse, logging, sys, os
from pathlib import Path
from monica import DynamicMonitorSteerProcessor, generate_monica_answers
from monica_tools import *
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def main():
    parser = argparse.ArgumentParser(description="Run MONICA on Qwen3-1.7B with AIME")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--steer_layers", type=int, nargs="+", default=[16, 17, 18, 19])
    parser.add_argument("--monitor_layers", type=int, nargs="+", default=[16, 17, 18])
    parser.add_argument("--steer_min", type=float, default=-2)
    parser.add_argument("--steer_max", type=float, default=-4)
    parser.add_argument("--hs_tokens", type=int, default=5)
    parser.add_argument("--steer_layer_weights", type=float, nargs="+", default=None)
    parser.add_argument("--artifact_file", type=str, default="data/artifacts/qwen3_1b.monica")
    parser.add_argument("--datasets", type=str, nargs="+", default=["aime_2024_multichoice"])
    parser.add_argument("--cue_types", type=str, nargs="+", default=["metadata"])
    parser.add_argument("--steer_method", type=str, default="monica", choices=["monica", "default_steer"])
    parser.add_argument("--debug_topk", type=int, default=30)
    parser.add_argument("--max_tokens", type=int, default=16800)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--file_tag", type=str, default="qwen3_1b_aime_metadata")
    parser.add_argument("--model_tag", type=str, default="qwen3_1b")
    parser.add_argument("--vec_tag", type=str, default="latest")
    args = parser.parse_args()

    root = ROOT
    model_name_val = args.model_name
    run_cfg = {
        "file_tag": args.file_tag,
        "model_tag": args.model_tag,
        "vec_tag": args.vec_tag,
        "model_name": model_name_val,
        "steer_layers": args.steer_layers,
        "monitor_layers": args.monitor_layers,
        "steer_max": args.steer_max,
        "steer_min": args.steer_min,
        "hs_tokens": args.hs_tokens,
        "steer_layer_weights": args.steer_layer_weights,
        "artifact_file": args.artifact_file,
    }

    artifact_path = root / run_cfg["artifact_file"]
    if not artifact_path.exists():
        raise FileNotFoundError("Missing artifact file: {}".format(artifact_path))

    print("Loading MONICA artifact from {} ...".format(artifact_path))
    calibrator, monitor, _ = load_monica_artifact(artifact_path)
    calibrator_vec = {i: vec for i, vec in enumerate(calibrator)}
    print("Loaded calibrator with {} layers".format(len(calibrator_vec)))

    monitor_layers = [int(x) for x in run_cfg["monitor_layers"]]
    monitor_vec = {}
    for layer in monitor_layers:
        key = layer if layer in monitor else str(layer)
        if key in monitor:
            monitor_vec[layer] = load_probe(monitor[key])
    
    missing = [layer for layer in monitor_layers if layer not in monitor_vec]
    if missing:
        raise KeyError("Probe missing for layers: {}".format(missing))
    print("Loaded {} monitor probes".format(len(monitor_vec)))

    print("Loading tokenizer and model: {} ...".format(model_name_val))
    model_path = "/models/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print("Loading model on {} ({})...".format(device, dtype))

    base = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cuda" and not hasattr(base, "device"):
        base.to(device)
    base.eval()
    print("Model loaded.")

    model = steerLRM(base, [int(x) for x in run_cfg["steer_layers"]])
    punctuation_ids = get_punctuation_token_ids(tokenizer)
    print("Punctuation tokens: {}".format(len(punctuation_ids)))

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

    generate_monica_answers(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        calibrator_vec=calibrator_vec,
        lrm_config=run_cfg,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        datasets=args.datasets,
        cue_types=args.cue_types,
        steer_method=args.steer_method,
        max_new_tokens=args.max_tokens,
        debug_topk=args.debug_topk,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        start_idx=0,
        no_monitor_log=True,
    )

    print("Done. Outputs: {}".format(output_dir))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
