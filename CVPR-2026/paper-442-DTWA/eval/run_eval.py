#!/usr/bin/env python3
import os, json, re, argparse, sys
import numpy as np, pandas as pd, torch
from PIL import Image; from io import BytesIO; from tqdm import tqdm
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default="/vepfs-mlp2/queue014/public/liyu/AutoSota-6/auto-pipeline-ab/workdir/gpu23/Autosota/models_g442/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--data_parquet", type=str, default="/vepfs-mlp2/queue014/public/liyu/AutoSota-6/auto-pipeline-ab/workdir/gpu23/Autosota/datasets_g442/HallusionBench/data/image-00000-of-00001_with_index.parquet")
    p.add_argument("--output_dir", type=str, default="/vepfs-mlp2/queue014/public/liyu/AutoSota-6/auto-pipeline-ab/optimizer/papers/paper-442/runs/run_20260604_111810/work/results")
    p.add_argument("--modify", type=str, default="modify_att"); p.add_argument("--max_new_tokens", type=int, default=2000)
    p.add_argument("--device", type=int, default=0); p.add_argument("--skip_eval", action="store_true")
    return p.parse_args()

def load_model(mp, dev):
    m = Qwen2_5_VLForConditionalGeneration.from_pretrained(mp, torch_dtype=torch.bfloat16, attn_implementation="eager").eval().to(dev)
    m._validate_model_kwargs = lambda mk: mk
    return m, AutoProcessor.from_pretrained(mp)

def load_data(pp):
    df = pd.read_parquet(pp); recs = df.to_dict(orient="records"); data = []
    for e in recs:
        eid = e.get("filename", f"{e.get('set_id','')}_{e.get('figure_id','')}_{e.get('question_id','')}")
        entry = {"id": eid, "question": e["question"], "answer": e.get("gt_answer_details", ""), "gt_answer": e.get("gt_answer", ""), "type": f"{e.get('category', '')}_{e.get('subcategory', '')}"}
        if "image" in e: entry["image"] = e["image"]
        elif "image_bytes" in e: entry["image"] = {"bytes": e["image_bytes"]}
        elif "bytes" in e: entry["image"] = {"bytes": e["bytes"]}
        data.append(entry)
    return data

def load_image(ii):
    if isinstance(ii, dict) and "bytes" in ii: return Image.open(BytesIO(ii["bytes"])).convert("RGB")
    elif isinstance(ii, bytes): return Image.open(BytesIO(ii)).convert("RGB")
    elif isinstance(ii, str): return Image.open(ii).convert("RGB")
    else: return ii

def generate_inputs(proc, model, ipath, q, modify=""):
    instruct = "The final answer MUST BE in <answer> </answer> tags."
    content = [{"type": "image", "image": load_image(ipath), "max_pixels": 1024 * 28 * 28 * 2}] if ipath else []
    content.append({"type": "text", "text": f"{q}\n{instruct}"})
    msgs = [{"role": "user", "content": content}]
    imgs, _ = process_vision_info(msgs) if ipath else (None, None)
    tq_text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[tq_text], images=imgs if ipath else None, padding=True, return_tensors="pt")
    if ipath:
        iids = inputs["input_ids"][0]
        vs = (iids == proc.tokenizer.convert_tokens_to_ids("<|vision_start|>")).nonzero(as_tuple=True)[0].item()
        ve = (iids == proc.tokenizer.convert_tokens_to_ids("<|vision_end|>")).nonzero(as_tuple=True)[0].item()
        qids = proc.tokenizer(q, add_special_tokens=False).input_ids
        qep = ve + len(qids)
        if modify:
            inputs[modify] = True; inputs["q_end_pos"] = qep
            inputs["vision_start"] = vs; inputs["vision_end"] = ve
            inputs["grid_w"] = inputs["image_grid_thw"][0][2].item() // 2
    return inputs.to(model.device)

def make_jsonable(obj):
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, torch.Tensor): return obj.detach().cpu().tolist()
    if isinstance(obj, (list, tuple)): return [make_jsonable(x) for x in obj]
    if isinstance(obj, dict): return {k: make_jsonable(v) for k, v in obj.items()}
    return obj

def get_output_text(proc, output, inputs):
    gids = output["sequences"]
    gids_t = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, gids)]
    return proc.batch_decode(gids_t, skip_special_tokens=True, clean_up_tokenization_spaces=False)

def run_inference(model, proc, data, odir, modify, mnt):
    os.makedirs(odir, exist_ok=True)
    tag = f"HallusionBench_Qwen2.5-VL-3B-Instruct_{modify}_maxNew{mnt}"
    sp = os.path.join(odir, f"{tag}.jsonl")
    hj = set()
    if os.path.exists(sp):
        with open(sp) as f:
            for line in f: hj.add(json.loads(line)["id"])
    total = len(data); print(f"Loaded {total} samples, {len(hj)} already judged")
    with tqdm(total=total, desc="Evaluating") as pbar:
        i = 0
        while i < total:
            s = data[i]
            if s.get("image") is None: i+=1; pbar.update(1); continue
            did = s["id"]
            if did in hj: i+=1; pbar.update(1); continue
            try:
                with torch.no_grad():
                    oi = generate_inputs(proc, model, s["image"], s["question"], modify=modify)
                    oo = model.generate(**oi, max_new_tokens=mnt, output_attentions=True, output_hidden_states=False, return_dict_in_generate=True)
                    ot = get_output_text(proc, oo, oi)
                r = {"id": did, "question": s["question"], "ori_response": ot, "answer": s["answer"], "gt_answer": s.get("gt_answer", ""), "type": s.get("type", "")}
                rs = make_jsonable(r)
                with open(sp, "a") as f: json.dump(rs, f); f.write("\n")
            except Exception as exc:
                print(f"\nError on {did}: {exc}")
                r = {"id": did, "question": s["question"], "ori_response": [f"ERROR: {str(exc)}"], "answer": s["answer"], "gt_answer": s.get("gt_answer", ""), "type": s.get("type", "")}
                with open(sp, "a") as f: json.dump(make_jsonable(r), f); f.write("\n")
            i+=1; pbar.update(1)
    print(f"\nResults saved to {sp}"); return sp

def score_results(rp):
    results = [json.loads(l) for l in open(rp)]
    total = len(results); correct = 0; irr_scores = []
    for r in results:
        resp = r.get("ori_response", [""])[0] if isinstance(r.get("ori_response"), list) else r.get("ori_response", "")
        gt = str(r.get("gt_answer", r.get("answer", ""))).strip()
        gtv = 1 if gt in ("yes","1","true") else (0 if gt in ("no","0","false") else -1)
        am = re.search(r'<answer>\s*(Yes|No|yes|no|1|0)\s*</answer>', resp, re.IGNORECASE)
        if am: pred = 1 if am.group(1).strip().lower() in ("yes","1") else 0
        else:
            ym = re.search(r'\byes\b', resp, re.IGNORECASE); nm = re.search(r'\bno\b', resp, re.IGNORECASE)
            pred = 1 if (ym and not nm) else (0 if (nm and not ym) else -1)
        if pred >= 0 and gtv >= 0 and pred == gtv: correct += 1
        acm = re.search(r'<answer>(.*?)</answer>', resp, re.DOTALL)
        if acm:
            at = acm.group(1); tt = len(resp.split()); atk = len(at.split())
            irr = 1.0 - (atk / max(tt, 1))
        else: irr = 1.0
        irr_scores.append(irr)
    acc = (correct / total * 100) if total > 0 else 0.0
    i_s = np.mean(irr_scores) if irr_scores else 1.0
    s_s = (acc / 100.0) * (1.0 - i_s)
    return acc, s_s, i_s

if __name__ == "__main__":
    args = parse_args(); dev = f"cuda:{args.device}"
    print(f"Device: {dev}\nModel: {args.model_path}\nData: {args.data_parquet}\nModify: {args.modify}")
    if not args.skip_eval:
        m, p = load_model(args.model_path, dev); d = load_data(args.data_parquet)
        rp = run_inference(m, p, d, args.output_dir, args.modify, args.max_new_tokens)
    else:
        tag = f"HallusionBench_Qwen2.5-VL-3B-Instruct_{args.modify}_maxNew{args.max_new_tokens}"
        rp = os.path.join(args.output_dir, f"{tag}.jsonl")
        if not os.path.exists(rp): print(f"Results not found: {rp}"); sys.exit(1)
    acc, ss, ii = score_results(rp)
    print(f"\n{'='*50}\nACC: {acc:.2f}\nS: {ss:.4f}\nI: {ii:.4f}\n{'='*50}")
