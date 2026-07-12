import sys, os, json, time, shutil
sys.path.insert(0, '/repo/src')
import torch, numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset as HFDataset, load_from_disk
from progressive_cramming.train.arguments import MyTrainingArguments
from progressive_cramming.train.trainers.progressive_cramming import ProgressiveCrammingTrainer

import argparse
p = argparse.ArgumentParser()
p.add_argument('--model_path', default='/models/pythia-1.4b')
p.add_argument('--output_dir', default='/autosota_cache/paper-4707-hellaswag-results-v2')
p.add_argument('--max_samples', type=int, default=None)
p.add_argument('--lr', type=float, default=0.5)
p.add_argument('--max_steps', type=int, default=5000)
p.add_argument('--max_steps_per_token', type=int, default=500)
p.add_argument('--warmup', type=int, default=100)
p.add_argument('--seed', type=int, default=42)
# --- Optimization flags ---
p.add_argument('--embedding_init_method', default='random0.02',
               choices=['random', 'random0.02', 'random0.002', 'random_norm', 'random_norm_0.02',
                        'mvnormal', 'zeros', 'random0.2', 'random5', 'neg_random', 'random_norm_0.2'])
p.add_argument('--attn_implementation', default='eager',
               choices=['eager', 'sdpa', 'flash_attention_2'])
p.add_argument('--loss_type', default='cross_entropy',
               choices=['cross_entropy', 'l2', 'l1', 'cosine'])
p.add_argument('--hybrid_alpha', type=float, default=None)
p.add_argument('--num_alignment_layers', type=int, default=0)
p.add_argument('--low_dim_projection', action='store_true', default=False)
p.add_argument('--low_dim_size', type=int, default=32)
p.add_argument('--low_dim_projection_train', action='store_true', default=True)
p.add_argument('--number_of_mem_tokens', type=int, default=1)
p.add_argument('--progressive_geometric_growth', action='store_true', default=False)
p.add_argument('--progressive_geometric_backoff', default='bisect',
               choices=['bisect', 'linear'])
p.add_argument('--leading_token_loss_weight', type=float, default=1.0)
p.add_argument('--leading_token_loss_count', type=int, default=0)
p.add_argument('--adam_beta2', type=float, default=0.9)
p.add_argument('--gradient_checkpointing', action='store_true', default=False)
args = p.parse_args()

torch.manual_seed(args.seed); np.random.seed(args.seed)
DEV = 'cuda:0'
os.makedirs(args.output_dir, exist_ok=True)
print('Loading model...')
model = AutoModelForCausalLM.from_pretrained(
    args.model_path, torch_dtype=torch.bfloat16, device_map=DEV,
    local_files_only=True, attn_implementation=args.attn_implementation)
tok = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
tok.pad_token = tok.eos_token
print(f'Model: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params')

print('Loading HellaSwag...')
dps = load_dataset('parquet', data_files={'validation': '/autosota_cache/hf/datasets/Rowan___hellaswag/data/validation-00000-of-000001.parquet'})
ds = dps['validation']
if args.max_samples:
    ds = ds.select(range(min(args.max_samples, len(ds))))
N = len(ds)
print(f'Samples: {N}')

persistent_tmp = os.path.join(args.output_dir, 'tmp_persistent')
ta = MyTrainingArguments(
    output_dir=persistent_tmp, model_checkpoint=args.model_path,
    learning_rate=args.lr, per_device_train_batch_size=1,
    max_optimization_steps_per_sample=args.max_steps,
    max_optimization_steps_per_token=args.max_steps_per_token,
    warmup_steps=args.warmup,
    progressive_train=True, progressive_step=1, progressive_min_seq_len=1,
    embedding_init_method=args.embedding_init_method,
    number_of_mem_tokens=args.number_of_mem_tokens,
    attn_implementation=args.attn_implementation, dtype='bfloat16', seed=args.seed,
    fix_position_ids=True, no_bos_token=False,
    loss_type=args.loss_type,
    hybrid_alpha=args.hybrid_alpha,
    num_alignment_layers=args.num_alignment_layers,
    low_dim_projection=args.low_dim_projection,
    low_dim_size=args.low_dim_size,
    low_dim_projection_train=args.low_dim_projection_train,
    progressive_geometric_growth=args.progressive_geometric_growth,
    progressive_geometric_backoff=args.progressive_geometric_backoff,
    leading_token_loss_weight=args.leading_token_loss_weight,
    leading_token_loss_count=args.leading_token_loss_count,
    adam_beta2=args.adam_beta2,
    gradient_checkpointing=args.gradient_checkpointing,
)

class Batch:
    def __init__(self, d):
        self.input_ids = torch.tensor(d['input_ids'])
        self.attention_mask = torch.tensor(d['attention_mask'])
def collate_fn(batch):
    return Batch({'input_ids': [item['input_ids'] for item in batch], 'attention_mask': [item['attention_mask'] for item in batch]})

results = []
conv_count = 0
correct_conv = 0
start_t = time.time()

for idx, sample in enumerate(tqdm(ds, desc='Eval')):
    ctx = sample.get('ctx', '')
    endings = sample.get('endings', [])
    label = int(sample.get('label', 0))
    if not ctx or len(endings) != 4:
        results.append({'idx': idx, 'converged': False})
        continue
    ctx_enc = tok.encode(ctx, add_special_tokens=False)
    if len(ctx_enc) < 1:
        results.append({'idx': idx, 'converged': False})
        continue

    ppath = os.path.join(persistent_tmp, 'progressive_prefixes')
    if os.path.exists(ppath):
        shutil.rmtree(ppath, ignore_errors=True)

    try:
        trainer = ProgressiveCrammingTrainer(model=model, processing_class=tok, args=ta)
        temp_ds = HFDataset.from_dict({'input_ids': [ctx_enc], 'attention_mask': [[1]*len(ctx_enc)]})
        trainer._create_dataloader = lambda: torch.utils.data.DataLoader(temp_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)
        trainer.writer = None
        save_path = trainer.train()

        if save_path:
            res_ds = load_from_disk(save_path)
            if len(res_ds) > 0:
                row = res_ds[-1]
                fc = row.get('final_convergence', 0.0)
                emb = torch.tensor(row['embedding'], device=DEV, dtype=torch.bfloat16)
                if fc >= 1.0 and emb.shape[0] >= 1:
                    conv_count += 1
                    nlls = []
                    for ending in endings:
                        full_text = ctx + ' ' + ending
                        full_enc = tok.encode(full_text, return_tensors='pt').to(DEV)
                        with torch.no_grad():
                            tok_emb = model.get_input_embeddings()(full_enc)
                        comp_emb = emb.unsqueeze(0)
                        united = torch.cat([comp_emb, tok_emb], dim=1)
                        am = torch.ones(1, united.size(1), device=DEV)
                        with torch.no_grad():
                            out = model(inputs_embeds=united, attention_mask=am)
                            logits = out.logits.float()
                        cl = len(ctx_enc)
                        cs = 1 + cl
                        sl = logits[:, cs-1:-1, :]
                        sb = full_enc[:, cl:]
                        if sb.numel() > 0:
                            lf = torch.nn.CrossEntropyLoss(reduction='sum')
                            ns = lf(sl.reshape(-1, sl.size(-1)), sb.reshape(-1))
                            nlls.append((ns / sb.numel()).item())
                        else:
                            nlls.append(float('inf'))
                    pred = int(np.argmin(nlls))
                    correct = (pred == label)
                    if correct:
                        correct_conv += 1
                    results.append({'idx': idx, 'converged': True, 'conv': fc, 'correct': correct, 'pred': pred, 'label': label})
                else:
                    results.append({'idx': idx, 'converged': False, 'conv': fc})
            else:
                results.append({'idx': idx, 'converged': False})
        else:
            results.append({'idx': idx, 'converged': False})
    except Exception as e:
        results.append({'idx': idx, 'converged': False, 'error': str(e)[:200]})

    if (idx+1) % 10 == 0:
        el = time.time() - start_t
        cp = conv_count/(idx+1)*100
        ac = correct_conv/conv_count*100 if conv_count>0 else 0
        tqdm.write(f'  [{idx+1}/{N}] Conv%={cp:.1f}% Acc={ac:.1f}% ({correct_conv}/{conv_count}) {el/60:.1f}min')

total = len(results)
cp = conv_count/total*100 if total>0 else 0
acc = correct_conv/conv_count*100 if conv_count>0 else 0
el = time.time() - start_t
print()
print('='*60)
print(f'Total: {total}, Converged: {conv_count}/{total} ({cp:.2f}%)')
print(f'Acc (converged): {correct_conv}/{conv_count} ({acc:.2f}%)')
print(f'Paper: Acc=37.63%, Conv%=97.07%')
print(f'CI Lower: Acc=36.99%')
print(f'Time: {el/60:.1f}min ({el/3600:.1f}hr)')
outf = os.path.join(args.output_dir, 'results.json')
with open(outf, 'w') as f:
    json.dump({'total': total, 'converged': conv_count, 'conv_pct': round(cp,2), 'correct_on_converged': correct_conv, 'acc': round(acc,2), 'paper_acc': 37.63, 'paper_conv_pct': 97.07, 'runtime_min': round(el/60,1), 'config': vars(args)}, f, indent=2)
print(f'Saved to {outf}')
