import time
import argparse
import pickle
import random
import os
import numpy as np
import torch
from tqdm import tqdm
from model import SASRec, trans_to_cuda, trans_to_cpu
from utils import Data, get_num_node_unique
from log import setup_logger, redirect_tqdm_to_console
from coral import CoralAgent

def init_seed(seed=None):
    if seed is None: seed = int(time.time() * 1000 // 1000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if torch.cuda.is_available():
    torch.cuda.set_device(0)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='ML1m')
parser.add_argument('--hiddenSize', type=int, default=100)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_dc', type=float, default=0.1)
parser.add_argument('--lr_dc_step', type=int, default=3)
parser.add_argument('--l2', type=float, default=1e-5)
parser.add_argument('--n_layers', type=int, default=2)
parser.add_argument('--n_heads', type=int, default=2)
parser.add_argument('--dropout_local', type=float, default=0.2)
parser.add_argument('--max_len', type=int, default=200)
parser.add_argument('--mask_prob', type=float, default=0.2)
parser.add_argument('--validation', action='store_true')
parser.add_argument('--valid_portion', type=float, default=0.1)
parser.add_argument('--patience', type=int, default=10)

# CORAL Args
parser.add_argument('--no_coral', action='store_true', help='Disable CORAL pipeline')
parser.add_argument('--lambda_max', type=float, default=0.7) 
parser.add_argument('--kappa', type=float, default=1)
parser.add_argument('--rho', type=float, default=2)
parser.add_argument('--tau', type=float, default=2)
parser.add_argument('--delta_conf', type=float, default=0.1)

opt = parser.parse_args()

def build_maps(dataset_raw):
    item_to_cat = {}
    cat_to_items = {}
    seqs, _, cats, _, _, _ = dataset_raw
    for s, c in zip(seqs, cats):
        for item_id, cat_id in zip(s, c):
            if item_id != 0:
                item_to_cat[item_id] = cat_id
                if cat_id not in cat_to_items:
                    cat_to_items[cat_id] = set()
                cat_to_items[cat_id].add(item_id)
    for c in cat_to_items:
        cat_to_items[c] = list(cat_to_items[c])
    num_cats = max(item_to_cat.values()) + 1
    return item_to_cat, cat_to_items, num_cats

def forward_train(model, data):
    from model import forward
    return forward(model, data, mode='train')

def analyze_tail_risk_stratified(records, metric_keys, percentiles=[0.20, 0.10, 0.05]):
    """
    records: List of dicts [{'uid': int, 'risk_val': float, 'metrics': {...}}, ...]
    """
    if not records:
        return {}
    
    # Sort by Risk Value descending (Highest HD first -> Most objective risk)
    sorted_records = sorted(records, key=lambda x: x['risk_val'], reverse=True)
    n = len(sorted_records)
    
    summary = {}
    def is_user_max_metric(k):
        # Saturation metrics use "Max per User" then "Avg over Users"
        return k.startswith('Sat_')

    def get_metric_val(subset_records, key):
        if is_user_max_metric(key):
            # Group by User -> Max -> Mean
            user_max_val = {}
            for r in subset_records:
                uid = r['uid']
                val = r['metrics'].get(key, 0.0)
                if uid not in user_max_val:
                    user_max_val[uid] = val
                else:
                    user_max_val[uid] = max(user_max_val[uid], val)
            if not user_max_val:
                return 0.0
            return np.mean(list(user_max_val.values()))
        else:
            # Default: Mean over all moments
            vals = [r['metrics'][key] for r in subset_records]
            return np.mean(vals)

    # Global Mean
    for k in metric_keys:
        summary[f"Global_{k}"] = get_metric_val(sorted_records, k)
        
    # Percentiles
    for p in percentiles:
        cutoff = int(np.ceil(n * p))
        cutoff = max(1, cutoff)
        subset = sorted_records[:cutoff]
        avg_risk = np.mean([x['risk_val'] for x in subset])
        
        for k in metric_keys:
            summary[f"Top{int(p*100)}%_Risk_{k}"] = get_metric_val(subset, k)
        summary[f"Top{int(p*100)}%_AvgHD"] = avg_risk
            
    return summary

def evaluate_dynamic_stream(model, test_data, agent=None, shadow_agent=None, cat_to_items=None, item_to_cat=None, num_cats=0, desc="Eval"):
    """
    Evaluation with Objective Risk Stratification (History Dominance)
    AND Triggered-Only Analysis (Counterfactual Comparison: CORAL vs Baseline).
    """
    model.eval()
    loader = torch.utils.data.DataLoader(test_data, num_workers=4, batch_size=opt.batch_size, shuffle=False)

    k_list = [5, 10]
    MIN_HISTORY_LEN = 4 
    
    moment_records = []
    
    # Additional list to store ONLY triggered moments for direct impact analysis
    triggered_moments_stats = []
    
    active_agent = agent if agent is not None else shadow_agent
    is_intervention_active = (agent is not None)

    debug_stats = {"trigger": 0, "success": 0, "total_steps": 0}

    with torch.no_grad():
        for data in tqdm(loader, desc=desc):
            inputs, mask, targets, cat_inputs, rate_inputs, user_ids = data
            inputs = trans_to_cuda(inputs).long()
            
            hidden = model(inputs, attention_mask=None) 
            all_logits = model.compute_scores(hidden) 
            all_logits = trans_to_cpu(all_logits).detach().numpy()
            
            inputs_np = inputs.cpu().numpy()
            cat_inputs_np = cat_inputs.numpy()
            rate_inputs_np = rate_inputs.numpy()
            user_ids_np = user_ids.numpy()
            
            batch_size = inputs.shape[0]
            
            for b in range(batch_size):
                uid = user_ids_np[b]
                u_items = inputs_np[b]
                u_cats = cat_inputs_np[b]
                u_rates = rate_inputs_np[b]
                
                valid_idx = np.where(u_items != 0)[0]
                if len(valid_idx) <= MIN_HISTORY_LEN:
                    continue
                
                seq_items = u_items[valid_idx]
                seq_cats = u_cats[valid_idx]
                seq_rates = u_rates[valid_idx]
                seq_logits = all_logits[b, valid_idx, :] 
                
                # --- Agent State Initialization ---
                intensities = None
                backup_N_t = None
                backup_r_hat = None

                if active_agent is not None:
                    intensities = np.copy(active_agent.mu[uid])
                    backup_N_t = active_agent.N_t[uid].copy()
                    backup_r_hat = active_agent.r_hat[uid].copy()
                    
                    # Warm-up phase
                    for t in range(MIN_HISTORY_LEN - 1):
                        curr_cat = seq_cats[t]
                        curr_rate = seq_rates[t]
                        user_beta = active_agent.beta[uid]
                        decay = np.exp(-user_beta)
                        
                        intensities = (intensities - active_agent.mu[uid]) * decay + active_agent.mu[uid]
                        if curr_rate >= active_agent.tau:
                            intensities += active_agent.alpha[uid, :, curr_cat]
                        
                        active_agent.N_t[uid, curr_cat] += 1
                        n = active_agent.N_t[uid, curr_cat]
                        active_agent.r_hat[uid, curr_cat] += (1.0/n) * ((curr_rate/5.0) - active_agent.r_hat[uid, curr_cat])

                # === Step-by-Step Inference ===
                for t in range(MIN_HISTORY_LEN - 1, len(seq_items) - 1):
                    debug_stats["total_steps"] += 1
                    target_item = seq_items[t+1]
                    
                    # 1. Objective Risk: History Dominance (HD)
                    window_len = 20
                    window_start = max(0, t - window_len + 1)
                    recent_cats = seq_cats[window_start:t+1]
                    valid_recent = [c for c in recent_cats if c < num_cats]
                    
                    hd_val = 0.0
                    local_dominant_cat = -1
                    if valid_recent:
                        counts = np.bincount(valid_recent)
                        local_dominant_cat = np.argmax(counts)
                        hd_val = counts[local_dominant_cat] / len(valid_recent)
                    
                    # 2. Hawkes Update (Pre-Rec Status)
                    D_t_pre = 0.0
                    intensities_pre = None 
                    
                    if active_agent is not None:
                        curr_cat = seq_cats[t]
                        curr_rate = seq_rates[t]
                        
                        user_beta = active_agent.beta[uid]
                        decay = np.exp(-user_beta)
                        intensities = (intensities - active_agent.mu[uid]) * decay + active_agent.mu[uid]
                        
                        if curr_rate >= active_agent.tau:
                            intensities += active_agent.alpha[uid, :, curr_cat]
                        
                        active_agent.current_intensities[uid] = intensities
                        intensities_pre = intensities.copy()
                        
                        active_agent.N_t[uid, curr_cat] += 1
                        n = active_agent.N_t[uid, curr_cat]
                        reward = 1.0
                        active_agent.r_hat[uid, curr_cat] += (1.0/n) * (reward - active_agent.r_hat[uid, curr_cat])
                        
                        D_t_pre = active_agent.get_saturation_D(uid, intensities)

                    # 3. Base Recommendation
                    step_scores = seq_logits[t].copy()
                    step_scores[0] = -np.inf 
                    rerank_k = 50
                    ind = np.argpartition(step_scores, -rerank_k)[-rerank_k:]
                    sorted_ind = ind[np.argsort(-step_scores[ind])]
                    baseline_list = sorted_ind.tolist()
                    final_list = list(baseline_list)

                    # 4. Intervention Logic
                    was_triggered = False
                    
                    if is_intervention_active and D_t_pre > active_agent.lambda_max:
                        debug_stats["trigger"] += 1
                        target_bad_cat = local_dominant_cat
                        
                        if target_bad_cat != -1:
                            c_policy, _ = active_agent.get_policy_target_category(
                                uid, t, strategy='argmax', exclude_cat=target_bad_cat 
                            )
                            
                            if c_policy != target_bad_cat and c_policy in cat_to_items:
                                cands = cat_to_items[c_policy]
                                valid_cands_idx = [x for x in cands if x < len(step_scores)]
                                
                                if valid_cands_idx:
                                    cand_vals = step_scores[valid_cands_idx]
                                    local_top_idx = np.argsort(-cand_vals)[:10] 
                                    policy_candidates = [valid_cands_idx[i] for i in local_top_idx]

                                    if policy_candidates:
                                        debug_stats["success"] += 1
                                        was_triggered = True # Mark as successfully intervened
                                        new_list = []
                                        policy_idx = 0
                                        for item in baseline_list:
                                            item_cat = item_to_cat.get(item, -1)
                                            if item_cat == target_bad_cat:
                                                if policy_idx < len(policy_candidates):
                                                    new_item = policy_candidates[policy_idx]
                                                    if new_item not in new_list:
                                                        new_list.append(new_item)
                                                    policy_idx += 1
                                                else:
                                                    new_list.append(item)
                                            else:
                                                if item not in new_list:
                                                    new_list.append(item)
                                        final_list = new_list

                    # 5. Calculate Post-Rec Saturation (Counterfactual Comparison)
                    D_t_post = D_t_pre # Default for metrics
                    
                    # A. Actual Outcome Simulation (for general step metrics)
                    if active_agent is not None and len(final_list) > 0:
                        top1_item = final_list[0]
                        top1_cat = item_to_cat.get(top1_item, -1)
                        if top1_cat != -1:
                            # Simulate a jump for Top-1 consumption
                            intensities_post = intensities_pre.copy()
                            intensities_post += active_agent.alpha[uid, :, top1_cat]
                            D_t_post = active_agent.get_saturation_D(uid, intensities_post)
                    
                    # B. Counterfactual Analysis (CORAL vs SASRec at same step)
                    if was_triggered:
                        # 1. Baseline Outcome (If we did NOT intervene)
                        D_baseline_post = D_t_pre 
                        if len(baseline_list) > 0:
                            base_item = baseline_list[0]
                            base_cat = item_to_cat.get(base_item, -1)
                            if base_cat != -1:
                                i_base = intensities_pre.copy() + active_agent.alpha[uid, :, base_cat]
                                D_baseline_post = active_agent.get_saturation_D(uid, i_base)
                        
                        # 2. CORAL Outcome (Already calculated as D_t_post above)
                        D_coral_post = D_t_post 

                        triggered_moments_stats.append({
                            'Sat_Baseline': float(D_baseline_post),
                            'Sat_Coral': float(D_coral_post),
                            'Delta': float(D_coral_post - D_baseline_post) # Negative means CORAL reduced risk compared to Baseline
                        })

                    # 6. Metrics Calculation for Stratified Report
                    step_metrics = {}
                    step_metrics['Sat_Pre'] = float(D_t_pre)
                    step_metrics['Sat_Post'] = float(D_t_post)
                    
                    for K in k_list:
                        topk = final_list[:K]
                        hit = 1.0 if target_item in topk else 0.0
                        step_metrics[f'Recall@{K}'] = hit
                        if hit:
                            step_metrics[f'MRR@{K}'] = 1.0 / (topk.index(target_item) + 1)
                        else:
                            step_metrics[f'MRR@{K}'] = 0.0
                            
                        if local_dominant_cat != -1:
                            match_count = 0
                            for x in topk:
                                xc = item_to_cat.get(x, -1)
                                if xc == local_dominant_cat:
                                    match_count += 1
                            tcc = match_count / K
                        else:
                            tcc = 0.0
                        step_metrics[f'TCC@{K}'] = tcc

                    moment_records.append({'uid': int(uid), 'risk_val': float(hd_val), 'metrics': step_metrics})
                
                # Restore Frozen State
                if active_agent is not None:
                    active_agent.N_t[uid] = backup_N_t
                    active_agent.r_hat[uid] = backup_r_hat

    # --- PART 1: Standard Stratified Reporting (Fair Comparison) ---
    print("\n" + "="*20 + f" {desc} Stratified Risk Analysis (By History Dominance) " + "="*20)
    print(f"Total Moments: {len(moment_records)}")
    if is_intervention_active:
        tr = debug_stats['trigger'] / max(1, debug_stats['total_steps']) * 100
        print(f"Intervention Trigger Rate (Dt-based): {tr:.2f}%")
    
    metric_keys = []
    for K in k_list:
        metric_keys.extend([f'Recall@{K}', f'MRR@{K}', f'TCC@{K}'])
    
    metric_keys.append('Sat_Pre')
    metric_keys.append('Sat_Post')
    
    results = analyze_tail_risk_stratified(moment_records, metric_keys, percentiles=[0.20, 0.10, 0.05])
    
    headers = ["Metric", "Global Avg", "Top 20% HD", "Top 10% HD", "Top 5% HD"]
    row_fmt = "{:<12} | {:<10.4f} | {:<12.4f} | {:<12.4f} | {:<12.4f}"
    
    print("-" * 75)
    print("{:<12} | {:<10} | {:<12} | {:<12} | {:<12}".format(*headers))
    print("-" * 75)
    
    for m_type in ['Recall', 'MRR', 'TCC']:
        for K in k_list:
            base_key = f"{m_type}@{K}"
            val_global = results.get(f"Global_{base_key}", 0.0)
            val_20 = results.get(f"Top20%_Risk_{base_key}", 0.0)
            val_10 = results.get(f"Top10%_Risk_{base_key}", 0.0)
            val_05 = results.get(f"Top5%_Risk_{base_key}", 0.0)
            print(row_fmt.format(base_key, val_global, val_20, val_10, val_05))
    
    print("-" * 75)
    
    # Stratified Saturation
    for s_key, s_name in [('Sat_Pre', 'Sat(Pre)'), ('Sat_Post', 'Sat(Post)')]:
        val_global = results.get(f"Global_{s_key}", 0.0)
        val_20 = results.get(f"Top20%_Risk_{s_key}", 0.0)
        val_10 = results.get(f"Top10%_Risk_{s_key}", 0.0)
        val_05 = results.get(f"Top5%_Risk_{s_key}", 0.0)
        print(row_fmt.format(s_name, val_global, val_20, val_10, val_05))
            
    print("-" * 75)
    print(f"Avg HD (Top 5%): {results.get('Top5%_AvgHD', 0):.2f}")
    
    # --- PART 2: Triggered Moments Only Analysis (Counterfactual) ---
    if is_intervention_active and triggered_moments_stats:
        print("\n" + "*"*15 + " INTERVENTION EFFECTIVENESS (Counterfactual: CORAL vs Baseline) " + "*"*15)
        n_trig = len(triggered_moments_stats)
        print(f"Number of Triggered Interventions: {n_trig}")
        
        avg_base = np.mean([x['Sat_Baseline'] for x in triggered_moments_stats])
        avg_coral = np.mean([x['Sat_Coral'] for x in triggered_moments_stats])
        avg_delta = np.mean([x['Delta'] for x in triggered_moments_stats])
        
        print(f"Triggered Avg Sat (Baseline Path): {avg_base:.4f} (What would have happened)")
        print(f"Triggered Avg Sat (CORAL Path):    {avg_coral:.4f} (Actual outcome)")
        print(f"Avg Improvement (CORAL - Base):    {avg_delta:.4f} <--- (Negative = Risk Reduced)")
        print("*"*75 + "\n")
    elif is_intervention_active:
         print("\n[Warning] CORAL active but no interventions triggered.\n")
         
    print("="*75 + "\n")
    
    return results.get(f"Global_Recall@10", 0.0), results.get(f"Global_MRR@10", 0.0)

def train_sasrec_pipeline(model, train_data, test_data, best_model_path):
    print("\n[Step 1] Start Training SASRec Baseline...")
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
    
    best_hit = 0
    bad_counter = 0
    train_loader = torch.utils.data.DataLoader(train_data, num_workers=4, batch_size=opt.batch_size, shuffle=True)
    
    for epoch in range(opt.epoch):
        model.train()
        total_loss = 0.0
        for data in tqdm(train_loader, desc=f"Epoch {epoch}"):
            optimizer.zero_grad()
            targets, scores = forward_train(model, data)
            loss = model.loss_function(scores, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        
        # Valid logic
        hit, mrr = evaluate_dynamic_stream(model, test_data, agent=None, shadow_agent=None, desc="Valid")
        
        if hit > best_hit:
            best_hit = hit
            torch.save(model.state_dict(), best_model_path)
            bad_counter = 0
        else:
            bad_counter += 1
            if bad_counter >= opt.patience:
                print("Early stopping.")
                break

def train_hawkes_mle(agent, train_data_raw):
    print("\n[Step 2] Fitting Hawkes Parameters (MLE)...")
    seqs = train_data_raw[0]
    cats = train_data_raw[2]
    rates = train_data_raw[5]
    for uid in tqdm(range(len(seqs)), desc="MLE Fitting"):
        u_s = seqs[uid]
        u_c = cats[uid]
        u_r = rates[uid]
        valid_indices = [i for i, x in enumerate(u_s) if x != 0]
        if not valid_indices: continue
        clean_cats = [u_c[i] for i in valid_indices]
        clean_rates = [u_r[i] for i in valid_indices]
        agent.fit_user_hawkes(uid, clean_cats, clean_rates)

def main():
    setup_logger(opt.dataset, opt)
    redirect_tqdm_to_console()
    init_seed(2020)
    
    train_data_raw = pickle.load(open(os.path.join('../../datasets/', opt.dataset, 'processed_data/train.txt'), "rb"))
    test_data_raw = pickle.load(open(os.path.join('../../datasets/', opt.dataset, 'processed_data/test.txt'), "rb"))
    
    num_node = max(get_num_node_unique(train_data_raw), get_num_node_unique(test_data_raw))
    item_to_cat, cat_to_items, num_cats = build_maps(train_data_raw)
    
    train_data = Data(train_data_raw, num_node, opt.max_len, opt.mask_prob, mode='train')
    test_data = Data(test_data_raw, num_node, opt.max_len, opt.mask_prob, mode='test')
    
    model = trans_to_cuda(SASRec(opt, num_node))
    save_dir = 'checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, f'sasrec_best_model_{opt.dataset}.pth')
    
    if not os.path.exists(best_model_path):
        train_sasrec_pipeline(model, train_data, test_data, best_model_path)
    
    print(f"Loading best SASRec model from {best_model_path}...")
    model.load_state_dict(torch.load(best_model_path))
    
    # Prepare Agent
    agent = CoralAgent(len(train_data_raw[0]), num_cats, item_to_cat, opt)
    hawkes_path = os.path.join(save_dir, f'hawkes_params_{opt.dataset}_tau{opt.tau}.pth')
    
    if not os.path.exists(hawkes_path):
        train_hawkes_mle(agent, train_data_raw)
        agent.save_params(hawkes_path)
    else:
        agent.load_params(hawkes_path)

    # 1. Evaluate Baseline with Shadow Agent
    print("\n=== Baseline (SASRec) Stratified Evaluation ===")
    evaluate_dynamic_stream(
        model, 
        test_data, 
        agent=None,
        shadow_agent=agent,
        cat_to_items=cat_to_items,
        item_to_cat=item_to_cat,
        num_cats=num_cats, 
        desc="Baseline"
    )

    # 2. Evaluate CORAL with Active Agent
    if not opt.no_coral:
        print("\n=== CORAL Stratified Evaluation ===")
        evaluate_dynamic_stream(
            model, 
            test_data, 
            agent=agent,
            shadow_agent=None,
            cat_to_items=cat_to_items, 
            item_to_cat=item_to_cat, 
            num_cats=num_cats, 
            desc="CORAL"
        )
    else:
        print("CORAL pipeline disabled.")

if __name__ == '__main__':
    main()