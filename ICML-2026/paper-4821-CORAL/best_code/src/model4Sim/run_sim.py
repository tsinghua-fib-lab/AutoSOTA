"""
run_sim_paper.py – Echo-chamber closed-loop simulation aligned with the ICML paper.

Implements Section 5.2.3 Environment-Robustness experiment (Figure 4):
  - LLM: Gemma-3-12B-IT  (paper Appendix C.3)
  - Users initialised with 5 real historical items  (paper Section 5.2.3)
  - Personality: fatigue-prone / obsessive, alternating  (paper Appendix C.3)
  - Echo-chamber prompt variant for adversarial feedback  (paper Appendix C.3)
  - Hawkes parameters updated every 10 steps via online MLE  (paper Section 5.2.3)
  - Tracks: exposure saturation D-curve, Cumulative Reward (CR), SCC  (paper Eq. 18-19)
"""

import os
import re
import random
import pickle
import argparse

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import pipeline

from model import SASRec, trans_to_cuda, trans_to_cpu
from coral import CoralAgent

# ============================================================
# Global configuration
# ============================================================

os.environ["TOKENIZERS_PARALLELISM"] = "false"

NUM_USERS = 50
TIME_STEPS = 100               
TOP_K = 5
INIT_HISTORY_LEN = 5           
HAWKES_UPDATE_FREQ = 10        
MODEL_ID = "google/gemma-3-12b-it"   #
BATCH_SIZE_LLM = 16
SEED = 2026

DATASET_NAME = "ML1m"
DATA_DIR = f"../../datasets/{DATASET_NAME}/processed_data/"
CHECKPOINT_DIR = "checkpoints"
MOVIES_FILE = f"../../datasets/{DATASET_NAME}/movielens_sequential_dataset_with_title.csv"

DOMAIN_MAP = {"ML1m": "movie", "Steam": "game", "Amazon": "shopping"}


PERSONAS = {
    "fatigue":   "get bored very quickly",
    "obsessive": "have obsessive personality",
}


# ============================================================
# Item text mapper
# ============================================================

class ItemMapper:
    """Maps item IDs to human-readable text for LLM prompts."""

    def __init__(self, item_to_cat_id_map: dict, movies_file: str = None):
        self.item_to_cat_id = item_to_cat_id_map
        self.item_text_map: dict = {}
        self.cat_name_map: dict = {}

        if movies_file and os.path.exists(movies_file):
            print(f"Loading metadata from {movies_file}...")
            try:
                df = pd.read_csv(movies_file)
                cols = {c.lower(): c for c in df.columns}
                if all(r in cols for r in ["item_id", "title", "category"]):
                    meta = df[
                        [cols["item_id"], cols["title"], cols["category"]]
                    ].drop_duplicates(subset=[cols["item_id"]])
                    for _, row in meta.iterrows():
                        iid = int(row[cols["item_id"]])
                        title = str(row[cols["title"]]).strip()
                        cat = str(row[cols["category"]]).strip()
                        self.item_text_map[iid] = f"{title} ({cat})"
                        cat_id = item_to_cat_id_map.get(iid, -1)
                        if cat_id != -1 and cat_id not in self.cat_name_map:
                            self.cat_name_map[cat_id] = cat
            except Exception as exc:
                print(f"[Warning] Metadata load failed: {exc}")

    def get_item_text(self, item_id: int) -> str:
        if item_id in self.item_text_map:
            return self.item_text_map[item_id]
        cat_id = self.item_to_cat_id.get(item_id, -1)
        cat_name = self.cat_name_map.get(cat_id, f"Cat_{cat_id}")
        return f"Item_{item_id} ({cat_name})"

    def get_cat_name(self, cat_id: int) -> str:
        return self.cat_name_map.get(cat_id, f"Cat_{cat_id}")


# ============================================================
# Prompt builder – aligned with paper Appendix C.3
# ============================================================

def build_prompt(
    history_texts: list,
    rec_texts: list,
    preferred_cats: list,
    persona: str,
    domain: str = "movie",
    echo_chamber: bool = True,
) -> list:
    """
    Constructs the LLM evaluation prompt following paper Appendix C.3.

    Standard prompt format:
        Role: [domain] User simulator.
        Context: You have a history of [history_str] and prefer categories [cate_str].
                 You [persona_desc].
        Task: Evaluate [rec_str].
              Select its ID (1-K) if interested. Select 0 to reject.
        Output: integer only.

    Echo-chamber variant (used for Figure 4 / environment robustness):
        Appends "you strongly prefer items similar to your history" to the context.
    """
    history_str = "\n".join(f"- {t}" for t in history_texts[-INIT_HISTORY_LEN:])
    cate_str = ", ".join(preferred_cats) if preferred_cats else "various"
    rec_str = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(rec_texts))
    item_num = len(rec_texts)
    persona_desc = PERSONAS.get(persona, PERSONAS["fatigue"])

    # Echo-chamber suffix (paper Appendix C.3, highlighted constraint)
    echo_suffix = (
        ", you strongly prefer items similar to your history" if echo_chamber else ""
    )

    content = (
        f"Role: {domain} User simulator.\n"
        f"Context: You have a history of:\n{history_str}\n"
        f"and prefer categories {cate_str}. "
        f"You {persona_desc}{echo_suffix}.\n\n"
        f"Task: Evaluate the recommendation list:\n{rec_str}\n"
        f"Select its ID (1-{item_num}) if you are interested.\n"
        f"Select 0 if you reject it (due to boredom or mismatch).\n"
        f"Output: choose integer decision only (0 or 1-{item_num})."
    )

    return [{"role": "user", "content": content}]


# ============================================================
# SASRec batch inference
# ============================================================

def get_recommendations_batch(
    model: torch.nn.Module, seqs: list, k: int = 5
) -> np.ndarray:
    """Returns top-k item indices from SASRec for a batch of sequences."""
    model.eval()
    max_len = model.max_len
    padded = []
    for seq in seqs:
        s = seq[-max_len:]
        pad_len = max_len - len(s)
        padded.append(s + [0] * pad_len if pad_len > 0 else s)

    inp = trans_to_cuda(torch.tensor(padded, dtype=torch.long))
    with torch.no_grad():
        hidden = model(inp, attention_mask=None)
        last_h = hidden[:, -1, :]
        scores = model.compute_scores(last_h)
        scores = trans_to_cpu(scores).detach().numpy()
        scores[:, 0] = -np.inf  # mask padding index

    return np.argsort(-scores, axis=1)[:, :k]


# ============================================================
# LLM call
# ============================================================

def call_llm(llm_pipe, prompts: list) -> list:
    """
    Runs LLM batch inference and parses integer choices.
    Paper: "The model outputs an integer decision: the item ID (click) or 0 (rejection)."
    """
    outputs = llm_pipe(
        prompts,
        batch_size=BATCH_SIZE_LLM,
        max_new_tokens=5,
        do_sample=False,  # greedy for reproducibility
    )
    choices = []
    for out in outputs:
        content = out[0]["generated_text"] if isinstance(out, list) else out["generated_text"]
        if isinstance(content, list):
            content = content[-1]["content"]
        nums = re.findall(r"\b[0-9]\b", str(content))
        choices.append(int(nums[-1]) if nums else 0)
    return choices


# ============================================================
# CORAL re-ranking
# ============================================================

def apply_coral_reranking(
    agent: CoralAgent,
    uids: list,
    base_recs: np.ndarray,
    seqs: list,
    item_to_cat: dict,
) -> np.ndarray:
    """Re-ranks SASRec candidates using CORAL policy scores."""
    reranked = []
    for idx, uid in enumerate(uids):
        t = len(seqs[idx])
        _, _D, cat_scores = agent.get_policy_target_category(
            uid, t, strategy="argmax", return_scores=True
        )
        sorted_cats = np.argsort(-cat_scores)
        buckets: dict = {}
        for item in base_recs[idx]:
            c = item_to_cat.get(item, 0)
            buckets.setdefault(c, []).append(item)

        ordered = []
        for c in sorted_cats:
            if c in buckets:
                ordered.extend(buckets.pop(c))
        for remaining in buckets.values():
            ordered.extend(remaining)
        reranked.append(ordered)

    return np.array(reranked)


# ============================================================
# SASRec shadow Hawkes intensity update
# ============================================================

def hawkes_step(
    current_intensity: np.ndarray,
    mu: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    chosen_cat: int,
    engaged: bool,
) -> np.ndarray:
    """
    Single-step Hawkes intensity update for the SASRec shadow tracker.

    Follows paper Eq. (2): λ_c(t) = μ_c + Σ_k α_{c,c(a_k)} · exp(−β(t−k)) · e_k
    where e_k = I(g(y_k) ≥ τ).  Excitation is applied only on engagement events.
    """
    decay = np.exp(-beta)
    decayed = (current_intensity - mu) * decay + mu
    if engaged:
        excitation = alpha[:, chosen_cat] if alpha.ndim == 2 else alpha[chosen_cat]
        decayed = decayed + excitation
    return np.clip(decayed, 0.0, 1e6)


# ============================================================
# Metric helpers
# ============================================================

def compute_scc(cat_sequences: list, num_cats: int) -> float:
    """
    Sequence Category Coverage (SCC), paper Eq. (19).
    SCC = (1/|U|) * Σ_u |Unique(C_{u,T})| / |C|
    """
    if num_cats == 0 or not cat_sequences:
        return 0.0
    return float(np.mean([len(set(seq)) / num_cats for seq in cat_sequences if seq]))


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="CORAL closed-loop simulation – paper-aligned (ICML)"
    )
    # SASRec hyper-params
    parser.add_argument("--dataset", default="ML1m")
    parser.add_argument("--hiddenSize", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr_dc", type=float, default=0.1)
    parser.add_argument("--lr_dc_step", type=int, default=10)
    parser.add_argument("--l2", type=float, default=1e-5)
    parser.add_argument("--max_len", type=int, default=200)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=2)
    parser.add_argument("--dropout_local", type=float, default=0.2)
    # CORAL hyper-params
    parser.add_argument("--lambda_max", type=float, default=0.7)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--rho", type=float, default=2.0)
    parser.add_argument("--tau", type=float, default=3.5)
    parser.add_argument("--delta_conf", type=float, default=0.1)
    # Simulation flags
    parser.add_argument(
        "--echo_chamber",
        action="store_true",
        default=True,
        help="Inject echo-chamber confirmation bias into user prompt (Figure 4).",
    )
    opt = parser.parse_args()

    # Reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.set_device(0)

    # ----------------------------------------------------------
    # 1. Load data
    # ----------------------------------------------------------
    print(f"Loading data from {DATA_DIR}...")
    data_path = os.path.join(DATA_DIR, "train.txt")
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return
    with open(data_path, "rb") as f:
        train_data_raw = pickle.load(f)

    seqs, _, cats, _, _, _ = train_data_raw

    item_to_cat: dict = {}
    all_items: set = set()
    for s, c in zip(seqs, cats):
        for item_id, cat_id in zip(s, c):
            if item_id != 0:
                item_to_cat[item_id] = cat_id
                all_items.add(item_id)

    num_cats = max(item_to_cat.values()) + 1
    num_node = max(all_items)
    domain = DOMAIN_MAP.get(opt.dataset, "movie")
    mapper = ItemMapper(item_to_cat, movies_file=MOVIES_FILE)

    # ----------------------------------------------------------
    # 2. Load SASRec backbone
    # ----------------------------------------------------------
    print("Loading SASRec...")
    model = SASRec(opt, num_node)
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"sasrec_best_model_{opt.dataset}.pth")
    if not os.path.exists(ckpt_path):
        print(f"Error: SASRec checkpoint not found at {ckpt_path}")
        return
    model.load_state_dict(torch.load(ckpt_path))
    model = trans_to_cuda(model)

    # ----------------------------------------------------------
    # 3. Load CORAL agent
    # ----------------------------------------------------------
    print("Loading CORAL agent...")
    agent = CoralAgent(len(seqs), num_cats, item_to_cat, opt)
    hawkes_path = os.path.join(
        CHECKPOINT_DIR, f"hawkes_params_{opt.dataset}_tau{opt.tau}.pth"
    )
    if os.path.exists(hawkes_path):
        agent.load_params(hawkes_path)
    else:
        print("[Warning] Hawkes checkpoint not found – using default parameters.")

    # ----------------------------------------------------------
    # 4. Load LLM  (paper Appendix C.3: Gemma-3-12B-IT)
    # ----------------------------------------------------------
    print(f"Loading LLM: {MODEL_ID}...")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    llm = pipeline("text-generation", model=MODEL_ID, device_map="auto", torch_dtype=dtype)

    # ----------------------------------------------------------
    # 5. Initialise simulated users with 5 real historical items
    #    Paper Section 5.2.3: "initialized with five historical items
    #    derived from the history"
    # ----------------------------------------------------------
    print(f"Initialising {NUM_USERS} users with {INIT_HISTORY_LEN} seed items each...")

    # Only select users whose sequences have enough non-padding items
    eligible = [
        i for i, s in enumerate(seqs)
        if len([x for x in s if x != 0]) >= INIT_HISTORY_LEN
    ]
    selected_indices = random.sample(eligible, min(NUM_USERS, len(eligible)))

    sim_users = []
    sas_intensities: dict = {}

    for slot, data_idx in enumerate(selected_indices):
        uid = slot
        raw_seq = [x for x in seqs[data_idx] if x != 0]
        raw_cat = [cats[data_idx][j] for j, x in enumerate(seqs[data_idx]) if x != 0]

        # First INIT_HISTORY_LEN interactions as seed (paper: "five historical items")
        seed_items = raw_seq[:INIT_HISTORY_LEN]
        seed_cats = raw_cat[:INIT_HISTORY_LEN]
        seed_ratings = [5.0] * INIT_HISTORY_LEN  # historical clicks assumed positive

        # Preferred categories: unique cats from seed history (preserve order)
        pref_cats = list(dict.fromkeys(seed_cats))
        pref_cat_names = [mapper.get_cat_name(c) for c in pref_cats]

        # Alternating personality assignment (paper: fatigue-prone / obsessive)
        persona = "fatigue" if uid % 2 == 0 else "obsessive"

        sim_users.append({
            "uid": uid,
            "persona": persona,
            "pref_cat_names": pref_cat_names,
            # SASRec shadow branch
            "sas_seq": list(seed_items),
            "sas_cat_seq": list(seed_cats),
            "sas_reward": 0.0,
            # CORAL active branch
            "coral_seq": list(seed_items),
            "coral_cat_seq": list(seed_cats),
            "coral_reward": 0.0,
            # Rolling buffer for online Hawkes update every HAWKES_UPDATE_FREQ steps
            "coral_cat_buf": list(seed_cats),
            "coral_rate_buf": list(seed_ratings),
        })

        # Initialise SASRec shadow Hawkes intensity from seed history
        intensity = np.copy(agent.mu[uid])
        for cat in seed_cats:
            # Seed items are positive engagements (e_k = 1)
            intensity = hawkes_step(
                intensity, agent.mu[uid], agent.alpha[uid], agent.beta[uid],
                chosen_cat=cat, engaged=True,
            )
        sas_intensities[uid] = intensity

        # Initialise CORAL agent state with seed history
        agent.reconstruct_history(uid, seed_items, seed_cats, seed_ratings)

    # ----------------------------------------------------------
    # 6. Simulation loop
    # ----------------------------------------------------------
    sas_d_curve: list = []
    coral_d_curve: list = []
    sas_cr_curve: list = []
    coral_cr_curve: list = []

    print(
        f"\n=== Starting Simulation | echo_chamber={opt.echo_chamber} "
        f"| T={TIME_STEPS} | users={len(sim_users)} ===\n"
    )

    for t in tqdm(range(TIME_STEPS), desc="Simulation"):

        # ---- Phase A: SASRec shadow tracking ----
        sas_seqs = [u["sas_seq"] for u in sim_users]
        sas_recs = get_recommendations_batch(model, sas_seqs, k=TOP_K)

        sas_prompts = [
            build_prompt(
                history_texts=[mapper.get_item_text(x) for x in u["sas_seq"]],
                rec_texts=[mapper.get_item_text(x) for x in sas_recs[i]],
                preferred_cats=u["pref_cat_names"],
                persona=u["persona"],
                domain=domain,
                echo_chamber=opt.echo_chamber,
            )
            for i, u in enumerate(sim_users)
        ]
        sas_choices = call_llm(llm, sas_prompts)

        step_sas_d = []
        for i, choice in enumerate(sas_choices):
            u = sim_users[i]
            uid = u["uid"]

            if 1 <= choice <= TOP_K:
                # Engagement: update sequence history and apply Hawkes excitation
                item = sas_recs[i][choice - 1]
                cat = item_to_cat.get(item, 0)
                u["sas_seq"].append(item)
                u["sas_cat_seq"].append(cat)
                u["sas_reward"] += 1.0
                sas_intensities[uid] = hawkes_step(
                    sas_intensities[uid], agent.mu[uid], agent.alpha[uid],
                    agent.beta[uid], chosen_cat=cat, engaged=True,
                )
            else:
                # Rejection: decay only (e_k = 0, no excitation per paper Eq. 3)
                sas_intensities[uid] = hawkes_step(
                    sas_intensities[uid], agent.mu[uid], agent.alpha[uid],
                    agent.beta[uid], chosen_cat=0, engaged=False,
                )

            D = agent.get_saturation_D(uid, sas_intensities[uid])
            step_sas_d.append(D)

        sas_d_curve.append(float(np.mean(step_sas_d)))
        sas_cr_curve.append(float(np.mean([u["sas_reward"] for u in sim_users])))

        # ---- Phase B: CORAL active intervention ----
        coral_seqs = [u["coral_seq"] for u in sim_users]
        uids = [u["uid"] for u in sim_users]
        base_recs = get_recommendations_batch(model, coral_seqs, k=50)
        coral_recs = apply_coral_reranking(agent, uids, base_recs, coral_seqs, item_to_cat)
        coral_recs = coral_recs[:, :TOP_K]

        coral_prompts = [
            build_prompt(
                history_texts=[mapper.get_item_text(x) for x in u["coral_seq"]],
                rec_texts=[mapper.get_item_text(x) for x in coral_recs[i]],
                preferred_cats=u["pref_cat_names"],
                persona=u["persona"],
                domain=domain,
                echo_chamber=opt.echo_chamber,
            )
            for i, u in enumerate(sim_users)
        ]
        coral_choices = call_llm(llm, coral_prompts)

        step_coral_d = []
        for i, choice in enumerate(coral_choices):
            u = sim_users[i]
            uid = u["uid"]

            if 1 <= choice <= TOP_K:
                item = coral_recs[i][choice - 1]
                cat = item_to_cat.get(item, 0)
                u["coral_seq"].append(item)
                u["coral_cat_seq"].append(cat)
                u["coral_reward"] += 1.0
                u["coral_cat_buf"].append(cat)
                u["coral_rate_buf"].append(5.0)
                # rating=5.0 (>= tau=3.5) triggers Hawkes excitation in update_single_step
                current_D = agent.update_single_step(uid, cat, rating=5.0, time_gap=1.0)
            else:
                # Rejection: decay only; use top-1 category to track which was exposed
                exposed_cat = item_to_cat.get(coral_recs[i][0], 0)
                u["coral_cat_buf"].append(exposed_cat)
                u["coral_rate_buf"].append(0.0)
                # rating=0.0 (< tau) signals no excitation
                current_D = agent.update_single_step(
                    uid, exposed_cat, rating=0.0, time_gap=1.0
                )

            step_coral_d.append(current_D)

        coral_d_curve.append(float(np.mean(step_coral_d)))
        coral_cr_curve.append(float(np.mean([u["coral_reward"] for u in sim_users])))

        # ---- Online Hawkes update every HAWKES_UPDATE_FREQ steps ----
        # Paper Section 5.2.3: "Hawkes parameters updated every 10 steps
        # via mini-batch online learning"
        if (t + 1) % HAWKES_UPDATE_FREQ == 0:
            for u in sim_users:
                uid = u["uid"]
                if len(u["coral_cat_buf"]) >= 2:
                    agent.fit_user_hawkes(
                        uid, u["coral_cat_buf"], u["coral_rate_buf"]
                    )
                # Reset rolling buffer for next window
                u["coral_cat_buf"] = []
                u["coral_rate_buf"] = []

        # ---- Progress log ----
        if t == 0 or (t + 1) % 10 == 0:
            print(
                f"  Step {t + 1:3d} | "
                f"SASRec D={sas_d_curve[-1]:.4f}  CR={sas_cr_curve[-1]:.2f} | "
                f"CORAL  D={coral_d_curve[-1]:.4f}  CR={coral_cr_curve[-1]:.2f}"
            )

    # ----------------------------------------------------------
    # 7. Compute final SCC  (paper Eq. 19)
    # ----------------------------------------------------------
    sas_scc = compute_scc([u["sas_cat_seq"] for u in sim_users], num_cats)
    coral_scc = compute_scc([u["coral_cat_seq"] for u in sim_users], num_cats)
    print(f"\nFinal SCC  – SASRec: {sas_scc:.4f}  |  CORAL: {coral_scc:.4f}")

    # ----------------------------------------------------------
    # 8. Save results
    # ----------------------------------------------------------
    print("Saving results...")

    # Per-step D-curve and cumulative reward (used for Figure 4 and CR analysis)
    pd.DataFrame({
        "Step": range(1, TIME_STEPS + 1),
        "SASRec_Intensity_D": sas_d_curve,
        "CORAL_Intensity_D": coral_d_curve,
        "SASRec_CR": sas_cr_curve,
        "CORAL_CR": coral_cr_curve,
    }).to_csv("simulation_metrics.csv", index=False)

    # Per-user category sequences (for SCC analysis)
    pd.DataFrame({
        "User_ID": range(len(sim_users)),
        "Persona": [u["persona"] for u in sim_users],
        "SASRec_Category_Sequence": [str(u["sas_cat_seq"]) for u in sim_users],
        "CORAL_Category_Sequence": [str(u["coral_cat_seq"]) for u in sim_users],
    }).to_csv("simulation_category_sequences.csv", index=False)

    # Aggregated summary
    pd.DataFrame({
        "Method": ["SASRec", "CORAL"],
        "Final_SCC": [sas_scc, coral_scc],
        "Total_CR": [
            float(np.mean([u["sas_reward"] for u in sim_users])),
            float(np.mean([u["coral_reward"] for u in sim_users])),
        ],
        "Final_D": [sas_d_curve[-1], coral_d_curve[-1]],
    }).to_csv("simulation_summary.csv", index=False)

    print(
        "Done. Saved:\n"
        "  simulation_metrics.csv           (D-curve + CR per step)\n"
        "  simulation_category_sequences.csv (per-user category history)\n"
        "  simulation_summary.csv            (final SCC, CR, D)"
    )


if __name__ == "__main__":
    main()
