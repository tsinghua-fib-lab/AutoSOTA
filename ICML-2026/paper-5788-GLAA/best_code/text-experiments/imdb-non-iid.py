"""
xFL on IMDb (Non-IID)
Clean, structured implementation for multi-seed experiments.
"""

import os
import time
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb
import matplotlib.pyplot as plt

# ----------------------------
# Global Configuration
# ----------------------------
BASE_SEED = 1337
VOCAB_SIZE = 20000
MAXLEN = 200
PAD, START, UNK, UNUSED = 0, 1, 2, 3

# ----------------------------
# Configuration Classes
# ----------------------------
class FLConfig:
    def __init__(self, n_clients=6, rounds=4, local_epochs=1, batch_size=128, alpha=0.2):
        self.n_clients = n_clients
        self.rounds = rounds
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.alpha = alpha

class XFLConfig:
    def __init__(self, topk=800, quant_bits=8, clip_radius=5.0, dp_sigma=0.05,
                 temperature=3.0, beta_align_final=0.80, align_warmup_rounds=1,
                 surrogate_every_R=1, l1_lambda=5e-7, hybrid_alpha=0.35):
        self.topk = topk
        self.quant_bits = quant_bits
        self.clip_radius = clip_radius
        self.dp_sigma = dp_sigma
        self.temperature = temperature
        self.beta_align_final = beta_align_final
        self.align_warmup_rounds = align_warmup_rounds
        self.surrogate_every_R = surrogate_every_R
        self.l1_lambda = l1_lambda
        self.hybrid_alpha = hybrid_alpha

# ----------------------------
# Reproducibility
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# ----------------------------
# Data & Preprocessing
# ----------------------------
def load_data():
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)
    X_train = keras.preprocessing.sequence.pad_sequences(
        X_train, maxlen=MAXLEN, padding="post", truncating="post"
    )
    X_test  = keras.preprocessing.sequence.pad_sequences(
        X_test,  maxlen=MAXLEN, padding="post", truncating="post"
    )
    y_train = np.array(y_train).astype("int32")
    y_test  = np.array(y_test).astype("int32")
    
    word_index = imdb.get_word_index()
    word_index = {k:(v+3) for k,v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2
    word_index["<UNUSED>"] = 3
    index_word = {v:k for k,v in word_index.items()}
    
    return X_train, y_train, X_test, y_test, index_word

def decode(ids, index_word):
    toks = []
    for t in ids:
        if int(t) == 0: continue
        toks.append(index_word.get(int(t), "<UNK>"))
    return toks

# ----------------------------
# Non-IID Logic
# ----------------------------
def dirichlet_split_noniid_text(y, n_clients, alpha=0.2):
    classes = np.unique(y)
    idx_by_class = [np.where(y == c)[0] for c in classes]
    for arr in idx_by_class:
        np.random.shuffle(arr)
    client_indices = [[] for _ in range(n_clients)]
    for idx in idx_by_class:
        props = np.random.dirichlet(alpha * np.ones(n_clients))
        props = props / props.sum()
        splits = (np.cumsum(props) * len(idx)).astype(int)[:-1]
        parts = np.split(idx, splits)
        for i in range(n_clients):
            client_indices[i].extend(parts[i].tolist())
    return [np.array(sorted(ci), dtype=np.int64) for ci in client_indices]

def build_client_covariate_shifts(n_clients, vocab_size, seed, jargon_frac=0.06):
    rng = np.random.RandomState(seed + 42)
    base_vocab = np.arange(4, vocab_size, dtype=np.int32)
    masked_sets = []
    drop_probs = []
    for cid in range(n_clients):
        rng_c = np.random.RandomState(seed + 1000 + cid)
        k = max(1, int(jargon_frac * len(base_vocab)))
        masked = rng_c.choice(base_vocab, size=k, replace=False)
        masked_sets.append(set(int(t) for t in masked))
        p_drop = 0.08 + 0.02 * (cid % 5)
        drop_probs.append(p_drop)
    return masked_sets, drop_probs

def apply_client_shift(X, masked_set, p_drop, rng):
    X = X.copy()
    B, L = X.shape
    mask_jargon = np.isin(X, list(masked_set))
    X[mask_jargon] = UNK
    drop_mask = (rng.rand(B, L) < p_drop) & (X != PAD)
    X[drop_mask] = PAD
    return X

def make_clients_noniid(X, y, n_clients, alpha, seed):
    splits = dirichlet_split_noniid_text(y, n_clients, alpha)
    masked_sets, drop_probs = build_client_covariate_shifts(n_clients, VOCAB_SIZE, seed)
    clients = []
    for cid, idx in enumerate(splits):
        Xi, yi = X[idx], y[idx]
        rng = np.random.RandomState(seed + 5000 + cid)
        Xi_shift = apply_client_shift(Xi, masked_sets[cid], drop_probs[cid], rng)
        clients.append((Xi_shift, yi))
    return clients

# ----------------------------
# Models
# ----------------------------
def build_task_model(vocab_size=20000, embed_dim=64, lstm_units=64, maxlen=200):
    inp = layers.Input(shape=(maxlen,), dtype="int32")
    emb = layers.Embedding(vocab_size, embed_dim, name="embedding")(inp)
    x = layers.Bidirectional(layers.LSTM(lstm_units), name="bilstm")(emb)
    x = layers.Dense(64, activation="relu", name="dense")(x)
    out = layers.Dense(1, activation="sigmoid", name="logit")(x)
    m = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy",
              metrics=["accuracy"])
    return m

def build_linear_logreg(vocab_size=20000, l1=2e-3):
    inp = layers.Input(shape=(vocab_size,), dtype="float32")
    out = layers.Dense(1, activation="sigmoid",
                       kernel_regularizer=keras.regularizers.l1(l1),
                       name="logreg")(inp)
    m = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(8e-3),
              loss="binary_crossentropy",
              metrics=["accuracy"])
    return m

# ----------------------------
# Utilities
# ----------------------------
def get_weights(model):
    return [w.copy() for w in model.get_weights()]

def set_weights(model, weights):
    model.set_weights([w.copy() for w in weights])

def avg_weights(weight_list):
    out = []
    for weights in zip(*weight_list):
        out.append(np.mean(np.stack(weights, axis=0), axis=0))
    return out

def bow_matrix(seqs, vocab_size):
    N = seqs.shape[0]
    X = np.zeros((N, vocab_size), dtype="float32")
    for i, row in enumerate(seqs):
        uniq = np.unique(row[row>0])
        uniq = uniq[uniq < vocab_size]
        X[i, uniq] = 1.0
    return X

def normalize_simplex(v, eps=1e-12):
    v = np.abs(v)
    s = np.sum(v, axis=-1, keepdims=True).clip(min=eps)
    return v / s

def jsd(p, q, eps=1e-12):
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    m = 0.5*(p+q)
    kl_pm = np.sum(p*np.log(p/m), axis=-1)
    kl_qm = np.sum(q*np.log(q/m), axis=-1)
    return 0.5*(kl_pm + kl_qm)

def compute_edi(per_client_dists, reference):
    ds = []
    for pc in per_client_dists:
        d0 = jsd(pc[0], reference[0])
        d1 = jsd(pc[1], reference[1])
        ds.extend([d0, d1])
    return float(np.mean(ds)) if ds else 0.0

# ----------------------------
# Saliency & Fidelity
# ----------------------------
def saliency_token_scores(model, X_batch, y_batch):
    Xb = tf.convert_to_tensor(X_batch, dtype=tf.int32)
    yb = tf.convert_to_tensor(y_batch.reshape(-1, 1), dtype=tf.float32)

    embed = model.get_layer("embedding")
    tail_layers = []
    seen_emb = False
    for lyr in model.layers:
        if lyr.name == "embedding":
            seen_emb = True
            continue
        if seen_emb:
            tail_layers.append(lyr)

    with tf.GradientTape() as tape:
        emb_out = embed(Xb)
        tape.watch(emb_out)
        x = emb_out
        for lyr in tail_layers: x = lyr(x)
        preds = x
        score = yb*preds + (1.0 - yb)*(1.0 - preds)

    grads = tape.gradient(score, emb_out)
    sal = tf.reduce_sum(tf.abs(grads * emb_out), axis=-1)
    return sal.numpy()

def model_prob_true(model, Xb, yb, preproc=None):
    Xin = preproc(Xb) if preproc is not None else Xb
    p = model.predict(Xin, verbose=0).reshape(-1)
    return float(np.mean(yb*p + (1 - yb)*(1 - p)))

def auc_area(xs, ys):
    area = 0.0
    for i in range(1, len(xs)):
        dx = xs[i] - xs[i-1]
        area += 0.5*(ys[i] + ys[i-1])*dx
    return float(area)

def deletion_insertion_auc_text(model, Xb, yb, imp_scores, steps=20, pad_id=0, preproc=None):
    B, L = Xb.shape
    order = np.argsort(-imp_scores, axis=1)
    xs = [0.0]
    del_scores = [model_prob_true(model, Xb, yb, preproc=preproc)]
    ins_scores = []

    X0 = np.full_like(Xb, pad_id)
    ins_scores.append(model_prob_true(model, X0, yb, preproc=preproc))

    for s in range(1, steps+1):
        frac = s/steps
        k = int(frac*L)
        xs.append(frac)

        Xd = Xb.copy()
        for i in range(B):
            if k>0: Xd[i, order[i, :k]] = pad_id
        del_scores.append(model_prob_true(model, Xd, yb, preproc=preproc))

        Xi = X0.copy()
        for i in range(B):
            if k>0:
                idx_keep = order[i, :k]
                Xi[i, idx_keep] = Xb[i, idx_keep]
        ins_scores.append(model_prob_true(model, Xi, yb, preproc=preproc))

    return auc_area(xs, del_scores), auc_area(xs, ins_scores)

# ----------------------------
# xFL Components
# ----------------------------
def _logit(p, eps=1e-7):
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p/(1.0 - p))

def distill_to_sparse_surrogate(task_model, X_client, cfg: XFLConfig):
    p = task_model.predict(X_client, verbose=0).reshape(-1)
    zT = _logit(p) / cfg.temperature
    p_soft = 1.0/(1.0 + np.exp(-zT))

    Xb = bow_matrix(X_client, VOCAB_SIZE)
    surr = build_linear_logreg(vocab_size=VOCAB_SIZE, l1=cfg.l1_lambda)
    surr.compile(optimizer=keras.optimizers.Adam(5e-3),
                 loss=lambda y_true, y_pred: keras.losses.binary_crossentropy(y_true, y_pred),
                 metrics=["accuracy"])
    surr.fit(Xb, p_soft, epochs=2, batch_size=256, verbose=0)
    W, b = surr.get_layer("logreg").get_weights()
    W = W.reshape(-1)
    w_pos = np.maximum(W, 0.0)
    w_neg = np.maximum(-W, 0.0)
    return np.stack([w_neg, w_pos], axis=0)

def artifact_from_surrogate(Wc, cfg: XFLConfig):
    W = Wc.copy()
    for c in range(2):
        idx = np.argsort(-np.abs(W[c]))[:cfg.topk]
        mask = np.zeros_like(W[c])
        mask[idx] = 1.0
        W[c] = W[c]*mask
        n = np.linalg.norm(W[c]) + 1e-8
        W[c] = W[c]*min(1.0, cfg.clip_radius/n)
    mx = np.max(np.abs(W)) + 1e-8
    scale = (2**(cfg.quant_bits-1)-1)/mx
    Wq = np.round(W*scale)/scale
    Wq += np.random.normal(0.0, cfg.dp_sigma, size=Wq.shape)
    Wq = normalize_simplex(Wq, eps=1e-12)
    return Wq

def robust_median(artifacts):
    A = np.stack(artifacts, axis=0)
    med = np.median(A, axis=0)
    return normalize_simplex(med)

# ----------------------------
# Explanation Builders
# ----------------------------
def token_importance_blb(model, Xb, yb):
    S = saliency_token_scores(model, Xb, yb)
    S = S / np.maximum(S.sum(axis=1, keepdims=True), 1e-8)
    return S

def token_importance_blc(global_hist, Xb, model):
    preds = (model.predict(Xb, verbose=0).reshape(-1) >= 0.5).astype("int32")
    B, L = Xb.shape
    imp = np.zeros((B, L), dtype="float32")
    for i in range(B):
        cls = preds[i]
        for j in range(L):
            tok = int(Xb[i, j])
            if tok == 0 or tok >= VOCAB_SIZE: continue
            imp[i, j] = global_hist[cls, tok]
    imp = imp / np.maximum(imp.sum(axis=1, keepdims=True), 1e-8)
    return imp

def token_importance_bld(lin_model, Xb):
    W, b = lin_model.get_layer("logreg").get_weights()
    W = W.reshape(-1)
    pred = (lin_model.predict(bow_matrix(Xb, VOCAB_SIZE), verbose=0).reshape(-1) >= 0.5).astype("int32")
    B, L = Xb.shape
    imp = np.zeros((B, L), dtype="float32")
    for i in range(B):
        cls = pred[i]
        for j in range(L):
            tok = int(Xb[i, j])
            if tok == 0 or tok >= VOCAB_SIZE: continue
            wpos = max(W[tok], 0.0)
            wneg = max(-W[tok], 0.0)
            imp[i, j] = wpos if cls==1 else wneg
    imp = imp / np.maximum(imp.sum(axis=1, keepdims=True), 1e-8)
    return imp

def token_importance_xfl_hybrid(task_model, Xb, yb, Pi, surr_weights_per_client, client_id, cfg: XFLConfig):
    alpha = cfg.hybrid_alpha
    S = saliency_token_scores(task_model, Xb, yb)
    S = S / np.maximum(S.sum(axis=1, keepdims=True), 1e-8)

    Wc = surr_weights_per_client[client_id]
    preds = (task_model.predict(Xb, verbose=0).reshape(-1) >= 0.5).astype("int32")
    B, L = Xb.shape
    M = np.zeros((B, L), dtype="float32")
    for i in range(B):
        cls = preds[i]
        for j in range(L):
            tok = int(Xb[i, j])
            if tok == 0 or tok >= VOCAB_SIZE: continue
            M[i, j] = 0.8*Wc[cls, tok] + 0.2*Pi[cls, tok]
    M = M / np.maximum(M.sum(axis=1, keepdims=True), 1e-8)

    imp = alpha*S + (1.0 - alpha)*M
    imp = imp / np.maximum(imp.sum(axis=1, keepdims=True), 1e-8)
    return imp

# ----------------------------
# Runners
# ----------------------------
def run_bl_a(flcfg, clients, X_test, y_test):
    global_model = build_task_model(vocab_size=VOCAB_SIZE, maxlen=MAXLEN)
    client_models = [build_task_model(vocab_size=VOCAB_SIZE, maxlen=MAXLEN) for _ in range(flcfg.n_clients)]
    set_weights(global_model, get_weights(global_model))
    for r in range(flcfg.rounds):
        w_updates = []
        for i,(Xi, yi) in enumerate(clients):
            set_weights(client_models[i], get_weights(global_model))
            client_models[i].fit(Xi, yi, epochs=flcfg.local_epochs,
                                 batch_size=flcfg.batch_size, verbose=0)
            w_updates.append(get_weights(client_models[i]))
        set_weights(global_model, avg_weights(w_updates))
    acc = global_model.evaluate(X_test, y_test, verbose=0)[1]
    return acc, global_model

def run_bl_b(flcfg, clients, X_test, y_test):
    acc, model = run_bl_a(flcfg, clients, X_test, y_test)
    per_client_vocab_dists = []
    for (Xi, yi) in clients:
        n = min(2000, len(Xi))
        Xs, ys = Xi[:n], yi[:n]
        S = saliency_token_scores(model, Xs, ys)
        hist = np.zeros((2, VOCAB_SIZE), dtype="float32")
        for i in range(n):
            lbl = int(ys[i])
            for j in range(Xs.shape[1]):
                tok = int(Xs[i, j])
                if tok==0 or tok>=VOCAB_SIZE: continue
                hist[lbl, tok] += float(S[i, j])
        hist = normalize_simplex(hist)
        per_client_vocab_dists.append(hist)
    return acc, model, per_client_vocab_dists

def run_bl_c(flcfg, clients, X_test, y_test):
    acc, model = run_bl_a(flcfg, clients, X_test, y_test)
    client_hist = []
    for (Xi, yi) in clients:
        n = min(2000, len(Xi))
        Xs, ys = Xi[:n], yi[:n]
        S = saliency_token_scores(model, Xs, ys)
        hist = np.zeros((2, VOCAB_SIZE), dtype="float32")
        preds = (model.predict(Xs, verbose=0).reshape(-1) >= 0.5).astype("int32")
        for i in range(n):
            cls = int(preds[i])
            for j in range(Xs.shape[1]):
                tok = int(Xs[i, j])
                if tok==0 or tok>=VOCAB_SIZE: continue
                hist[cls, tok] += float(S[i, j])
        hist = normalize_simplex(hist)
        client_hist.append(hist)
    global_hist = robust_median(client_hist)
    return acc, model, global_hist

def run_bl_d(flcfg, clients, X_test, y_test):
    global_lin = build_linear_logreg(vocab_size=VOCAB_SIZE, l1=2e-3)
    client_lin = [build_linear_logreg(vocab_size=VOCAB_SIZE, l1=2e-3) for _ in range(flcfg.n_clients)]
    set_weights(global_lin, get_weights(global_lin))
    for r in range(flcfg.rounds):
        w_updates = []
        for i,(Xi, yi) in enumerate(clients):
            set_weights(client_lin[i], get_weights(global_lin))
            Xb = bow_matrix(Xi, VOCAB_SIZE)
            client_lin[i].fit(Xb, yi, epochs=flcfg.local_epochs,
                              batch_size=256, verbose=0)
            w_updates.append(get_weights(client_lin[i]))
        set_weights(global_lin, avg_weights(w_updates))
    acc = global_lin.evaluate(bow_matrix(X_test, VOCAB_SIZE), y_test, verbose=0)[1]
    per_client_dists = []
    for (Xi, yi) in clients:
        m = build_linear_logreg(vocab_size=VOCAB_SIZE, l1=2e-3)
        set_weights(m, get_weights(global_lin))
        Xb = bow_matrix(Xi, VOCAB_SIZE)
        m.fit(Xb, yi, epochs=1, batch_size=256, verbose=0)
        W, b = m.get_layer("logreg").get_weights()
        W = W.reshape(-1)
        w_pos = np.maximum(W, 0.0)
        w_neg = np.maximum(-W, 0.0)
        dist = normalize_simplex(np.stack([w_neg, w_pos], axis=0))
        per_client_dists.append(dist)
    Wg, _ = global_lin.get_layer("logreg").get_weights()
    Wg = Wg.reshape(-1)
    Wg_dist = normalize_simplex(np.stack([np.maximum(-Wg,0.0), np.maximum(Wg,0.0)], axis=0))
    return acc, global_lin, Wg_dist, per_client_dists

def run_xfl(flcfg, xcfg, clients, X_test, y_test):
    global_model = build_task_model(vocab_size=VOCAB_SIZE, maxlen=MAXLEN)
    client_models = [build_task_model(vocab_size=VOCAB_SIZE, maxlen=MAXLEN) for _ in range(flcfg.n_clients)]
    Pi = normalize_simplex(np.ones((2, VOCAB_SIZE), dtype="float32"))
    round_times = []
    final_client_aligned = []

    for r in range(flcfg.rounds):
        t0 = time.time()
        artifacts = []
        for i,(Xi, yi) in enumerate(clients):
            set_weights(client_models[i], get_weights(global_model))
            client_models[i].fit(Xi, yi, epochs=flcfg.local_epochs,
                                 batch_size=flcfg.batch_size, verbose=0)
            if (r % xcfg.surrogate_every_R) == 0:
                Wc = distill_to_sparse_surrogate(client_models[i], Xi, xcfg)
                beta = xcfg.beta_align_final * min(1.0, (r+1)/max(1, xcfg.align_warmup_rounds))
                S_i = normalize_simplex(Wc)
                S_mix = normalize_simplex((1.0 - beta)*S_i + beta*Pi)
                artifacts.append(artifact_from_surrogate(S_mix, xcfg))
                if r == flcfg.rounds - 1:
                    final_client_aligned.append(S_mix)
        set_weights(global_model, avg_weights([get_weights(cm) for cm in client_models]))
        if len(artifacts)>0:
            Pi = robust_median(artifacts)
        round_times.append(time.time() - t0)

    acc = global_model.evaluate(X_test, y_test, verbose=0)[1]
    overhead_bytes = flcfg.n_clients * 2 * xcfg.topk * 3
    avg_time = np.mean(round_times)
    if not final_client_aligned:
        for (Xi, yi) in clients:
            Wc = distill_to_sparse_surrogate(global_model, Xi, xcfg)
            beta = xcfg.beta_align_final
            S_i = normalize_simplex(Wc)
            final_client_aligned.append(normalize_simplex((1.0 - beta)*S_i + beta*Pi))
    return acc, global_model, Pi, final_client_aligned, overhead_bytes, avg_time

# ----------------------------
# Experiment Runner
# ----------------------------
def run_once(seed: int, X_train, y_train, X_test, y_test):
    keras.backend.clear_session()
    set_seed(seed)
    print(f"\n{'='*40}\n Seed {seed}\n{'='*40}")

    flcfg_base = FLConfig(n_clients=6, rounds=4, local_epochs=1, batch_size=128, alpha=0.2)
    flcfg_x    = FLConfig(n_clients=6, rounds=8, local_epochs=1, batch_size=128, alpha=0.2)
    xcfg  = XFLConfig(topk=800, dp_sigma=0.05, surrogate_every_R=1, beta_align_final=0.80, align_warmup_rounds=1, l1_lambda=5e-7, hybrid_alpha=0.35)

    clients = make_clients_noniid(X_train, y_train, flcfg_base.n_clients, flcfg_base.alpha, seed)

    print("Running BL-A …")
    acc_A, model_A = run_bl_a(flcfg_base, clients, X_test, y_test)
    print(f"BL-A acc: {acc_A:.4f}")

    print("Running BL-B …")
    acc_B, model_B, pc_b = run_bl_b(flcfg_base, clients, X_test, y_test)
    print(f"BL-B acc: {acc_B:.4f}")

    print("Running BL-C …")
    acc_C, model_C, ghist_C = run_bl_c(flcfg_base, clients, X_test, y_test)
    print(f"BL-C acc: {acc_C:.4f}")

    print("Running BL-D …")
    acc_D, model_D, Wmap_D_global, pc_d = run_bl_d(flcfg_base, clients, X_test, y_test)
    print(f"BL-D acc: {acc_D:.4f}")

    print("Running xFL …")
    acc_X, model_X, Pi_X, pc_x_aligned, overhead_X, avg_round_time_X = run_xfl(flcfg_x, xcfg, clients, X_test, y_test)
    print(f"xFL acc: {acc_X:.4f}")

    # Metrics
    ref_B = normalize_simplex(np.mean(np.stack(pc_b, axis=0), axis=0))
    edi_B = compute_edi(pc_b, ref_B)
    edi_C = compute_edi(pc_b, ghist_C)
    edi_D = compute_edi(pc_d, Wmap_D_global)
    edi_X = compute_edi(pc_x_aligned, Pi_X)
    print(f"EDI -> BL-B {edi_B:.4f} | BL-C {edi_C:.4f} | BL-D {edi_D:.4f} | xFL {edi_X:.4f}")

    # Fidelity
    BATCH_N = 256
    Xb = X_test[:BATCH_N]
    yb = y_test[:BATCH_N]

    maps_B = token_importance_blb(model_B, Xb, yb)
    maps_C = token_importance_blc(ghist_C, Xb, model_C)
    maps_D = token_importance_bld(model_D, Xb)
    maps_X = token_importance_xfl_hybrid(model_X, Xb, yb, Pi_X, pc_x_aligned, client_id=0, cfg=xcfg)

    bow_preproc = lambda X: bow_matrix(X, VOCAB_SIZE)

    del_B, ins_B = deletion_insertion_auc_text(model_B, Xb, yb, maps_B, steps=20)
    del_C, ins_C = deletion_insertion_auc_text(model_C, Xb, yb, maps_C, steps=20)
    del_D, ins_D = deletion_insertion_auc_text(model_D, Xb, yb, maps_D, steps=20, preproc=bow_preproc)
    del_X, ins_X = deletion_insertion_auc_text(model_X, Xb, yb, maps_X, steps=20)

    print(f"Del AUC: BL-B {del_B:.3f} | BL-C {del_C:.3f} | BL-D {del_D:.3f} | xFL {del_X:.3f}")
    print(f"Ins AUC: BL-B {ins_B:.3f} | BL-C {ins_C:.3f} | BL-D {ins_D:.3f} | xFL {ins_X:.3f}")

    return {
        "acc": {"BL-A": acc_A, "BL-B": acc_B, "BL-C": acc_C, "BL-D": acc_D, "xFL": acc_X},
        "edi": {"BL-B": edi_B, "BL-C": edi_C, "BL-D": edi_D, "xFL": edi_X},
        "del_auc": {"BL-B": del_B, "BL-C": del_C, "BL-D": del_D, "xFL": del_X},
        "ins_auc": {"BL-B": ins_B, "BL-C": ins_C, "BL-D": ins_D, "xFL": ins_X},
        "maps": {"BL-B": maps_B, "BL-C": maps_C, "BL-D": maps_D, "xFL": maps_X},
        "Xb": Xb
    }

if __name__ == "__main__":
    X_train, y_train, X_test, y_test, index_word = load_data()
    seeds = [BASE_SEED + i for i in range(5)]
    
    metrics = {
        "acc": {m: [] for m in ["BL-A", "BL-B", "BL-C", "BL-D", "xFL"]},
        "edi": {m: [] for m in ["BL-B", "BL-C", "BL-D", "xFL"]},
        "del_auc": {m: [] for m in ["BL-B", "BL-C", "BL-D", "xFL"]},
        "ins_auc": {m: [] for m in ["BL-B", "BL-C", "BL-D", "xFL"]},
    }
    
    last_res = None
    for s in seeds:
        res = run_once(s, X_train, y_train, X_test, y_test)
        last_res = res
        for m in metrics["acc"]: metrics["acc"][m].append(res["acc"][m])
        for m in metrics["edi"]: metrics["edi"][m].append(res["edi"][m])
        for m in metrics["del_auc"]: metrics["del_auc"][m].append(res["del_auc"][m])
        for m in metrics["ins_auc"]: metrics["ins_auc"][m].append(res["ins_auc"][m])

    def summarize(arr):
        arr = np.array(arr, dtype=np.float64)
        return float(arr.mean()), float(arr.std())

    print("\n===== Summary over seeds =====")
    print("Accuracy:")
    for m in metrics["acc"]:
        mean_v, std_v = summarize(metrics["acc"][m])
        print(f"  {m}: {mean_v:.4f} ± {std_v:.4f}")

    print("\nEDI (lower better):")
    for m in metrics["edi"]:
        mean_v, std_v = summarize(metrics["edi"][m])
        print(f"  {m}: {mean_v:.4f} ± {std_v:.4f}")

    print("\nDeletion AUC (lower better):")
    for m in metrics["del_auc"]:
        mean_v, std_v = summarize(metrics["del_auc"][m])
        print(f"  {m}: {mean_v:.4f} ± {std_v:.4f}")

    print("\nInsertion AUC (higher better):")
    for m in metrics["ins_auc"]:
        mean_v, std_v = summarize(metrics["ins_auc"][m])
        print(f"  {m}: {mean_v:.4f} ± {std_v:.4f}")

    # Plots
    os.makedirs("imdb_noniid_outputs", exist_ok=True)
    
    def plot_tokens_with_importance(sample_ids, imp, title, fname, topn=10):
        toks = decode(sample_ids, index_word)
        vals = imp[:len(sample_ids)]
        idxs = [i for i,t in enumerate(sample_ids) if int(t)!=0]
        scored = sorted([(i, float(vals[i])) for i in idxs], key=lambda x:-x[1])[:topn]
        labels = [toks[i] if i < len(toks) else "<PAD>" for i,_ in scored]
        scores = [s for _,s in scored]
        plt.figure(figsize=(6,3))
        plt.bar(range(len(scores)), scores)
        plt.xticks(range(len(scores)), labels, rotation=45, ha="right")
        plt.title(title)
        plt.tight_layout()
        path = os.path.join("imdb_noniid_outputs", fname)
        plt.savefig(path, bbox_inches="tight")
        plt.close()

    sample_i = 3
    Xb = last_res["Xb"]
    plot_tokens_with_importance(Xb[sample_i], last_res["maps"]["BL-B"][sample_i], "BL-B top tokens", "blb_tokens.png")
    plot_tokens_with_importance(Xb[sample_i], last_res["maps"]["BL-C"][sample_i], "BL-C top tokens", "blc_tokens.png")
    plot_tokens_with_importance(Xb[sample_i], last_res["maps"]["BL-D"][sample_i], "BL-D top tokens", "bld_tokens.png")
    plot_tokens_with_importance(Xb[sample_i], last_res["maps"]["xFL"][sample_i], "xFL (hybrid) top tokens", "xfl_tokens.png")

    def save_bar(vals, labels, title, fname, ylabel):
        plt.figure(figsize=(5,3)); plt.bar(range(len(labels)), vals)
        plt.xticks(range(len(labels)), labels); plt.ylabel(ylabel); plt.title(title)
        plt.tight_layout(); p=os.path.join("imdb_noniid_outputs", fname); plt.savefig(p, bbox_inches="tight"); plt.close()

    methods=["BL-B","BL-C","BL-D","xFL"]
    mean_del = [summarize(metrics["del_auc"][m])[0] for m in methods]
    mean_ins = [summarize(metrics["ins_auc"][m])[0] for m in methods]
    mean_edi = [summarize(metrics["edi"][m])[0] for m in methods]

    save_bar(mean_del, methods, "Deletion AUC (mean)", "bar_deletion_auc.png", "AUC")
    save_bar(mean_ins, methods, "Insertion AUC (mean)", "bar_insertion_auc.png", "AUC")
    save_bar(mean_edi, methods, "Consistency (EDI, mean)", "bar_edi.png", "EDI")

    print("Done. See plots under imdb_noniid_outputs/")
