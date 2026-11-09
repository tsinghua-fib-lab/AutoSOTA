"""Nullrecurrent (Kaggle OpenVaccine 1st place) features + Keras model.

Feature engineering and architecture follow KaggleOpenVaccine/scripts/nullrecurrent_inference.py
(itself the cleaned-up 1st place notebook), vectorised over a batch of equal-length constructs so
the released .h5 weights load unchanged.
"""
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L

from .paths import MODEL_COLS

TOKEN2INT = {x: i for i, x in enumerate("().ACGUBEHIMSXse")}


def _ohe(n, i):
    v = [0] * n
    v[i] = 1
    return v


def _seq_loop_features(seqs, loops):
    """One-hot sequence (+start/end tokens), loop type and their interaction (30 combos)."""
    len_app = 28
    seq_app = "AGCU" * 7
    loop_app = "SSSSMMMMIIIIBBBBHHHHEEEEXXXX"
    seqs = [s + seq_app for s in seqs]
    loops = [l + loop_app for l in loops]

    voc = ["A", "G", "C", "U", "s", "e"]
    m = {s: _ohe(len(voc), i) for i, s in enumerate(voc)}
    x_node = np.array([[m[c] for c in ["s"] + list(s) + ["e"]] for s in seqs], dtype=np.float32)

    voc = ["S", "M", "I", "B", "H", "E", "X"]
    m = {s: _ohe(len(voc), i) for i, s in enumerate(voc)}
    x_loop = np.array([[m[c] for c in l] for l in loops], dtype=np.float32)
    pad = np.zeros((x_loop.shape[0], 1, x_loop.shape[2]), dtype=np.float32)
    x_loop = np.concatenate([pad, x_loop, pad], axis=1)

    x = np.concatenate([x_node, x_loop], axis=2)
    a = np.sum(x * (2 ** np.arange(x.shape[2])[None, None, :]), axis=2)
    vocab = sorted(set(a.flatten()))
    ohes = np.stack([a == v for v in vocab], axis=2)
    x = np.concatenate([x, ohes], axis=2).astype(np.float32)
    x = np.concatenate([x[:, :(-len_app - 1), :], x[:, -1, :][:, None, :]], axis=1)
    return x


def _pair_idx(struct):
    out = np.arange(len(struct))
    stack = []
    for c, ch in enumerate(struct):
        if ch == "(":
            stack.append(c)
        elif ch == ")":
            t = stack.pop()
            out[c], out[t] = t, c
    return out


def _dist_to(struct, paired):
    n = len(struct)
    hit = np.array([(ch != ".") == paired for ch in struct])
    out = np.full(n, 10000.0)
    cur = 10000
    for c in range(n):
        cur += 1
        if hit[c]:
            out[c] = 1
            cur = 0
        else:
            out[c] = min(out[c], cur)
    cur = 10000
    for c in range(n - 1, -1, -1):
        cur += 1
        if hit[c]:
            out[c] = 0
            cur = 0
        else:
            out[c] = min(out[c], cur)
    return out


def _bpp_features(bpp, struct):
    p = bpp
    srt = np.sort(p, axis=1)
    m0 = p.max(axis=0)
    pi = _pair_idx(struct)
    pij = p[np.arange(len(pi)), pi]
    s0 = p.sum(axis=0)
    feats = [
        p.max(-1),
        1 - p.sum(axis=1),
        srt[:, -1] - srt[:, -2],
        m0[np.argmax(p, axis=0)] - m0,
        pij,
        m0 - pij,
        m0[pi] - pij,
        s0[pi] - pij,
    ]
    for paired in (True, False):
        d = 1 / (_dist_to(struct, paired) + 1)
        feats += [d, np.sqrt(d)]
    return np.stack(feats, axis=1)


def _extra_features(bpps, structs):
    n = len(structs[0]) + 2
    out = np.zeros((len(structs), n, 12), dtype=np.float32)
    for k, (b, s) in enumerate(zip(bpps, structs)):
        out[k, 1:-1] = _bpp_features(b, s)
    return out


def _structure_adj(structs):
    n = len(structs[0])
    out = np.zeros((len(structs), n + 2, n + 2, 1), dtype=np.float32)
    for k, s in enumerate(structs):
        stack = []
        for i, ch in enumerate(s):
            if ch == "(":
                stack.append(i)
            elif ch == ")":
                j = stack.pop()
                out[k, j + 1, i + 1, 0] = 1
                out[k, i + 1, j + 1, 0] = 1
    return out


def _seq_distance(b, n):
    idx = np.arange(n)
    d = 1 / (np.abs(idx[:, None] - idx[None, :]) + 1)
    d = np.stack([d ** i for i in (1, 2, 4)], axis=2).astype(np.float32)
    return np.broadcast_to(d[None], (b, n, n, 3))


def _calc_neighbor(d, dim, n):
    xs, ys = np.where(d == n)
    for x, y in zip(xs, ys):
        if x + 1 < dim:
            d[x + 1, y] = min(d[x + 1, y], n + 1)
        if y + 1 < dim:
            d[x, y + 1] = min(d[x, y + 1], n + 1)
        if x - 1 >= 0:
            d[x - 1, y] = min(d[x - 1, y], n + 1)
        if y - 1 >= 0:
            d[x, y - 1] = min(d[x, y - 1], n + 1)
    return d


def _structure_distance(ss):
    out = []
    dim = ss.shape[1]
    for k in range(ss.shape[0]):
        s = ss[k, :, :, 0]
        d = 10 + np.zeros_like(s)
        d[s == 1] = 1
        d[np.arange(dim), np.arange(dim)] = 0
        for x in range(9):
            d = _calc_neighbor(d, dim, x)
        out.append(d)
    d = 1 / (np.array(out) + 1)
    return np.stack([d ** i for i in (1, 2, 4)], axis=3).astype(np.float32)


def build_inputs(seqs, structs, loops, bpps):
    """Equal-length batch -> (X_node [B, N+2, 59], As [B, N+2, N+2, 8])."""
    x = np.concatenate([_seq_loop_features(seqs, loops), _extra_features(bpps, structs)], axis=2)
    n = len(seqs[0]) + 2
    a = np.zeros((len(seqs), n, n), dtype=np.float32)
    for k, b in enumerate(bpps):
        a[k, 1:-1, 1:-1] = b
    ss = _structure_adj(structs)
    a = np.concatenate([a[..., None], ss, _seq_distance(len(seqs), n), _structure_distance(ss)], axis=3)
    return x.astype(np.float32), a.astype(np.float32)


# ---------------------------------------------------------------- model


def gru_layer(hidden_dim, dropout):
    return L.Bidirectional(L.GRU(hidden_dim, dropout=dropout, return_sequences=True, kernel_initializer="orthogonal"))


def lstm_layer(hidden_dim, dropout):
    return L.Bidirectional(L.LSTM(hidden_dim, dropout=dropout, return_sequences=True, kernel_initializer="orthogonal"))


def wave_block(x, filters, kernel_size, n):
    x = L.Conv1D(filters=filters, kernel_size=1, padding="same")(x)
    res_x = x
    for rate in [2 ** i for i in range(n)]:
        t = L.Conv1D(filters=filters, kernel_size=kernel_size, padding="same", activation="tanh", dilation_rate=rate)(x)
        s = L.Conv1D(filters=filters, kernel_size=kernel_size, padding="same", activation="sigmoid", dilation_rate=rate)(x)
        x = L.Multiply()([t, s])
        x = L.Conv1D(filters=filters, kernel_size=1, padding="same")(x)
        res_x = L.Add()([res_x, x])
    return res_x


def attention(x_inner, x_outer, n_factor, dropout):
    kw = dict(activation="linear", kernel_initializer="glorot_uniform", bias_initializer="glorot_uniform")
    x_q = L.Conv1D(n_factor, 1, **kw)(x_inner)
    x_k = L.Conv1D(n_factor, 1, **kw)(x_outer)
    x_v = L.Conv1D(n_factor, 1, **kw)(x_outer)
    x_kt = L.Permute((2, 1))(x_k)
    res = L.Lambda(lambda c: K.batch_dot(c[0], c[1]) / np.sqrt(n_factor))([x_q, x_kt])
    att = L.Lambda(lambda c: K.softmax(c, axis=-1))(res)
    return L.Lambda(lambda c: K.batch_dot(c[0], c[1]))([att, x_v])


def multi_head_attention(x, y, n_factor, n_head, dropout):
    if n_head == 1:
        att = attention(x, y, n_factor, dropout)
    else:
        heads = [attention(x, y, n_factor // n_head, dropout) for _ in range(n_head)]
        att = L.Concatenate()(heads)
        att = L.Dense(n_factor, kernel_initializer="glorot_uniform", bias_initializer="glorot_uniform")(att)
    x = L.Add()([x, att])
    x = L.LayerNormalization()(x)
    if dropout > 0:
        x = L.Dropout(dropout)(x)
    return x


def res(x, unit, kernel=3, rate=0.1):
    h = L.Conv1D(unit, kernel, 1, padding="same", activation=None)(x)
    h = L.LayerNormalization()(h)
    h = L.LeakyReLU()(h)
    h = L.Dropout(rate)(h)
    return L.Add()([x, h])


def forward(x, unit, kernel=3, rate=0.1):
    h = L.Conv1D(unit, kernel, 1, padding="same", activation=None)(x)
    h = L.LayerNormalization()(h)
    h = L.Dropout(rate)(h)
    h = L.LeakyReLU()(h)
    return res(h, unit, kernel, rate)


def adj_attn(x, adj, unit, n=2, rate=0.1):
    x_a = x
    x_as = []
    for _ in range(n):
        x_a = forward(x_a, unit)
        x_a = tf.matmul(adj, x_a)
        x_as.append(x_a)
    x_a = x_as[0] if n == 1 else L.Concatenate()(x_as)
    return forward(x_a, unit)


def get_base(n_node_feat, n_adj_feat):
    node = tf.keras.Input(shape=(None, n_node_feat), name="node")
    adj = tf.keras.Input(shape=(None, None, n_adj_feat), name="adj")
    adj_learned = L.Dense(1, "relu")(adj)
    adj_all = L.Concatenate(axis=3)([adj, adj_learned])

    xs = [node]
    x1 = forward(node, 128 * 2, kernel=3, rate=0.1)
    x2 = forward(x1, 64 * 2, kernel=6, rate=0.1)
    x3 = forward(x2, 32 * 2, kernel=15, rate=0.1)
    x4 = forward(x3, 16 * 2, kernel=30, rate=0.1)
    x = L.Concatenate()([x1, x2, x3, x4])

    for unit in [64 * 2, 32 * 2]:
        x_as = [adj_attn(x, adj_all[:, :, :, i], unit, rate=0) for i in range(adj_all.shape[3])]
        x_c = forward(x, unit, kernel=30)
        x = L.Concatenate()(x_as + [x_c])
        x = forward(x, unit)
        x = multi_head_attention(x, x, unit, 4, 0.0)
        xs.append(x)
    x = L.Concatenate()(xs)
    return tf.keras.Model(inputs=[node, adj], outputs=[x])


def get_ae_model(base, n_node_feat, n_adj_feat):
    """Denoising autoencoder pre-training head from mrkmakr's kernel."""
    node = tf.keras.Input(shape=(None, n_node_feat), name="node")
    adj = tf.keras.Input(shape=(None, None, n_adj_feat), name="adj")
    x = base([L.SpatialDropout1D(0.3)(node), adj])
    x = forward(x, 64 * 2, rate=0.2)
    p = L.Dense(n_node_feat, "sigmoid")(x)
    node_1 = tf.where(node > 1 - 1e-8, node, tf.zeros_like(node))
    node_0 = tf.where(node < 1e-8, node, tf.ones_like(node))
    node_float = tf.where((node <= 1 - 1e-8) & (node >= 1e-8), node, p)
    loss = -tf.reduce_mean(20 * node_1 * tf.math.log(p + 1e-4) + (1 - node_0) * tf.math.log(1 - p + 1e-4)
                           - 5 * (node_float - p) ** 2)
    return tf.keras.Model(inputs=[node, adj], outputs=[loss])


def get_model(base, head, n_node_feat, n_adj_feat):
    node = tf.keras.Input(shape=(None, n_node_feat), name="node")
    adj = tf.keras.Input(shape=(None, None, n_adj_feat), name="adj")
    x = base([node, adj])
    if head != "forward":
        x = forward(x, 128 * 2, rate=0.2)
    if head == "gru":
        x = gru_layer(128 * 2, dropout=0.2)(x)
    elif head == "lstm":
        x = lstm_layer(128 * 2, dropout=0.2)(x)
    elif head == "forward":
        x = forward(x, 128 * 4, kernel=5, rate=0.1)
        x = forward(x, 128 * 4, kernel=3, rate=0.1)
        x = forward(x, 128 * 4, kernel=1, rate=0.1)
    elif head == "wave":
        for f, n in [(16 * 2, 12), (32 * 2, 8), (64 * 2, 4), (128 * 2, 1)]:
            x = wave_block(x, f, 3, n)
            x = L.BatchNormalization()(x)
            x = L.Dropout(0.1)(x)
    else:
        raise ValueError(head)
    x = x[:, 1:-1, :]
    x = L.Dense(len(MODEL_COLS))(x)
    return tf.keras.Model(inputs=[node, adj], outputs=[x])


OFFICIAL = [("lstm", "ov-v40032-wgts"), ("gru", "ov-v40131-wgts"), ("forward", "ov-v40237-wgts"), ("wave", "ov-v40334-wgts")]
