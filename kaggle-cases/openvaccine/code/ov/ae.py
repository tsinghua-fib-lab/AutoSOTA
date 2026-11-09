"""Denoising-autoencoder pre-training of the shared base (mrkmakr's covid-ae-pretrain-gnn-attn-cnn).

Usage: python -m ov.ae --run runs/<name>   -> writes <run>/base_ae.h5"""
import argparse
import os

import numpy as np
import yaml

from ov.trainer import _rows, load_sets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()

    import tensorflow as tf
    from ov import nullrec

    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
    with open(os.path.join(args.run, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    ae = cfg["ae"]
    sets = load_sets(args.run)
    names = [n for n in ae.get("sets", ["train", "test107", "test130"]) if n in sets]
    rng = np.random.default_rng(cfg["seed"])
    tf.random.set_seed(cfg["seed"])

    n_feat, n_adj = sets["train"]["X"].shape[2], sets["train"]["A"].shape[3]
    base = nullrec.get_base(n_feat, n_adj)
    ae_model = nullrec.get_ae_model(base, n_feat, n_adj)
    opt = tf.keras.optimizers.Adam()
    sig = [tf.TensorSpec([None, None, None], tf.float32), tf.TensorSpec([None, None, None, None], tf.float32)]

    @tf.function(input_signature=sig)
    def step(x, a):
        with tf.GradientTape() as tape:
            loss = ae_model([x, a], training=True)
        g = tape.gradient(loss, ae_model.trainable_variables)
        opt.apply_gradients(zip(g, ae_model.trainable_variables))
        return loss

    for rnd in range(ae["epochs"] // ae["each"]):
        for name in names:
            X, A = sets[name]["X"], sets[name]["A"]
            for _ in range(ae["each"]):
                losses = []
                order = rng.permutation(len(X))
                for s in range(0, len(order), ae["batch_size"]):
                    ids = order[s:s + ae["batch_size"]]
                    losses.append(float(step(X[ids], _rows(A, ids))))
            print("ae round %d %s loss %.4f" % (rnd, name, np.mean(losses)), flush=True)
    base.save_weights(os.path.join(args.run, "base_ae.h5"))


if __name__ == "__main__":
    main()
