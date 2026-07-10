import os
import gc
import sys
import json
import functools
import time

import numpy as np
import jax
import jax.numpy as jnp
import optax

import pcx as px
import pcx.predictive_coding as pxc
import pcx.nn as pxnn
import pcx.utils as pxu
import pcx.functional as pxf

import stune
from omegacli import OmegaConf

from model_BN import get_model
from dataset import get_loader, get_datasetinfo


def _collect_vodes(model, layer_keys):
    vode_list = []
    for key in layer_keys:
        v = model.vodes[key]
        if isinstance(v, list):
            vode_list.extend(v)
        else:
            vode_list.append(v)
    return vode_list


@pxf.vmap(pxu.M(pxc.VodeParam | pxc.VodeParam.Cache).to((None, 0)), in_axes=(0, 0), out_axes=0, axis_name="batch")
def forward(x, y, *, model):
    return model(x, y, init_step=True)


@pxf.vmap(pxu.M(pxc.VodeParam | pxc.VodeParam.Cache).to((None, 0)), in_axes=(0, 0), out_axes=0, axis_name="batch")
def eval_forward(x, y, *, model):
    return model(x, y, inference=True)


@pxf.vmap(pxu.M(pxc.VodeParam | pxc.VodeParam.Cache).to((None, 0)), in_axes=(0,), out_axes=(None, 0), axis_name="batch")
def energy(x, *, model, t=0):
    N = len(model.vodes)
    start_idx = max(N - t - 1, 0)
    y_ = model(x, None, ind=start_idx, init_step=True)
    layer_keys = [f'layer_{i}' for i in range(start_idx, N)]
    vode_list = _collect_vodes(model, layer_keys)
    energies = [vode.energy_step(t=t) for vode in vode_list]
    total_energy = functools.reduce(lambda a, b: a + b, energies)
    return jax.lax.psum(total_energy, "batch"), y_


@pxf.vmap(pxu.M(pxc.VodeParam | pxc.VodeParam.Cache).to((None, 0)), in_axes=(0,), out_axes=(None, 0), axis_name="batch")
def energyW_FU(x, *, model, T=0, out_key="h0"):
    y_ = model(x, None, out_key=out_key)
    N = len(model.vodes)
    layer_keys = [f'layer_{i}' for i in range(N)]
    vode_list = _collect_vodes(model, layer_keys)
    energies = [vode.energy() for vode in vode_list]
    total_energy = functools.reduce(lambda a, b: a + b, energies)
    return jax.lax.psum(total_energy, "batch"), y_


@pxf.vmap(pxu.M(pxc.VodeParam | pxc.VodeParam.Cache).to((None, 0)), in_axes=(0,), out_axes=(None, 0), axis_name="batch")
def energyW_PC(x, *, model, T=0, out_key="h"):
    y_ = model(x, None, out_key=out_key)
    N = len(model.vodes)
    layer_keys = [f'layer_{i}' for i in range(N)]
    vode_list = _collect_vodes(model, layer_keys)
    energies = [vode.energy() for vode in vode_list]
    total_energy = functools.reduce(lambda a, b: a + b, energies)
    return jax.lax.psum(total_energy, "batch"), y_


def _make_train_on_batch(energyW_fn):
    @pxf.jit(static_argnums=0, donate_argnames=("model", "optim"))
    def _train_on_batch(T: int, x: jax.Array, y: jax.Array, *, model, optim_w: pxu.Optim, optim_h: pxu.Optim, beta: float = 1.0):
        model.train()

        with pxu.step(model, (pxc.STATUS.INIT, None), clear_params=pxc.VodeParam.Cache):
            logits = forward(x, y, model=model)

        optim_h.init(pxu.M_hasnot(pxc.VodeParam, frozen=True)(model))

        for t in range(T):
            def _energy(x, model, ind=t):
                return energy(x, model=model, t=ind)
            with pxu.step(model, clear_params=pxc.VodeParam.Cache):
                mask = pxu.M_hasnot(pxc.VodeParam, frozen=True).to([False, True])(model, is_pytree=True)
                _, g = pxf.value_and_grad({"model": mask}, has_aux=True)(_energy)(x, model=model)
            optim_h.step(model, g["model"])

        optim_h.clear()

        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            (e, _), g = pxf.value_and_grad(
                pxu.M(pxnn.LayerParam | pxnn.BatchNormParam).to([False, True]), has_aux=True
            )(energyW_fn)(x, model=model, T=T)
        optim_w.step(model, g["model"], scale_by=1.0 / x.shape[0], apply_updates=True)

        return logits, None

    return _train_on_batch


@pxf.jit()
def eval_on_batch(x: jax.Array, y: jax.Array, *, model):
    model.eval()

    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache | pxc.VodeParam):
        outputs = eval_forward(x, None, model=model)
    top1_pred = outputs.argmax(axis=-1)
    top5_indices = jax.lax.top_k(outputs, k=5)[1]

    top1_acc = (top1_pred == y).mean()
    top5_acc = jnp.any(top5_indices == y[:, None], axis=-1).mean()

    return top1_acc, top5_acc, top1_pred


def train(dl, T, *, model, train_on_batch_fn, optim_w: pxu.Optim, optim_h: pxu.Optim, beta: float = 1.0):
    start_time = time.time()
    for i, (x, y) in enumerate(dl):
        train_on_batch_fn(
            T, x, jax.nn.one_hot(y, model.nm_classes.get()),
            model=model, optim_w=optim_w, optim_h=optim_h, beta=beta,
        )
    end_time = time.time()
    return end_time - start_time


def eval(dl, *, model):
    acc = []
    acc5 = []
    ys_ = []

    for x, y in dl:
        a, a5, y_ = eval_on_batch(x, y, model=model)
        acc.append(a)
        acc5.append(a5)
        ys_.append(y_)

    return np.mean(acc), np.mean(acc5), np.concatenate(ys_)


def main(run_info: stune.RunInfo, save_model: bool, savepath: str = None, seed_idx: int = 0):
    batch_size = run_info["hp/batch_size"]
    nm_epochs = run_info["hp/epochs"]
    dataset_name = run_info["hp/dataset"]
    model_name = run_info["hp/model"]

    nm_classes, input_size = get_datasetinfo(dataset_name, model_name)

    model = get_model(
        model_name=model_name,
        nm_classes=nm_classes,
        act_fn=getattr(jax.nn, run_info["hp/act_fn"]),
        input_size=input_size,
        se_flag=run_info["hp/se_flag"],
        T=run_info["hp/T"],
        alpha=run_info["hp/alpha"],
        precision_type=run_info["hp/precision_type"],
        lr_h=run_info["hp/optim/x/lr"],
    )

    train_dataloader, test_dataloader = get_loader(dataset_name, batch_size, input_size)

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=run_info["hp/optim/w/lr"],
        peak_value=1.1 * run_info["hp/optim/w/lr"],
        warmup_steps=0.1 * len(train_dataloader) * nm_epochs,
        decay_steps=len(train_dataloader) * nm_epochs,
        end_value=0.1 * run_info["hp/optim/w/lr"],
        exponent=1.0,
    )

    try:
        nesterov = run_info["hp/optim/x/nesterov"]
    except (KeyError, AttributeError):
        nesterov = True

    try:
        forward_type = run_info["hp/Forward_type"]
    except (KeyError, AttributeError):
        forward_type = "FU"

    energyW_fn = energyW_FU if forward_type == "FU" else energyW_PC
    train_on_batch_fn = _make_train_on_batch(energyW_fn)

    optim_h = pxu.Optim(
        lambda: optax.sgd(run_info["hp/optim/x/lr"], momentum=run_info["hp/optim/x/momentum"], nesterov=nesterov),
    )
    optim_w = pxu.Optim(
        lambda: optax.adamw(schedule, weight_decay=run_info["hp/optim/w/wd"]),
        pxu.M(pxnn.LayerParam | pxnn.BatchNormParam)(model),
    )

    best_accuracy = 0
    best_accuracy5 = 0
    accuracies = []
    accuracies5 = []

    below_times = 0
    best_times = 0
    for e in range(50):
        train_time = train(train_dataloader, T=run_info["hp/T"], model=model,
                           train_on_batch_fn=train_on_batch_fn, optim_w=optim_w, optim_h=optim_h, beta=1.0)
        a, a5, y = eval(test_dataloader, model=model)
        if e > 5 and float(a) < 0.1:
            below_times += 1
        else:
            below_times = 0
        print(f"Epoch {e}: top1={a}, top5={a5}")

        if a > best_accuracy:
            best_times = 0
            best_accuracy = a
            if save_model:
                save_name = f"./weights/{model_name}_{dataset_name}_{run_info['hp/precision_type']}_{run_info['hp/Forward_type']}_BN"
                pxu.save_params(model, save_name)
                print("Model saved")
        else:
            best_times += 1
        if a5 > best_accuracy5:
            best_accuracy5 = a5
        accuracies.append(float(a))
        accuracies5.append(float(a5))

        if savepath and (e + 1) % 10 == 0:
            data = {"status": "running", "seed_idx": seed_idx, "epoch": e + 1}
            if os.path.exists(savepath):
                with open(savepath, "r") as f:
                    data = json.load(f)
            data[f"seed_{seed_idx}"] = {
                "epoch": e + 1,
                "best_accuracy": float(best_accuracy),
                "best_accuracy5": float(best_accuracy5),
                "accuracies": accuracies,
                "accuracies5": accuracies5,
            }
            data["status"] = "running"
            with open(savepath, "w") as f:
                json.dump(data, f, indent=4)

        if below_times >= 5 or best_times >= 10:
            break

    del train_dataloader
    del test_dataloader
    gc.collect()

    return float(best_accuracy), float(best_accuracy5), accuracies, accuracies5


if __name__ == "__main__":
    import seed

    run_info = stune.RunInfo(OmegaConf.load(sys.argv[1]))
    seed.run(main)(run_info)
