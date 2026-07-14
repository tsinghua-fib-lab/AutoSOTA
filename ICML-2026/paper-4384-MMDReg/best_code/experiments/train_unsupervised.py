import argparse
import h5py
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jaxopt.linear_solve import solve_lu
from mmd_reg.mmd_unweighted import batched_inner_objective_solutions, expm_skew
from neural_mmd_reg.dataset_without_overlap_masks import PointCloudDataset
from neural_mmd_reg.dataset_without_overlap_masks import RandomRotateSource
from neural_mmd_reg.dataset_without_overlap_masks import RandomShuffle
from neural_mmd_reg.dataset_without_overlap_masks import RandomTranslateSource
from neural_mmd_reg.losses_and_metrics import get_rotation_errors
from neural_mmd_reg.losses_and_metrics import get_rotation_loss
from neural_mmd_reg.losses_and_metrics import get_translation_errors
from neural_mmd_reg.losses_and_metrics import get_translation_loss
from neural_mmd_reg.set_transformer import UnsupervisedModel
from neural_mmd_reg.set_transformer import count_params, save_model
from torch.utils.data import DataLoader, default_collate
from torchvision.transforms import v2


def get_args():
    dists = ["gaussian", "laplace"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist", type=str, default="laplace", choices=dists)
    args = parser.parse_args()
    return args


def get_modelnet40_clean_dataloaders(
    batch_size,
    train_data_path="datasets/processed/modelnet40_clean_train.hdf5",
    val_data_path="datasets/processed/modelnet40_clean_val.hdf5",
):
    """
    Return the training and validation PyTorch DataLoaders.

    Args:
        batch_size: Batch size.
        train_data_path: Path to HDF5 file containing processed training data.
        val_data_path: Path to HDF5 file containing processed validation data.

    Returns:
        Training and validation PyTorch DataLoaders.
    """
    collate_fn = lambda batch: jax.tree.map(jnp.asarray, default_collate(batch))
    train_tr = v2.Compose(
        [
            RandomRotateSource(),
            RandomTranslateSource(),
            RandomShuffle(),
        ]
    )
    train_ds = PointCloudDataset(train_data_path, transform=train_tr)
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_tr = None
    val_ds = PointCloudDataset(val_data_path, transform=val_tr)
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
    )
    return train_dl, val_dl


def loss_fn(model, Ws, batch):
    Xs, Ys, _, _ = batch
    batch_expm_skew = jax.vmap(expm_skew)

    pred_rs, pred_ts = model(Xs, Ys)
    pred_Rs = batch_expm_skew(pred_rs)
    Xs = Xs @ jnp.swapaxes(pred_Rs, 1, 2) + jnp.reshape(pred_ts, (-1, 1, 3))

    init_inner_params = jnp.zeros((len(Ws), 6))
    inner_params = batched_inner_objective_solutions(
        init_inner_params,
        Ws,
        Xs,
        Ys,
        maxiter=100,
        solver="cholesky",
        materialize_jac=True,
        damping_parameter=1e-0,
        implicit_diff_solve=solve_lu,
    )
    mmd_rs = inner_params[:, :3]
    mmd_ts = inner_params[:, 3:]
    mmd_Rs = batch_expm_skew(mmd_rs)
    identity_Rs = jnp.zeros_like(mmd_Rs) + jnp.eye(3)
    identity_ts = jnp.zeros_like(mmd_ts)

    loss = 0.0
    loss = loss + get_rotation_loss(identity_Rs, mmd_Rs)
    loss = loss + get_translation_loss(identity_ts, mmd_ts)

    composed_Rs = jnp.matmul(mmd_Rs, pred_Rs)
    composed_ts = jnp.matvec(mmd_Rs, pred_ts) + mmd_ts

    predictions = (pred_Rs, pred_ts, composed_Rs, composed_ts)
    return loss, predictions


@nnx.jit
def train_step(model, optimizer, Ws, batch):
    value_and_grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, predictions), grads = value_and_grad_fn(model, Ws, batch)
    optimizer.update(model, grads)
    return loss


@nnx.jit
def eval_step(model, Ws, batch):
    Xs, Ys, true_Rs, true_ts = batch
    loss, predictions = loss_fn(model, Ws, batch)
    pred_Rs, pred_ts, composed_Rs, composed_ts = predictions
    net_errors_R = get_rotation_errors(pred_Rs, true_Rs)
    net_errors_t = get_translation_errors(pred_ts, true_ts)
    mmd_errors_R = get_rotation_errors(composed_Rs, true_Rs)
    mmd_errors_t = get_translation_errors(composed_ts, true_ts)
    return loss, net_errors_R, net_errors_t, mmd_errors_R, mmd_errors_t


if __name__ == "__main__":
    args = get_args()
    dist = args.dist
    dist_fn = jax.random.laplace if dist == "laplace" else jax.random.normal

    batch_size = 256
    num_epochs = 10000
    mmd_D = 32

    train_dl, val_dl = get_modelnet40_clean_dataloaders(batch_size)
    M = len(train_dl.dataset)
    N = len(val_dl.dataset)

    num_steps = num_epochs * len(train_dl)
    learning_rate_fn = optax.linear_onecycle_schedule(
        num_steps, 1e-3, 0.01, 0.99
    )
    length_scale_fn = optax.linear_schedule(0.1, 0.05, num_epochs)

    model = UnsupervisedModel(
        dim_i=3,
        dim_h=128,
        num_h=4,
        num_i=64,
        drop=0.1,
        rngs=nnx.Rngs(0),
    )
    print(f"Number of model parameters: {count_params(model):,}")

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.lamb(learning_rate_fn),
    )
    optimizer = nnx.Optimizer(model, optimizer, wrt=nnx.Param)

    # We use changing weights in training and constant weights in validation.
    key = jax.random.key(0)
    key, subkey = jax.random.split(key)
    Ws_eval = dist_fn(subkey, (1, mmd_D, 3))
    Ws_eval = jnp.tile(Ws_eval, (batch_size, 1, 1))

    epochs_save_path = f"results/epochs_unsupervised_{dist}_trained.hdf5"
    params_save_path = f"results/params_unsupervised_{dist}_trained.msgpack"

    for epoch in range(num_epochs):
        length_scale = jnp.asarray(length_scale_fn(epoch))

        model.train()
        train_loss = jnp.asarray(0.0)
        for batch in train_dl:
            B = jnp.size(batch[0], axis=0)
            key, subkey = jax.random.split(key)
            Ws = dist_fn(subkey, (B, mmd_D, 3)) / length_scale
            loss = train_step(model, optimizer, Ws, batch)
            train_loss += loss * B / M

        model.eval()
        val_loss = jnp.asarray(0.0)
        all_net_errors_R = jnp.zeros((N,))
        all_net_errors_t = jnp.zeros((N,))
        all_mmd_errors_R = jnp.zeros((N,))
        all_mmd_errors_t = jnp.zeros((N,))
        idx = 0
        for batch in val_dl:
            B = jnp.size(batch[0], axis=0)  # B is at most batch_size.
            Ws = Ws_eval[:B] / length_scale
            evals = eval_step(model, Ws, batch)
            loss, net_errors_R, net_errors_t, mmd_errors_R, mmd_errors_t = evals
            val_loss += loss * B / N
            all_net_errors_R = all_net_errors_R.at[idx:idx+B].set(net_errors_R)
            all_net_errors_t = all_net_errors_t.at[idx:idx+B].set(net_errors_t)
            all_mmd_errors_R = all_mmd_errors_R.at[idx:idx+B].set(mmd_errors_R)
            all_mmd_errors_t = all_mmd_errors_t.at[idx:idx+B].set(mmd_errors_t)
            idx += B

        print(f"Epoch {epoch+1}/{num_epochs} Averages:")
        print(f"  Length Scale: {length_scale:.5f}")
        print(f"  Train Loss: {train_loss:.5f}")
        print(f"  Val Loss: {val_loss:.5f}")
        print(f"  Val Net Rotation Error: {jnp.mean(all_net_errors_R):.5f}")
        print(f"  Val Net Translation Error: {jnp.mean(all_net_errors_t):.5f}")
        print(f"  Val MMD Rotation Error: {jnp.mean(all_mmd_errors_R):.5f}")
        print(f"  Val MMD Translation Error: {jnp.mean(all_mmd_errors_t):.5f}")

        es = f"epoch_{epoch}"  # Epoch string.
        with h5py.File(epochs_save_path, "x" if epoch == 0 else "r+") as f:
            f.create_dataset(f"{es}/length_scale", data=length_scale)
            f.create_dataset(f"{es}/train_loss", data=train_loss)
            f.create_dataset(f"{es}/val_loss", data=val_loss)
            f.create_dataset(f"{es}/all_net_errors_R", data=all_net_errors_R)
            f.create_dataset(f"{es}/all_net_errors_t", data=all_net_errors_t)
            f.create_dataset(f"{es}/all_mmd_errors_R", data=all_mmd_errors_R)
            f.create_dataset(f"{es}/all_mmd_errors_t", data=all_mmd_errors_t)

    save_model(model, params_save_path)
