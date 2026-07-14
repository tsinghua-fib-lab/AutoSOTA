import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jaxopt.linear_solve import solve_lu
from mmd_reg.mmd_weighted import batched_inner_objective_solutions, expm_skew
from neural_mmd_reg.dataset_with_overlap_masks import PointCloudDataset
from neural_mmd_reg.dataset_with_overlap_masks import RandomCrop
from neural_mmd_reg.dataset_with_overlap_masks import RandomJitter
from neural_mmd_reg.dataset_with_overlap_masks import RandomShuffle
from neural_mmd_reg.dataset_with_overlap_masks import RandomRotateSource
from neural_mmd_reg.dataset_with_overlap_masks import RandomTranslateSource
from neural_mmd_reg.losses_and_metrics import get_rotation_errors
from neural_mmd_reg.losses_and_metrics import get_rotation_loss
from neural_mmd_reg.losses_and_metrics import get_translation_errors
from neural_mmd_reg.losses_and_metrics import get_translation_loss
from neural_mmd_reg.set_transformer import SupervisedModel
from neural_mmd_reg.set_transformer import count_params, restore_model
from neural_mmd_reg.set_transformer import save_model
from torch.utils.data import DataLoader, default_collate
from torchvision.transforms import v2


def get_modelnet40_partial_dataloaders(
    batch_size,
    train_data_path="datasets/processed/modelnet40_partial_train.hdf5",
    val_data_path="datasets/processed/modelnet40_partial_val.hdf5",
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
            RandomCrop(),
            RandomJitter(),
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
    Xs, Ys, true_Rs, true_ts, true_Xos, true_Yos = batch
    batch_expm_skew = jax.vmap(expm_skew)
    pred_rs, pred_ts, pred_ls, pred_Xo_logits, pred_Yo_logits = model(Xs, Ys)
    pred_Rs = batch_expm_skew(pred_rs)
    Ws = Ws / jnp.reshape(pred_ls, (-1, 1, 1))

    init_inner_params = jnp.concatenate([pred_rs, pred_ts], axis=1)
    inner_params = batched_inner_objective_solutions(
        init_inner_params,
        Ws,
        Xs,
        Ys,
        nnx.sigmoid(pred_Xo_logits) >= 0.5,
        nnx.sigmoid(pred_Yo_logits) >= 0.5,
        tol=1e-5,
        maxiter=100,
        solver="cholesky",
        materialize_jac=True,
        damping_parameter=1e-0,
        implicit_diff_solve=solve_lu,
    )
    mmd_rs = inner_params[:, :3]
    mmd_ts = inner_params[:, 3:]
    mmd_Rs = batch_expm_skew(mmd_rs)

    loss = 0.0
    loss = loss + get_rotation_loss(mmd_Rs, true_Rs)
    loss = loss + get_translation_loss(mmd_ts, true_ts)

    predictions = (pred_Rs, pred_ts, mmd_Rs, mmd_ts)
    return loss, predictions


@nnx.jit
def train_step(model, optimizer, Ws, batch):
    wrt = nnx.All(nnx.Param, nnx.PathContains("head_ls"))
    argnums = nnx.DiffState(0, wrt)
    value_and_grad_fn = nnx.value_and_grad(
        loss_fn, argnums=argnums, has_aux=True
    )
    (loss, predictions), grads = value_and_grad_fn(model, Ws, batch)
    optimizer.update(model, grads)
    return loss


@nnx.jit
def eval_step(model, Ws, batch):
    Xs, Ys, true_Rs, true_ts, true_Xos, true_Yos = batch
    loss, predictions = loss_fn(model, Ws, batch)
    pred_Rs, pred_ts, composed_Rs, composed_ts = predictions
    net_errors_R = get_rotation_errors(pred_Rs, true_Rs)
    net_errors_t = get_translation_errors(pred_ts, true_ts)
    mmd_errors_R = get_rotation_errors(composed_Rs, true_Rs)
    mmd_errors_t = get_translation_errors(composed_ts, true_ts)
    return loss, net_errors_R, net_errors_t, mmd_errors_R, mmd_errors_t


if __name__ == "__main__":
    batch_size = 16
    num_epochs = 20
    mmd_D = 16384

    train_dl, val_dl = get_modelnet40_partial_dataloaders(batch_size)
    M = len(train_dl.dataset)
    N = len(val_dl.dataset)

    num_steps = num_epochs * len(train_dl)
    learning_rate_fn = optax.cosine_decay_schedule(1e-7, num_steps)

    model = SupervisedModel(
        dim_i=3,
        dim_h=512,
        num_h=4,
        drop=0.0,
        rngs=nnx.Rngs(0),
    )
    model = restore_model(model, "results/params_supervised_trained.msgpack")
    print(f"Number of model parameters: {count_params(model):,}")

    wrt = nnx.All(nnx.Param, nnx.PathContains("head_ls"))
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate_fn),
    )
    optimizer = nnx.Optimizer(model, optimizer, wrt=wrt)

    # We use changing weights in training and constant weights in validation.
    key = jax.random.key(0)
    key, subkey = jax.random.split(key)
    Ws_eval = jax.random.laplace(subkey, (1, mmd_D, 3))
    Ws_eval = jnp.tile(Ws_eval, (batch_size, 1, 1))

    for epoch in range(num_epochs):
        model.train()
        train_loss = jnp.asarray(0.0)
        for batch in train_dl:
            B = jnp.size(batch[0], axis=0)
            key, subkey = jax.random.split(key)
            Ws = jax.random.laplace(subkey, (B, mmd_D, 3))
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
            Ws = Ws_eval[:B]
            evals = eval_step(model, Ws, batch)
            loss, net_errors_R, net_errors_t, mmd_errors_R, mmd_errors_t = evals
            val_loss += loss * B / N
            all_net_errors_R = all_net_errors_R.at[idx:idx+B].set(net_errors_R)
            all_net_errors_t = all_net_errors_t.at[idx:idx+B].set(net_errors_t)
            all_mmd_errors_R = all_mmd_errors_R.at[idx:idx+B].set(mmd_errors_R)
            all_mmd_errors_t = all_mmd_errors_t.at[idx:idx+B].set(mmd_errors_t)
            idx += B

        print(f"Epoch {epoch+1}/{num_epochs} Averages:")
        print(f"  Train Loss: {train_loss:.5f}")
        print(f"  Val Loss: {val_loss:.5f}")
        print(f"  Val Net Rotation Error: {jnp.mean(all_net_errors_R):.5f}")
        print(f"  Val Net Translation Error: {jnp.mean(all_net_errors_t):.5f}")
        print(f"  Val MMD Rotation Error: {jnp.mean(all_mmd_errors_R):.5f}")
        print(f"  Val MMD Translation Error: {jnp.mean(all_mmd_errors_t):.5f}")

    save_model(model, "results/params_supervised_tuned.msgpack")
