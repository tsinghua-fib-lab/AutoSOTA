import argparse
import jax
import jax.numpy as jnp
from flax import nnx
from jaxopt.linear_solve import solve_lu
from mmd_reg.mmd_unweighted import batched_inner_objective_solutions, expm_skew
from neural_mmd_reg.dataset_without_overlap_masks import PointCloudDataset
from neural_mmd_reg.losses_and_metrics import get_rotation_errors
from neural_mmd_reg.losses_and_metrics import get_rotation_loss
from neural_mmd_reg.losses_and_metrics import get_translation_errors
from neural_mmd_reg.losses_and_metrics import get_translation_loss
from neural_mmd_reg.set_transformer import UnsupervisedModel
from neural_mmd_reg.set_transformer import count_params, restore_model
from torch.utils.data import DataLoader, default_collate


def get_args():
    dists = ["gaussian", "laplace"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist", type=str, default="laplace", choices=dists)
    args = parser.parse_args()
    return args


def get_modelnet40_clean_dataloader(
    batch_size,
    test_data_path="datasets/processed/modelnet40_clean_test.hdf5",
):
    """
    Return the testing PyTorch DataLoader.

    Args:
        batch_size: Batch size.
        test_data_path: Path to HDF5 file containing processed test data.

    Returns:
        Testing PyTorch DataLoader.
    """
    collate_fn = lambda batch: jax.tree.map(jnp.asarray, default_collate(batch))
    test_tr = None
    test_ds = PointCloudDataset(test_data_path, transform=test_tr)
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
    )
    return test_dl


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
        tol=2e-5,
        maxiter=100,
        solver="cholesky",
        materialize_jac=True,
        damping_parameter=1.0,
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

    batch_size = 32
    mmd_D = 128
    length_scale = 0.05

    test_dl = get_modelnet40_clean_dataloader(batch_size)
    N = len(test_dl.dataset)

    model = UnsupervisedModel(
        dim_i=3,
        dim_h=128,
        num_h=4,
        num_i=64,
        drop=0.1,
        rngs=nnx.Rngs(0),
    )
    model = restore_model(
        model, f"results/params_unsupervised_{dist}_trained.msgpack"
    )
    print(f"Number of model parameters: {count_params(model):,}")

    key = jax.random.key(0)

    model.eval()
    test_loss = jnp.asarray(0.0)
    all_net_errors_R = jnp.zeros((N,))
    all_net_errors_t = jnp.zeros((N,))
    all_mmd_errors_R = jnp.zeros((N,))
    all_mmd_errors_t = jnp.zeros((N,))
    idx = 0
    for batch in test_dl:
        B = jnp.size(batch[0], axis=0)
        key, subkey = jax.random.split(key)
        Ws = dist_fn(subkey, (B, mmd_D, 3)) / length_scale
        evals = eval_step(model, Ws, batch)
        loss, net_errors_R, net_errors_t, mmd_errors_R, mmd_errors_t = evals
        test_loss += loss * B / N
        all_net_errors_R = all_net_errors_R.at[idx:idx+B].set(net_errors_R)
        all_net_errors_t = all_net_errors_t.at[idx:idx+B].set(net_errors_t)
        all_mmd_errors_R = all_mmd_errors_R.at[idx:idx+B].set(mmd_errors_R)
        all_mmd_errors_t = all_mmd_errors_t.at[idx:idx+B].set(mmd_errors_t)
        idx += B

    print(f"Test Loss: {test_loss:.5f}")
    print(f"Test Net Rotation Error: {jnp.mean(all_net_errors_R):.5f}")
    print(f"Test Net Translation Error: {jnp.mean(all_net_errors_t):.5f}")
    print(f"Test MMD Rotation Error: {jnp.mean(all_mmd_errors_R):.5f}")
    print(f"Test MMD Translation Error: {jnp.mean(all_mmd_errors_t):.5f}")
