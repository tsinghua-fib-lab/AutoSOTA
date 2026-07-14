import h5py
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from mmd_reg.mmd_weighted import expm_skew
from neural_mmd_reg.dataset_with_overlap_masks import PointCloudDataset
from neural_mmd_reg.dataset_with_overlap_masks import RandomCrop
from neural_mmd_reg.dataset_with_overlap_masks import RandomJitter
from neural_mmd_reg.dataset_with_overlap_masks import RandomShuffle
from neural_mmd_reg.dataset_with_overlap_masks import RandomRotateSource
from neural_mmd_reg.dataset_with_overlap_masks import RandomTranslateSource
from neural_mmd_reg.losses_and_metrics import get_overlap_accuracy
from neural_mmd_reg.losses_and_metrics import get_overlap_loss
from neural_mmd_reg.losses_and_metrics import get_rotation_errors
from neural_mmd_reg.losses_and_metrics import get_rotation_loss
from neural_mmd_reg.losses_and_metrics import get_translation_errors
from neural_mmd_reg.losses_and_metrics import get_translation_loss
from neural_mmd_reg.set_transformer import SupervisedModel
from neural_mmd_reg.set_transformer import count_params, save_model
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


def loss_fn(model, batch):
    Xs, Ys, true_Rs, true_ts, true_Xos, true_Yos = batch
    batch_expm_skew = jax.vmap(expm_skew)
    pred_rs, pred_ts, pred_ls, pred_Xo_logits, pred_Yo_logits = model(Xs, Ys)
    pred_Rs = batch_expm_skew(pred_rs)
    loss = 0.0
    loss = loss + get_rotation_loss(pred_Rs, true_Rs)
    loss = loss + get_translation_loss(pred_ts, true_ts)
    loss = loss + get_overlap_loss(pred_Xo_logits, true_Xos)
    loss = loss + get_overlap_loss(pred_Yo_logits, true_Yos)
    predictions = (pred_Rs, pred_ts, pred_Xo_logits, pred_Yo_logits)
    return loss, predictions


@nnx.jit
def train_step(model, optimizer, batch):
    value_and_grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, predictions), grads = value_and_grad_fn(model, batch)
    optimizer.update(model, grads)
    return loss


@nnx.jit
def eval_step(model, batch):
    Xs, Ys, true_Rs, true_ts, true_Xos, true_Yos = batch
    loss, predictions = loss_fn(model, batch)
    pred_Rs, pred_ts, pred_Xo_logits, pred_Yo_logits = predictions
    net_errors_R = get_rotation_errors(pred_Rs, true_Rs)
    net_errors_t = get_translation_errors(pred_ts, true_ts)
    overlap_accuracy = get_overlap_accuracy(
        pred_Xo_logits, pred_Yo_logits, true_Xos, true_Yos
    )
    return loss, overlap_accuracy, net_errors_R, net_errors_t


if __name__ == "__main__":
    batch_size = 16
    num_epochs = 20000

    train_dl, val_dl = get_modelnet40_partial_dataloaders(batch_size)
    M = len(train_dl.dataset)
    N = len(val_dl.dataset)

    num_steps = num_epochs * len(train_dl)
    learning_rate_fn = optax.warmup_cosine_decay_schedule(
        1e-6, 1e-4, 1000, num_steps
    )

    model = SupervisedModel(
        dim_i=3,
        dim_h=512,
        num_h=4,
        drop=0.0,
        rngs=nnx.Rngs(0),
    )
    print(f"Number of model parameters: {count_params(model):,}")

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate_fn),
    )
    optimizer = nnx.Optimizer(model, optimizer, wrt=nnx.Param)

    epochs_save_path = "results/epochs_supervised_trained.hdf5"
    params_save_path = "results/params_supervised_trained.msgpack"

    for epoch in range(num_epochs):
        model.train()
        train_loss = jnp.asarray(0.0)
        for batch in train_dl:
            B = jnp.size(batch[0], axis=0)
            loss = train_step(model, optimizer, batch)
            train_loss += loss * B / M
        
        model.eval()
        val_loss = jnp.asarray(0.0)
        val_overlap_acc = jnp.asarray(0.0)
        all_net_errors_R = jnp.zeros((N,))
        all_net_errors_t = jnp.zeros((N,))
        idx = 0
        for batch in val_dl:
            B = jnp.size(batch[0], axis=0)
            evals = eval_step(model, batch)
            loss, overlap_accuracy, net_errors_R, net_errors_t = evals
            val_loss += loss * B / N
            val_overlap_acc += overlap_accuracy * B / N
            all_net_errors_R = all_net_errors_R.at[idx:idx+B].set(net_errors_R)
            all_net_errors_t = all_net_errors_t.at[idx:idx+B].set(net_errors_t)
            idx += B

        print(f"Epoch {epoch+1:,}/{num_epochs:,} Averages:")
        print(f"  Train Loss: {train_loss:.5f}")
        print(f"  Val Loss: {val_loss:.5f}")
        print(f"  Val Net Overlap Accuracy: {val_overlap_acc:.5f}")
        print(f"  Val Net Rotation Error: {jnp.mean(all_net_errors_R):.5f}")
        print(f"  Val Net Translation Error: {jnp.mean(all_net_errors_t):.5f}")

        es = f"epoch_{epoch}"  # Epoch string.
        with h5py.File(epochs_save_path, "x" if epoch == 0 else "r+") as f:
            f.create_dataset(f"{es}/train_loss", data=train_loss)
            f.create_dataset(f"{es}/val_loss", data=val_loss)
            f.create_dataset(f"{es}/val_overlap_acc", data=val_overlap_acc)
            f.create_dataset(f"{es}/all_net_errors_R", data=all_net_errors_R)
            f.create_dataset(f"{es}/all_net_errors_t", data=all_net_errors_t)

    save_model(model, params_save_path)
