import jax
import jax.numpy as jnp
from flax import nnx
from flax.serialization import to_bytes, from_bytes


class MAB(nnx.Module):
    """Multihead Attention Block (MAB).

    Args:
        dim_Q: Dimension of input query features.
        dim_K: Dimension of input key/value features.
        dim_V: Dimension of output features. Must be divisible by `num_h`.
        num_h: Number of attention heads.
        drop: Dropout rate.
        rngs: RNG key.
    """

    def __init__(self, dim_Q, dim_K, dim_V, num_h, drop, rngs: nnx.Rngs):
        self.num_heads = num_h
        self.dim_split = dim_V // num_h
        kernel_init = nnx.nn.initializers.he_normal()
        self.li_q = nnx.Linear(dim_Q, dim_V, kernel_init=kernel_init, rngs=rngs)
        self.li_k = nnx.Linear(dim_K, dim_V, kernel_init=kernel_init, rngs=rngs)
        self.li_v = nnx.Linear(dim_K, dim_V, kernel_init=kernel_init, rngs=rngs)
        self.li_o = nnx.Linear(dim_V, dim_V, kernel_init=kernel_init, rngs=rngs)
        self.dr_a = nnx.Dropout(drop, rngs=rngs)
        self.dr_o = nnx.Dropout(drop, rngs=rngs)
        self.rms_0 = nnx.LayerNorm(dim_V, rngs=rngs)
        self.rms_1 = nnx.LayerNorm(dim_V, rngs=rngs)

    def __call__(self, Q, K):
        Q_ = self.li_q(Q)
        K_ = self.li_k(K)
        V_ = self.li_v(K)
        Q_ = jnp.concatenate(jnp.split(Q_, self.num_heads, axis=2), axis=0)
        K_ = jnp.concatenate(jnp.split(K_, self.num_heads, axis=2), axis=0)
        V_ = jnp.concatenate(jnp.split(V_, self.num_heads, axis=2), axis=0)
        A = jnp.matmul(Q_, jnp.swapaxes(K_, 1, 2)) / jnp.sqrt(self.dim_split)
        A = jax.nn.softmax(A, axis=2)
        A = self.dr_a(A)
        O = Q_ + jnp.matmul(A, V_)
        O = jnp.split(O, self.num_heads, axis=0)
        O = jnp.concatenate(O, axis=2)
        O = self.rms_0(O)
        O = O + self.dr_o(nnx.relu(self.li_o(O)))
        O = self.rms_1(O)
        return O


class SAB(nnx.Module):
    """Set Attention Block (SAB).

    Args:
        dim_i: Dimension of input features.
        dim_o: Dimension of output features. Must be divisible by `num_h`.
        num_h: Number of attention heads.
        drop: Dropout rate.
        rngs: RNG key.
    """

    def __init__(self, dim_i, dim_o, num_h, drop, rngs: nnx.Rngs):
        self.mab = MAB(dim_i, dim_i, dim_o, num_h, drop, rngs=rngs)

    def __call__(self, X):
        return self.mab(X, X)


class ISAB(nnx.Module):
    """Induced Set Attention Block (ISAB).

    Args:
        dim_i: Dimension of input features.
        dim_o: Dimension of output features. Must be divisible by `num_h`.
        num_h: Number of attention heads.
        num_i: Number of inducing points.
        drop: Dropout rate.
        rngs: RNG key.
    """

    def __init__(self, dim_i, dim_o, num_h, num_i, drop, rngs: nnx.Rngs):
        initializer = nnx.nn.initializers.he_normal()
        self.I = nnx.Param(
            initializer(rngs.params(), (1, num_i, dim_o), jnp.float32)
        )
        self.mab_0 = MAB(dim_o, dim_i, dim_o, num_h, drop, rngs=rngs)
        self.mab_1 = MAB(dim_i, dim_o, dim_o, num_h, drop, rngs=rngs)

    def __call__(self, X):
        I_tile = jnp.tile(self.I, (jnp.size(X, axis=0), 1, 1))
        H = self.mab_0(I_tile, X)
        return self.mab_1(X, H)


class CAB(nnx.Module):
    """Cross Attention Block (CAB).

    Args:
        dim_i: Dimension of input features.
        dim_o: Dimension of output features. Must be divisible by `num_h`.
        num_h: Number of attention heads.
        drop: Dropout rate.
        rngs: RNG key.
    """

    def __init__(self, dim_i, dim_o, num_h, drop, rngs: nnx.Rngs):
        self.mab = MAB(dim_i, dim_i, dim_o, num_h, drop, rngs=rngs)

    def __call__(self, X, Y):
        return self.mab(X, Y)


class ICAB(nnx.Module):
    """Induced Cross Attention Block (ICAB).

    Args:
        dim_i: Dimension of input features.
        dim_o: Dimension of output features. Must be divisible by `num_h`.
        num_h: Number of attention heads.
        num_i: Number of inducing points.
        drop: Dropout rate.
        rngs: RNG key.
    """

    def __init__(self, dim_i, dim_o, num_h, num_i, drop, rngs: nnx.Rngs):
        initializer = nnx.nn.initializers.he_normal()
        self.I = nnx.Param(
            initializer(rngs.params(), (1, num_i, dim_o), jnp.float32)
        )
        self.mab_0 = MAB(dim_o, dim_i, dim_o, num_h, drop, rngs=rngs)
        self.mab_1 = MAB(dim_i, dim_o, dim_o, num_h, drop, rngs=rngs)

    def __call__(self, X, Y):
        I_tile = jnp.tile(self.I, (jnp.size(X, axis=0), 1, 1))
        H = self.mab_0(I_tile, Y)
        return self.mab_1(X, H)


class PMA(nnx.Module):
    """Pooling by Multihead Attention (PMA).

    Args:
        dim: Dimension of input/output features. Must be divisible by `num_h`.
        num_h: Number of attention heads.
        num_s: Number of learnable seed vectors used for pooling.
        drop: Dropout rate.
        rngs: RNG key.
    """

    def __init__(self, dim, num_h, num_s, drop, rngs: nnx.Rngs):
        initializer = nnx.nn.initializers.he_normal()
        self.S = nnx.Param(
            initializer(rngs.params(), (1, num_s, dim), jnp.float32)
        )
        self.mab = MAB(dim, dim, dim, num_h, drop, rngs=rngs)

    def __call__(self, X):
        S_tile = jnp.tile(self.S, (jnp.size(X, axis=0), 1, 1))
        return self.mab(S_tile, X)


class Residual(nnx.Module):
    """Residual connection.

    Args:
        dim_i: Dimension of input features.
        dim_o: Dimension of output features.
        rngs: RNG key.
    """

    def __init__(self, dim_i, dim_o, rngs: nnx.Rngs):
        if dim_i != dim_o:
            kernel_init = nnx.nn.initializers.he_normal()
            self.layer = nnx.Linear(
                dim_i, dim_o, kernel_init=kernel_init, rngs=rngs
            )
        else:
            self.layer = lambda X: X

    def __call__(self, X):
        return self.layer(X)


class ResEncoderBlock(nnx.Module):
    """Residual Encoder Block.

    Args:
        dim_i: Dimension of input features.
        dim_o: Dimension of output features. Must be divisible by `num_h`.
        num_h: Number of attention heads.
        num_i: Number of inducing points.
        drop: Dropout rate.
        rngs: RNG key.
    """

    def __init__(self, dim_i, dim_o, num_h, num_i, drop, rngs: nnx.Rngs):
        self.isab_residual = Residual(dim_i, dim_o, rngs=rngs)
        self.isab = ISAB(dim_i, dim_o, num_h, num_i, drop, rngs=rngs)
        self.icab = ICAB(dim_o, dim_o, num_h, num_i, drop, rngs=rngs)

    def __call__(self, X, Y):
        feat_X = self.isab(X) + self.isab_residual(X)
        feat_Y = self.isab(Y) + self.isab_residual(Y)
        feat_X = self.icab(feat_X, feat_Y) + feat_X
        feat_Y = self.icab(feat_Y, feat_X) + feat_Y
        return feat_X, feat_Y


class EncoderS(nnx.Module):
    """Encoder for supervised model.

    Args:
        dim_i: Dimension of input features.
        dim_h: Dimension of hidden features. Must be divisible by `num_h`.
        num_h: Number of attention heads.
        drop: Dropout rate.
        rngs: RNG key.
    """

    def __init__(self, dim_i, dim_h, num_h, drop, rngs: nnx.Rngs):
        self.cab_0 = CAB(dim_i, dim_h, num_h, drop, rngs=rngs)
        self.cab_1 = CAB(dim_h, dim_h, num_h, drop, rngs=rngs)
        self.cab_2 = CAB(dim_h, dim_h, num_h, drop, rngs=rngs)
        self.cab_3 = CAB(dim_h, dim_h, num_h, drop, rngs=rngs)
        self.cab_4 = CAB(dim_h, dim_h, num_h, drop, rngs=rngs)
        self.cab_5 = CAB(dim_h, dim_h, num_h, drop, rngs=rngs)
        self.cab_6 = CAB(dim_h, dim_h, num_h, drop, rngs=rngs)
        self.cab_7 = CAB(dim_h, dim_h, num_h, drop, rngs=rngs)
        self.sab = SAB(2 * dim_h, dim_h, num_h, drop, rngs=rngs)

    def __call__(self, X, Y):
        feat_X = self.cab_0(X, Y)
        feat_Y = self.cab_0(Y, X)
        feat_X = self.cab_1(feat_X, feat_Y)
        feat_Y = self.cab_1(feat_Y, feat_X)
        feat_X = self.cab_2(feat_X, feat_Y)
        feat_Y = self.cab_2(feat_Y, feat_X)
        feat_X = self.cab_3(feat_X, feat_Y)
        feat_Y = self.cab_3(feat_Y, feat_X)
        feat_X = self.cab_4(feat_X, feat_Y)
        feat_Y = self.cab_4(feat_Y, feat_X)
        feat_X = self.cab_5(feat_X, feat_Y)
        feat_Y = self.cab_5(feat_Y, feat_X)
        feat_X = self.cab_6(feat_X, feat_Y)
        feat_Y = self.cab_6(feat_Y, feat_X)
        feat_X = self.cab_7(feat_X, feat_Y)
        feat_Y = self.cab_7(feat_Y, feat_X)
        fusion = jnp.concatenate([feat_X, feat_Y], axis=2)
        fusion = self.sab(fusion)
        return fusion


class EncoderU(nnx.Module):
    """Encoder for Unsupervised Model.

    Args:
        dim_i: Dimension of input features.
        dim_h: Dimension of hidden features. Must be divisible by `num_h`.
        num_h: Number of attention heads.
        num_i: Number of inducing points.
        drop: Dropout rate.
        rngs: RNG key.
    """

    def __init__(self, dim_i, dim_h, num_h, num_i, drop, rngs: nnx.Rngs):
        self.eb_0 = ResEncoderBlock(dim_i, dim_h, num_h, num_i, drop, rngs=rngs)
        self.eb_1 = ResEncoderBlock(dim_h, dim_h, num_h, num_i, drop, rngs=rngs)
        self.eb_2 = ResEncoderBlock(dim_h, dim_h, num_h, num_i, drop, rngs=rngs)
        self.eb_3 = ResEncoderBlock(dim_h, dim_h, num_h, num_i, drop, rngs=rngs)
        self.eb_4 = ResEncoderBlock(dim_h, dim_h, num_h, num_i, drop, rngs=rngs)
        self.eb_5 = ResEncoderBlock(dim_h, dim_h, num_h, num_i, drop, rngs=rngs)
        self.eb_6 = ResEncoderBlock(dim_h, dim_h, num_h, num_i, drop, rngs=rngs)
        self.eb_7 = ResEncoderBlock(dim_h, dim_h, num_h, num_i, drop, rngs=rngs)
        self.icab = ICAB(dim_h, dim_h, num_h, num_i, drop, rngs=rngs)
        self.isab = ISAB(dim_h, dim_h, num_h, num_i, drop, rngs=rngs)

    def __call__(self, X, Y):
        feat_X, feat_Y = self.eb_0(X, Y)  # (B, N, dim_h), (B, N, dim_h)
        feat_X, feat_Y = self.eb_1(feat_X, feat_Y)
        feat_X, feat_Y = self.eb_2(feat_X, feat_Y)
        feat_X, feat_Y = self.eb_3(feat_X, feat_Y)
        feat_X, feat_Y = self.eb_4(feat_X, feat_Y)
        feat_X, feat_Y = self.eb_5(feat_X, feat_Y)
        feat_X, feat_Y = self.eb_6(feat_X, feat_Y)
        feat_X, feat_Y = self.eb_7(feat_X, feat_Y)
        fusion = self.icab(feat_X, feat_Y)
        fusion = self.isab(fusion)
        return fusion


class SABHeadS(nnx.Module):
    """Decoder head for local features for supervised model.

    Args:
        dim_i: Dimension of input features.
        dim_h: Dimension of hidden features. Must be divisible by `num_h`.
        dim_o: Dimension of output features. Must be divisible by `num_h`.
        num_h: Number of attention heads.
        num_i: Number of inducing points.
        drop: Dropout rate.
        rngs: RNG key.
    """

    def __init__(self, dim_i, dim_h, dim_o, num_h, drop, rngs: nnx.Rngs):
        kernel_init = nnx.nn.initializers.he_normal()
        self.sab_0 = SAB(dim_i, dim_h, num_h, drop, rngs=rngs)
        self.li_0 = nnx.Linear(dim_i, dim_o, kernel_init=kernel_init, rngs=rngs)

    def __call__(self, fusion):
        fusion = self.sab_0(fusion) # (B, N, dim_h)
        fusion = self.li_0(fusion)  # (B, N, dim_o)
        return fusion


class PoolHeadS(nnx.Module):
    """Decoder head for global features for supervised model.

    Args:
        dim_i: Dimension of input features.
        dim_h: Dimension of hidden features. Must be divisible by `num_h`.
        dim_o: Dimension of output features. Must be divisible by `num_h`.
        num_h: Number of attention heads.
        num_i: Number of inducing points.
        drop: Dropout rate.
        rngs: RNG key.
    """

    def __init__(self, dim_i, dim_h, dim_o, num_h, drop, rngs: nnx.Rngs):
        kernel_init = nnx.nn.initializers.he_normal()
        self.dim_h = dim_h
        self.sab_0 = SAB(dim_i, dim_h, num_h, drop, rngs=rngs)
        self.pool = PMA(dim_i, num_h, 1, drop, rngs=rngs)
        self.li_0 = nnx.Linear(dim_h, dim_o, kernel_init=kernel_init, rngs=rngs)

    def __call__(self, fusion):
        fusion = self.sab_0(fusion) # (B, N, dim_h)
        fusion = self.pool(fusion)  # (B, 1, dim_h)
        fusion = jnp.reshape(fusion, (-1, self.dim_h))  # (B, dim_h)
        fusion = self.li_0(fusion)  # (B, dim_o)
        return fusion


class PoolHeadU(nnx.Module):
    """Decoder head for global features for unsupervised model.

    Args:
        dim_i: Dimension of input features.
        dim_h: Dimension of hidden features. Must be divisible by `num_h`.
        dim_o: Dimension of output features. Must be divisible by `num_h`.
        num_h: Number of attention heads.
        num_i: Number of inducing points.
        drop: Dropout rate.
        rngs: RNG key.
    """

    def __init__(self, dim_i, dim_h, dim_o, num_h, num_i, drop, rngs: nnx.Rngs):
        kernel_init = nnx.nn.initializers.he_normal()
        self.dim_h = dim_h
        self.isab = ISAB(dim_i, dim_h, num_h, num_i, drop, rngs=rngs)
        self.pool = PMA(dim_i, num_h, 1, drop, rngs=rngs)
        self.li_0 = nnx.Linear(dim_h, dim_o, kernel_init=kernel_init, rngs=rngs)

    def __call__(self, fusion):
        fusion = self.isab(fusion)  # (B, N, dim_h)
        fusion = self.pool(fusion)  # (B, 1, dim_h)
        fusion = jnp.reshape(fusion, (-1, self.dim_h))  # (B, dim_h)
        fusion = self.li_0(fusion)  # (B, dim_o)
        return fusion


class UnsupervisedModel(nnx.Module):
    """Set Transformer for unsupervised model.

    Args:
        dim_i: Dimension of input features.
        dim_h: Dimension of hidden features. Must be divisible by `num_h`.
        num_h: Number of attention heads.
        num_i: Number of inducing points.
        drop: Dropout rate.
        rngs: RNG key.
    """

    def __init__(self, dim_i, dim_h, num_h, num_i, drop, rngs: nnx.Rngs):
        self.encoder = EncoderU(dim_i, dim_h, num_h, num_i, drop, rngs=rngs)
        self.head_rs = PoolHeadU(dim_h, dim_h, 3, num_h, num_i, drop, rngs=rngs)
        self.head_ts = PoolHeadU(dim_h, dim_h, 3, num_h, num_i, drop, rngs=rngs)

    def __call__(self, X, Y):
        fusion = self.encoder(X, Y)  # (B, N, dim_h)
        pred_rs = self.head_rs(fusion) * 0.5  # (B, 3)
        pred_ts = self.head_ts(fusion) * 0.5  # (B, 3)
        return pred_rs, pred_ts


class SupervisedModel(nnx.Module):
    """Set Transformer for supervised model.

    Args:
        dim_i: Dimension of input features.
        dim_h: Dimension of hidden features. Must be divisible by `num_h`.
        num_h: Number of attention heads.
        num_i: Number of inducing points.
        drop: Dropout rate.
        rngs: RNG key.
    """

    def __init__(self, dim_i, dim_h, num_h, drop, rngs: nnx.Rngs):
        self.encoder = EncoderS(dim_i, dim_h, num_h, drop, rngs=rngs)
        self.head_rs = PoolHeadS(dim_h, dim_h, 3, num_h, drop, rngs=rngs)
        self.head_ts = PoolHeadS(dim_h, dim_h, 3, num_h, drop, rngs=rngs)
        self.head_ls = PoolHeadS(dim_h, dim_h, 1, num_h, drop, rngs=rngs)
        self.head_os = SABHeadS(dim_h, dim_h, 2, num_h, drop, rngs=rngs)

    def __call__(self, X, Y):
        fusion = self.encoder(X, Y)  # (B, N, dim_h)
        logits = self.head_os(fusion)  # (B, N, 2)
        pred_rs = self.head_rs(fusion) * 0.5  # (B, 3)
        pred_ts = self.head_ts(fusion) * 0.5  # (B, 3)
        pred_ls = self.head_ls(fusion)  # (B, 1)
        pred_ls = nnx.softplus(pred_ls) + 1e-6
        pred_Xo_logits = logits[:, :, 0]  # (B, N)
        pred_Yo_logits = logits[:, :, 1]  # (B, N)
        return pred_rs, pred_ts, pred_ls, pred_Xo_logits, pred_Yo_logits


def count_params(model):
    params = nnx.state(model, nnx.Param)
    return sum(p.size for p in jax.tree.leaves(params))


def save_model(model, path="model_params.msgpack"):
    """
    Save the parameters state of ``model`` to ``path``.

    Args:
        model: The Flax model to be saved.
        path: The path where the state will be saved to.
    """
    graphdef, param_state, rng_state = nnx.split(model, nnx.Param, nnx.RngState)
    param_state_dict = nnx.to_pure_dict(param_state)
    with open(path, "wb") as f:
        f.write(to_bytes(param_state_dict))


def restore_model(model, path="model_params.msgpack"):
    """
    Restore the parameters state from ``path`` to ``model``.

    Args:
        model: The Flax model to restore into.
        path: The path where the state will be loaded from.

    Returns:
        The restored Flax model.
    """
    graphdef, param_state, rng_state = nnx.split(model, nnx.Param, nnx.RngState)
    param_state_dict = nnx.to_pure_dict(param_state)
    with open(path, "rb") as f:
        param_state_dict = from_bytes(param_state_dict, f.read())
    nnx.replace_by_pure_dict(param_state, param_state_dict)
    model = nnx.merge(graphdef, param_state, rng_state)
    return model


if __name__ == "__main__":
    model_params_path = "example_model_params.msgpack"
    Xs = jax.random.normal(jax.random.key(0), (2, 5, 3))
    Ys = jax.random.normal(jax.random.key(1), (2, 5, 3))

    model = UnsupervisedModel(
        dim_i=3,
        dim_h=128,
        num_h=4,
        num_i=16,
        drop=0.1,
        rngs=nnx.Rngs(0),
    )
    model.eval()
    pred_rs, pred_ts = model(Xs, Ys)
    save_model(model, path=model_params_path)

    restored_model = UnsupervisedModel(
        dim_i=3,
        dim_h=128,
        num_h=4,
        num_i=16,
        drop=0.1,
        rngs=nnx.Rngs(1),  # Different parameter initialization
    )
    restored_model = restore_model(restored_model, path=model_params_path)
    restored_model.eval()
    restored_pred_rs, _ = restored_model(Xs, Ys)

    print(f"Same eval inference: {jnp.allclose(pred_rs, restored_pred_rs)}")
    print(f"Number of model parameters: {count_params(model):,}")
