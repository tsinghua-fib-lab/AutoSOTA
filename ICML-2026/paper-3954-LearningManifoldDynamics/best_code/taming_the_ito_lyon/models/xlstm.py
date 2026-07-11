"""Single-layer xLSTM sequence model using the mLSTM block."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from stochastax.manifolds import Manifold
from stochastax.manifolds.spd import SPDManifold

from .extrapolation import ExtrapolationScheme


def _pick_weight(linear: eqx.nn.Linear) -> jax.Array:
    return linear.weight


def _pick_bias(linear: eqx.nn.Linear) -> jax.Array | None:
    return linear.bias


def small_init(key: jax.Array, shape: tuple[int, ...], dim: int) -> jax.Array:
    return jr.normal(key, shape) * math.sqrt(2.0 / (5.0 * dim))


def wang_init(
    key: jax.Array,
    shape: tuple[int, ...],
    dim: int,
    n_layers: int,
) -> jax.Array:
    return jr.normal(key, shape) * (2.0 / (n_layers * math.sqrt(dim)))


def set_linear_weight(
    key: jax.Array,
    in_features: int,
    out_features: int,
    use_bias: bool,
    init_fn: str,
    *,
    n_layers: int | None = None,
) -> eqx.nn.Linear:
    linear = eqx.nn.Linear(
        in_features=in_features,
        out_features=out_features,
        use_bias=use_bias,
        key=key,
    )

    if init_fn == "small":
        weight = small_init(key, (out_features, in_features), in_features)
    elif init_fn == "wang":
        if n_layers is None:
            raise ValueError("n_layers must be provided for Wang initialisation.")
        weight = wang_init(key, (out_features, in_features), in_features, n_layers)
    elif init_fn == "zero":
        weight = jnp.zeros_like(linear.weight)
    else:
        raise ValueError(f"Unknown init_fn={init_fn!r}")

    return eqx.tree_at(_pick_weight, linear, weight)


def _build_input_sequence(
    control_values: jax.Array,
    extrapolation_scheme: ExtrapolationScheme | None,
    n_recon: int | None,
) -> jax.Array:
    if extrapolation_scheme is None:
        return control_values

    assert n_recon is not None, "n_recon must be set when using extrapolation_scheme"
    length = int(control_values.shape[0])
    ts = jnp.linspace(0.0, 1.0, length, dtype=control_values.dtype)
    control, _ = extrapolation_scheme.create_control(ts, control_values, n_recon)
    return jax.vmap(control.evaluate)(ts)


def _retract_output(manifold: Manifold, y: jax.Array) -> jax.Array:
    if isinstance(manifold, SPDManifold):
        return SPDManifold.retract(SPDManifold.unvech(y))
    return manifold.retract(y)


@dataclass(frozen=True)
class XLSTMArgs:
    d_model: int
    n_heads: int
    n_layers: int
    d_conv: int = 4
    xlstm_expand: int = 2
    ffn_expand: int = 2
    use_ln_bias: bool = False
    use_proj_bias: bool = True
    use_qkv_bias: bool = False
    use_conv_bias: bool = True
    use_ffn_bias: bool = False
    use_ffn: bool = True
    activation: Callable[[jax.Array], jax.Array] = jax.nn.gelu

    def __post_init__(self) -> None:
        d_inner = int(self.d_model) * int(self.xlstm_expand)
        if d_inner % int(self.n_heads) != 0:
            raise ValueError(
                f"d_inner={d_inner} must be divisible by n_heads={self.n_heads}"
            )
        object.__setattr__(self, "d_inner", d_inner)
        object.__setattr__(self, "d_hidden", int(self.d_model) * int(self.ffn_expand))
        object.__setattr__(self, "d_head", d_inner // int(self.n_heads))


class GatedFFN(eqx.Module):
    win: eqx.nn.Linear
    wout: eqx.nn.Linear
    d_hidden: int = eqx.field(static=True)
    activation: Callable[[jax.Array], jax.Array] = eqx.field(static=True)

    def __init__(
        self,
        key: jax.Array,
        *,
        d_model: int,
        d_hidden: int,
        use_ffn_bias: bool,
        n_layers: int,
        activation: Callable[[jax.Array], jax.Array],
    ) -> None:
        win_key, wout_key = jr.split(key)
        self.d_hidden = int(d_hidden)
        self.activation = activation
        self.win = set_linear_weight(
            win_key,
            d_model,
            2 * d_hidden,
            use_ffn_bias,
            "small",
        )
        self.wout = set_linear_weight(
            wout_key,
            d_hidden,
            d_model,
            use_ffn_bias,
            "wang",
            n_layers=n_layers,
        )
        if use_ffn_bias:
            assert self.win.bias is not None
            assert self.wout.bias is not None
            self.win = eqx.tree_at(_pick_bias, self.win, jnp.zeros_like(self.win.bias))
            self.wout = eqx.tree_at(
                _pick_bias, self.wout, jnp.zeros_like(self.wout.bias)
            )

    def __call__(self, x: jax.Array) -> jax.Array:
        x_proj = self.win(x)
        x_main, x_gate = jnp.split(x_proj, [self.d_hidden], axis=-1)
        return self.wout(self.activation(x_main) * x_gate)


class HeadLinear(eqx.Module):
    weight: jax.Array
    bias: jax.Array | None
    n_heads: int = eqx.field(static=True)

    def __init__(
        self,
        key: jax.Array,
        *,
        in_features: int,
        n_heads: int,
        use_bias: bool,
    ) -> None:
        if in_features % n_heads != 0:
            raise ValueError(
                f"in_features={in_features} must be divisible by n_heads={n_heads}"
            )
        d_head = in_features // n_heads
        self.n_heads = int(n_heads)
        self.weight = small_init(key, (n_heads, d_head, d_head), d_head)
        self.bias = jnp.zeros(in_features) if use_bias else None

    def __call__(self, x: jax.Array) -> jax.Array:
        x_heads = x.reshape(self.n_heads, -1)
        y_heads = jnp.einsum("hi,hij->hj", x_heads, self.weight)
        y = y_heads.reshape(-1)
        return y if self.bias is None else y + self.bias


def mlstm_forward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    ig_pre_act: jax.Array,
    fg_pre_act: jax.Array,
    mask: jax.Array,
) -> jax.Array:
    _, length, d_head = q.shape

    log_fg_act = jax.nn.log_sigmoid(fg_pre_act)
    log_fg_cumsum = jnp.pad(jnp.cumsum(log_fg_act, axis=-1), ((0, 0), (1, 0)))
    rep_log_fg_cumsum = jnp.repeat(
        log_fg_cumsum[..., None], repeats=length + 1, axis=-1
    )
    transpose_diff = rep_log_fg_cumsum - jnp.swapaxes(rep_log_fg_cumsum, 1, 2)
    log_fg = jnp.where(mask[None, :, :], transpose_diff[:, 1:, 1:], -jnp.inf)

    log_d = log_fg + ig_pre_act[:, None, :]
    max_log_d = jnp.max(log_d, axis=-1, keepdims=True)
    d = jnp.exp(log_d - max_log_d)

    scores = q @ jnp.swapaxes(k / math.sqrt(d_head), -1, -2)
    c = scores * d
    norm = jnp.maximum(jnp.abs(jnp.sum(c, axis=-1, keepdims=True)), jnp.exp(-max_log_d))
    c_normed = c / (norm + 1e-6)
    return c_normed @ v


def mlstm_step(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    ig_pre_act: jax.Array,
    fg_pre_act: jax.Array,
    c_state: jax.Array,
    n_state: jax.Array,
    m_state: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
    _, d_head = q.shape

    log_fg_act = jax.nn.log_sigmoid(fg_pre_act)
    next_m_state = jnp.maximum(log_fg_act + m_state, ig_pre_act)
    fg_act = jnp.exp(log_fg_act + m_state - next_m_state)
    ig_act = jnp.exp(ig_pre_act - next_m_state)

    k_scaled = k / math.sqrt(d_head)
    next_c_state = fg_act[:, None, None] * c_state + ig_act[:, None, None] * jnp.einsum(
        "hi,hj->hij", k_scaled, v
    )
    next_n_state = fg_act[:, None] * n_state + ig_act[:, None] * k_scaled

    qn = jnp.einsum("hi,hi->h", q, next_n_state)[:, None]
    mv = jnp.exp(-next_m_state)[:, None]
    h = jnp.einsum("hi,hij->hj", q, next_c_state) / (
        jnp.maximum(jnp.abs(qn), mv) + 1e-6
    )
    return h, (next_c_state, next_n_state, next_m_state)


class XLSTMLayer(eqx.Module):
    xlstm_norm: eqx.nn.LayerNorm
    win: eqx.nn.Linear
    wout: eqx.nn.Linear
    conv_kernel: jax.Array
    conv_bias: jax.Array | None
    wq: HeadLinear
    wk: HeadLinear
    wv: HeadLinear
    igate: eqx.nn.Linear
    fgate: eqx.nn.Linear
    group_norm: eqx.nn.GroupNorm
    skip: jax.Array
    ffn_norm: eqx.nn.LayerNorm | None
    ffn: GatedFFN | None
    args: XLSTMArgs = eqx.field(static=True)

    def __init__(self, key: jax.Array, *, args: XLSTMArgs) -> None:
        self.args = args
        (
            win_key,
            wout_key,
            conv_key,
            wq_key,
            wk_key,
            wv_key,
            igate_key,
            fgate_key,
            ffn_key,
        ) = jr.split(key, 9)

        self.xlstm_norm = eqx.nn.LayerNorm(
            args.d_model, use_weight=True, use_bias=args.use_ln_bias
        )

        self.win = set_linear_weight(
            win_key,
            args.d_model,
            2 * args.d_inner,
            args.use_proj_bias,
            "small",
        )
        if args.use_proj_bias:
            assert self.win.bias is not None
            self.win = eqx.tree_at(_pick_bias, self.win, jnp.zeros_like(self.win.bias))
        self.wout = set_linear_weight(
            wout_key,
            args.d_inner,
            args.d_model,
            False,
            "wang",
            n_layers=args.n_layers,
        )

        self.conv_kernel = small_init(
            conv_key, (args.d_inner, args.d_conv), args.d_inner
        )
        self.conv_bias = jnp.zeros(args.d_inner) if args.use_conv_bias else None

        self.wq = HeadLinear(
            wq_key,
            in_features=args.d_inner,
            n_heads=args.n_heads,
            use_bias=args.use_qkv_bias,
        )
        self.wk = HeadLinear(
            wk_key,
            in_features=args.d_inner,
            n_heads=args.n_heads,
            use_bias=args.use_qkv_bias,
        )
        self.wv = HeadLinear(
            wv_key,
            in_features=args.d_inner,
            n_heads=args.n_heads,
            use_bias=args.use_qkv_bias,
        )

        self.igate = set_linear_weight(
            igate_key,
            3 * args.d_inner,
            args.n_heads,
            True,
            "small",
        )
        assert self.igate.bias is not None
        self.igate = eqx.tree_at(
            _pick_bias, self.igate, jnp.linspace(0.0, 0.1, args.n_heads)
        )
        self.fgate = set_linear_weight(
            fgate_key,
            3 * args.d_inner,
            args.n_heads,
            True,
            "zero",
        )
        assert self.fgate.bias is not None
        self.fgate = eqx.tree_at(
            _pick_bias, self.fgate, jnp.linspace(3.0, 6.0, args.n_heads)
        )

        self.group_norm = eqx.nn.GroupNorm(args.n_heads, args.d_inner)
        self.skip = jnp.ones(args.d_inner)

        if args.use_ffn:
            self.ffn_norm = eqx.nn.LayerNorm(
                args.d_model, use_weight=True, use_bias=args.use_ln_bias
            )
            self.ffn = GatedFFN(
                ffn_key,
                d_model=args.d_model,
                d_hidden=args.d_hidden,
                use_ffn_bias=args.use_ffn_bias,
                n_layers=args.n_layers,
                activation=args.activation,
            )
        else:
            self.ffn_norm = None
            self.ffn = None

    def __call__(self, x: jax.Array, mask: jax.Array) -> jax.Array:
        h = jax.vmap(self.xlstm_norm)(x)
        x = x + self.xlstm(h, mask)
        if self.ffn is not None and self.ffn_norm is not None:
            h = jax.vmap(self.ffn_norm)(x)
            x = x + jax.vmap(self.ffn)(h)
        return x

    def xlstm(self, x: jax.Array, mask: jax.Array) -> jax.Array:
        x_proj, res = jnp.split(
            jax.vmap(self.win)(x),
            [self.args.d_inner],
            axis=-1,
        )

        conv_input = jnp.pad(x_proj, ((self.args.d_conv - 1, 0), (0, 0))).T
        xc = jax.vmap(
            lambda signal, kernel: jnp.convolve(signal, kernel, mode="valid")
        )(conv_input, self.conv_kernel).T
        if self.conv_bias is not None:
            xc = xc + self.conv_bias
        xc = jax.nn.silu(xc)

        q = jax.vmap(self.wq)(xc)
        k = jax.vmap(self.wk)(xc)
        v = jax.vmap(self.wv)(x_proj)
        gate_input = jnp.concatenate([q, k, v], axis=-1)

        q_heads = q.reshape(q.shape[0], self.args.n_heads, self.args.d_head).transpose(
            1, 0, 2
        )
        k_heads = k.reshape(k.shape[0], self.args.n_heads, self.args.d_head).transpose(
            1, 0, 2
        )
        v_heads = v.reshape(v.shape[0], self.args.n_heads, self.args.d_head).transpose(
            1, 0, 2
        )

        ig_pre_act = jax.vmap(self.igate)(gate_input).T
        fg_pre_act = jax.vmap(self.fgate)(gate_input).T
        h = mlstm_forward(q_heads, k_heads, v_heads, ig_pre_act, fg_pre_act, mask)
        h = h.transpose(1, 0, 2).reshape(x.shape[0], self.args.d_inner)
        h = jax.vmap(self.group_norm)(h) + self.skip * xc
        return jax.vmap(self.wout)(h * jax.nn.silu(res))

    def step(
        self,
        x: jax.Array,
        state: tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]] | None,
    ) -> tuple[
        jax.Array,
        tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]],
    ]:
        h, next_state = self.xlstm_step(self.xlstm_norm(x), state)
        x = x + h
        if self.ffn is not None and self.ffn_norm is not None:
            x = x + self.ffn(self.ffn_norm(x))
        return x, next_state

    def xlstm_step(
        self,
        x: jax.Array,
        state: tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]] | None,
    ) -> tuple[
        jax.Array,
        tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]],
    ]:
        if state is None:
            state = (
                jnp.zeros((self.args.d_inner, self.args.d_conv - 1), dtype=x.dtype),
                (
                    jnp.zeros(
                        (self.args.n_heads, self.args.d_head, self.args.d_head),
                        dtype=x.dtype,
                    ),
                    jnp.zeros((self.args.n_heads, self.args.d_head), dtype=x.dtype),
                    jnp.zeros((self.args.n_heads,), dtype=x.dtype),
                ),
            )

        conv_state, cnm_state = state
        x_proj, res = jnp.split(self.win(x), [self.args.d_inner], axis=-1)

        conv_input = jnp.concatenate([conv_state, x_proj[:, None]], axis=-1)
        xc = jnp.einsum("ij,ij->i", conv_input, jnp.flip(self.conv_kernel, axis=-1))
        if self.conv_bias is not None:
            xc = xc + self.conv_bias
        xc = jax.nn.silu(xc)

        q = self.wq(xc)
        k = self.wk(xc)
        v = self.wv(x_proj)
        gate_input = jnp.concatenate([q, k, v], axis=-1)

        q_heads = q.reshape(self.args.n_heads, self.args.d_head)
        k_heads = k.reshape(self.args.n_heads, self.args.d_head)
        v_heads = v.reshape(self.args.n_heads, self.args.d_head)
        ig_pre_act = self.igate(gate_input)
        fg_pre_act = self.fgate(gate_input)

        h, next_cnm_state = mlstm_step(
            q_heads, k_heads, v_heads, ig_pre_act, fg_pre_act, *cnm_state
        )
        h = self.group_norm(h.reshape(-1)) + self.skip * xc
        h = self.wout(h * jax.nn.silu(res))
        next_state = (conv_input[:, 1:], next_cnm_state)
        return h, next_state


class XLSTM(eqx.Module):
    """Single-layer xLSTM adapted for continuous sequence regression."""

    input_proj: eqx.nn.Linear
    layer: XLSTMLayer
    norm: eqx.nn.LayerNorm
    readout_layer: eqx.nn.Linear

    manifold: Manifold = eqx.field(static=True)
    readout_activation: Callable[[jax.Array], jax.Array] = eqx.field(static=True)
    evolving_out: bool = eqx.field(static=True)
    extrapolation_scheme: ExtrapolationScheme | None = eqx.field(static=True)
    n_recon: int | None = eqx.field(static=True)
    args: XLSTMArgs = eqx.field(static=True)

    def __init__(
        self,
        input_path_dim: int,
        output_path_dim: int,
        *,
        d_model: int,
        n_heads: int,
        key: jax.Array,
        manifold: Manifold,
        d_conv: int = 4,
        xlstm_expand: int = 2,
        ffn_expand: int = 2,
        use_ffn: bool = True,
        readout_activation: Callable[[jax.Array], jax.Array] = lambda x: x,
        evolving_out: bool = True,
        extrapolation_scheme: ExtrapolationScheme | None = None,
        n_recon: int | None = None,
    ) -> None:
        proj_key, layer_key, readout_key = jr.split(key, 3)

        self.args = XLSTMArgs(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=1,
            d_conv=d_conv,
            xlstm_expand=xlstm_expand,
            ffn_expand=ffn_expand,
            use_ffn=use_ffn,
        )
        self.manifold = manifold
        self.readout_activation = readout_activation
        self.evolving_out = bool(evolving_out)
        self.extrapolation_scheme = extrapolation_scheme
        self.n_recon = n_recon

        self.input_proj = eqx.nn.Linear(
            input_path_dim,
            d_model,
            use_bias=True,
            key=proj_key,
        )
        self.layer = XLSTMLayer(layer_key, args=self.args)
        self.norm = eqx.nn.LayerNorm(d_model, use_weight=True, use_bias=True)
        self.readout_layer = eqx.nn.Linear(
            d_model,
            output_path_dim,
            use_bias=True,
            key=readout_key,
        )

    def _forward_from_x(self, x: jax.Array) -> jax.Array:
        h = jax.vmap(self.input_proj)(x)
        mask = jnp.tril(jnp.ones((h.shape[0], h.shape[0]), dtype=bool))
        h = self.layer(h, mask)
        return jax.vmap(self.norm)(h)

    def _apply_readout(self, hidden_states: jax.Array) -> jax.Array:
        def apply_single(h: jax.Array) -> jax.Array:
            y = self.readout_activation(self.readout_layer(h))
            return _retract_output(self.manifold, y)

        return jax.vmap(apply_single)(hidden_states)

    def __call__(self, control_values: jax.Array) -> jax.Array:
        x_eval = _build_input_sequence(
            control_values, self.extrapolation_scheme, self.n_recon
        )
        hidden = self._forward_from_x(x_eval)

        if self.evolving_out:
            return self._apply_readout(hidden)

        y = self.readout_activation(self.readout_layer(hidden[-1]))
        return _retract_output(self.manifold, y)

    @eqx.filter_jit
    def step(
        self,
        x_t: jax.Array,
        state: tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]] | None,
    ) -> tuple[jax.Array, tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]]:
        h, next_state = self.layer.step(self.input_proj(x_t), state)
        h = self.norm(h)
        y = self.readout_activation(self.readout_layer(h))
        return _retract_output(self.manifold, y), next_state
