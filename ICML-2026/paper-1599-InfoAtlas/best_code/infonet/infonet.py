from typing import Optional

import torch
import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder
from .query import Query_Gen_transformer


def log_mean_exp(a: torch.Tensor) -> torch.Tensor:
    """
    a: (B, S) or (B, S, 1)
    return: (B,)
    """
    if a.dim() == 3 and a.size(-1) == 1:
        a = a.squeeze(-1)  # (B, S)
    b = torch.max(a, dim=1, keepdim=True).values  # (B, 1)
    return torch.log(torch.mean(torch.exp(a - b), dim=1)) + b[:, 0]


class HypoGen2HyperNet(nn.Module):
    """
    Hypernetwork implemented by HyPoGen2 HyperTr.

    Changes:
      - prepare(): generate the B sets of target weights once (cached in HyperTr.generated_weights)
      - score(): reuse the generated weights to score (B,Dx) or (B,S,Dx) without re-running the hypernetwork
    """

    def __init__(
        self,
        meta_dim: int,
        base_input_dim: int,
        hidden_dim: int = 512,
        z_dim: Optional[int] = None,
        horizon: int = 1,
        weight_dim: int = 512,
        enc_dec_dim: int = 512,
        opt_block_dim: int = 768,
        num_opt_mlp_layer: int = 3,
        num_enc_dec_layer: int = 3,
        num_layers: int = 4,
        weight_split_dim: int = 768,
        nhead: int = 4,
        lr_scheme_method: str = "cosine",
        use_compile: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        from .hylap.hypogen2_impl.hypernetworks import HyperTr

        self.meta_dim = meta_dim
        self.base_input_dim = base_input_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim if z_dim is not None else meta_dim
        self.horizon = horizon

        self.language_processor = nn.Linear(meta_dim, self.z_dim)

        self.hyper_tr = HyperTr(
            ftask_dim=self.z_dim,
            dim2=base_input_dim,
            dim3=hidden_dim,
            dim4=1 * horizon,
            num_layers=num_layers,
            weight_dim=weight_dim,
            enc_dec_dim=enc_dec_dim,
            opt_block_dim=opt_block_dim,
            num_opt_mlp_layer=num_opt_mlp_layer,
            num_enc_dec_layer=num_enc_dec_layer,
            weight_split_dim=weight_split_dim,
            nhead=nhead,
            lr_scheme_method=lr_scheme_method,
            use_compile=use_compile,
            **kwargs,
        )

        # cache z from the most recent prepare() (optional, kept for clarity)
        self._cached_z: Optional[torch.Tensor] = None

    def prepare(self, meta_v: torch.Tensor) -> torch.Tensor:
        """
        meta_v: (B,Q,meta_dim) or (B,meta_dim)
        Returns z_sequence: (B,Q,z_dim)
        and generates + caches generated_weights (B weight sets) inside HyperTr
        """
        if meta_v.dim() == 2:
            meta_v = meta_v.unsqueeze(1)  # (B,1,meta_dim)
        if meta_v.dim() != 3 or meta_v.size(-1) != self.meta_dim:
            raise ValueError(f"meta_v expected (B,Q,{self.meta_dim}), got {tuple(meta_v.shape)}")

        z_sequence = self.language_processor(meta_v)  # (B,Q,z_dim)

        # generate weights once (B sets), cached in hyper_tr.generated_weights
        if not hasattr(self.hyper_tr, "generate_weights"):
            raise RuntimeError("HyperTr does not have generate_weights(); cannot reuse target weights.")
        self.hyper_tr.generate_weights(z_sequence)
        self._cached_z = z_sequence
        return z_sequence

    def score(self, base_v: torch.Tensor, early_sup: bool = False) -> torch.Tensor:
        """
        base_v:
          - (B,Dx) -> (B,)
          - (B,S,Dx) -> (B,S)
        Reuse hyper_tr.generated_weights via functional_call; do not regenerate.
        """
        if self._cached_z is None:
            raise RuntimeError("HypoGen2HyperNet.score() called before prepare().")

        ht = self.hyper_tr
        if not hasattr(ht, "generated_weights"):
            raise RuntimeError("HyperTr has no generated_weights cache; cannot reuse weights.")
        if not hasattr(ht, "target_net"):
            raise RuntimeError("HyperTr has no target_net; cannot functional_call target.")

        # generated_weights is a list (per layer/step); forward_with_weights uses the last one
        weights_list = ht.generated_weights
        if not isinstance(weights_list, (list, tuple)) or len(weights_list) == 0:
            raise RuntimeError("HyperTr.generated_weights is empty. Did generate_weights() run correctly?")

        # early_sup: run every step's weights and stack them
        if early_sup:
            outs = []
            for w in weights_list:
                outs.append(self._apply_weights_to_base(ht.target_net, w, base_v))
            out = torch.stack(outs, dim=0)  # (T,B,...)  or (T,B,S,...)
        else:
            w = weights_list[-1]
            out = self._apply_weights_to_base(ht.target_net, w, base_v)

        # squeeze last dim if (...,1)
        if out.dim() >= 1 and out.size(-1) == 1:
            out = out.squeeze(-1)
        return out

    @staticmethod
    def _apply_weights_to_base(target_net: nn.Module, weights: dict, base_v: torch.Tensor) -> torch.Tensor:
        """
        target_net: nn.Module
        weights: dict with batched params: each value shape (B, ...)
        base_v: (B,Dx) or (B,S,Dx)
        return: (B,1) or (B,S,1) (depending on target_net output)
        """
        fcall = torch.func.functional_call

        if base_v.dim() == 2:
            # (B,Dx): vmap over B
            return torch.vmap(fcall, in_dims=(None, 0, 0), randomness="different")(target_net, weights, base_v)

        if base_v.dim() == 3:
            # (B,S,Dx): nested vmap: over B first, then S
            B, S, Dx = base_v.shape

            def run_one_task(w_b: dict, x_b: torch.Tensor) -> torch.Tensor:
                # x_b: (S,Dx), w_b: unbatched params (no leading B)
                return torch.vmap(lambda x_i: fcall(target_net, w_b, x_i), randomness="different")(x_b)

            # vmap over B: each w_b corresponds to one task/sample
            return torch.vmap(run_one_task, in_dims=(0, 0))(weights, base_v)

        raise ValueError(f"base_v must be (B,Dx) or (B,S,Dx), got {tuple(base_v.shape)}")

    def forward(self, meta_v: torch.Tensor, base_v: torch.Tensor, early_sup: bool = False) -> torch.Tensor:
        # backward-compatible call: forward = prepare + score
        self.prepare(meta_v)
        out = self.score(base_v, early_sup=early_sup)
        # (B,) is returned directly; (B,S) is also returned
        return out


class InfoNet(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        query_gen: Query_Gen_transformer,
        decoder_query_dim: int,
        input_dim_x: int = 3,
        input_dim_y: int = 3,
        num_mlp_layer: int = 1,
        hidden_dim: int = 512,
        dropout: float = 0.0,
        targetnet_hiddim: int = 256,
        hypermlp_hiddim: int = 2048,
        hypogen_z_dim: Optional[int] = None,
        **hypogen_kwargs,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.query_gen = query_gen

        self.num_mlp_layer = num_mlp_layer
        self.input_dim_x = input_dim_x
        self.input_dim_y = input_dim_y
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.decoder_query_dim = decoder_query_dim
        self.targetnet_hiddim = targetnet_hiddim

        base_input_dim = self.input_dim_x + self.input_dim_y

        self.weight_gen = HypoGen2HyperNet(
            meta_dim=decoder_query_dim,
            base_input_dim=base_input_dim,
            hidden_dim=targetnet_hiddim,
            z_dim=hypogen_z_dim if hypogen_z_dim is not None else decoder_query_dim,
            horizon=1,
            **hypogen_kwargs,
        )

    def _score_points(
        self,
        xy: torch.Tensor,          # (B,S,Dx)
        early_sup: bool = False,
    ) -> torch.Tensor:
        """
        Score only: reuse the already-prepared target weights
        return: (B,S)
        """
        out = self.weight_gen.score(xy, early_sup=early_sup)  # (B,S) or (T,B,S)
        if out.dim() == 3:
            # early_sup=True: (T,B,S) -> not expanded here (early_sup typically unused)
            raise RuntimeError(f"early_sup=True produced shape {tuple(out.shape)}; InfoNet expects (B,S) when early_sup=False.")
        if out.dim() != 2:
            raise RuntimeError(f"Expected score shape (B,S), got {tuple(out.shape)}")
        return out

    def forward(
        self,
        inputs: Optional[torch.Tensor],   # (B,S,Dx)
        query: Optional[torch.Tensor] = None,
        input_mask: Optional[torch.Tensor] = None,
        query_mask: Optional[torch.Tensor] = None,
        early_sup: bool = False,
    ):
        if inputs is None:
            raise ValueError("inputs cannot be None")

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            latents = self.encoder(inputs, input_mask)
            query = self.query_gen(inputs)

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            meta_seq = self.decoder(
                x_q=query,
                latents=latents,
                query_mask=query_mask,
            )

        if meta_seq.dim() != 3:
            raise ValueError(f"decoder output must be 3D (B,Q,D), got shape={tuple(meta_seq.shape)}")

        B, Q, D = meta_seq.shape
        if D != self.decoder_query_dim or Q != self.decoder_query_dim:
            raise ValueError(
                f"Expected decoder output shape (B,{self.decoder_query_dim},{self.decoder_query_dim}), "
                f"got (B,{Q},{D})."
            )

        # generate the B sets of target weights once (cached in HyperTr)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            self.weight_gen.prepare(meta_seq)

        log_mean_exp_et = 0.0
        for _ in range(3):
            perm = torch.randperm(inputs.shape[1], device=inputs.device)
            marginal = torch.cat(
                (
                    inputs[:, :, 0:self.input_dim_x],
                    inputs[:, perm, self.input_dim_x:],
                ),
                dim=2,
            )  # (B,S,Dx)

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                et = self._score_points(marginal, early_sup=early_sup)  # (B,S)
            log_mean_exp_et = log_mean_exp_et + log_mean_exp(et)    # (B,)

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            t = self._score_points(inputs, early_sup=early_sup)         # (B,S)

        mi_lb = torch.mean(t, dim=1) - log_mean_exp_et / 3.0        # (B,)
        return mi_lb

