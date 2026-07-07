"""Unified QuITE-equipped MTS backbone wrapper.

The same `Model` class handles both forecasting and classification, dispatched
by `args.task`. Embedding logic is delegated to `models.embeddings.get_embedding`
so that 'quite' / 'mean' / 'mtand' / 'add' / 'concat' / vanilla baselines are
each a single self-contained module.

Backbones (selected by `args.model`):
    Patch family   : patchtst, patchmixer, tmix
    Variate family : itransformer, s_mamba
    Hybrid         : timexer
"""
import torch
import torch.nn as nn

# from mamba_ssm import Mamba  # lazy import

from models.common import PatchMixerLayer, Transpose, TempBlock
from models.embeddings import get_embedding, LearnableTE
from models.embeddings.baseline import VariateBaselineEmbedding
from models.layers.transformer import (
    Encoder, EncoderLayer, TimeXer_Encoder, TimeXer_EncoderLayer, CrossAttentionFFNLayer,
)
from models.layers.attention import FullAttention, AttentionLayer
from models.layers.mamba import Encoder as MambaEncoder, EncoderLayer as MambaEncoderLayer


PATCH_MODELS = {"patchtst", "patchmixer", "tmix"}
VARIATE_MODELS = {"itransformer", "s_mamba"}


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.task = args.task
        self.model = args.model
        self.mode = args.mode
        self.irr_emb = args.irr_emb
        self.device = args.device
        self.d_model = args.hid_dim
        self.factor = 5

        # Time embedding used by the forecasting decoder for future-time queries
        self.te = LearnableTE(self.d_model)

        # ----- Input embedding (delegated to models/embeddings/) -----
        if self.model == 'timexer':
            self.patch_embedding = get_embedding(args, family='patch')
            if args.irr_emb:
                # Irregular embedding (self/mean/mtand) handles its own input shape.
                self.variate_embedding = get_embedding(args, family='variate')
            else:
                # Non-irregular baseline: TimeXer flattens patches before the
                # exogenous projection, so its c_in is maxlen*npatch (not maxlen).
                mode = args.mode if args.mode in ('add', 'concat') else 'vanilla'
                self.variate_embedding = VariateBaselineEmbedding(
                    args, mode=mode, c_in=args.maxlen * args.npatch,
                )
            self.var_id_emb = nn.Embedding(args.ndim, self.d_model)
        else:
            family = 'variate' if self.model in VARIATE_MODELS else 'patch'
            self.embedding = get_embedding(args, family=family)

        # ----- Backbone encoder -----
        self._build_encoder(args)

        # ----- Output head (task-specific) -----
        if self.task == 'forecasting':
            self.cross_transformer = CrossAttentionFFNLayer(
                AttentionLayer(
                    FullAttention(False, self.factor, attention_dropout=args.dropout, output_attention=False),
                    self.d_model, args.nhead),
                self.d_model, self.d_model,
                dropout=args.dropout, activation='gelu',
            )
            self.decoder = nn.Sequential(
                nn.Linear(self.d_model, self.d_model), nn.ReLU(inplace=True),
                nn.Linear(self.d_model, self.d_model), nn.ReLU(inplace=True),
                nn.Linear(self.d_model, 1),
            )
        else:  # classification
            d_static = args.d_static
            if d_static != 0:
                self.emb = nn.Linear(d_static, args.ndim)
                self.classifier = nn.Sequential(
                    nn.Linear(args.ndim * 2, 200), nn.ReLU(),
                    nn.Linear(200, args.n_class),
                )
            else:
                self.classifier = nn.Sequential(
                    nn.Linear(args.ndim, 200), nn.ReLU(),
                    nn.Linear(200, args.n_class),
                )

    # ----------------------------------------------------------------
    def _build_encoder(self, args):
        if self.model == 'patchtst':
            norm_layer = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(self.d_model), Transpose(1, 2))
            self.encoder = Encoder(
                [EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, self.factor, attention_dropout=args.dropout, output_attention=False),
                        self.d_model, args.nhead),
                    self.d_model, self.d_model,
                    dropout=args.dropout, activation='gelu')
                 for _ in range(args.nlayer)],
                norm_layer=norm_layer,
            )

        elif self.model in VARIATE_MODELS:
            norm_layer = nn.LayerNorm(self.d_model)
            if self.model == "s_mamba":
                from mamba_ssm import Mamba  # lazy import, only for s_mamba backbone
                d_state = getattr(args, 'd_state', 16)
                d_conv = getattr(args, 'd_conv', 2)
                expand = getattr(args, 'expand', 1)
                self.encoder = MambaEncoder(
                    [MambaEncoderLayer(
                        Mamba(d_model=self.d_model, d_state=d_state, d_conv=d_conv, expand=expand),
                        Mamba(d_model=self.d_model, d_state=d_state, d_conv=d_conv, expand=expand),
                        self.d_model, getattr(args, 'd_ff', 4 * self.d_model),
                        dropout=args.dropout, activation='gelu')
                     for _ in range(args.nlayer)],
                    norm_layer=norm_layer,
                )
            else:  # itransformer
                self.encoder = Encoder(
                    [EncoderLayer(
                        AttentionLayer(
                            FullAttention(False, self.factor, attention_dropout=args.dropout, output_attention=False),
                            self.d_model, args.nhead),
                        self.d_model, self.d_model,
                        dropout=args.dropout, activation='gelu')
                     for _ in range(args.nlayer)],
                    norm_layer=norm_layer,
                )

        elif self.model == 'timexer':
            norm_layer = nn.LayerNorm(self.d_model)
            self.encoder = TimeXer_Encoder(
                [TimeXer_EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, self.factor, attention_dropout=args.dropout, output_attention=False),
                        self.d_model, args.nhead),
                    AttentionLayer(
                        FullAttention(False, self.factor, attention_dropout=args.dropout, output_attention=False),
                        self.d_model, args.nhead),
                    self.d_model, self.d_model,
                    dropout=args.dropout, activation='gelu')
                 for _ in range(args.nlayer)],
                norm_layer=norm_layer,
            )

        elif self.model == 'patchmixer':
            self.dropout_layer = nn.Dropout(args.dropout)
            self.encoder_blocks = nn.ModuleList(
                [PatchMixerLayer(dim=args.npatch, a=args.npatch) for _ in range(1)]
            )

        elif self.model == 'tmix':
            # Unified TempBlock for both forecasting and classification (paper consistent).
            self.encoder_blocks = nn.ModuleList([TempBlock(args) for _ in range(args.nlayer)])

    # ================================================================
    # Embedding + backbone encoder → (B*N, M_eff, D)
    # ================================================================
    def _embed_encode(self, X, truth_time_steps, mask):
        """Returns (irr_z_flat: (B*N, M_eff, D), B, M_eff, N)."""

        # ----- Variate family (iTransformer / S-Mamba) -----
        if self.model in VARIATE_MODELS:
            # Patch-shaped input (B, M, L_in, N) gets flattened to (B, L, N).
            if X.dim() == 4:
                B, M, L_in, N = X.shape
                X = X.reshape(B, M * L_in, N)
                truth_time_steps = truth_time_steps.reshape(B, M * L_in, N)
                mask = mask.reshape(B, M * L_in, N)

            variable_embeddings = self.embedding(X, truth_time_steps, mask)   # (B, N, D)
            irr_z, _ = self.encoder(variable_embeddings)
            B, N = irr_z.shape[0], irr_z.shape[1]
            return irr_z.unsqueeze(2).reshape(B * N, 1, self.d_model), B, 1, N

        # ----- Patch family (PatchTST / PatchMixer / TMix) -----
        if self.model in PATCH_MODELS:
            B, M, L_in, N = X.shape
            patch_emb = self.embedding(X, truth_time_steps, mask)             # (B*N, M, D)

            if self.model == 'patchtst':
                patch_emb, _ = self.encoder(patch_emb)
            elif self.model == 'patchmixer':
                patch_emb = self.dropout_layer(patch_emb)
                for blk in self.encoder_blocks:
                    patch_emb = blk(patch_emb)
            elif self.model == 'tmix':
                # (B*N, M, D) → (B, N, M*D) → blocks → (B*N, M, D)
                patch_emb = patch_emb.reshape(B, N, M * self.d_model)
                for blk in self.encoder_blocks:
                    patch_emb = blk(patch_emb)
                patch_emb = patch_emb.reshape(B * N, M, self.d_model)

            return patch_emb, B, M, N

        # ----- Hybrid (TimeXer) -----
        if self.model == 'timexer':
            B, M, L_in, N = X.shape

            patch_emb = self.patch_embedding(X, truth_time_steps, mask)       # (B*N, M, D)

            # Prepend a per-variable identity token → (B*N, M+1, D)
            var_id = self.var_id_emb(torch.arange(N, device=self.device))
            var_flat = var_id.unsqueeze(0).expand(B, -1, -1).unsqueeze(2).reshape(B * N, 1, self.d_model)
            en_embed = torch.cat([patch_emb, var_flat], dim=1)

            # Exogenous variate context, flatten patch dim
            X_flat = X.reshape(B, M * L_in, N)
            time_flat = truth_time_steps.reshape(B, M * L_in, N)
            mask_flat = mask.reshape(B, M * L_in, N)
            ex_embed = self.variate_embedding(X_flat, time_flat, mask_flat)   # (B, N, D)

            irr_z_flat = self.encoder(en_embed, ex_embed)                     # (B*N, M+1, D)
            return irr_z_flat, B, M + 1, N

        raise ValueError(f"Unknown model: {self.model}")

    # ================================================================
    # Forecasting head (cross-attention decoder + MLP regressor)
    # ================================================================
    def forecasting(self, time_steps_to_predict, X, truth_time_steps, mask=None):
        irr_z_flat, B, _, N = self._embed_encode(X, truth_time_steps, mask)

        L_pred = time_steps_to_predict.shape[1]
        te_pred = self.te(time_steps_to_predict.unsqueeze(-1))                # (B, L_pred, D)
        te_pred_flat = te_pred.unsqueeze(1).expand(-1, N, -1, -1).reshape(B * N, L_pred, self.d_model)

        out_flat, _ = self.cross_transformer(te_pred_flat, irr_z_flat, irr_z_flat)
        out = out_flat.reshape(B, N, L_pred, self.d_model).permute(0, 2, 1, 3)
        return self.decoder(out).squeeze(-1)                                   # (B, L_pred, N)

    # ================================================================
    # Classification head (sum-over-D pool + MLP classifier)
    # ================================================================
    def classification(self, X, truth_time_steps, mask=None, P_static=None, feature=False):
        irr_z_flat, B, M_eff, N = self._embed_encode(X, truth_time_steps, mask)
        irr_z = irr_z_flat.reshape(B, N, M_eff * self.d_model)

        if feature:
            return irr_z

        h = torch.sum(irr_z, dim=-1)                                          # (B, N)
        if P_static is not None:
            static_emb = self.emb(P_static)
            return self.classifier(torch.cat([h, static_emb], dim=-1))
        return self.classifier(h)
