import torch
import torch.nn as nn

from models.modules import SelfAttentionBlock, TransformerEncoderBlock, CrossAttentionBlock


class Model(nn.Module):
    """QuITE++: hierarchical encoder + cross-attention decoder for IMTS forecasting.

    Hierarchical encoder (L layers):
      1. Query-based patch embedding (Eq. 10-13): a learnable patch token aggregates
         observation tokens within each (patch, variable) via self-attention.
      2. Patch-level self-attention across patches per variable (with a prepended
         variable token), Eq. 15-17.
      3. Variable-level self-attention across variables, Eq. 18-19.

    Decoder (Eq. 22-26): future-time queries attend to both the variable-level
    summary (Global Context) and the per-patch representations (Local Context).
    """

    def __init__(self, args):
        super().__init__()
        d_model = args.hid_dim
        self.device = args.device
        self.hid_dim = args.hid_dim
        self.N = args.ndim
        self.n_layer = args.nlayer
        self.npatch = args.npatch
        dropout = args.dropout

        # Harmonic time embedding (Shukla & Marlin 2021), Eq. 4
        self.te_scale = nn.Linear(1, 1)
        self.te_periodic = nn.Linear(1, args.hid_dim - 1)

        # Value + identity embeddings
        self.val_emb = nn.Linear(1, d_model)
        self.patch_emb = nn.Embedding(args.npatch, d_model)
        self.var_emb = nn.Embedding(args.ndim, d_model)

        # Query-based patch embedding layer (Eq. 10-13)
        self.embedding_layer = SelfAttentionBlock(d_model, args.nhead, dropout=dropout)

        # Hierarchical encoder: (patch-level self-attn, variable-level self-attn) x L
        self.encoder = nn.ModuleList([
            nn.Sequential(
                SelfAttentionBlock(d_model, args.nhead, dropout=dropout),
                TransformerEncoderBlock(d_model, args.nhead, 2 * d_model, dropout=dropout),
            )
            for _ in range(args.nlayer)
        ])

        # Decoder: shared cross-attention block for both Global and Local contexts
        self.cross_transformer = CrossAttentionBlock(d_model, args.nhead, 2 * d_model, dropout=dropout)

        self.decoder = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
            nn.Linear(d_model, 1),
        )

    def LearnableTE(self, tt):
        out1 = self.te_scale(tt)
        out2 = torch.sin(self.te_periodic(tt))
        return torch.cat([out1, out2], -1)

    # ----------------------------------------------------------------
    # Hierarchical encoder
    # ----------------------------------------------------------------
    def patch_extractor(self, X, truth_time_steps, mask):
        B, M, L_in, N = X.shape
        d_model = self.hid_dim

        X = X.permute(0, 1, 3, 2)                          # (B, M, N, L_in)
        mask = mask.permute(0, 1, 3, 2)                    # (B, M, N, L_in)
        truth_time_steps = truth_time_steps.permute(0, 1, 3, 2)

        # Observation tokens: value + time embedding (Eq. 5)
        patch_vs = self.val_emb(X.unsqueeze(-1))                          # (B, M, N, L_in, D)
        time_emb = self.LearnableTE(truth_time_steps.unsqueeze(-1))       # (B, M, N, L_in, D)
        patch_vs = patch_vs + time_emb

        # Learnable patch query tokens (one per (patch, variable))
        patch_idx = torch.arange(M, dtype=torch.long, device=self.device)
        patch_query = self.patch_emb(patch_idx).view(1, M, 1, 1, d_model).expand(B, -1, N, -1, -1)

        # Learnable variable query tokens (one per variable), initialized for the encoder
        var_idx = torch.arange(N, dtype=torch.long, device=self.device)
        cls_reprs = self.var_emb(var_idx).view(1, N, 1, d_model).expand(B, -1, 1, -1)

        # Query-based aggregation within each (patch, variable): Eq. 11-12
        z = torch.cat([patch_query, patch_vs], dim=3)                     # (B, M, N, L_in+1, D)
        z_flat = z.view(B * M * N, L_in + 1, d_model)
        attn_mask = torch.cat(
            [torch.ones(B, M, N, 1, device=self.device), mask], dim=3
        ).view(B * M * N, L_in + 1)
        processed_z_flat = self.embedding_layer(z_flat, attn_mask=attn_mask.unsqueeze(-1))
        processed_z = processed_z_flat.view(B, M, N, L_in + 1, d_model)

        patch_reprs = processed_z[:, :, :, 0, :]                          # (B, M, N, D)  e_{m,n}

        # Hierarchical encoder
        for n in range(self.n_layer):
            # ----- Patch-level self-attention (Eq. 15-17) -----
            z = patch_reprs.permute(0, 2, 1, 3)                           # (B, N, M, D)
            z = torch.cat([cls_reprs, z], dim=2)                          # (B, N, M+1, D)
            z_flat = z.reshape(B * N, M + 1, d_model)

            processed_z_flat = self.encoder[n][0](z_flat)
            processed_z = processed_z_flat.view(B, N, M + 1, d_model)

            cls_reprs = processed_z[:, :, 0, :]                           # (B, N, D)
            patch_reprs = processed_z[:, :, 1:, :].permute(0, 2, 1, 3)    # (B, M, N, D)

            # ----- Variable-level self-attention (Eq. 18-19) -----
            cls_reprs = self.encoder[n][1](cls_reprs).unsqueeze(2)        # (B, N, 1, D)

        return cls_reprs, patch_reprs

    # ----------------------------------------------------------------
    # Forecasting (Eq. 22-26)
    # ----------------------------------------------------------------
    def forecasting(self, time_steps_to_predict, X, truth_time_steps, mask=None):
        B, M, _, N = X.shape
        d_model = self.hid_dim

        # cls_reprs: (B, N, 1, D) — variable-level summary  (C_n)
        # patch_reprs: (B, M, N, D) — per-patch representations (e_{m,n}, encoded)
        cls_reprs, patch_reprs = self.patch_extractor(X, truth_time_steps, mask)

        # Future-time query embedding (Eq. 22)
        L_pred = time_steps_to_predict.shape[1]
        te_pred = self.LearnableTE(time_steps_to_predict.unsqueeze(-1))                # (B, L_pred, D)
        te_pred_flat = te_pred.unsqueeze(1).expand(-1, N, -1, -1).reshape(B * N, L_pred, d_model)

        # Global Context (Eq. 23)
        global_summary_flat = cls_reprs.squeeze(2).reshape(B * N, 1, d_model)
        global_context_flat = self.cross_transformer(te_pred_flat, global_summary_flat, global_summary_flat)

        # Local Context (Eq. 24): use the encoded patch representations
        local_details_flat = patch_reprs.permute(0, 2, 1, 3).reshape(B * N, M, d_model)
        local_details_flat = self.cross_transformer(te_pred_flat, local_details_flat, local_details_flat)

        # Final prediction (Eq. 25-26)
        global_context = global_context_flat.view(B, N, L_pred, d_model).permute(0, 2, 1, 3)
        local_details = local_details_flat.view(B, N, L_pred, d_model).permute(0, 2, 1, 3)
        combined = torch.cat([global_context, local_details], dim=-1)
        return self.decoder(combined).squeeze(-1)
