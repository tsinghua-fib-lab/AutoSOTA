"""Full cramming where the compression embedding lives in a low-rank subspace."""

from __future__ import annotations

import os

import torch

from progressive_cramming.train.embedding_init import create_compression_embedding
from progressive_cramming.train.parametrization import LowDimProjectedParametrization
from progressive_cramming.train.trainers.full_cramming import (
    FullCrammingTrainer,
    _BatchInputs,
    _RunContext,
)


class LowDimTrainer(FullCrammingTrainer):
    """Full-cramming-style trainer with a learned rank-k projection on the compression embedding."""

    # ------------------------------------------------------------------
    # Parametrization (overrides the embedding-init shape and the factory kwargs).
    # ------------------------------------------------------------------

    def _init_compression_token_embeddings(self, inputs: _BatchInputs, ctx: _RunContext) -> torch.nn.Parameter:
        """Sample a fresh coefficient tensor in the low-dim subspace (shape [batch, num_comp, low_dim_size])."""
        return create_compression_embedding(
            batch_size=inputs.batch_size,
            num_compression_tokens=ctx.num_compression_tokens,
            # Initialize in the low-dim coefficient space, not the hidden space:
            # the Linear projection will lift it to ``hidden_size`` inside the
            # parametrization on every forward.
            hidden_size=self.args.low_dim_size,
            init_method=ctx.init_method,
            mvn_dist=ctx.mvn_dist,
            token_embeddings=inputs.token_embeddings,
            single_compression_token_embeddings_initialization=ctx.single_compression_token_init,
            pca_components=ctx.pca_components,
            pca_mean=ctx.pca_mean,
            loaded_embeddings=ctx.loaded_embeddings,
        )

    def _extra_parametrization_kwargs(self, ctx: _RunContext) -> dict:
        """Tell ``build_parametrization`` to construct a ``LowDimProjectedParametrization``.

        Re-uses the same centralized projection builder
        (``BaseTrainer._build_low_dim_projection_module``) as
        :class:`ProgressiveCrammingTrainer`, so the checkpoint-loading and
        ``low_dim_projection_train`` semantics cannot drift between trainers.
        Full cramming has no batch-spanning training (one batch = one
        independent compression problem), so a fresh Linear is built per
        batch — global mode does not apply.
        """
        projection = self._build_low_dim_projection_module(
            hidden_size=ctx.hidden_size,
            device=ctx.device,
        )
        return {
            "low_dim_train": True,
            "low_dim_size": self.args.low_dim_size,
            "train_projection": self.args.low_dim_projection_train,
            "projection_module": projection,
        }

    def _on_training_complete(self, parametrization, ctx: _RunContext) -> None:
        """Persist the projection weights of the last batch's parametrization."""
        if not isinstance(parametrization, LowDimProjectedParametrization):
            return
        if not self.args.low_dim_projection_train:
            return
        output_dir = self.args.output_dir
        if not output_dir:
            return
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "low_dim_projection.pt")
        torch.save(
            {
                "low_dim_projection": parametrization.shared_state_dict(),
                "low_dim_size": self.args.low_dim_size,
                "hidden_size": ctx.hidden_size,
            },
            save_path,
        )
        print(f"Saved low-dimensional projection weights to {save_path}")
