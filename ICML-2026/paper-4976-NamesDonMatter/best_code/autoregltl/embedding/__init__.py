import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Optional

from .normalization_methods import get_normalization_method


@dataclass
class EmbedderConfig:
    tie_embeddings: bool = True
    pad_vocab_size_multiple: int = 8

    # Number of dimensions for the Atomic Propositions
    d_ap: int = 0
    base_normalization: str = "l2"
    ap_normalization: str = "l2"
    final_normalization: str = "l2"

    feature_normalization: str = "disabled"
    
    embed_scaling: Optional[str] = None

    def build(self, d_model, vocab, **factory_kwargs):
        """
        Build and return the embedder.
        """
        return DynamicEmbedder(
            d_model=d_model,
            d_ap=self.d_ap,
            vocab=vocab,
            base_normalization=self.base_normalization,
            ap_normalization=self.ap_normalization,
            final_normalization=self.final_normalization,
            feature_normalization=self.feature_normalization,
            embed_scaling=self.embed_scaling,
            **factory_kwargs,
        )

    @classmethod
    def from_args(cls, args):
        return cls(
            d_ap = args.d_ap,
            base_normalization = args.embed_base_normalization,
            ap_normalization = args.embed_ap_normalization,
            final_normalization = args.embed_final_normalization,
            feature_normalization = args.feature_normalization,
            embed_scaling = args.embed_scaling,
        )


class EmbedScaler(nn.Module):
    def __init__(self, method, d_model, **factory_kwargs):
        super().__init__()
        if method is None:
            self.forward = lambda x: x
        elif method == "learnable":
            self.multiplier = nn.parameter.Parameter(torch.empty((1), **factory_kwargs))
            self.forward = lambda x: x * self.multiplier
        elif method == "sqrtd":
            # sqrt of embeddings dim
            multiplier = math.sqrt(d_model)
            self.forward = lambda x: x * multiplier
        else:
            raise ValueError(f"Unknown EmbedScaler method: {method}")

    def reset_parameters(self) -> None:
        try:
            self.multiplier.copy_(1.0)
            print("Reset embed:", self.multiplier)
        except AttributeError:
            pass


class DynamicEmbedder(nn.Module):
    def __init__(
            self,
            d_model,
            d_ap,
            vocab,
            base_normalization="l2",
            ap_normalization="l2",
            final_normalization="l2",
            feature_normalization="disabled",
            embed_scaling: Optional[str] = None,
            **factory_kwargs,
        ):
        self.vocab = vocab
        self.d_model = d_model
        self.d_ap = d_ap
        self.base_normalization = get_normalization_method(base_normalization)
        self.ap_normalization = get_normalization_method(ap_normalization)
        self.final_normalization = get_normalization_method(final_normalization)
        self.feature_normalization = get_normalization_method(feature_normalization)

        super().__init__()

        assert d_ap == 0
        # Vocab size without AP tokens
        self.base_vocab_size = vocab.size() - 26  # excluding AP tokens
        # Two tokens for APs: actual and placeholder
        vocab_size = self.base_vocab_size + 2
        self.base_weight = nn.parameter.Parameter(torch.empty((vocab_size, d_model), **factory_kwargs))

        # Effective weights of the embedding/projection matrix
        self.register_buffer("w", torch.empty((vocab_size, d_model), **factory_kwargs), persistent=False)

        self.embed_scaler = EmbedScaler(embed_scaling, d_model, **factory_kwargs)

        self.reset_parameters()
        self.prepare()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        torch.nn.init.normal_(self.base_weight)
        if (ap_weight := getattr(self, "ap_weight", None)) is not None:
            torch.nn.init.normal_(ap_weight)
        self.embed_scaler.reset_parameters()
        if hasattr(self, "ap_method") and hasattr(self.ap_method, "reset_parameters"):
            self.ap_method.reset_parameters()
    
    def prepare(self):
        self.ap_count = len(self.vocab.aps)
        w = self.base_weight

        # self.w rows must be normalized (L2 or another)
        # Better: normalize the base embeddings, then normalize the AP embeddings (d_ap), then normalize all (self.w)
        # Because we have two sides to balance: constant base embeddings and random AP embeddings
        # They should not override each other
        self.w = self.final_normalization(w).to(self.base_weight.device)

    def shrink_w(self):
        """
        Resize the vocabulary to the current vocab size.
        Returns the number of tokens removed.
        """
        old_vocab_size = self.output_vocab_size
        self.ap_count = len(self.vocab.aps)
        return old_vocab_size - self.output_vocab_size
    
    def embed(self, input_ids):
        # Shape of input_ids: (batch_size, seq_length)
        # Output shape: (batch_size, ap_tokens, seq_length, d_model)
        embeddings = []
        # Second output: a mask of shape (batch_size, ap_tokens, seq_length)
        # indicating where each AP token is located
        masks = []
        if self.ap_count == 0:
            emb = self.embed_scaler(F.embedding(input_ids, self.w)).unsqueeze(1)
            return emb, torch.zeros(emb.size(0), 1, emb.size(2), dtype=torch.int, device=emb.device)
        for i in range(self.ap_count):
            ap_token_id = self.base_vocab_size + i
            # Process input_ids such that:
            # 1. ap_token is replaced with self.base_vocab_size (actual AP embedding)
            # 2. all other AP tokens are replaced with self.base_vocab_size + 1 (placeholder embedding)
            modified_ids = input_ids.clone()
            modified_ids[input_ids >= self.base_vocab_size] = self.base_vocab_size + 1
            modified_ids[input_ids == ap_token_id] = self.base_vocab_size
            emb = self.embed_scaler(F.embedding(modified_ids, self.w)).unsqueeze(1)
            embeddings.append(emb)
            masks.append((input_ids == ap_token_id).unsqueeze(1))
        return torch.cat(embeddings, dim=1), torch.cat(masks, dim=1).int()
    
    def project(self, hidden, seq_ap_mask):
        hidden = self.feature_normalization(hidden)
        logits = F.linear(hidden, self.w)
        # Disallow start/pad tokens
        if self.vocab.use_start_token:
            logits[..., self.vocab.start_id] = -float("inf")
        if self.vocab.use_pad_token:
            logits[..., self.vocab.pad_id] = -float("inf")
        # current logits shape: (batch_size, ap_count, seq_length, base_vocab_size + 2)
        # output logits shape: (batch_size, seq_length, base_vocab_size + num_aps)
        output_logits = torch.zeros(
            (logits.size(0), logits.size(2), self.output_vocab_size),
            device=logits.device,
            dtype=logits.dtype,
        )
        # Base tokens: average across ap_count
        output_logits[:, :, :self.base_vocab_size] = logits[:, :, :, :self.base_vocab_size].mean(dim=1)
        # AP tokens
        if self.ap_count == 0:
            return output_logits
        # We need to convert this to bool so that ~ operator works correctly
        seq_ap_mask = seq_ap_mask.bool()
        for i in range(self.ap_count):
            ap_token_id = self.base_vocab_size + i
            output_logits[:, :, ap_token_id] = logits[:, i, :, self.base_vocab_size]
            # Disallow AP tokens if they don't appear in the input
            mask = ~seq_ap_mask[:, i].expand(output_logits.size(0))
            output_logits[mask, :, ap_token_id] = -float("inf")
        return output_logits
    
    def _get_output_vocab_size(self):
        return self.base_vocab_size + self.ap_count
    
    output_vocab_size = property(fget=_get_output_vocab_size)