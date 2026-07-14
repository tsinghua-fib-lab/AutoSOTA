#!/usr/bin/env python3
"""
Description: This module implements the `MultiModalTransformer` class, a
bidirectional transformer encoder designed to fuse multimodal information
by processing  representations from modality-specific encoders. It supports
token masking and provides flexibility in weight sharing across different
modalities. The model is the foundation of UrbanFusion.

Modes of operation:
-------------------
1. Single model:
   - A shared-weight transformer where all modalities use the same parameters.

2. LoRA Mixture of Experts (MoE):
   - Tokens from each modality utilize separate weights in the feed-forward
     layers.

3. LoRA Attention Experts:
   - Tokens from different modalities have distinct projections in the multi-
     head attention mechanism.

4. LoRA Mixture of Transformers (MoT):
   - Both the feed-forward and attention layers are modality-specific,
     allowing the highest level of specialization.
"""

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer
from torch import Tensor

from srl.multi_modal_encoder.lora_experts import LoRAExperts


class MultiModalTransformer(nn.Module):
    def __init__(
        self,
        encoders: list,
        transformer_type: str = "single",
        embed_dim: int = 256,
        only_cls: bool = True,
        avg_pool: bool = False,
        depth: int = 3,
        num_heads: int = 4,
        head: str = "identity",
        head_contrastive_dim: int = 256,
        head_hidden_dim: int = 256,
        hourglass_dim: int = 128,
        name_vit_architecture: str = None,
        pretrained: bool = False,
        first_n_layers: int = 3,
        reg_tokens: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        proj_bias: bool = True,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        add_positional_encodings: bool = True,
        lora_init: str = "xavier_uniform",
        lora_init_settings: dict = None,
        lora_rank: int = 8,
        lora_chunk_size: int = None,
        reconstruction_head_dim: int = None,
        normalize_embedding: bool = True,
    ) -> None:
        """
        A Transformer-Based Multimodal Fusion Model. The model can operate in
        different modes, including a single model, LoRA Mixture of Experts
        (MoE), LoRA Attention Experts, and LoRA Mixture of Transformers (MoT).
        It takes a list of modality-specific encoders as input and fuses their
        representations using a shared transformer architecture.

        Parameters
        ----------
        encoders : list
            List of modality-specific encoders, each producing a tensor of
            shape (batch_size, seq_len, embed_dim). The list should contain
            one encoder per modality as nn.Module instances.
        transformer_type : str, optional
            The type of model to use:
            "single" for a shared-weight transformer,
            "LoRA_MoE" for LoRA Mixture of Experts,
            "LoRA_attention" for LoRA Attention Experts,
            "LoRA_MoT" for LoRA Mixture of Transformers, by default "single".
        embed_dim : int, optional
            The embedding dimension of the model, by default 256.
        only_cls : bool, optional
            If True, only the CLS token is used for contrastive learning, by
            default True. Otherwise, each token is returned and used for
            contrastive learning.
        avg_pool : bool, optional
            If True, average pooling is applied to the output of the
            transformer before the projection head, by default False. If set,
            no CLS token added to the input. In this case, only_cls should be
            True if avg_pool is True. Default is False.
        depth : int, optional
            The number of transformer blocks to use, by default 3.
        num_heads : int, optional
            The number of attention heads to use, by default 4.
        head : str, optional
            The type of head to use: "mlp", "linear", or "identity", by
            default "identity". The head refers to the projection layer
            applied to the output of the transformer.
        head_contrastive_dim : int, optional
            The output dimension of the head for contrastive learning, by
            default 256. Only used if head is "mlp" or "linear".
        head_hidden_dim : int, optional
            The hidden dimension of the head for contrastive learning, by
            default 256. Only used if head is "mlp".
        hourglass_dim : int, optional
            The hidden dimension of the hourglass head for contrastive
            learning, by default 128. Only used if head is "hourglass".
        name_vit_architecture : str, optional
            The name of a Vision Transformer architecture to use, by default
            None. If provided, the model will use the specified architecture
            instead of the custom transformer settings.
        pretrained : bool, optional
            Whether to use a pretrained Vision Transformer, by default False.
            Only used if name_vit_architecture is provided.
        first_n_layers : int, optional
            The number of layers to use from the Vision Transformer, by
            default 3. Only used if name_vit_architecture is provided.
        reg_tokens : int, optional
            The number of register tokens to use, by default 0.
        mlp_ratio : float, optional
            The ratio of hidden to input dimensions in the feed-forward layers,
            by default 4.
        qkv_bias : bool, optional
            Whether to include bias in the query, key, and value projections,
            by default True.
        qk_norm : bool, optional
            Whether to normalize the query and key vectors, by default False.
        proj_bias : bool, optional
            Whether to include bias in the projection layers, by default True.
        proj_drop_rate : float, optional
            The dropout rate to apply to the projection layers, by default 0.
        attn_drop_rate : float, optional
            The dropout rate to apply to the attention layers, by default 0.
        add_positional_encodings : bool, optional
            Whether to add learnable positional encodings to the model, by
            default True.
        lora_init : str, optional
            The type of initialization to use for LoRA layers: "normal",
            "kaiming_uniform", "xavier_uniform", by default "xavier_uniform".
        lora_init_settings : dict, optional
            Custom initialization settings for LoRA layers, by default None.
            This can include:
            - "mean" / "std" for "normal" initialization.
            - "gain" for "xavier_uniform" initialization.
            - "a_squared" for "kaiming_uniform" initialization.
        lora_rank : int, optional
            The rank of the LoRA module (low-rank adaptation), by default 8.
        lora_chunk_size : int, optional
            The chunk size for processing modalities, by default None.
            If None, all modalities are processed in parallel.
            If 1, they are processed sequentially like a for-loop.
            Has influence on memory consumption and speed.
        reconstruction_head_dim : int, optional
            The dimension of the reconstruction head, by default None.
            Only supported if head is "hourglass" and only_cls is True.
        normalize_embedding : bool, optional
            Whether to normalize the output embeddings to hypersphere, by
            default True.
        """
        super().__init__()

        # Check if encoders are provided
        assert encoders is not None, "You must provide a list of encoders."

        # Check if right output setting for average pooling
        if avg_pool:
            assert only_cls, "If avg_pool is True, only_cls should be True."

        # Store parameters
        self.transformer_type = transformer_type
        self.encoders = nn.ModuleList(encoders)
        self.embed_dim = embed_dim
        self.only_cls = only_cls
        self.avg_pool = avg_pool
        self.depth = depth
        self.num_heads = num_heads
        self.head = head
        self.head_type = head
        self.head_contrastive_dim = head_contrastive_dim
        self.head_hidden_dim = head_hidden_dim
        self.hourglass_dim = hourglass_dim
        self.name_vit_architecture = name_vit_architecture
        self.pretrained = pretrained
        self.first_n_layers = first_n_layers
        self.reg_tokens = reg_tokens
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm
        self.proj_bias = proj_bias
        self.proj_drop_rate = proj_drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.add_positional_encodings = add_positional_encodings
        self.lora_init = lora_init
        self.lora_init_settings = lora_init_settings
        self.lora_rank = lora_rank
        self.lora_chunk_size = lora_chunk_size
        self.reconstruction_head_dim = reconstruction_head_dim
        self.normalize_embedding = normalize_embedding

        # Calculate sequence length per modality (including CLS and Register
        # Tokens) used for individual sequence length per modality in LoRA
        # experts.
        self.sequence_modalities = self._get_sequence_per_modality(encoders)
        self.reg_tokens_bool = int(bool(reg_tokens))  # Store whether exist

        # Compute total number of tokens (CLS + Register Tokens + Modalities)
        if self.avg_pool:
            self.cls_tokens = 0
        else:
            self.cls_tokens = 1
        self.number_tokens = (
            self.cls_tokens
            + reg_tokens
            + sum(encoder.seq_len for encoder in encoders)
        )

        # Compute number of modalities (CLS + Register Tokens + Encoders)
        self.n_modalities = (
            len(encoders) + self.cls_tokens + self.reg_tokens_bool
        )

        # Initialize transformer blocks
        self._get_blocks()

        # Initialize head
        self._get_head()

        # Initialize learnable positional encodings
        self.positional_embedding = (
            nn.Parameter(torch.zeros(1, self.number_tokens, embed_dim))
            if add_positional_encodings
            else None
        )

        # Initialize learnable CLS token
        if not self.avg_pool:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Initialize learnable register tokens
        self.reg_token = (
            nn.Parameter(torch.zeros(1, reg_tokens, embed_dim))
            if reg_tokens
            else None
        )
        self._init_tokens()

        if self.transformer_type != "single":
            self._add_lora_experts()

    def _get_sequence_per_modality(self, encoders: list) -> dict:
        """
        Initializes a dictionary mapping modality indices to their respective
        sequence lengths, including CLS and optional register tokens.

        Parameters
        ----------
        encoders : list
            A list of encoder objects, each containing a `seq_len` attribute.
        reg_tokens : int, optional
            Number of register tokens, if any (default: None).

        Returns
        -------
        sequence_modalities : dict
            A dictionary where keys represent modality indices and values are
            sequence lengths.
        """
        if not self.avg_pool:
            sequence_modalities = {0: 1}  # CLS token
            if self.reg_tokens:
                sequence_modalities[1] = self.reg_tokens
        else:
            sequence_modalities = {}
            if self.reg_tokens:
                sequence_modalities[0] = self.reg_tokens

        # Assign sequence lengths from encoders
        for encoder in encoders:
            sequence_modalities[len(sequence_modalities)] = encoder.seq_len

        return sequence_modalities

    def _get_blocks(self):
        """
        Initializes the transformer blocks based on the specified settings.
        Either uses a custom transformer or existing (pretrained) ViT model
        for extracting the transformer blocks. Implementation is based on
        the timm library.
        """
        if self.name_vit_architecture is not None:
            timm_vit = timm.create_model(
                self.name_vit_architecture,
                pretrained=self.pretrained,
                proj_drop_rate=self.proj_drop_rate,
                attn_drop_rate=self.attn_drop_rate,
            )
            self.blocks = timm_vit.blocks[0 : self.first_n_layers]
            self.embed_dim = timm_vit.embed_dim  # update embed_dim
        else:
            timm_vit = VisionTransformer(
                embed_dim=self.embed_dim,
                depth=self.depth,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                qk_norm=self.qk_norm,
                proj_bias=self.proj_bias,
                proj_drop_rate=self.proj_drop_rate,
                attn_drop_rate=self.attn_drop_rate,
            )
            self.blocks = timm_vit.blocks

    def _get_head(self):
        """
        Initializes the head of the model, supporting different head types
        ('mlp', 'linear', 'identity').
        """
        self.head_name = self.head
        # Mapping of head types to corresponding constructors
        head_constructors = {
            "mlp": self._create_mlp_head,
            "linear": self._create_linear_head,
            "identity": self._create_identity_head,
            "hourglass": self._create_hourglass_decoder,
            "hourglass_small": self._create_hourglass_small_decoder,
            "hourglass_tiny": self._create_hourglass_tiny_decoder,
        }

        if self.head not in head_constructors:
            raise ValueError(f"Head type {self.head} not supported")

        if self.only_cls:
            if self.head == "hourglass":
                self.head_encoder = self._create_hourglass_encoder()
            elif self.head == "hourglass_small":
                self.head_encoder = self._create_hourglass_small_encoder()
            elif self.head == "hourglass_tiny":
                self.head_encoder = self._create_hourglass_tiny_encoder()
            self.head = head_constructors[self.head](self.head_contrastive_dim)
            if self.reconstruction_head_dim is not None:
                if self.head_name == "hourglass":
                    self.head_reconstruction = self._create_hourglass_decoder(
                        self.reconstruction_head_dim
                    )
                elif self.head_name == "hourglass_small":
                    self.head_reconstruction = (
                        self._create_hourglass_small_decoder(
                            self.reconstruction_head_dim
                        )
                    )
                elif self.head_name == "hourglass_tiny":
                    self.head_reconstruction = (
                        self._create_hourglass_tiny_decoder(
                            self.reconstruction_head_dim
                        )
                    )
        else:
            if self.head == "hourglass":
                # One head per token
                self.head_encoder = nn.ModuleList(
                    [
                        self._create_hourglass_encoder()
                        for _ in range(self.number_tokens)
                    ]
                )
            # One head per token
            self.head = nn.ModuleList(
                [
                    head_constructors[self.head](self.head_contrastive_dim)
                    for _ in range(self.number_tokens)
                ]
            )

    def _create_mlp_head(self):
        """
        Creates an MLP-based head after the transformer.
        """
        return nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.head_hidden_dim),
            nn.LayerNorm(self.head_hidden_dim),
            nn.GELU(),
            nn.Linear(self.head_hidden_dim, self.head_contrastive_dim),
        )

    def _create_linear_head(self):
        """
        Creates a Linear-based head after the transformer.
        """
        return nn.Sequential(
            nn.Linear(self.embed_dim, self.head_contrastive_dim)
        )

    @staticmethod
    def _create_identity_head():
        """
        Creates an Identity head after the transformer.
        """
        return nn.Identity()

    def _create_hourglass_encoder(self) -> nn.Sequential:
        """
        Creates an hourglass head encoder after the transformer.
        Used for reducing the dimensionality of the transformer output, before
        increasing it again for the contrastive loss. The lower dimensional
        represenation is potentially better for downstream tasks.
        """
        return nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.head_hidden_dim),
            nn.LayerNorm(self.head_hidden_dim),
            nn.GELU(),
            nn.Linear(self.head_hidden_dim, self.hourglass_dim),
        )

    def _create_hourglass_small_encoder(self) -> nn.Sequential:
        """
        Creates a tiny hourglass head encoder after the transformer.
        Used for reducing the dimensionality of the transformer output, before
        increasing it again for the contrastive loss. The lower dimensional
        represenation is potentially better for downstream tasks.
        """
        return nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.hourglass_dim),
        )

    def _create_hourglass_tiny_encoder(self) -> nn.Sequential:
        """
        Creates a tiny hourglass head encoder after the transformer.
        Used for reducing the dimensionality of the transformer output, before
        increasing it again for the contrastive loss. The lower dimensional
        represenation is potentially better for downstream tasks. Tiny is
        just identity
        """
        return nn.Identity()

    def _create_hourglass_decoder(self, ouput_dim) -> nn.Sequential:
        """
        Creates an hourglass head decoder after the transformer.
        Used for increasing the dimensionality of the transformer output, after
        reducing it for the contrastive loss. The higher dimensional
        represenation is potentially better for downstream tasks.
        """
        return nn.Sequential(
            nn.LayerNorm(self.hourglass_dim),
            nn.GELU(),
            nn.Linear(self.hourglass_dim, self.head_hidden_dim),
            nn.LayerNorm(self.head_hidden_dim),
            nn.GELU(),
            nn.Linear(self.head_hidden_dim, ouput_dim),
        )

    def _create_hourglass_small_decoder(self, ouput_dim) -> nn.Sequential:
        """
        Creates a tiny hourglass head decoder after the transformer.
        Used for increasing the dimensionality of the transformer output, after
        reducing it for the contrastive loss. The higher dimensional
        represenation is potentially better for downstream tasks.
        """
        return nn.Sequential(
            nn.LayerNorm(self.hourglass_dim),
            nn.GELU(),
            nn.Linear(self.hourglass_dim, ouput_dim),
        )

    def _create_hourglass_tiny_decoder(self, ouput_dim) -> nn.Sequential:
        """
        Creates a tiny hourglass head decoder after the transformer.
        Used for increasing the dimensionality of the transformer output, after
        reducing it for the contrastive loss. The higher dimensional
        represenation is potentially better for downstream tasks. Tiny is
        just identity
        """
        return nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, ouput_dim),
        )

    def _init_tokens(self) -> None:
        """
        Initializes the special tokens and positional embeddings.

        Based on the timm ViT implementation.
        https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
        """
        # Initialize weights of the special tokens
        if not self.avg_pool:
            nn.init.normal_(self.cls_token, std=1e-6)
        if self.reg_token is not None:
            nn.init.normal_(self.reg_token, std=1e-6)
        if self.positional_embedding is not None:
            nn.init.normal_(self.positional_embedding, std=1e-6)

    def _add_lora_experts(self):
        """
        Adds LoRA experts to the model based on the specified transformer type.

        The following transformer types are supported:
        - LoRA Mixture of Experts (MoE)
        - LoRA Attention Experts
        - LoRA Mixture of Transformers (MoT)
        """

        # Add LoRA experts to the model based on the transformer type
        for layer_id, enc_layer in enumerate(self.blocks):

            if self.transformer_type in ["LoRA_attention", "LoRA_MoT"]:
                # Replace query, key, and value projections with LoRA experts
                setattr(
                    enc_layer.attn,
                    "qkv",
                    LoRAExperts(
                        getattr(enc_layer.attn, "qkv"),
                        rank=self.lora_rank,
                        dim=self.embed_dim,
                        out_dim=self.embed_dim * 3,
                        n_modalities=self.n_modalities,
                        initialize=True,
                        init_type=self.lora_init,
                        init_settings=self.lora_init_settings,
                        chunk_size=self.lora_chunk_size,
                        sequence_modalities=self.sequence_modalities,
                    ),
                )

                # Replace output projection with LoRA experts
                setattr(
                    enc_layer.attn,
                    "proj",
                    LoRAExperts(
                        getattr(enc_layer.attn, "proj"),
                        rank=self.lora_rank,
                        dim=self.embed_dim,
                        out_dim=self.embed_dim,
                        n_modalities=self.n_modalities,
                        initialize=True,
                        init_type=self.lora_init,
                        init_settings=self.lora_init_settings,
                        chunk_size=self.lora_chunk_size,
                        sequence_modalities=self.sequence_modalities,
                    ),
                )

            if self.transformer_type in ["LoRA_MoE", "LoRA_MoT"]:
                # Replace feed-forward layers with LoRA experts
                setattr(
                    enc_layer.mlp,
                    "fc1",
                    LoRAExperts(
                        getattr(enc_layer.mlp, "fc1"),
                        rank=self.lora_rank,
                        dim=self.embed_dim,
                        out_dim=self.embed_dim * 4,
                        n_modalities=self.n_modalities,
                        initialize=True,
                        init_type=self.lora_init,
                        init_settings=self.lora_init_settings,
                        chunk_size=self.lora_chunk_size,
                        sequence_modalities=self.sequence_modalities,
                    ),
                )
                setattr(
                    enc_layer.mlp,
                    "fc2",
                    LoRAExperts(
                        getattr(enc_layer.mlp, "fc2"),
                        rank=self.lora_rank,
                        dim=self.embed_dim * 4,
                        out_dim=self.embed_dim,
                        n_modalities=self.n_modalities,
                        initialize=True,
                        init_type=self.lora_init,
                        init_settings=self.lora_init_settings,
                        chunk_size=self.lora_chunk_size,
                        sequence_modalities=self.sequence_modalities,
                    ),
                )

    def forward(
        self,
        inputs: list,
        mask_indices: list = None,
        return_representations: bool = False,
        return_modality_tokens: bool = False,
        return_backbone_features: bool = False,
        modality_dropout: float = 0.0,
    ) -> Tensor:
        """
        Forward pass of the model. Optionally applies modality masking.
        For contrastive learning, the model returns either the CLS token or
        all tokens, depending on the `only_cls` attribute. Optionally, the
        representations before the projection head can be returned.

        Parameters
        ----------
        inputs : list
            One tensor per modality, each of shape (batch_size, seq_len,
            input_dim).
        mask_indices : list, optional
            List of modality indices to be masked, by default None.
        return_representations : bool, optional
            If true, return representations before the projection head, by
            default False.
        return_backbone_features : bool, optional
            If true, return the features from the backbone encoders,
            by default False.

        Returns
        -------
        x : Tensor
            The output tensor of shape (batch_size, number_tokens,
            contrastive_dim / embed_dim).
        """

        assert len(inputs) == len(
            self.encoders
        ), "Mismatch between inputs and encoders"
        if modality_dropout > 0.0 and mask_indices is not None:
            total_modalities = len(self.encoders)
            device = (
                mask_indices.device
                if isinstance(mask_indices, torch.Tensor)
                else "cpu"
            )
            all_indices = torch.arange(total_modalities, device=device)
            mask_tensor = torch.tensor(list(mask_indices), device=device)
            mask_bool = torch.zeros(
                total_modalities, dtype=torch.bool, device=device
            )
            mask_bool[mask_tensor] = True
            remaining = all_indices[~mask_bool]
            keep_mask = torch.rand(len(remaining), device=device) < (
                modality_dropout
            )
            selected = remaining[keep_mask]
            if selected.numel() == remaining.numel() and selected.numel() > 0:
                idx_to_drop = torch.randint(0, selected.numel(), (1,))
                selected = torch.cat(
                    [selected[:idx_to_drop], selected[idx_to_drop + 1 :]]
                )
            mask_indices = set(mask_indices) | set(selected.tolist())

        # Apply modality-specific encoders
        encoded_tokens = []
        if return_backbone_features:
            backbone_features = []
        for i, encoder in enumerate(self.encoders):

            # Check if modality should be masked
            # if yes create a tensor of zeros with the same shape
            if mask_indices and i in mask_indices:
                # Get batch size based on input type
                if isinstance(inputs[i], torch.Tensor):
                    # Regular tensor
                    batch_size = inputs[i].shape[0]
                    device = inputs[i].device
                else:
                    # CLIP tokenized text
                    batch_size = inputs[i]["input_ids"].shape[0]
                    device = inputs[i]["input_ids"].device

                # Get encoder's dtype
                dtype = next(self.blocks[0].parameters()).dtype

                # Create masked tensor based on encoder's properties
                masked_shape = (batch_size, encoder.seq_len, self.embed_dim)
                encoded_tokens.append(
                    torch.zeros(masked_shape, device=device, dtype=dtype)
                )
                if return_backbone_features:
                    encoded_output, features = encoder(
                        inputs[i], return_features=True
                    )
                    del encoded_output
                    backbone_features.append(features)
            else:
                #  Apply encoder to generate tokens
                #  if i == 4:
                #    print("encoder ", i)
                #    print(inputs[i])
                #    print(encoder(inputs[i]))
                if return_backbone_features:
                    encoded_output, features = encoder(
                        inputs[i], return_features=True
                    )
                    backbone_features.append(features)
                else:
                    encoded_output = encoder(inputs[i])
                encoded_tokens.append(encoded_output)

        # Concatenate encoded tokens along sequence dimension
        x = torch.cat(
            encoded_tokens, dim=1
        )  # Shape: (batch_size, total_seq_len, embed_dim)

        if return_backbone_features:
            multi_modal_repr = torch.cat(backbone_features, dim=1)

        if return_modality_tokens:
            # Flatten along embedding dimension
            combined_representation = x.view(
                x.size(0), -1
            ).detach()  # Shape: (batch_size, total_seq_len * embed_dim)

            # Normalize combined representation
            combined_representation = F.normalize(
                combined_representation, p=2, dim=-1
            )

        # Add CLS Token
        if not self.avg_pool:
            batch_size = x.shape[0]
            cls_token = self.cls_token.expand(
                batch_size, -1, -1
            )  # Expand CLS token to batch size

        # Add Register Tokens
        reg_tokens = (
            self.reg_token.expand(batch_size, -1, -1)
            if self.reg_token is not None
            else None
        )

        token_list = []
        # Concatenate CLS and Register Tokens
        if not self.avg_pool:
            token_list.append(cls_token)
        if reg_tokens is not None:
            token_list.append(reg_tokens)
        token_list.append(x)

        # (batch_size, 1 + reg_tokens + seq_len, embed_dim)
        x = torch.cat(token_list, dim=1)

        # Add positional encodings
        if self.positional_embedding is not None:
            x = x + self.positional_embedding[:, : x.shape[1], :]

        # Pass input through Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        # Return representations before head if requested
        if return_representations:
            # Only return CLS token
            if self.only_cls and not self.avg_pool:
                x = x[:, 0]
            elif self.avg_pool:
                # Average pooling over all tokens
                x = x.mean(dim=1)
            if (
                self.head_type == "hourglass"
                or self.head_type == "hourglass_small"
                or self.head_type == "hourglass_tiny"
            ):
                x = self.head_encoder(x)
            return x

        # Apply projection head
        else:
            # Apply head to the CLS token of the transformer
            if self.only_cls:
                if not self.avg_pool:
                    x = x[:, 0]  # shape (B, contrastive_dim)
                # Average pooling over all tokens
                elif self.avg_pool:
                    x = x.mean(dim=1)
                # If hourglass head is used, apply encoder first
                if (
                    self.head_type == "hourglass"
                    or self.head_type == "hourglass_small"
                    or self.head_type == "hourglass_tiny"
                ):
                    low_dim = self.head_encoder(x)
                # Apply decoder head
                x = self.head(low_dim)
                if self.reconstruction_head_dim is not None:
                    rec = self.head_reconstruction(low_dim)

            # Apply heads to all tokens of the transformer
            else:
                tokens = []
                if (
                    self.head_type == "hourglass"
                    or self.head_type == "hourglass_small"
                    or self.head_type == "hourglass_tiny"
                ):
                    enocder_tokens = []
                    for i in range(self.number_tokens):
                        enocder_tokens.append(self.head_encoder[i](x[:, i]))
                    for i in range(self.number_tokens):
                        tokens.append(
                            self.head[i](enocder_tokens[i])
                        )  # shape (B, contrastive_dim)
                else:
                    for i in range(self.number_tokens):
                        tokens.append(
                            self.head[i](x[:, i])
                        )  # shape (B, contrastive_dim)

                x = torch.stack(
                    tokens, dim=1
                )  # shape (B, number_tokens, contrastive_dim)

        # Apply L2 normalization to get embeddings on hypersphere
        if self.normalize_embedding:
            x = F.normalize(x, p=2, dim=-1)

        if return_modality_tokens:
            if return_backbone_features:
                x, combined_representation, multi_modal_repr, rec
            else:
                return x, combined_representation
        else:
            if return_backbone_features:
                return x, multi_modal_repr, rec
            else:
                return x
