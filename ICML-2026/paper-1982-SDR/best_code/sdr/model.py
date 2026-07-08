import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from typing import Optional
from transformers import AutoModelForCausalLM, PreTrainedModel
from safetensors.torch import save_file

from attn_mask_utils import _prepare_4d_causal_attention_mask

def pad_embeddings_with_attention_mask(
    embeddings,
    attention_mask,
):
    """
    embeddings: List[Tensor(seq_len_i, hidden_dim)]
    attention_mask: Tensor(B, L)
    """

    batch_size = len(embeddings)
    assert attention_mask.dim() == 2
    B, L = attention_mask.shape
    assert B == batch_size

    hidden_dim = embeddings[0].size(-1)

    max_len = max(e.size(0) for e in embeddings)
    if max_len != L:
        raise ValueError(
            f"Max embedding length ({max_len}) != attention_mask.shape[1] ({L})"
        )
    
    def infer_pad_direction(mask_row):
        first_one = (mask_row == 1).nonzero(as_tuple=True)[0][0].item()
        last_one = (mask_row == 1).nonzero(as_tuple=True)[0][-1].item()
        if first_one == 0:
            return "right"
        elif last_one == L - 1:
            return "left"
        else:
            raise ValueError("Attention mask is not contiguous")

    pad_direction = infer_pad_direction(attention_mask[0])

    padded = torch.zeros(
        (batch_size, L, hidden_dim),
        dtype=embeddings[0].dtype,
        device=embeddings[0].device,
    )

    for i, emb in enumerate(embeddings):
        seq_len = emb.size(0)

        if pad_direction == "right":
            padded[i, :seq_len, :] = emb
        else:  # left padding
            padded[i, L - seq_len :, :] = emb

    return padded

class GeneralHead(nn.Module):
    def __init__(self, config, base_layers, rotary_emb, input_dim, output_dim, dtype, head_model, multi_remote_strategy, dropout_rate=0.1, device="cuda"):
        super().__init__()
        
        self.config = config
        self.device = torch.device(device)
        
        self.rotary_emb = rotary_emb

        # different head designs
        if head_model.startswith('transformer'):
            self.transformer_layer = copy.deepcopy(base_layers[-1])
            self.transformer_layer.apply(self._init_weights)
        else:
            self.transformer_layer = None
        
        if head_model == 'linear':
            self.mlp_head = nn.Linear(input_dim, 1)
            self.mlp_head.apply(self._init_weights).to(dtype)
        else: # mlp or transformer(+mlp)
            if multi_remote_strategy == 'head':
                self.mlp_head = nn.Sequential(
                    nn.Linear(input_dim, 1024),
                    nn.GELU(),
                    nn.Dropout(dropout_rate),

                    nn.Linear(1024, 512),
                    nn.GELU(),
                    nn.Dropout(dropout_rate),

                    nn.Linear(512, 256),
                    nn.GELU(),
                    nn.Dropout(dropout_rate),

                    nn.Linear(256, output_dim)
                )
            else:
                self.mlp_head = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(input_dim, 256),
                        nn.GELU(),
                        nn.Dropout(dropout_rate),

                        nn.Linear(1024, 512),
                        nn.GELU(),
                        nn.Dropout(dropout_rate),

                        nn.Linear(512, 256),
                        nn.GELU(),
                        nn.Dropout(dropout_rate),

                        nn.Linear(256, 1)
                    )
                    for _ in range(output_dim)
                ])
            self.mlp_head.apply(self._init_weights).to(dtype)

    def _init_weights(self, module):
        initializer_range = self._get_config_attr("initializer_range", default=0.02)
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def _get_config_attr(self, *args, default=None):
        for attr in args:
            if hasattr(self.config, attr):
                return getattr(self.config, attr)
        if default is not None:
            return default
        raise AttributeError(f"Config does not contain any of {args}")

    def forward(self, hidden_states, attention_mask):
        if self.transformer_layer is not None:
            assert hidden_states.ndim == 3, "transformer-based scoring model needs all tokens' hidden"
            # Prepare Rotary Embeddings (RoPE)
            position_embeddings = None
            if self.rotary_emb is not None:
                batch_size = hidden_states.shape[0]
                seq_len = hidden_states.shape[1]
                device = hidden_states.device
                
                position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
                cos, sin = self.rotary_emb(hidden_states, position_ids)
                
                position_embeddings = (cos, sin)

            # Prepare 4D causal attention mask
            if attention_mask is not None:
                batch_size, seq_length = attention_mask.shape
                attention_mask_4d = _prepare_4d_causal_attention_mask(
                    attention_mask, (batch_size, seq_length), hidden_states, 0
                )

            hidden_states = self.transformer_layer(
                hidden_states, 
                attention_mask=attention_mask_4d, 
                position_embeddings=position_embeddings
            )

        if hidden_states.ndim == 3: # all tokens' hidden, need to extract the last one
            batch_idx = torch.arange(hidden_states.shape[0], device=hidden_states.device)
            last_token_ids = attention_mask.sum(dim=1) - 1
            hidden_states = hidden_states[batch_idx, last_token_ids, :]

        if isinstance(self.mlp_head, nn.Sequential) or isinstance(self.mlp_head, nn.Linear):     
            head_outputs = self.mlp_head(hidden_states)
        else:
            outputs = [mlp(hidden_states) for mlp in self.mlp_head]   # list of (B, 1)
            head_outputs = torch.cat(outputs, dim=-1)   
        
        return head_outputs


class LLMWithScorer(nn.Module):
    def __init__(self, base_model_name, input_layer, num_remote, head_model, multi_remote_strategy, embed_path=None, device="cuda"):
        super().__init__()
        
        self.device = torch.device(device)
        
        print(f"Loading base model: {base_model_name}...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True # Qwen often needs this
        ).to(self.device)
        self.config = self.base_model.config
        self.dtype = next(self.base_model.parameters()).dtype
        
        # Freeze base model
        for p in self.base_model.parameters():
            p.requires_grad_(False)

        # Find Layers
        base_layers = self._find_transformer_layers(self.base_model)
        if base_layers is None:
            raise ValueError("Could not automatically find the transformer layer list.")
        
        # Extract Rotary Embedding Module
        rotary_emb = self._find_rotary_embedding(self.base_model)
            
        # load embeds if provided
        if embed_path is not None:
            print("Loading embeddings from disk...")
            data = torch.load(embed_path, map_location="cpu")
            self.embeddings = data["embeddings"]
            self.input_layer_index = data["layer_id"]
            del self.base_model # no further use
        else:
            print("args.embed_path not provided, need to generate hidden states online during training")
            # Handle layer indexing
            if input_layer > 0 and input_layer < 1: # proportion of all layers
                self.input_layer_index = int(len(base_layers) * input_layer)
            elif input_layer < 0:
                self.input_layer_index = int(input_layer + len(base_layers))

        hidden_size = self._get_config_attr("hidden_size", "n_embd")
        
        self.scorer = GeneralHead(self.config, base_layers, rotary_emb, hidden_size, num_remote, self.dtype, head_model, multi_remote_strategy)
        
        for p in self.scorer.parameters():
            p.requires_grad_(True)


    def _get_config_attr(self, *args, default=None):
        for attr in args:
            if hasattr(self.config, attr):
                return getattr(self.config, attr)
        if default is not None:
            return default
        raise AttributeError(f"Config does not contain any of {args}")

    def _find_transformer_layers(self, model: PreTrainedModel) -> Optional[nn.ModuleList]:
        search_paths = ["model.layers", "model.h", "transformer.h", "bert.encoder.layer", "layers"]
        for path in search_paths:
            module = model
            try:
                for attr in path.split("."):
                    module = getattr(module, attr)
                if isinstance(module, (nn.ModuleList, list)):
                    return module
            except AttributeError:
                continue
        return None

    def _find_rotary_embedding(self, model: PreTrainedModel):
        """
        Locate the RoPE module. Usually model.model.rotary_emb or similar.
        """
        # Common paths for Qwen2, Llama, Mistral
        search_paths = ["model.rotary_emb", "transformer.rotary_emb", "rotary_emb"]
        for path in search_paths:
            module = model
            try:
                for attr in path.split("."):
                    module = getattr(module, attr)
                # Check if it has a forward method (it's a module)
                if hasattr(module, "forward"):
                    return module
            except AttributeError:
                continue
        return None

    def forward(self, input_ids, attention_mask, **kwargs):
        base_outputs = None
        if "embeddings" in self.__dict__: # embeddings provided
            index = kwargs.pop("index")
            batch_embeddings = [self.embeddings[id] for id in index]
            if batch_embeddings[0].ndim > 1: # all tokens
                hidden_states = pad_embeddings_with_attention_mask(batch_embeddings, attention_mask).to(self.device)
            else: # last tokens only
                hidden_states = torch.stack(batch_embeddings, dim=0).to(self.device)
        else: # Run Base Model
            with torch.no_grad():
                base_outputs = self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    **kwargs
                )
            if self.input_layer_index >= len(base_outputs.hidden_states):
                hidden_states = base_outputs.hidden_states[-1]
            else:
                hidden_states = base_outputs.hidden_states[self.input_layer_index]
        
        score = self.scorer(hidden_states, attention_mask)
        
        return base_outputs, score
        