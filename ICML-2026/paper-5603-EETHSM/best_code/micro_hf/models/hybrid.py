""" PyTorch GPTNeoX model."""

from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    ModelOutput,
    logging,
)
from transformers.configuration_utils import PretrainedConfig
from transformers.generation import GenerationMixin

from .rope import GPTNeoXLayer
from .mamba import MambaBlock, MambaCache

import math
from dataclasses import dataclass
from typing import Optional, Union

logger = logging.get_logger(__name__)


class HybridConfig(PretrainedConfig):

    model_type = "hybrid"

    def __init__(
        self,
        layers=[],

        vocab_size=50432,
        hidden_size=6144,
        num_attention_heads=64,
        intermediate_size=24576,
        hidden_act="silu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        rotary_pct=0.25,
        rotary_emb_base=10000,
        max_position_embeddings=2048,
        initializer_range=0.02,
        layer_norm_epsilon=1e-5,
        use_cache=True,
        bos_token_id=0,
        pad_token_id=0,
        eos_token_id=2,
        tie_word_embeddings=False,
        
        state_size=16,
        expand=2,
        conv_kernel=4,
        use_bias=False,
        use_conv_bias=True,
        residual_in_fp32=True,
        time_step_rank="auto",
        time_step_scale=1.0,
        time_step_min=0.001,
        time_step_max=0.1,
        time_step_init_scheme="random",
        time_step_floor=1e-4,
        rescale_prenorm_residual=False,
        use_mambapy=False,
        
        **kwargs
    ):
        self.layers = layers

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = len(layers)
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.rotary_pct = rotary_pct
        self.rotary_emb_base = rotary_emb_base
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_epsilon
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings

        self.state_size = state_size
        self.layer_norm_epsilon = layer_norm_epsilon
        self.conv_kernel = conv_kernel
        self.expand = expand
        self.intermediate_size = int(expand * self.hidden_size)
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.use_bias = use_bias
        self.use_conv_bias = use_conv_bias
        self.time_step_rank = math.ceil(self.hidden_size / 16) if time_step_rank == "auto" else time_step_rank
        self.time_step_scale = time_step_scale
        self.time_step_min = time_step_min
        self.time_step_max = time_step_max
        self.time_step_init_scheme = time_step_init_scheme
        self.time_step_floor = time_step_floor
        self.rescale_prenorm_residual = rescale_prenorm_residual
        self.residual_in_fp32 = residual_in_fp32
        self.use_mambapy = use_mambapy

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, pad_token_id=pad_token_id, **kwargs)


class HybridPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = HybridConfig
    base_model_prefix = "hybrid"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GPTNeoXLayer", "MambaBlock"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, HybridModel):
            module.gradient_checkpointing = value



@dataclass
class HybridOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[torch.FloatTensor] = None
    cache_params: Optional[MambaCache] = None


@dataclass
class HybridCausalLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    cache_params: Optional[MambaCache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    past_key_values: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    attentions: Optional[torch.FloatTensor] = None



class HybridModel(HybridPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embed_in = nn.Embedding(config.vocab_size, config.hidden_size)
        self.emb_dropout = nn.Dropout(config.hidden_dropout_prob)

        layer_types = [MambaBlock, GPTNeoXLayer]

        modules = []
        for idx, layer in enumerate(config.layers):
            if layer == "SSM":
                modules.append(MambaBlock(config, layer_idx=idx))
            elif layer == "TF":
                modules.append(GPTNeoXLayer(config))
            else:
                raise NotImplementedError

        self.layers = nn.ModuleList(modules)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_in

    def set_input_embeddings(self, value):
        self.embed_in = value
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,

        cache_params: Optional[MambaCache] = None, # From Mamba
        cache_poisition: Optional[torch.LongTensor] = None, # From Mamba
        # cache_position: Optional[torch.LongTensor] = None, # From Mamba
    ) -> Union[Tuple, HybridOutput]:
        r"""
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * self.config.num_hidden_layers)
        else:
            past_length = past_key_values[0][0].size(-2)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(past_length, seq_length + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)

        # Attention mask.
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            # attention_mask = attention_mask.view(batch_size, -1)
            # # We create a 3D attention mask from a 2D tensor mask.
            # # Sizes are [batch_size, 1, 1, to_seq_length]
            # # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # # this attention mask is more simple than the triangular masking of causal attention
            # # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            # attention_mask = attention_mask[:, None, None, :]

            # # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # # masked positions, this operation will create a tensor which is 0.0 for
            # # positions we want to attend and -10000.0 for masked positions.
            # # Since we are adding it to the raw scores before the softmax, this is
            # # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        if inputs_embeds is None:
            inputs_embeds = self.embed_in(input_ids)

        hidden_states = self.emb_dropout(inputs_embeds)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # From Mamba
        if use_cache:
            if cache_params is None:
                cache_params = MambaCache(
                    self.config, inputs_embeds.size(0), device=inputs_embeds.device, dtype=inputs_embeds.dtype
                )
                cache_position = torch.arange(0, self.config.conv_kernel, device=inputs_embeds.device)
            elif cache_position is None:
                # cases when we do manual forward instead of using `model.generate` which will initiate
                # `cache_position` and makes sure it is not None, throw error here instead of doing some
                # hack to conjecture the current cache position
                raise ValueError(
                    "You have to specify the `cache_position` manually when `use_cache=True` and `cache_params` is passed, "
                    "you don't have to pass a `cache_params` if you are in prefilling stage because in that case it will "
                    "be initialized for you automatically"
                )
        else:
            cache_params = None
            cache_position = None

            

        presents = () if use_cache else None
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, (layer, layer_past) in enumerate(zip(self.layers, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            if isinstance(layer, GPTNeoXLayer):
                outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    layer_past=layer_past,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
                hidden_states = outputs[0]
                if use_cache is True:
                    presents = presents + (outputs[1],)
                if output_attentions:
                    all_attentions = all_attentions + (outputs[2 if use_cache else 1],)

                    
            elif isinstance(layer, MambaBlock):
                hidden_states = layer(
                    hidden_states,
                    cache_params=cache_params,
                    cache_position=cache_position,
                    # attention_mask=attention_mask,
                )

                if use_cache is True:
                    presents = presents + (None,)
                if output_attentions:
                    all_attentions = all_attentions + (None,)

            else:
                assert False, "Unexpected Layer"


        hidden_states = self.final_layer_norm(hidden_states)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_attentions, cache_params] if v is not None)

        return HybridOutput(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cache_params=cache_params
        )



class HybridForCausalLM(HybridPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["embed_out.weight"]

    def __init__(self, config):
        super().__init__(config)

        self.hybrid = HybridModel(config)
        self.embed_out = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.embed_out

    def set_output_embeddings(self, new_embeddings):
        self.embed_out = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        
        cache_params: Optional[MambaCache] = None, # From Mamba
        cache_position: Optional[torch.LongTensor] = None, # From Mamba
        **kwargs,  # for now we need this for generation
    ) -> Union[Tuple, HybridOutput]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.hybrid(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_params=cache_params,
            cache_poisition=cache_position,
        )

        hidden_states = outputs[0]
        lm_logits = self.embed_out(hidden_states)

        lm_loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shift_logits = lm_logits[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return HybridCausalLMOutput(
            loss=lm_loss,
            logits=lm_logits,
            cache_params=outputs.cache_params,
            hidden_states=outputs.hidden_states,
            past_key_values=outputs.past_key_values,
            last_hidden_state=outputs.last_hidden_state,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        inputs_embeds=None,
        use_cache=None,
        cache_params: Optional[MambaCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        # Overwritten -- uses `cache_params` as opposed to `past_key_values`
        model_inputs = {"input_ids": input_ids.contiguous()}
        if use_cache and cache_params is None:
            # we initialize the `cache_position` to full size of `conv_states` at prefill stage
            # considering padding will be applied when input length is shorter, and truncation
            # will be applied when it is longer, so it will be equivalent to always have it match
            # the length of `cache_params.conv_states`, which is `config.conv_kernel`
            cache_position = torch.arange(0, self.hybrid.config.conv_kernel, device=input_ids.device)
            if inputs_embeds is not None:
                model_inputs = {"inputs_embeds": inputs_embeds}
                max_batch_size = inputs_embeds.size(0)
            else:
                max_batch_size = input_ids.size(0)
            cache_params = MambaCache(self.hybrid.config, max_batch_size, device=self.device, dtype=self.dtype)

        if use_cache and cache_position[0] > 0:
            model_inputs["input_ids"] = input_ids[:, -1].unsqueeze(-1).contiguous()
            attention_mask = None

        if not use_cache and inputs_embeds is not None:
            model_inputs = {"inputs_embeds": inputs_embeds}

        model_inputs.update(
            {
                "cache_params": cache_params,
                "use_cache": use_cache,
                "cache_position": cache_position,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    
    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past[:2])
                + layer_past[2:],
            )
        return reordered_past


