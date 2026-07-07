
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from .dream.modeling_dream import DreamBaseModel,DreamConfig,DreamModel,Cache
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import (
    BaseModelOutput,
    # MaskedLMOutput,
    ModelOutput
)
from torch.nn import CrossEntropyLoss

# from .llada.modeling_llada import LLaDAModel,LLaDAModelLM,LLaDAConfig,create_model_config_from_pretrained_config
from .llada.generate import generate as llada_generate
from llava.model.language_model.llada.log_likelyhood import get_log_likelihood as get_log_likelihood
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from dataclasses import dataclass
import os
ENFORCE_NUM_ITEMIN_BATCH = os.environ.get("ENFORCE_NUM_ITEMIN_BATCH", False)
@dataclass
class MaskedLMOutput(ModelOutput):
    """
    Base class for masked language models outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Masked language modeling (MLM) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    past_key_values: Optional[Cache] = None

class LlavaDreamConfig(DreamConfig):
    model_type = "llava_dream"
    # temperature: float = 0.0  # reset to 0.0, previously 0.9 for Vicuna
    # max_new_tokens: int = 1024
    # do_sample: bool = False
    # top_p: Optional[float] = None
    # rope_scaling: Optional[dict] = {}
    
    
class LlavaDreamModel(LlavaMetaModel,DreamBaseModel):
    config_class = LlavaDreamConfig
    dtype = torch.bfloat16 # hack

    def __init__(self, pretrained_config,init_params=None,vision_kwargs=None):
        # breakpoint()
        super(LlavaDreamModel, self).__init__(pretrained_config,vision_kwargs=vision_kwargs)
        # LLaDAModel.__init__(self, llada_config)
        #LlavaMetaModel.__init__(self, pretrained_config,vision_kwargs=vision_kwargs,skip_init=True)
        
    # @property
    # def embed_tokens(self):
    #     breakpoint()
    #     return self.model.model.embed_tokens

def forward_process(bsz,seq_len,device, eps=1e-3):
    b, l = bsz,seq_len
    t = torch.rand(b, device=device)
    # t = torch.sigmoid(t)
    p_mask = (1 - eps) * t + eps
    
    p_mask = p_mask[:, None]#.repeat(1, l)
    
    masked_indices = torch.rand((b, l), device=device)
    mask_cutoff =  torch.max(p_mask,masked_indices.min(-1,keepdim=True).values)
    masked_indices = masked_indices <= mask_cutoff
    # mask at least one token
    # 126336 is used for [MASK] token
    #noisy_batch = torch.where(masked_indices, 126336, input_ids)
    
    return masked_indices, p_mask

class LlavaDreamForMaskedDiffusion(DreamModel,LlavaMetaForCausalLM):
    
    config_class = LlavaDreamConfig
    supports_gradient_checkpointing = True
    
    def __init__(self, config: DreamConfig, model: Optional[DreamModel] = None, init_params: bool = False,vision_kwargs=None,**kwargs):
        DreamModel.__init__(self, config)

        # configure default generation settings
        config.model_type = "llava_dream"
        # config.rope_scaling = None

        if not model:
            self.model = LlavaDreamModel(config, init_params=init_params,vision_kwargs=vision_kwargs)
        else:
            self.model = model
        #self.model.set_activation_checkpointing('whole_layer')
        
        self.post_init() # TODO
        
    def get_model(self):
        return self.model
    
    def forward_dream(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **loss_kwargs,
    ) -> Union[Tuple, MaskedLMOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        attention_mask = None
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **loss_kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            past_key_values=outputs.past_key_values,
        )
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = None,
        cache_position=None,
        policy='uniform',
        policy_args=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        eos_id = 151643 # hack
        mask_id = 151666
        
        raw_inputs_ids = input_ids
        # if attention_mask is not None:
        #     attention_mask_raw = attention_mask.clone()
        # elif isinstance(attention_mask)
        # else:
        #     attention_mask_raw = None
        non_padding = ~(raw_inputs_ids==eos_id)
        attention_mask[raw_inputs_ids==eos_id] = True # no sequence attention mask per Sec B.1
        labels[raw_inputs_ids==eos_id] = eos_id # revert target
        # fix attention mask
        input_ids == input_ids
        # pad_len = torch.randint(0,pad_len_max,(1,)).item()
        # padding = torch.full((bsz,pad_len),eos_id,dtype=labels.dtype,device=labels.device) 
                
        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels,new_input_ids) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes,return_inputs=True)
        #prompt_lengths = 
        #breakpoint()
        # hack starts here
        # 1. Get the mask of trget tokens 
        # if we have labels, run forward process
        if labels is not None:
            labels_mask = ~(labels == -100) # targets mask
            noise_embeddings = self.get_model().get_input_embeddings()(torch.tensor([mask_id]).to(raw_inputs_ids))
            # noise_embeddings is 1, 4096
            bsz,seq_len = labels_mask.shape
            noise_embeddings = noise_embeddings.view(1,1,-1)#.repeat(bsz,seq_len,1)
            # t = torch.rand(b, device=input_ids.device)
            masked_indices, p_mask = forward_process(bsz,seq_len,raw_inputs_ids.device)
            # torch.where()
            final_masked_indices = masked_indices&labels_mask
            final_masked_indices_inv = (~masked_indices)&labels_mask
            # breakpoint()
            # breakpoint()
            # boardcast goingon here
            # final_masked_indices: B X L X 1
            # noise_embeddings: 1 X 1 X D
            # inputs_embeds:  B X L X D
            inputs_embeds_inv = torch.where(final_masked_indices_inv.view(bsz,seq_len,1),noise_embeddings,inputs_embeds)
            inputs_embeds = torch.where(final_masked_indices.view(bsz,seq_len,1),noise_embeddings,inputs_embeds)
            # inputs_embeds_inv = torch.where(final_masked_indices_inv.view(bsz,seq_len,1),noise_embeddings,inputs_embeds)
            # print(final_masked_indices.float().mean(-1).cpu())
            # new_input_ids
            # breakpoint()
            
            labels_inv = labels.clone()
            labels_inv[~final_masked_indices_inv] = -100
            labels[~final_masked_indices] = -100
            
            inputs_embeds = torch.cat([inputs_embeds,inputs_embeds_inv])
            labels =  torch.cat([labels,labels_inv])
            final_masked_indices = torch.cat([final_masked_indices,final_masked_indices_inv])
            seq_len = labels.shape[-1]
            print(seq_len)
            # CUFOFF=5000
            # if seq_len > CUFOFF:
            #     print(seq_len,labels.shape)
            #     labels = labels[:,:CUFOFF]
            #     inputs_embeds = inputs_embeds[:,:CUFOFF]
            #     attention_mask = attention_mask[:,:CUFOFF]
            #     if position_ids is not None:
            #         position_ids = position_ids[:,:CUFOFF]
            #     assert input_ids is None
            #     assert past_key_values is None
            # elif seq_len < CUFOFF:
            #     pass
                # raise ValueError("Out of Length")
                # pad_len_max = 128 #torch.randint(0, 128, (1,)).item()
                # if pad_len_max > 0:
                #     pad_len = torch.randint(0,pad_len_max,(1,)).item()
                #     padding = torch.full((bsz,pad_len),eos_id,dtype=labels.dtype,device=labels.device) 
                #     labels = torch.cat([labels,padding],dim=-1)
                #     new_input_ids  = torch.cat([new_input_ids,padding],dim=-1)
                #     padding = torch.full((bsz,pad_len,inputs_embeds.shape[-1]),0,dtype=inputs_embeds.dtype,device=inputs_embeds.device)
                #     inputs_embeds = torch.cat([inputs_embeds,padding],dim=-2)
                #     padding = torch.full((bsz,pad_len),1,dtype=attention_mask.dtype,device=attention_mask.device)
                #     attention_mask = torch.cat([attention_mask,padding],dim=-1)
                #     if position_ids is not None:
                #         padding = torch.full((bsz,padding),0,dtype=position_ids.dtype,device=position_ids.device)
                #         position_ids = torch.cat([position_ids,padding],dim=-1)
        if dpo_forward:
            raise NotImplementedError() 
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            return logits, labels

        else:
            # breakpoint()
            attention_mask = None
            num_items_in_batch = None
            if ENFORCE_NUM_ITEMIN_BATCH:
                num_items_in_batch = labels.ne(-100).sum()
                num_items_in_batch = torch.distributed.reduce(num_items_in_batch)
            output =  super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                num_items_in_batch=num_items_in_batch,
            )
            output['new_input_ids']=new_input_ids
            output['labels'] = labels
            output['final_masked_indices']=final_masked_indices
            output['p_mask'] = p_mask
            return output

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        max_new_tokens=512,
        steps=512,
        temperature=0.2,
        top_p=0.95,
        alg_temp=0.,
        alg="entropy",
        output_history=False,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        modalities = kwargs.pop("modalities", None) if "modalities" in kwargs and modalities is None else modalities
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
        else:
            # breakpoint()
            inputs_embeds = self.get_model().embed_tokens(inputs)

        #return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)
        #return llada_generate(self.get_model(),inputs_embeds=inputs_embeds,position_ids=position_ids,attention_mask=attention_mask,**kwargs)
        # breakpoint()
        return self.diffusion_generate(
            None,
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
            output_history=output_history,
            return_dict_in_generate=True,
            steps=steps,
            temperature=temperature,
            top_p=top_p,
            alg=alg,
            alg_temp=alg_temp,
            **kwargs
        )
    
    @torch.no_grad()
    def log_likelyhood_inference(
        self,
        inputs: Optional[torch.Tensor] = None,
        answer: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        mc_num=128,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        raise NotImplementedError("`log_likelyhood_inference` is not supported")
        modalities = kwargs.pop("modalities", None) if "modalities" in kwargs and modalities is None else modalities
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        max_seq_len = 5000
        #if inputs_embeds.shape[1] > max_seq_len:
        max_seq_len = max_seq_len[:,-max_seq_len:]
        answer = answer[:300]
        return get_log_likelihood(self.get_model(), None,inputs_embeds=inputs_embeds, answer=answer, mc_num=mc_num,**kwargs)
        #return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)
        return llada_generate(self.get_model(),inputs_embeds=inputs_embeds,position_ids=position_ids,attention_mask=attention_mask,**kwargs)
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


AutoConfig.register("llava_dream", LlavaDreamConfig)
AutoModelForCausalLM.register(LlavaDreamConfig, LlavaDreamForMaskedDiffusion)

    
    
            
    
