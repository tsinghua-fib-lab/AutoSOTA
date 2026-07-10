import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.stopping_criteria import StoppingCriteriaList, LLamaQaStoppingCriteria
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import LlamaTokenizer, LlamaForCausalLM
import numpy as np


class LLM:
    def __init__(self, model_name, device, num_gpus, auth_token=None, max_memory=40, **kwargs):
        self.model_name = model_name
        self.device = device
        self.num_gpus = num_gpus
        self.stopping_criteria = None
        self.max_memory = max_memory
        self.model, self.tokenizer = self.load_model(model_name=model_name, max_memory=max_memory, auth_token=auth_token)

    def load_model(self, model_name, max_memory, auth_token=None):
        if 'gpt2' in model_name:
            tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            model = GPT2LMHeadModel.from_pretrained(model_name)
            model.cuda()
            return model, tokenizer
        
        if self.device == "cuda":
            kwargs = {"torch_dtype": torch.float16, "offload_folder": f"offload/{model_name}"}
            if self.num_gpus == "auto":
                kwargs["device_map"] = "auto"
            else:
                self.num_gpus = int(self.num_gpus)
                if self.num_gpus != 1:
                    kwargs.update({
                        "device_map": "auto",
                        "max_memory": {i: f"{max_memory}GiB" for i in range(self.num_gpus)},
                    })
        elif self.device == "cpu":
            kwargs = {}
        else:
            raise ValueError(f"Invalid device: {self.device}")
        
        if 'mistral' in model_name.lower():
            if auth_token is not None:
                tokenizer = LlamaTokenizer.from_pretrained(model_name, token=auth_token)
                model = LlamaForCausalLM.from_pretrained(model_name, token=auth_token, **kwargs)
            else:
                tokenizer = LlamaTokenizer.from_pretrained(model_name)
                model = LlamaForCausalLM.from_pretrained(model_name, **kwargs)
        else:
            if auth_token is not None:
                tokenizer = AutoTokenizer.from_pretrained(model_name, token=auth_token, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_name, token=auth_token, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

        if self.device == "cuda" and self.num_gpus == 1:
            model.cuda()
        
        return model, tokenizer

    def set_stop_words(self, stop_words):
        self.stop_words = stop_words
        self.stopping_criteria = StoppingCriteriaList()
        list_stop_word_ids = []
        for stop_word in self.stop_words:
            if 'llama' in self.model_name.lower() or 'mistral' in self.model_name.lower():
                stop_word_ids = self.tokenizer.encode('\n' + stop_word)[3:]
            else:
                stop_word_ids = self.tokenizer.encode('\n' + stop_word)
            list_stop_word_ids.append(stop_word_ids)
        self.stopping_criteria.append(LLamaQaStoppingCriteria(list_stop_word_ids))

    def generate(self, input_text, max_new_tokens=256, top_p=0.95, top_k=0, temperature=0.8, 
                 mode='vanilla', verbose=False, return_attentions=False, 
                 teacher_forcing_seq=None, **kwargs):
        with torch.no_grad():
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
            if verbose:
                print(f'MODEL INPUT LENGTH: {input_ids.shape[-1]}')
            max_len = input_ids.shape[-1] + max_new_tokens

            outputs = self.model.generate(
                inputs=input_ids, 
                max_length=max_len, 
                num_return_sequences=1,
                output_scores=True, 
                return_dict_in_generate=True, 
                top_p=top_p, 
                top_k=top_k, 
                temperature=temperature, 
                stopping_criteria=self.stopping_criteria, 
                output_attentions=return_attentions, 
                teacher_forcing_seq=teacher_forcing_seq, 
                **kwargs
            )
            
            sequences = outputs.sequences
            gen_sequences = sequences[:, input_ids.shape[-1]:][0, :]
            gen_arr = gen_sequences.cpu().numpy()
            output_str = self.tokenizer.decode(gen_sequences, skip_special_tokens=True)

            if verbose:
                print(f'MODEL OUTPUT: \n{output_str}')

        if self.device:
            torch.cuda.empty_cache()
        
        if not return_attentions:
            return output_str, gen_arr
        else:
            return output_str, outputs.attentions, gen_arr
