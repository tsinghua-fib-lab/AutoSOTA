import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteriaList
from .helper import StopOnTokens
#import deepspeed
from together import Together
from vllm import LLM, SamplingParams
from openai import AzureOpenAI
try:
    import google.genai as genai
    from google.genai import types
    _HAS_GENAI = True
except Exception:
    genai = None
    types = None
    _HAS_GENAI = False

from torch import LongTensor, FloatTensor

import logging
logger = logging.getLogger('RRAG-main')

import litellm
from litellm import batch_completion
from litellm.exceptions import RateLimitError
from openai import OpenAI
import os 
import json
import time
import joblib
from .prompt_template import *

# for question-answering
MAX_NEW_TOKENS = 50

# for biogen
# MAX_NEW_TOKENS = 2048
CONTEXT_MAX_TOKENS = {'Mistral-7B-Instruct-v0.2': 8192, 
                      'Llama-3.2-3B-Instruct': 8192, 
                      'Llama-3.2-1B-Instruct': 4096, 
                      'Llama-3.1-8B-Instruct': 8192, 
                      'Mixtral-8x7B-Instruct-v0.1': 32000,
                      'DeepSeek-R1-Distill-Qwen-7B': 8192,
                      'gpt-4o':8192,
                      'o1-mini':8192,
                      'vicuna-7b-v1.5': 4096}


def create_model(model_name, model_dir, use_open_model_api=True, **kwargs):
    if not use_open_model_api:
        if model_name == 'mistral7b':
            return VLLMModel('Mistral-7B-Instruct-v0.2',model_dir,MISTRAL_TMPL,**kwargs) 
        elif model_name == 'deepseek7b':
            return VLLMModel('DeepSeek-R1-Distill-Qwen-7B',model_dir,DEEPSEEK_TMPL,**kwargs)
        elif model_name == 'llama3b':
            return VLLMModel('Llama-3.2-3B-Instruct',model_dir,LLAMA_TMPL,**kwargs) 
            # return HFModel('Llama-3.2-3B-Instruct',model_dir,LLAMA_TMPL,**kwargs)
        elif model_name == 'llama1b':
            return VLLMModel('Llama-3.2-1B-Instruct',model_dir,LLAMA_TMPL,**kwargs) 
        elif model_name == 'llama8b':
            return VLLMModel('Llama-3.1-8B-Instruct',model_dir,LLAMA_TMPL,**kwargs) 
        elif model_name == 'vicuna7b':
            return VLLMModel('vicuna-7b-v1.5',model_dir,VICUNA_TMPL,**kwargs)  
        elif model_name == 'mixtral8x7b':
            return VLLMModel('Mixtral-8x7B-Instruct-v0.1',model_dir,MISTRAL_TMPL,**kwargs)
        elif model_name == 'o1-mini':
            return GPTModel('o1-mini', GPT_TMPL, **kwargs) 
        elif model_name == 'gpt-4o':
            return GPTModel('gpt-4o', GPT_TMPL, **kwargs) 
        elif model_name == 'gpt-4o-mini':
            return GPTModel('gpt-4o-mini', GPT_TMPL, **kwargs) 
        else:
            raise NotImplementedError
    else:
        if model_name == 'tai_mistral7b':
            return TogetherAIModel('mistralai/Mistral-7B-Instruct-v0.2',MISTRAL_TMPL,**kwargs)  
        elif model_name == 'tai_llama8b':
            return TogetherAIModel('meta-llama/Llama-3-8b-chat-hf',LLAMA_TMPL,**kwargs) 
        elif model_name == 'o1-mini':
            return GPTModel('o1-mini', GPT_TMPL, **kwargs) 
        elif model_name == 'gpt-4o':
            return GPTModel('gpt-4o', GPT_TMPL, **kwargs) 
        elif model_name == 'gpt-4o-mini':
            return GPTModel('gpt-4o-mini', GPT_TMPL, **kwargs) 
        # 加入deepseek
        elif model_name == 'deepseek-chat':
            return DeepseekModel('deepseek-chat', GPT_TMPL, **kwargs)
        elif model_name == 'deepseek-reasoner':
            return DeepseekModel('deepseek-reasoner', GPT_TMPL, **kwargs)
        elif model_name == 'grok-4-fast':
            return GrokModel('grok-4-fast', GPT_TMPL, **kwargs)
        else:
            raise NotImplementedError

# 所有模型继承的父类，支持cache
# cache：避免对同一 prompt 重复调用后端 LLM，从而节省时间、网络请求次数和 API 成本；
# 可持久化到磁盘以跨次运行复用结果。
class BaseModel:
    def __init__(self,cache_path=None):
        # setup the LLM response cache if cache_path is not None
        self.use_cache = cache_path is not None 
        self.cache_path = cache_path
        if cache_path is not None and os.path.exists(cache_path):
            self.cache = self.load_cache()
        else:
            self.cache = {}
        self.hash = lambda x: x # for now directly string as the hash key
        # cache is a dict {hash(s):LLM response for input s}
        # 
        # end of sentence str
        self.clean_str = []

        # input prompt template
        self.prompt_template = {}

        self.total_input_tokens = 0
        self.total_output_tokens = 0

    # 先尝试从 cache 命中，否则调用子类实现的 _query / _batch_query，并在命中后写回 cache
    def query(self, prompt):
        if self.use_cache: # use cache if cache hits
            result = self.query_from_cache(prompt)
            if len(result)>0:
                return result

        # otherwise, do normal LLM query
        result = self._query(prompt)

        # store the response to self.cache
        if self.use_cache:
            self.cache[self.hash(prompt)]=result
        return result

    def _query(self,prompt): # will be implemented in each subclass
        raise NotImplementedError

    def batch_query(self, prompt_list):
        if self.use_cache: # use cache if cache hits
            results = self.batch_query_from_cache(prompt_list)
            if len(results)>0:
                return results

        # otherwise, do normal LLM batch query

        results = self._batch_query(prompt_list)

        # store the response to self.cache
        if self.use_cache:
            for p,r in zip(prompt_list,results):
                self.cache[self.hash(p)]=r
        return results

    def _batch_query(self, prompt_list):  # will be implemented in each subclass
        raise NotImplementedError

    def query_from_cache(self,prompt):
        # return cached responses
        h = self.hash(prompt)
        if h in self.cache:
            return self.cache[h]
        else:
            return ''

    def batch_query_from_cache(self,prompt_list):
        # return cached responses
        h_list = [self.hash(prompt) for prompt in prompt_list]
        if all([h in self.cache for h in h_list]):
            return [self.cache[h] for h in h_list]
        else:
            return []


    def dump_cache(self): # dump cache to disk
        joblib.dump(self.cache,self.cache_path)
    
    def load_cache(self): # load cache from disk
        return joblib.load(self.cache_path)
        
    def _clean_response(self,response): # clean response based on self.clean_str
        for pattern in self.clean_str:
            idx = response.find(pattern)
            if idx!=-1:
                response = response[:idx]
        return response.strip()

    def query_biogen(self, prompt):
        raise NotImplementedError

    # 把 data_item（由 dataset_utils.process_data_item 输出）与 prompt_template 结合生成最终 prompt；
    # 支持多选、是否使用检索上下文、long_gen、decode、hint 等模式及 separate（对每个 passage 单独生成 prompt）。

    def wrap_prompt(self,data_item,as_multi_choice=True,hints=None,seperate=False):  
        # use data_item and generate the input prompt to the LLM

        # data_item should be the output of DataUtils.process_data_item()
        # as_multi_choice: if use it as a multiple-choice QA
        # hints: if we are using hints in the last step of the keyword aggregation
        # seperate: if True, return a list of prompts for differnet passages; otherwise, concatenate all passages and return a single prompt

        # get info
        question = data_item['question']
        topk_content = data_item['topk_content']
        choices = data_item.get('choices',[])
        use_retrieval = len(topk_content) > 0 # if we use retrieved passage

        def fill_template(template,question,context_str,choices,use_retrieval,as_multi_choice,hints):
            filling = {'query_str': question}
            if use_retrieval:
                filling.update({'context_str': context_str})
            if as_multi_choice:
                filling.update({
                        'A': choices[0],
                        'B': choices[1],
                        'C': choices[2],
                        'D': choices[3]                
                    })
            if hints is not None:
                filling.update({'hints':hints})
            return template.format(**filling)

        # get corresponding template
        mode = 'qa'
        if as_multi_choice: mode += '-mc'
        if "long_gen" in data_item: mode += '-long'
        if not use_retrieval: mode += '-zero'
        if "decode" in data_item: mode += '-decode' 
        if "genhint" in data_item: mode += '-genhint'
        if hints is not None: mode += '-hint' 
        template = self.prompt_template[mode]

        if seperate:
            return [fill_template(template,question,context_str,choices,use_retrieval,as_multi_choice,hints) for context_str in topk_content]
        else:
            context_str = '\n\n'.join(topk_content)
            return fill_template(template,question,context_str,choices,use_retrieval,as_multi_choice,hints)

    def get_token_count(self):
        return {
            "input": self.total_input_tokens,
            "output": self.total_output_tokens,
        }

    def reset_token_count(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0

class VLLMModel(BaseModel):
    def __init__(self, model_name, model_dir, prompt_template, cache_path=None, max_output_tokens=None, seed=42, **kwargs):
        super().__init__(cache_path)

        self.max_output_tokens = MAX_NEW_TOKENS if max_output_tokens is None else max_output_tokens 
        self.model_name = model_name
        self.prompt_template = prompt_template

        self.llm = LLM(
            model=model_dir + model_name,
            gpu_memory_utilization=0.5
        )
        self.sampling_params = SamplingParams(
            # max_tokens=2048,
            temperature=0
        )
        
        self.clean_str = ['\n\n']

    def _query(self, prompt):
        outputs = self.llm.generate(prompt, self.sampling_params)
        result = outputs[0].outputs[0].text
        self.total_input_tokens += outputs[0].prompt_token_ids.__len__()
        self.total_output_tokens += outputs[0].outputs[0].token_ids.__len__()
        result = self._clean_response(result)
        return result

    def _batch_query(self, prompt_list):
        results = []
        outputs = self.llm.generate(prompt_list, self.sampling_params)
        for output in outputs:
            self.total_input_tokens += len(output.prompt_token_ids)
            self.total_output_tokens += len(output.outputs[0].token_ids)
            result = self._clean_response(output.outputs[0].text)
            results.append(result)
        return results

class HFModel(BaseModel):
    def __init__(self, model_name, model_dir, prompt_template, cache_path=None, max_output_tokens=None, **kwargs):
        super().__init__(cache_path)
        # set max number of output tokens
        self.max_output_tokens = MAX_NEW_TOKENS if max_output_tokens is None else max_output_tokens 

        # set up tokenizer and model
        model_path = model_dir + model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_path) 
        if "CohereForAI" in model_name:
            #'CohereForCausalLM' object has no attribute 'torch_dtype'
            self.model = AutoModelForCausalLM.from_pretrained(model_path,**kwargs) 
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map='cuda',**kwargs)
        
        self.prompt_template = prompt_template
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model_name = model_name

        with open("configs/ds_config.json", 'r') as f:
            ds_config = json.load(f)

        self.generation_kwargs = {
            'max_new_tokens':self.max_output_tokens,
            'pad_token_id':self.tokenizer.eos_token_id,
            'do_sample':True
        }
        if 'Llama-3' in model_name:
            self.generation_kwargs['eos_token_id']=[self.tokenizer.eos_token_id,self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]

        self.clean_str = ['\n\n'] 


    def _query(self, prompt):
        # get text prompt as input and return text responses
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        token_length = inputs.input_ids.size(1)
        self.total_input_tokens += token_length
        # check if the num of token exceeds the max context window ..
        # this only happens when we use the bio generation data without any defense (i.e., we concatenate all long passages together)
        # we did not implement this cut off for _batch_query
        if token_length >= (CONTEXT_MAX_TOKENS[self.model_name]-self.max_output_tokens):
            prompt_length = len(prompt)
            ratio = (CONTEXT_MAX_TOKENS[self.model_name]-self.max_output_tokens-100)/token_length # this is a rough cut off, 100 is a buffer
            ## this is left cut 
            # prompt_cut = prompt[:int(prompt_length*ratio)] + "..." + "\n\n" + prompt[prompt.rfind("Query: Tell me a bio of"):] 
            ## this is right cut 
            prompt_cut = "Context information is below.\n" + "---------------------\n ..."+ prompt[-int(prompt_length*ratio):]
            inputs = self.tokenizer(prompt_cut, return_tensors="pt").to("cuda")
            logger.warning(f"Prompt length exceeds the limit, cut the prompt to {inputs.input_ids.size(1)} tokens")
        with torch.no_grad():
            outputs = self.model.generate(**inputs,**self.generation_kwargs)
        output_token_count = outputs[0].size(0) - inputs.input_ids.size(1)
        self.total_output_tokens += output_token_count
        outputs = outputs[0][len(inputs[0]):]
        result = self.tokenizer.decode(outputs, skip_special_tokens=True)
        result = self._clean_response(result)
        return result
        

    def _batch_query(self, prompt_list):
        try:
            inputs = self.tokenizer(
                prompt_list, return_tensors="pt", padding=True, truncation=True
            ).to("cuda")
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **self.generation_kwargs)
            self.total_output_tokens += outputs.numel() - inputs['input_ids'].numel()
            results = self.tokenizer.batch_decode(
                outputs[:, len(inputs[0]):], skip_special_tokens=True
            )
            results = [self._clean_response(x) for x in results]
        except Exception as e:
            print(e)
            results = []
            for prompt in prompt_list:
                results.append(self._query(prompt))
        return results


class GPTModel(BaseModel):
    def __init__(
        self,
        model_name,
        prompt_template,
        cache_path=None,
        max_output_tokens=None,
        api_key=None,
        base_url="https://xinghuapi.com/v1",
        **kwargs
    ):
        super().__init__(cache_path)
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.temperature = 0.0
        self.max_output_tokens = MAX_NEW_TOKENS if max_output_tokens is None else max_output_tokens

        # 优先使用显式传入的 api_key，否则从环境变量读取
        # 你可以设置环境变量：XINGHUAPI_API_KEY 或 OPENAI_API_KEY
        resolved_key = (
            api_key
            or os.getenv("XINGHUAPI_API_KEY", "")
            or os.getenv("OPENAI_API_KEY", "")
        )

        # 使用你要求的“聊天”方式，并改请求地址
        self.client = OpenAI(
            base_url=base_url,
            api_key=resolved_key,
        )

    def _query(self, prompt: str) -> str:
        fallback = "I don't know"

        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user",   "content": prompt},
            ]

            chat = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
            )

            # ---- defensive checks -------------------------------------------
            if chat is None or not getattr(chat, "choices", None):
                return fallback

            msg = chat.choices[0].message
            content = getattr(msg, "content", None)
            if not content:
                return fallback

            # ---- token bookkeeping (only if usage is available) ------------
            usage = getattr(chat, "usage", None)
            if usage is not None:
                self.total_input_tokens  += getattr(usage, "prompt_tokens", 0)
                self.total_output_tokens += getattr(usage, "completion_tokens", 0)

            return content

        except Exception as exc:
            print("Chat query failed:", exc)
            return fallback

    def _batch_query(self, prompt_list):
        return [self._query(p) for p in prompt_list]

    def query_biogen(self, prompt):
        return self.query(prompt)

class GrokModel(BaseModel):
    """
    与 GPTModel 对齐：统一使用 OpenAI-compatible chat.completions
    默认 base_url 指向 xinghuapi 的 /v1 兼容接口（如你有自己的 Grok OpenAI 兼容网关，可用环境变量覆盖）。
    """

    def __init__(
        self,
        model_name,
        prompt_template,
        cache_path=None,
        max_output_tokens=None,
        api_key=None,
        base_url="https://xinghuapi.com/v1",
        **kwargs
    ):
        super().__init__(cache_path)
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.temperature = 0.0
        self.max_output_tokens = MAX_NEW_TOKENS if max_output_tokens is None else max_output_tokens

        # 优先使用显式传入的 api_key，否则从环境变量读取
        # 你可以设置环境变量：XINGHUAPI_API_KEY / GROK_API_KEY / OPENAI_API_KEY
        resolved_key = (
            api_key
            or os.getenv("XINGHUAPI_API_KEY", "")
            or os.getenv("GROK_API_KEY", "")
            or os.getenv("OPENAI_API_KEY", "")
        )
        if not resolved_key:
            raise RuntimeError(
                "GrokModel: 未找到 API key。请设置 XINGHUAPI_API_KEY / GROK_API_KEY / OPENAI_API_KEY 之一。"
            )

        # 允许用环境变量覆盖 base_url（可选）
        base_url = os.getenv("OPENAI_BASE_URL", base_url)

        self.client = OpenAI(
            base_url=base_url,
            api_key=resolved_key,
        )

        self.clean_str = ["\n\n"]

    def _query(self, prompt: str) -> str:
        fallback = "I don't know"

        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user",   "content": prompt},
            ]

            chat = self.client.chat.completions.create(
                model=self.model_name,  # 例如 "grok-2" / "grok-2-mini"（取决于你的网关命名）
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
            )

            # ---- defensive checks -------------------------------------------
            if chat is None or not getattr(chat, "choices", None):
                return fallback

            msg = chat.choices[0].message
            content = getattr(msg, "content", None)
            if not content:
                return fallback

            # ---- token bookkeeping (only if usage is available) ------------
            usage = getattr(chat, "usage", None)
            if usage is not None:
                self.total_input_tokens  += getattr(usage, "prompt_tokens", 0)
                self.total_output_tokens += getattr(usage, "completion_tokens", 0)

            return self._clean_response(content)

        except Exception as exc:
            print("Chat query failed:", exc)
            return fallback

    def _batch_query(self, prompt_list):
        return [self._query(p) for p in prompt_list]

    def query_biogen(self, prompt):
        return self.query(prompt)




# 加入deepseek model类
class DeepseekModel(BaseModel):
    """
    与 GPTModel 功能完全一致的 DeepSeek 版本：
    - 相同的构造入参与属性：model_name, prompt_template, cache_path, max_output_tokens
    - 相同的方法与行为：query/_query, batch_query/_batch_query, query_biogen
    - 相同的温度、最大输出 token、失败回退、usage 记账与本地 cache 机制
    - 唯一差异：使用 DeepSeek 的 OpenAI 兼容接口（base_url + DEEPSEEK_API_KEY）
    """
    def __init__(self, model_name, prompt_template, cache_path=None, max_output_tokens=None, **kwargs):
        super().__init__(cache_path)
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.temperature = 0.0
        self.max_output_tokens = MAX_NEW_TOKENS if max_output_tokens is None else max_output_tokens

        # 使用 DeepSeek 的 OpenAI 兼容接口
        # 需要环境变量：DEEPSEEK_API_KEY
        # 注意：OpenAI 官方 Python SDK v1+ 支持 base_url 参数
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            base_url="https://api.deepseek.com",
        )

    def _query(self, prompt: str) -> str:
        fallback = "I don't know"

        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user",   "content": prompt},
            ]

            chat = self.client.chat.completions.create(
                model=self.model_name,            # 例如：'deepseek-chat' 或 'deepseek-reasoner'
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
            )

            # ---- 防御式检查 -----------------------------------------------
            if chat is None or not chat.choices:
                return fallback

            content = chat.choices[0].message.content
            if not content:                       # 覆盖 None 或空字符串
                return fallback

            # ---- token 记账（如果 usage 可用）----------------------------
            if getattr(chat, "usage", None):
                self.total_input_tokens  += getattr(chat.usage, "prompt_tokens", 0)
                self.total_output_tokens += getattr(chat.usage, "completion_tokens", 0)

            return content

        except Exception as exc:
            print("DeepSeek query failed:", exc)
            return fallback

    def _batch_query(self, prompt_list):
        # 与 GPTModel 保持一致：逐条调用 _query
        return [self._query(p) for p in prompt_list]

    def query_biogen(self, prompt):
        # 与 GPTModel 保持一致：直接复用 query
        return self.query(prompt)


class TogetherAIModel(BaseModel):
    def __init__(self, model_name, prompt_template, cache_path=None, max_output_tokens=None, **kwargs):
        super().__init__(cache_path)
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.temperature = 0.0
        self.max_output_tokens = MAX_NEW_TOKENS if max_output_tokens is None else max_output_tokens

        self.client = Together(api_key=os.getenv("TOGETHER_API_KEY", ""))

    def _query(self, prompt):
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_output_tokens
            )

            self.total_input_tokens += response.usage.prompt_tokens
            self.total_output_tokens += response.usage.completion_tokens
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in Together _query: {e}")
            return ""

    def _batch_query(self, prompt_list):
        # TODO: implement batch query if supported
        return [self._query(prompt) for prompt in prompt_list]
    
