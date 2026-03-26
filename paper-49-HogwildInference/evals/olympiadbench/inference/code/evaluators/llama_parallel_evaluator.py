import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from evaluators.evaluator import Evaluator


class ParallelLlamaEvaluator(Evaluator):
    def __init__(self, model_name, cuda_device_id=0, k=-1):
        super(ParallelLlamaEvaluator, self).__init__(model_name, k)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
        self.device = f'cuda:{cuda_device_id}'
        self.model.generation_config = GenerationConfig.from_pretrained(model_name)

    def make_input(self, prompt, question_content):
        input = prompt + '\n' + question_content
        return input
    
    def make_parallel_chat_template(
        self, prompt: str, worker_outputs: dict, own_worker_id: int, input: str
    ):
        prompt_with_template = self.tokenizer.apply_chat_template(
            [dict(role='user', content=input + prompt)],
            add_generation_prompt=True, tokenize=False
        )
        print(input + prompt)
        assert prompt_with_template.endswith(SEP)

        formated_outputs = {
            k: f"{SPECIAL_TOKENS.worker_fmt.format(k)}\n{v}" if k == own_worker_id
            else f"{SPECIAL_TOKENS.worker_fmt.format(k)}\n{v} {SPECIAL_TOKENS.worker_end.format(k)}"
            for k, v in worker_outputs.items()
        }

        full_prompt = SEP.join((prompt_with_template.rstrip(SEP),
                    *(v for k, v in formated_outputs.items() if k != own_worker_id),
                    formated_outputs[own_worker_id]
                    ))

        return self.tokenizer(full_prompt, add_special_tokens=False, return_tensors='pt')
    
    def get_answer(self, input):
        # add calling to two workers

        worker_outputs: dict = {
            1: "", 
            2: ""
        }
        worker_output_ids = {k: self.tokenizer(v, add_special_tokens=False)['input_ids']
                            for k, v in  worker_outputs.items()}
        
        work_done = {1: False, 2: False}

        for i in range(2048):
            with torch.no_grad():
                for own_worker_id in worker_outputs.keys():
                    next_token_id: int = self.model(self.make_parallel_chat_template(
                        PROMPTS_TWO_WORKERS[own_worker_id], worker_outputs, own_worker_id, input)['input_ids'].to(self.device),
                    ).logits[:, -1, :].argmax(-1).item()
                    worker_output_ids[own_worker_id].append(next_token_id)
                    if next_token_id == 128009: # <|eot_id|>
                        work_done[own_worker_id] = True
                worker_outputs = {k: self.tokenizer.decode(v) for k, v in worker_output_ids.items()}
                    
                if work_done[1] and work_done[2]:
                    return concat_result(worker_outputs)
        return concat_result(worker_outputs)
        
    

def concat_result(worker_outputs):
    result = ""
    for k, v in worker_outputs.items():
        string = v.replace('\n', '\\n')
        result += string
    return result


SEP = '\n\n'

class SPECIAL_TOKENS:
    worker_fmt = "<worker {}>"
    worker_end = "</worker {}>"
    
    done = "<complete>"

PROMPTS_TWO_WORKERS = {
    1: f"""
You are a highly-paid expert LLM worker that solves tasks in the team with another worker. Your name is worker 1, your teammate name is worker 2. You and your teammate will be given a task that could be done in parallel. You and your teammate must solve different subtasks. 
Detailed Instructions:
- Use {SPECIAL_TOKENS.done} tag in your response when all of your subtask are solved.
- You must not write tags {SPECIAL_TOKENS.worker_fmt.format(1)}, {SPECIAL_TOKENS.worker_fmt.format(2)}, {SPECIAL_TOKENS.worker_end.format(1)}, {SPECIAL_TOKENS.worker_end.format(2)}.
- If you are worker 2, you only need to find the answer to your subtasks. You could see worker 1 reasoning between tags {SPECIAL_TOKENS.worker_fmt.format(1)} and {SPECIAL_TOKENS.worker_end.format(1)}
- If you are worker 1, you need to find the answer to your subtasks and to give the final answer to the whole task, using worker 2 response. The worker 2 current reasoning could be found after the prompt between tags {SPECIAL_TOKENS.worker_fmt.format(2)} and {SPECIAL_TOKENS.worker_end.format(2)}
If worker 2 has finished his subtasks, his reasoning ends with <done> tag: {SPECIAL_TOKENS.done}.
- Worker 2 must put tag {SPECIAL_TOKENS.done} when his subtasks are solved. Worker 2 must stop his reasoning after {SPECIAL_TOKENS.done}.
- Worker 1 must not think anything about worker 2 subtasks until worker 2 had put tag {SPECIAL_TOKENS.done}.
- Worker 2 must not think about worker 1 subtasks at any time.
- When worker 2 had put tag {SPECIAL_TOKENS.done} and worker 1 has solved all of his subtasks, worker 1 must give the final answer. Worker 1 must use worker 2 reasoning for subtask 2 instead of solving it by himself. After giving the final answer to the whole task, worker 1 must stop writing.
- Worker 2 must not write anything after tag {SPECIAL_TOKENS.done}.
- Worker 1 must come up with the shortest solution possible for his subtasks.
- Worker 2 must come up with the shortest solution possible for his subtasks.

IMPORTANT:
- You must solve another subtasks than worker 2.
- Worker 1 split subtasks between worker2 and worker1. Subtasks between workers should intersect as little as possible.
- Worker 1 must solve only subtasks that worker 1 has chosen for worker 1.

""".strip(),
    2: f"""
You are a highly-paid expert LLM worker that solves tasks in the team with another worker. Your name is worker 2, your teammate name is worker 1. You and your teammate will be given a task that could be done in parallel. You and your teammate must solve different subtasks. 
Detailed Instructions:
- Use {SPECIAL_TOKENS.done} tag in your response when all of your subtask are solved.
- You must not write tags {SPECIAL_TOKENS.worker_fmt.format(1)}, {SPECIAL_TOKENS.worker_fmt.format(2)}, {SPECIAL_TOKENS.worker_end.format(1)}, {SPECIAL_TOKENS.worker_end.format(2)}.
- If you are worker 2, you only need to find the answer to your subtasks. You could see worker 1 reasoning between tags {SPECIAL_TOKENS.worker_fmt.format(1)} and {SPECIAL_TOKENS.worker_end.format(1)}
- If you are worker 1, you need to find the answer to your subtasks and to give the final answer to the whole task, using worker 2 response. The worker 2 current reasoning could be found after the prompt between tags {SPECIAL_TOKENS.worker_fmt.format(2)} and {SPECIAL_TOKENS.worker_end.format(2)}
If worker 2 has finished his subtasks, his reasoning ends with <done> tag: {SPECIAL_TOKENS.done}.
- Worker 2 must put tag {SPECIAL_TOKENS.done} when his subtasks are solved. Worker 2 must stop his reasoning after {SPECIAL_TOKENS.done}.
- Worker 1 must not think anything about worker 2 subtasks until worker 2 had put tag {SPECIAL_TOKENS.done}.
- Worker 2 must not think about worker 1 subtasks at any time.
- When worker 2 had put tag {SPECIAL_TOKENS.done} and worker 1 has solved all of his subtasks, worker 1 must give the final answer. Worker 1 must use worker 2 reasoning for subtask 2 instead of solving it by himself. After giving the final answer to the whole task, worker 1 must stop writing.
- Worker 2 must not write anything after tag {SPECIAL_TOKENS.done}.
- Worker 1 must come up with the shortest solution possible for his subtasks.
- Worker 2 must come up with the shortest solution possible for his subtasks.

IMPORTANT:
- You must solve another subtasks than worker 1.
- Worker 1 split subtasks between worker2 and worker1. Subtasks between workers should intersect as little as possible.
- Worker 2 must solve only subtasks that worker 1 has chosen for worker 2.


""".strip()
}
