import torch.nn.functional as F
from sympy import *
from pydantic import BaseModel
import random
from typing import Optional, Dict, Any, List, Type
from transformers import PreTrainedTokenizer, PreTrainedModel
import torch
from tqdm import tqdm
import math


class ag_step(BaseModel):
    question: str = ""
    cur_step: str = ""
    action: str = ""
    action_input: str = ""
    previous_steps: str = ""
    value: Optional[float] = -100
    ans: str = ""
    num_step_tokens:list[int]=[]
    step_prob:float=0
    q_value:Optional[float] = -100
    final_value:Optional[float] = -100
    step_value: list[dict] = []
    whole_generate: str=""
    ag_prompt:str=""
    cur_step_tokens:tuple=()
    correct:Optional[bool] = False


def get_prompt(node,history,generate_model_name):
    prompt=f"Instruction:{node.question}\nResponse: Let’s think step by step.\n{node.previous_steps}"
    example=random.sample(history,1)
    if "gemma" in generate_model_name:
        generate_prompt = [{"role":"user","content":"You are a powerful agent with broad math knowledge and good at accurate calculation on math equations.Below is an instruction that describes a task. Continue to finish the response that appropriately completes the request Within a maximum of 40 steps. When outputting each step, mark the sequence number of each step at the beginning, and explicitly state the final answer after the final step following the format 'The final answer is:'.After outputting the final answer only once, be sure to stop outputting."},
                        {"role":"assistant","content":"OK, I understand."},
                        {"role":"user","content":example[0]["instruction"]},
                        {"role":"assistant","content":example[0]["response"]},
                        {"role":"user","content":"Instruction: A square has sides of length 10, and a circle centered at one of its vertices has radius 10.  What is the area of the union of the regions enclosed by the square and the circle? Express your answer in terms of $\\pi$.\nResponse: Let’s think step by step.\nStep 1:I want to find the area of the shaded region in this picture, where the blue is the square and the red is the circle.\nStep 2:I notice that the circle and the square share a quarter of the circle's area, which is $\\frac{{1}}{{4}}\\pi r^2$, where $r = 10$.\n"},
                        {"role":"assistant","content":"Step 3:So I can subtract that from the sum of the areas of the circle and the square to get the area of the union.\nStep 4:The area of the circle is $\\pi r^2 = 100\\pi$, and the area of the square is $s^2 = 100$, where $s = 10$.\nStep 5:So the area of the union is $100\\pi + 100 - \\frac{{1}}{{4}}100\\pi = 100 + \\frac{{3}}{{4}}100\\pi$.\nStep 6: The final answer is: 100 + \\frac{{3}}{{4}}100\\pi.\n"},
                        {"role":"user","content":prompt}]
    else:
        generate_prompt = [{"role":"system","content":"You are a powerful agent with broad math knowledge and good at accurate calculation on math equations.Below is an instruction that describes a task. Continue to finish the response that appropriately completes the request Within a maximum of 40 steps. When outputting each step, mark the sequence number of each step at the beginning, and explicitly state the final answer after the final step following the format 'The final answer is:'.After outputting the final answer only once, be sure to stop outputting."},
                        {"role":"user","content":example[0]["instruction"]},
                        {"role":"assistant","content":example[0]["response"]},
                        {"role":"user","content":"Instruction: A square has sides of length 10, and a circle centered at one of its vertices has radius 10.  What is the area of the union of the regions enclosed by the square and the circle? Express your answer in terms of $\\pi$.\nResponse: Let’s think step by step.\nStep 1:I want to find the area of the shaded region in this picture, where the blue is the square and the red is the circle.\nStep 2:I notice that the circle and the square share a quarter of the circle's area, which is $\\frac{{1}}{{4}}\\pi r^2$, where $r = 10$.\n"},
                        {"role":"assistant","content":"Step 3:So I can subtract that from the sum of the areas of the circle and the square to get the area of the union.\nStep 4:The area of the circle is $\\pi r^2 = 100\\pi$, and the area of the square is $s^2 = 100$, where $s = 10$.\nStep 5:So the area of the union is $100\\pi + 100 - \\frac{{1}}{{4}}100\\pi = 100 + \\frac{{3}}{{4}}100\\pi$.\nStep 6: The final answer is: 100 + \\frac{{3}}{{4}}100\\pi.\n"},
                        {"role":"user","content":prompt}]
    return generate_prompt


def remove_box(text):
    if "boxed" in text:
        start=text.find("boxed")+6
        end=start
        for i in range(1,len(text)-start):
            if text[len(text)-i] =='}':
                end=len(text)-i
                break
        return text[start:end]
    else:
        return "bad solution"

def remove_unit(text):
    unit=["calories","cm","customers","degrees","nickels","daps","gallons","^\circ","inches","seconds","cents","meters","beakers","inches/second","\u00b0","/second","cubic","feet"]
    for item in unit:
        if item in text:
            return text.replace(item,"").strip()
    return text

def extract_answer(text):
    split_ans = text.split('The final answer is')
    if len(split_ans) == 1 or split_ans[-1]=="":
        split_ans=text.split('the final answer is')
        if len(split_ans) == 1 or split_ans[-1]=="":
            if "boxed" not in text:
                return "bad_answer"
            else:
                return remove_box(text)
    if split_ans[-1][0] == ":":
        extract_ans = split_ans[-1][1:].strip()
    else:
        extract_ans = split_ans[-1].strip()
    extract_ans=extract_ans.replace("<|eot_id|>","")
    extract_ans=extract_ans.replace("<|start_header_id|>","")
    extract_ans=extract_ans.split('I hope it is correct')[-1]
    if extract_ans=="":
        return "bad_answer"
    extract_ans = remove_box(extract_ans)
    extract_ans=remove_unit(extract_ans)
    extract_ans = extract_ans.split('.\n')[0]
    if len(extract_ans) > 0 and extract_ans[-1] == '.':
        extract_ans = extract_ans[0:-1]
    extract_ans=extract_ans[1:] if extract_ans[0]=="$" and len(extract_ans)>1 else extract_ans
    extract_ans=extract_ans[:-1] if extract_ans[-1]=="$" and len(extract_ans)>1 else extract_ans
    if extract_ans[:2]=="\\(":
        extract_ans=extract_ans[2:]
    if extract_ans[-2:]=="\\)": 
        extract_ans=extract_ans[:-2]
    extract_ans=extract_ans if len(extract_ans)<200 else "bad_answer"
    return extract_ans


def get_sc_final_answer(solutions_node,answer_dict,part):
    for i,node in enumerate(solutions_node):
        answer=extract_answer(node.solution) if not part else extract_answer(node.cur_step)
        if answer in answer_dict:
            answer_dict[answer]+=1
        else:
            answer_dict[answer]=1
    return answer_dict


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: PreTrainedTokenizer,
        model: PreTrainedModel,
):
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg


def add_tokenizer(model, tokenizer):
    IGNORE_INDEX = -100
    DEFAULT_PAD_TOKEN = "[PAD]"
    DEFAULT_EOS_TOKEN = "</s>"
    DEFAULT_BOS_TOKEN = "<s>"
    DEFAULT_UNK_TOKEN = "<unk>"
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.unk_token_id = tokenizer.unk_token_id


def get_model_name(model_dir):
    model_name_list=["llama-3","llama3.1","qwen2","deepseek","llemma","phi3.5","phi-3","Qwen2.5","gemma-2-2b","gemma-2-9b","Ministral-8B-Instruct","Mistral-7B-Instruct-v0.3"]
    model_name = next((name for name in model_name_list if name in model_dir), None)
    return model_name


def get_dataset_name(data_dir):
    data_name_list=["test500","gaokao23","AIME","olympiadbench","minerva"]
    data_name = next((name for name in data_name_list if name in data_dir), None)
    return data_name


def tokenize_verify_input(verify_prompt, next_step, verify_tokenizer):
    input_ids_list = []
    for i in range(len(verify_prompt)):
        tokenized_next_step = verify_tokenizer(next_step[i], return_tensors="pt", max_length=8000, truncation=True).input_ids[0]
        tokenized_state = verify_tokenizer(verify_prompt[i], padding=False, max_length=verify_tokenizer.model_max_length - len(tokenized_next_step), truncation=True, return_tensors="pt").input_ids[0]
        input_ids_list.append(torch.cat((tokenized_state, tokenized_next_step), dim=0))
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=verify_tokenizer.pad_token_id).to("cuda")
    input_ids = dict(
        input_ids=input_ids,
        attention_mask=input_ids.ne(verify_tokenizer.pad_token_id),
    )
    return input_ids


def verify_solution(nodes,verify_llm, verify_tokenizer,num_labels,part,batch_size=1):
    select_steps_list=[]
    question_list=[]
    step_length_list=[]
    next_steps=[]
    for node in nodes:
        solution=node.cur_step
        if extract_answer(solution)=="" or extract_answer(solution)=="bad_answer":
            node.final_value=-1
            node.step_value=[-1.0]
            step_length_list.append(0)
            continue
        severed_solution=solution.split("Step")
        select_steps="\n" if not part else node.previous_steps
        severed_solution=severed_solution[1:] if severed_solution[0]=="" else severed_solution
        select_steps_list.append(select_steps)
        step_length_list.append(len(severed_solution))
        for i in range(len(severed_solution)):
            question_list.append(node.question)
            next_steps.append(f"Step"+severed_solution[i])
            select_steps+=next_steps[-1]
            if i!=len(severed_solution)-1:
                select_steps_list.append(select_steps)
    value_list=[]
    for indx in range(math.ceil(len(select_steps_list)/batch_size)):
        value_list+=verify(question_list[indx*batch_size:(indx+1)*batch_size], verify_llm, verify_tokenizer, select_steps_list[indx*batch_size:(indx+1)*batch_size], next_steps[indx*batch_size:(indx+1)*batch_size],num_labels)
    step_sum=0
    for i in range(len(nodes)):
        if step_length_list[i]!=0:
            values=value_list[step_sum:step_sum+step_length_list[i]]
            nodes[i].step_value=values
            nodes[i].final_value=min(values) if num_labels!=1 else max(values)
            step_sum+=step_length_list[i]
    return nodes



def verify(question, verify_llm, verify_tokenizer, select_steps_list, next_steps,num_labels):
    prompt = "### Instruct:\nThe steps in 'Selected steps' are the correct problem-solving steps of the problem, while the steps in 'Next step:' are the next problem-solving steps generated by an AI agent based on the steps in 'Selected steps:'.You need to rate the step in 'Next step:' based on it`s usefulness and correctness.\n\n"
    next_step_list = ["Next step:" + next_step for next_step in next_steps]
    for item in next_step_list:
        if "I can’t help with this question." in item:
            return [0.0]
    verify_prompt = [prompt + "Problem:" + question[i] + "\n" +"Select steps:"+select_steps_list[i] for i in range(len(select_steps_list))]
    input_ids = tokenize_verify_input(verify_prompt, next_step_list, verify_tokenizer)
    with torch.no_grad():
        output = verify_llm(**input_ids)
    pred = F.softmax(output.logits, dim=-1)
    return [float(pred[i][0]) for i in range(len(pred))]