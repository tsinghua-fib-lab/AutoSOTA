from sympy import *
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, PreTrainedTokenizer, PreTrainedModel
import torch
import json
from tqdm import tqdm
from mcts_math.agents.utils import math_is_equiv
import argparse
from utils import get_model_name,get_dataset_name,ag_step,get_sc_final_answer,verify_solution,add_tokenizer,get_prompt,extract_answer
from generate_train_data import generate_numi_steps_test,eval_answer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='')
    parser.add_argument("--save_dir", type=str, default='')
    parser.add_argument("--generate_model_dir", type=str, default='')
    parser.add_argument("--verify_model_dir", type=str, default='')
    parser.add_argument("--tokenizer_dir", type=str, default='')
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--test_n", type=int, default=64)
    parser.add_argument("--num_labels", type=int, default=2)
    parser.add_argument("--generate", action= "store_true")
    parser.add_argument("--sc", action= "store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--use_history", action= "store_true")
    return parser.parse_args()

# def eval_answer(pred_text,ground_truth):
#     import signal
#     timeout_seconds = 10*60
#     class TimeoutError(Exception):
#         pass
#     def handler(signum, frame):
#         raise TimeoutError()
#     signal.signal(signal.SIGALRM, handler)
#     signal.alarm(timeout_seconds)
#     try:
#         pred=extract_answer(pred_text)
#         if pred=="bad_answer":
#             return False
#         result = math_is_equiv(pred,ground_truth)
#     except TimeoutError as exc:
#         result = False
#     finally:
#         signal.alarm(0)
#         signal.signal(signal.SIGALRM, signal.SIG_DFL)
#     return result

class bon_search:

    def __init__(self, args)-> None:
        self.data_dir=args.data_dir
        self.save_dir=args.save_dir
        self.generate_model_dir=args.generate_model_dir
        self.verify_model_dir=args.verify_model_dir
        self.tokenizer_dir=args.tokenizer_dir
        self.temperature=args.temperature
        self.top_p=args.top_p
        self.test_n=args.test_n
        self.num_labels=args.num_labels
        self.batch_size=args.batch_size
        self.start=args.start
        self.use_history=args.use_history
        self.verifier_dir=args.verify_model_dir

    def generate_solutions(self):
        model_dir=self.generate_model_dir
        generate_llm = LLM(
            model = model_dir,
            tensor_parallel_size=1,
            trust_remote_code=True,
            gpu_memory_utilization=0.9,
            swap_space=4,
            seed=69,
            max_seq_len_to_capture=32768
        )
        generate_tokenizer = AutoTokenizer.from_pretrained(model_dir)
        ori_data=[]
        generate_model_name=get_model_name(model_dir)
        data_name=get_dataset_name(self.data_dir)
        with open(self.data_dir, "r") as f:
            for line in f:
                ori_data.append(json.loads(line))
        data = [ag_step(question=item["problem"], ans=item["answer"]) for item in ori_data]
        history=[]
        for indx, item in tqdm(enumerate(data),total=len(data),desc="Mian process"):
            if indx<self.start:
                continue
            solutions_node=[]
            total_tokens={"input":0,"output":0}
            ori_prompt=get_prompt(item,history,generate_model_name) if self.use_history and history else None
            solutions_node += generate_numi_steps_test(item, generate_tokenizer, generate_llm, self.test_n, -10,self.temperature,self.top_p,ori_prompt=ori_prompt,max_tokens=7000,gemma="gemma" in generate_model_name)
            total_tokens["input"]=solutions_node[0].num_step_tokens[0]
            save_data={"question":item.question,"ground_truth":item.ans,"level":ori_data[indx]["level"],"solutions":[]}
            for node in solutions_node:
                save_data["solutions"].append(node.cur_step)
                if self.use_history:
                    if eval_answer(node.cur_step,node.ans):
                        history.append({"instruction":f"Instruction:{node.question}\nResponse: Let’s think step by step.\n","response":node.cur_step})
                total_tokens["output"]+=node.num_step_tokens[1]
                save_data["used_tokens"]=total_tokens
            with open(self.save_dir,"a") as file:
                file.write(json.dumps(save_data) + '\n')


    def major_vote(self):
        total_q,sc_acc_list,=0,[0,0,0,0,0,0,0]
        with open(self.data_dir,"r") as f:
            for indx,line in tqdm(enumerate(f),total=500,desc="Main process"):
                data=json.loads(line)
                answer_count={"bad_answer":0}
                for d in range(1,8):
                    solutions=data["solutions"][2**(d-1):2**d]
                    solution_nodes=[ag_step(question=data["question"],ans=data["ground_truth"],cur_step=solution) for solution in solutions]
                    answer_count=get_sc_final_answer(solution_nodes,answer_count,True)
                    prediction=max(answer_count,key=lambda x:answer_count[x])
                    sc_acc_list[d-1]+=math_is_equiv(prediction,data["ground_truth"]) if prediction!="bad_answer" else 0
                total_q+=1
            self.save_result(self.data_dir,"",total_q,[],[],sc_acc_list)
            with open(self.save_dir,"a",encoding='utf-8') as f:
                f.write(f"generator_data:{self.data_dir},{total_q}个问题\n" )
                for i in range(len(sc_acc_list)):
                    f.write(f"major_vote_acc_pass{2**(i+1)}:{sc_acc_list[i]/total_q}\n")
                f.write("\n")


    def save_result(self,solution_data_dir,verifier_dir,total_q,min_value_acc_list,final_step_value_acc_list,sc_acc_list):
        for name in ["prm800k","shepherd","perplexity","openr"]:
            if name in verifier_dir:
                if verifier_dir!="":
                    with open(f"bon_search_reasult_{name}.txt","a",encoding='utf-8') as f:
                        f.write(f"generator_data:{solution_data_dir},{total_q}个问题\n" )
                        for i in range(len(min_value_acc_list)):
                            f.write(f"{verifier_dir} min_value_acc_pass{2**(i+1)}:{min_value_acc_list[i]/total_q}\n{verifier_dir} final_step_value_acc_pass{2**(i+1)}:{final_step_value_acc_list[i]/total_q}\n")
                        f.write("\n")
                else:
                    with open(f"bon_search_reasult_{name}.txt","a",encoding='utf-8') as f:
                        f.write(f"generator_data:{solution_data_dir},{total_q}个问题\n" )
                        for i in range(len(sc_acc_list)):
                            f.write(f"major_vote_acc_pass{2**(i+1)}:{sc_acc_list[i]/total_q}\n")
                        f.write("\n")
                break
            

    def select_solution(self):
        print(f"================={self.data_dir}=============={self.verifier_dir}=======================")
        verify_llm = AutoModelForSequenceClassification.from_pretrained(self.verifier_dir, num_labels=self.num_labels, torch_dtype=torch.bfloat16,device_map="auto").eval().to("cuda")
        verify_tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_dir)
        print(verify_tokenizer.model_max_length)
        add_tokenizer(verify_llm, verify_tokenizer)
        total_q,min_value_acc_list,final_step_value_acc_list=0,[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]
        with open(self.data_dir,"r") as f:
            for indx,line in tqdm(enumerate(f),total=500,desc="Main process"):
                if indx<self.start:
                    continue
                data=json.loads(line)
                solution_nodes=[ag_step(question=data["question"],ans=data["ground_truth"],cur_step=solution) for solution in data["solutions"]]
                solution_nodes=verify_solution(solution_nodes,verify_llm, verify_tokenizer,self.num_labels,True,batch_size=self.batch_size)
                for d in range(1,8):
                    nodes=solution_nodes[:2**d]
                    nodes.sort(key=lambda x: x.final_value, reverse=True)
                    select_node_minvalue=nodes[0]
                    nodes.sort(key=lambda x: x.step_value[-1], reverse=True)
                    select_node_finalstepvalue=nodes[0]
                    min_value_acc_list[d-1]+=eval_answer(select_node_minvalue.cur_step,data["ground_truth"])
                    final_step_value_acc_list[d-1]+=eval_answer(select_node_finalstepvalue.cur_step,data["ground_truth"])
                total_q+=1
        self.save_result(self.data_dir,self.verifier_dir,total_q,min_value_acc_list,final_step_value_acc_list,[])


if __name__=="__main__":
    args = parse_args()
    search=bon_search(args)
    if args.generate:
        search.generate_solutions()
    elif args.sc:
        search.major_vote()
    else:
        search.select_solution()