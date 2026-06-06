"""
This code is based on SpecBench
https://github.com/hemingkx/Spec-Bench
"""
import argparse
from fastchat.utils import str_to_torch_dtype
from evaluation.eval import run_eval, reorg_answer_file

from transformers import AutoModelForCausalLM, AutoTokenizer

from model.recycling.kv_cache import initialize_past_key_values
from model.recycling.modeling_llama_kv import LlamaForCausalLM
from model.recycling.tree_template_ import choose_tree_template

import torch


@torch.no_grad()
def recycling_forward(inputs, model, tokenizer, max_new_tokens, temperature=0.0, do_sample=False, output_id_topk=8, tree_version="2.2.2"):
    input_ids = inputs.input_ids.cuda()
    accept_length_list = []
    step = 0
    start_len = input_ids.size(1)
    cur_seq_len = start_len - 1
    verify_input_ids = input_ids[:, :-1]
    input_ids = input_ids[:, -1:]
    
    (
        past_key_values,
        past_key_values_data,
        current_length_data,
    ) = initialize_past_key_values(model)
        
    model(input_ids=verify_input_ids, past_key_values=past_key_values)
    
    tree_template = choose_tree_template(tree_version)
    pad_len = len(tree_template[-1]) + 1
    def prepare_data(template, input_ids):
        candi_attention_ids = torch.zeros([input_ids.size(0), len(template)+1, len(template)+1], dtype=torch.long)
        candi_positon_ids = torch.zeros([input_ids.size(0), len(template)+1], dtype=torch.long)

        candi_attention_ids[:, :, 0] = 1
        candi_positon_ids[:, 0] = 0
        search_path = []
        father_index = [-1]
        candi_index = [-1]
        deep_split = []
        
        for candi_i in range(len(template)):
            tree_deep = len(template[candi_i])
            tree_index = template[candi_i][-1]

            if tree_deep == 1:
                father_index.append(0)
            else:
                father_index.append(template.index(template[candi_i][:-1])+1)
            candi_index.append(tree_index)

            candi_positon_ids[:, candi_i+1] = tree_deep

            if tree_deep != candi_positon_ids[:, candi_i]:
                deep_split.append(candi_i+1)

            cur_path = [0]
            candi_attention_ids[:, candi_i+1, candi_i+1] = 1

            for candi_j in range(candi_i):
                if template[candi_j] == template[candi_i][:len(template[candi_j])]:
                    candi_attention_ids[:, candi_i+1, candi_j+1] = 1
                    cur_path.append(candi_j+1)
            cur_path.append(candi_i+1)

            for sub_i in range(1, len(cur_path)):
                if cur_path[:sub_i] in search_path:
                    search_path.remove(cur_path[:sub_i])
            search_path.append(cur_path)

        deep_split.append(len(template)+1)

        search_path = torch.tensor([path + [-1] * (pad_len-len(path)) for path in search_path], dtype=torch.long, device=torch.device('cuda:0'))
        father_index = torch.tensor(father_index, dtype=torch.long, device=torch.device('cuda:0'))
        candi_index = torch.tensor(candi_index, dtype=torch.long, device=torch.device('cuda:0'))
        
        return candi_attention_ids.cuda(), candi_positon_ids.cuda(), search_path, father_index, candi_index, deep_split

    def prepare_data_input_ids(cur_input_ids, adj_matrix, father_index, candi_index, deep_split):
        input_ids = torch.zeros_like(father_index, dtype=torch.long, device=cur_input_ids.device)
        input_ids[0] = cur_input_ids
        for layer in range(len(deep_split)-1):
            cur_father = input_ids[father_index[deep_split[layer]:deep_split[layer+1]]]
            cur_father_index = cur_father * output_id_topk + candi_index[deep_split[layer]:deep_split[layer+1]]
            input_ids[deep_split[layer]:deep_split[layer+1]] = adj_matrix.view(-1)[cur_father_index]
        return input_ids.unsqueeze(0)
    
    eos_token_id = torch.tensor(tokenizer.eos_token_id, dtype=torch.long, device=torch.device('cuda:0'))
    
    attention_mask, position_ids, search_path, father_index, candi_index, deep_split= prepare_data(tree_template, input_ids)

    for i in range(max_new_tokens):
        input_ids = prepare_data_input_ids(input_ids, adj_matrix, father_index, candi_index, deep_split)
        merge_attention_mask = torch.cat([torch.ones([input_ids.size(0), len(tree_template)+1, verify_input_ids.size(1)], dtype=torch.long, device=torch.device('cuda:0')), attention_mask], dim=-1)
        merge_positon_ids = position_ids + verify_input_ids.size(1)

        outputs = model(input_ids=input_ids, attention_mask=merge_attention_mask.unsqueeze(1), position_ids=merge_positon_ids, past_key_values=past_key_values)
        
        model_res = torch.argmax(outputs.logits, dim=-1)
        
        all_input_path = input_ids[0][search_path]
        all_input_path[search_path==-1] = -100
        all_output_path = model_res[0][search_path]
        reward = torch.cumprod(all_input_path[:, 1:].eq(all_output_path[:, :-1]), dim=-1).sum(dim=-1)
        best_reward = reward.max()
        
        accept_len = 1 + best_reward
        accept_length_list.append(accept_len.item())
        best_path_index = torch.argmax(reward, dim=-1).to(torch.long)
        index_path = search_path[best_path_index][:accept_len]
        best_path_input = torch.index_select(input_ids, index=index_path, dim=1)

        tgt = past_key_values_data[..., verify_input_ids.size(1)+index_path, :]
        dst = past_key_values_data[..., verify_input_ids.size(1) : verify_input_ids.size(1) + tgt.shape[-2], :]
        dst.copy_(tgt, non_blocking=True)
        
        current_length_data.fill_(verify_input_ids.size(1) + tgt.shape[-2])
                
        verify_input_ids = torch.cat([verify_input_ids, best_path_input], dim=-1)

        if (best_path_input == eos_token_id).any() or verify_input_ids.size(1) - start_len > max_new_tokens:
            break
        
        to_update = torch.topk(outputs.logits, k=output_id_topk, dim=-1)[1][0]
        adj_matrix[input_ids.squeeze(0)] = to_update
        
        input_ids = model_res[:, search_path[best_path_index][accept_len-1]].unsqueeze(-1).cuda()
        if (input_ids == eos_token_id).any():
            start_len -= 1
            break

        step += 1
    return verify_input_ids, verify_input_ids.size(1) - start_len, step, accept_length_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
    )
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end",
        type=int,
        help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="The temperature for medusa sampling.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float64", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU.",
    )
    parser.add_argument(
        "--output-id-topk",
        type=int,
        default=10,
        help="how many tokens from output_ids to be used to renew the tree.",
    )
    parser.add_argument(
        "--tree_version",
        type=str,
        default="2.3.3",
        help="the version of tree template.",
    )
    args = parser.parse_args()

    question_file = f"data/{args.bench_name}/question.jsonl"

    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    model = LlamaForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=str_to_torch_dtype(args.dtype),
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    
    adj_matrix = torch.zeros((model.vocab_size, args.output_id_topk), dtype=torch.long, device=model.device, requires_grad=False)
    adj_matrix_memory = adj_matrix.element_size() * adj_matrix.nelement()
    adj_matrix_memory_MB = adj_matrix_memory / (1024 ** 2)

    print(f'Adj_matrix size: ({model.vocab_size}, {args.output_id_topk}), memory: {adj_matrix_memory_MB:.2f} MB')
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if args.temperature > 0:
        do_sample = True
    else:
        do_sample = False

    run_eval(
        model=model,
        tokenizer=tokenizer,
        forward_func=recycling_forward,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_tokens=args.max_new_tokens,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        temperature=args.temperature,
        do_sample=do_sample,
        output_id_topk=args.output_id_topk,
        tree_version=args.tree_version,
    )

    reorg_answer_file(answer_file)