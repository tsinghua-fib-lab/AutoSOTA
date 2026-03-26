import os
import json
import random
import subprocess
import torch.multiprocessing as mp
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def get_free_gpus(threshold=71920):
    output = subprocess.check_output("nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits", shell=True)
    gpu_free_memory = [int(x) for x in output.decode("utf-8").strip().split("\n")]

    free_gpus = [i for i, mem in enumerate(gpu_free_memory) if mem > threshold]
    return free_gpus

def ask_llm_worker(gpu_ids, process_id, config, messages_chunk):
    # set GPU id
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

    # load to target GPU
    llm = LLM(
        model=config["model_path"],
        trust_remote_code=True,
        dtype=config.get("dtype", "half"),
        max_model_len=config.get("max_model_len", 8192),
        gpu_memory_utilization=config.get("gpu_memory_utilization", 0.85),
        tensor_parallel_size=config.get("tensor_parallel_size", 1),
        max_num_seqs=config.get("max_num_seqs", 256),
        seed=random.randint(0, 2**32 - 1),
    )
    tokenizer = llm.get_tokenizer()

    # set sampling parameters
    seed = random.randint(0, 2**32 - 1)
    sampling_params_config = config["sampling_params"]
    sampling_params = SamplingParams(
        n=sampling_params_config.get("n", 1),
        max_tokens=sampling_params_config.get("max_tokens", 512),
        top_p=sampling_params_config.get("top_p", 0.7),
        temperature=sampling_params_config.get("temperature", 0.8),
        stop=sampling_params_config.get("stop", None),
        seed=seed,
    )

    # generate prompt
    prompts = [tokenizer.apply_chat_template(messages, tokenize=False) for messages in messages_chunk]

    # generate output
    outputs_list = llm.generate(prompts, sampling_params)

    # ensure the output dir is exist
    output_dir = config.get("output_dir", "./output")
    os.makedirs(output_dir, exist_ok=True)

    # set the path of output
    output_file = os.path.join(output_dir, f"output_gpu_{process_id}.jsonl")

    # write file
    with open(output_file, 'w', encoding='utf-8') as f:
        for messages, prompt, outputs in zip(messages_chunk, prompts, outputs_list):
            prompt_responses = [output.text for output in outputs.outputs]
            result = {
                'messages': messages,
                'prompt': prompt,
                'responses': prompt_responses
            }
            f.write(json.dumps(result, ensure_ascii=False))
            f.write('\n')

def ask_llm_parallel(config):
    num_gpus = config.get("num_gpus", 1)
    free_gpus = config["free_gpus"]
    tp = config.get("tensor_parallel_size", 1)
    num_processes = num_gpus // tp
    
    # load messages
    messages_file = config.get("messages_file", "messages.jsonl")
    messages_list = []
    with open(messages_file, 'r', encoding='utf-8') as f:
        for line in f:
            messages = json.loads(line)["messages"]
            messages_list.append(messages)

    # divide the messages_list into equal part for each process
    chunk_size = (len(messages_list) + num_processes - 1) // num_processes
    messages_chunks = [messages_list[i*chunk_size:(i+1)*chunk_size] for i in range(num_processes)]

    # create process
    processes = []
    for i in range(num_processes):
        gpu_ids = ','.join(str(gpu_id) for gpu_id in free_gpus[i*tp:i*tp+tp])
        p = mp.Process(target=ask_llm_worker, args=(gpu_ids, i, config, messages_chunks[i]))
        processes.append(p)
        p.start()

    # wait for all process to finish
    for p in processes:
        p.join()

if __name__ == "__main__":
    import argparse

    # parse parameter
    parser = argparse.ArgumentParser(description="Parallel LLM Inference")
    parser.add_argument('--config', type=str, default='config.json', help='Path to the configuration file')
    args = parser.parse_args()

    # load config
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # find free gpu
    free_gpus = get_free_gpus(threshold=70000)
    assert len(free_gpus) >= config['num_gpus']
    config['free_gpus'] = free_gpus[:config['num_gpus']]

    # run multiprocess inference
    ask_llm_parallel(config)
