import os
import argparse
from vllm import LLM, SamplingParams
from query_utils import create_prompts_and_answers_zero_shot

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_nickname", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max_tokens", type=int, default=4096)
    args = parser.parse_args()

    # Initialize the LLM
    engine_kwargs = {
        "model": args.model_nickname,
        "dtype": "bfloat16",
        "gpu_memory_utilization": 0.95,
        "enable_prefix_caching": True,
        "trust_remote_code": True,
    }
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible:
        num_gpus = len(cuda_visible.split(","))
        engine_kwargs["tensor_parallel_size"] = num_gpus
    model = LLM(**engine_kwargs)

    # Sampling settings
    sampling_params = SamplingParams(
        n=10,
        temperature=0.6,
        max_tokens=args.max_tokens,
        seed=42
    )

    num_prompts = 3
    data = create_prompts_and_answers_zero_shot(args.dataset, num_prompts)
    prompts = data["prompts"]
    solutions = data["solutions"]

    output_dir = "../../data/query_smoke_test"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{args.dataset}_{args.model_nickname.split('/')[-1]}.txt")

    with open(out_path, "w") as out_f:
        for idx, (a, prompt) in enumerate(zip(solutions, prompts), start=1):
            outputs = model.generate(prompts=[prompt], sampling_params=sampling_params)
            responses = [out.text.strip() for req in outputs for out in req.outputs]

            out_f.write(f"=== Question {idx} ===\n{prompt}\n\n")
            out_f.write(f"=== Reference Answer {idx} ===\n{a}\n\n")
            for i, resp in enumerate(responses, start=1):
                out_f.write(f"--- Response {i} ---\n{resp}\n\n")