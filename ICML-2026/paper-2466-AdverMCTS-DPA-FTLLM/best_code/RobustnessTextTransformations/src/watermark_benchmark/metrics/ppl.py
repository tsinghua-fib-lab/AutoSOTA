import math
import os
from dataclasses import replace

from tqdm import tqdm

from .quality import RatingMetric

prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
You are given a prompt and a response, and you provide a grade out of 100 measuring the quality of the response, in terms of accuracy, level of details, and typographical, grammatical and lexical correctness. 
Remove points as soon as one of the criteria is missed. <|eot_id|> 
<|start_header_id|>user<|end_header_id|> 
Prompt: {}\nResponse: {}<|eot_id|> <|start_header_id|>assistant<|end_header_id|> Grade: """


class PPLRating(RatingMetric):
    def rate(self, generations, _):
        """
        Runs the model on the given generations and calculates the rating for each generation.

        Args:
            config (Config): Configuration object.
            generations (List[Generation]): List of generations to rate.
            writer_queue (Queue): Queue to write the rated generations to.
            device (int): Index of the GPU device to use.

        Returns:
            None
        """

        config = self.config
        writer_queue = self.writer_queue
        device = self.device

        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

        # Imports
        import torch

        from watermark_benchmark.servers import get_model
        from watermark_benchmark.utils import get_server_args, setup_randomness

        torch.set_num_threads(1)

        setup_randomness(config)

        # Setup server
        config.model = "meta-llama/Meta-Llama-3-8B-Instruct"
        config.max_new_tokens = 8
        config.dtype = "bfloat16"
        config.num_return_sequences = 1
        inference_engine = config.engine
        server = get_model(inference_engine, config, **get_server_args(config))
        tokenizer = server.tokenizer()
        tokenizer.pad_token = tokenizer.eos_token

        tasks = []
        for generation in generations:
            if "<<SYS>>" in generation.prompt:
                original_prompt = (
                    generation.prompt.split("<</SYS>>")[-1]
                    .replace("[INST]", "")
                    .replace("[/INST]", "")
                    .strip()
                )
                original_system_prompt = (
                    generation.prompt.split("<<SYS>>")[1]
                    .split("<</SYS>>")[0]
                    .strip()
                )
            else:
                original_prompt = (
                    generation.prompt.replace("[/INST]", "")
                    .replace("[INST]", "")
                    .replace("<<SYS>>", "")
                    .replace("<</SYS>>", "")
                    .replace(
                        "You are a helpful assistant. Always answer in the most accurate way.",
                        "",
                    )
                    .strip()
                )
                original_system_prompt = None

            original_response = generation.response

            if not original_system_prompt:
                full_prompt = f"""
<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{original_prompt} <|eot_id|> <|start_header_id|>assistant<|end_header_id|> 
{original_response} <|eot_id|>"""
            else:
                full_prompt = f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
{original_system_prompt} <|eot_id|> <|start_header_id|>user<|end_header_id|>
{original_prompt} <|eot_id|> <|start_header_id|>assistant<|end_header_id|> 
{original_response} <|eot_id|>"""

            tasks.append(full_prompt)

        # Clip sequences that are too long
        max_token_length = 8000
        for i in tqdm(range(len(tasks)), total=len(tasks), desc="Encoding"):
            task = tasks[i]
            if len(task) > max_token_length:
                encoded_task = tokenizer(task)["input_ids"]
                if len(encoded_task) > max_token_length:
                    print(
                        "Warning: Task too long ({} tokens), clipping to {} tokens".format(
                            len(encoded_task), max_token_length
                        )
                    )
                    task = tokenizer.decode(encoded_task[:max_token_length])
            tasks[i] = task

        # Run model
        outputs = server.run(
            tasks,
            config,
            1.0,
            use_tqdm=True,
            prompt_logprobs=1,
        )

        # Parse outputs
        end_header_token_str = "<|end_header_id|>"
        end_header_token = tokenizer.encode(end_header_token_str)[-1]
        for idx, gen in enumerate(outputs):
            input_task = tasks[idx]
            if (
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
                in input_task
            ):
                turn_index = 3
            else:
                turn_index = 2

            filtered_logprobs = [
                v[1]
                for v in get_items_after_nth_turn(
                    gen.prompt_logprobs, end_header_token, turn_index
                )[1:]
            ]

            ppl = math.exp(-sum(filtered_logprobs) / len(filtered_logprobs))
            generations[idx] = replace(generations[idx], rating=-ppl)

        # Write to file
        writer_queue.put(generations)


def get_items_after_nth_turn(array, value, n):
    occurrence_count = 0
    for index, pair in enumerate(array):
        number, _ = pair
        if number == value:
            occurrence_count += 1
            if occurrence_count == n:
                return array[index:]

    # If the nth occurrence is never found, return an empty list
    return []
