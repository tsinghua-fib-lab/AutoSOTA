import os
import random
import re
from dataclasses import replace

from ..utils import get_server_args, setup_randomness
from .quality import RatingMetric


class LLMCompareRating(RatingMetric):
    def rate(self, generations, baselines):
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

        writer_queue = self.writer_queue
        config = self.config
        device = self.device
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
        from vllm import LLM, SamplingParams

        setup_randomness(config)

        model = LLM(
            "meta-llama/Meta-Llama-3-8B-Instruct", **get_server_args(config)
        )

        tasks = []
        for generation in generations:
            if generation.watermark is not None:
                tasks.append(
                    generation.prompt.replace("[/INST]", "")
                    .replace("[INST]", "")
                    .replace("<<SYS>>", "")
                    .replace("<</SYS>>", "")
                    .strip()
                )

        inputs = [g.response for g in generations if g.watermark is not None]
        baselines = [
            baselines[(float(g.temp), g.id)]
            for g in generations
            if g.watermark is not None
        ]
        first = [
            random.random() < 0.5
            for g in generations
            if g.watermark is not None
        ]
        indices = [
            g_idx
            for g_idx, g in enumerate(generations)
            if g.watermark is not None
        ]
        prompts = []
        for i, task in enumerate(tasks):
            bl, assistant, local_first = baselines[i], inputs[i], first[i]
            if local_first:
                p = compare_prompt.format(task, assistant, bl)
            else:
                p = compare_prompt.format(task, bl, assistant)
            prompts.append(p)

        outputs = model.generate(
            prompts, SamplingParams(temperature=0, max_tokens=1024)
        )

        num_regex = re.compile(r"\[\[[ABC]\]\]")
        new_generations = []

        # Parse outputs
        for idx, gen in enumerate(outputs):
            local_first = first[idx]
            raw = 0.5
            matches = re.findall(num_regex, gen.outputs[0].text.strip())
            if matches and len(matches):
                if (
                    matches[0] == "[[A]]"
                    and local_first
                    or matches[0] == "[[B]]"
                    and not local_first
                ):
                    raw = 1
                elif (
                    matches[0] == "[[B]]"
                    and local_first
                    or matches[0] == "[[A]]"
                    and not local_first
                ):
                    raw = 0
                else:
                    raw = 0.5

            if idx >= len(generations):
                print(
                    "Warning: Received more outputs than generations ({} vs {})".format(
                        len(outputs), len(generations)
                    )
                )
                break

            new_generations.append(
                replace(generations[indices[idx]], rating=raw)
            )

        for i, g in enumerate(generations):
            if g.watermark is None:
                new_generations.append(replace(g, rating=0.5))

        # Write to file
        writer_queue.put(new_generations)


compare_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
Please act as an impartial judge and evaluate the quality of the responses provided by two
AI assistants to the user question displayed below. You should choose the assistant that
follows the user's instructions and answers the user's question better. Your evaluation
should consider factors such as the helpfulness, relevance, accuracy, depth, creativity,
and level of detail of their responses. Also account for typographical correctness, grammatical correctness and lexical correctness.
Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the
order in which the responses were presented does not influence your decision. Do not allow
the length of the responses to influence your evaluation. Do not favor certain names of
the assistants. Be as objective as possible. After providing your explanation, you must 
output your final verdict by strictly following this format: 
* "[[A]]" if assistant A is better, 
* "[[B]]" if assistant B is better, and 
* "[[C]]" for a tie. 
For example, "Verdict: [[C]]".

Prompt: 
{} <|eot_id|> <|start_header_id|>user<|end_header_id|> 

[[Start of Assistant A]]
{}
[[End of Assistant A's Answer]]

[[Start of Assistant B]]
{}
[[End of Assistant B's Answer]]<|eot_id|> <|start_header_id|>assistant<|end_header_id|>
"""
