import io
import logging
import os
import random
import time
from contextlib import redirect_stdout
from dataclasses import replace

import mauve
import numpy as np
from tqdm import tqdm

from .quality import RatingMetric

# Suppress logging for the mauve module

mauve_logger = logging.getLogger("mauve")
mauve_logger.setLevel(logging.ERROR)

faiss_logger = logging.getLogger("faiss")
faiss_logger.setLevel(logging.ERROR)


class MAUVERating(RatingMetric):
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
        import torch
        from vllm import LLM

        vllm_logger = logging.getLogger("vllm")
        vllm_logger.setLevel(logging.ERROR)

        specs = {
            (str(g.watermark.to_dict(True, True)), g.attack): []
            for g in generations
            if g.watermark is not None
        }

        for i, g in enumerate(generations):
            if g.watermark is not None:
                specs[(str(g.watermark.to_dict(True, True)), g.attack)].append(
                    (i, g)
                )

        embedding_model = LLM(
            model="gpt2-large", enforce_eager=True, gpu_memory_utilization=0.24
        )
        tokenizer = embedding_model.get_tokenizer()

        baseline_generations = [(k, v) for k, v in baselines.items()]
        baseline_tasks = [
            tokenizer(b[1])["input_ids"][:1000] for b in baseline_generations
        ]
        vllm_outputs = embedding_model.encode(prompt_token_ids=baseline_tasks)
        baseline_encodings = {}
        for i, (k, _) in enumerate(baseline_generations):
            baseline_encodings[k] = vllm_outputs[i].outputs.embedding

        new_generations = []
        for key, tasks in tqdm(
            list(specs.items()),
            desc="Computing MAUVE scores",
            total=len(specs),
        ):
            p_tokens = np.array(
                [baseline_encodings[(float(g.temp), g.id)] for _, g in tasks]
            )

            local_tasks = [
                tokenizer(g.response)["input_ids"][:1000] for _, g in tasks
            ]
            original_ids = [v for v, _ in tasks]

            try:
                vllm_outputs = embedding_model.encode(
                    prompt_token_ids=local_tasks, use_tqdm=False
                )
            except Exception as e:
                print(f"Error in encoding: {e}")
                continue

            q_tokens = np.array([v.outputs.embedding for v in vllm_outputs])

            seed = (
                config.seed
                if config.seed is not None
                else int(time.time() * 1000)
            ) % 100000

            # Redirect stderr at the file descriptor level
            stderr_fd = 2  # Standard error file descriptor is 2
            devnull_fd = os.open(os.devnull, os.O_RDWR)  # Open os.devnull
            saved_stderr_fd = os.dup(stderr_fd)  # Save the current stderr fd
            try:
                os.dup2(devnull_fd, stderr_fd)  # Replace stderr with devnull
                f_stdout = io.StringIO()
                with redirect_stdout(f_stdout):
                    mauve_score = mauve.compute_mauve(
                        p_features=p_tokens,
                        q_features=q_tokens,
                        device_id=0,
                        seed=seed,
                        verbose=False,
                    ).mauve
            finally:
                os.dup2(
                    saved_stderr_fd, stderr_fd
                )  # Restore the original stderr
                os.close(saved_stderr_fd)  # Close the saved fd
                os.close(devnull_fd)  # Close the devnull fd

            for i in original_ids:
                new_generations.append(
                    replace(generations[i], rating=mauve_score)
                )

        for i, g in enumerate(generations):
            if g.watermark is None:
                new_generations.append(replace(g, rating=1))

        # Write to file
        writer_queue.put(new_generations)
