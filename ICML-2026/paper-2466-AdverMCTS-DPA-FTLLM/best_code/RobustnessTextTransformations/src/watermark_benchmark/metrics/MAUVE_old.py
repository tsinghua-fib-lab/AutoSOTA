import os
from dataclasses import replace

from evaluate import load
from tqdm import tqdm

from .quality import RatingMetric


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

        specs = {
            (str(g.watermark.to_dict(True, False)), g.attack): []
            for g in generations
            if g.watermark is not None
        }

        for i, g in enumerate(generations):
            if g.watermark is not None:
                specs[(str(g.watermark.to_dict(True, False)), g.attack)].append(
                    (i, g)
                )

        mauve = load("mauve")
        for tasks in tqdm(
            specs.values(), desc="Computing MAUVE scores", total=len(specs)
        ):
            baseline_tasks = [
                baselines[(str(float(g.temp)), str(g.id))] for _, g in tasks
            ]
            local_tasks = [g.response for _, g in tasks]
            original_ids = [v for v, _ in tasks]

            mauve_score = mauve.compute(
                predictions=local_tasks,
                references=baseline_tasks,
                device_id=device,
                seed=config.seed,
            ).mauve

            for i in original_ids:
                generations[i] = replace(generations[i], rating=mauve_score)

        for i, g in enumerate(generations):
            if g.watermark is None:
                generations[i] = replace(g, rating=1)

        # Write to file
        writer_queue.put(generations)
