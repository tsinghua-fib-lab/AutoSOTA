from dataclasses import replace

from tqdm import tqdm

from .quality import RatingMetric


class RepetitionRating(RatingMetric):
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

        writer_queue = self.writer_queue

        tasks = [gen.response for gen in generations]

        for task_idx, task in tqdm(
            enumerate(tasks), desc="Rating generations", total=len(tasks)
        ):
            rep = [0, 0, 0, 0]
            words = task.split()  # Split task into words

            for n in range(2, 6):  # For n-grams of length 2 through 5
                ngrams = [
                    " ".join(words[i : i + n])
                    for i in range(len(words) - n + 1)
                ]
                seen = set()
                match = []

                for ngram in ngrams:
                    if ngram in seen:
                        match.append(1)
                    else:
                        match.append(0)
                        seen.add(ngram)

                rep[n - 1] = -sum(match) / len(match) if match else 0

            generations[task_idx] = replace(
                generations[task_idx], rating=tuple(rep)
            )

        # Write to file
        writer_queue.put(generations)
