import json
import os
from dataclasses import replace

import numpy as np
from apps import check_correctness

# from ..utils import get_server_args, setup_randomness
from .quality import RatingMetric


class CodeRating(RatingMetric):

    def __init__(self, config, writer_queue, device, timeout=10):
        super().__init__(config, writer_queue, device)
        self.timeout = timeout

    def rate(self, generations, _):
        """
        Run solutions from one problem.
        """

        # Load id to folder mapping
        with open(
            os.path.join(self.config.results, "response_id_to_folder.json"),
            "r",
            encoding="utf-8",
        ) as f:
            response_id_to_folder = json.load(f)

        # Iterate
        new_generations = []
        for g in generations:
            if str(g.id) not in response_id_to_folder:
                continue

            path = response_id_to_folder[str(g.id)]
            curr_res = check_correctness(
                prob_path=path,
                generation=g.response,
                timeout=self.timeout,
                debug=False,
            )
            fixed = []
            for e in curr_res:
                if isinstance(e, np.ndarray):
                    e = e.item(0)
                if isinstance(e, np.bool_):
                    e = bool(e)
                if not isinstance(e, bool):
                    e = e > 0
                fixed.append(e)
            curr_res = fixed

            if not len(curr_res):
                new_generations.append(replace(g, rating=0))
            else:
                new_generations.append(
                    replace(
                        g,
                        rating=len([v for v in curr_res if v]) / len(curr_res),
                    )
                )

        self.writer_queue.put(new_generations)
