import json
import random
import string
from pathlib import Path
from functools import partial

from .base import BaseDataset
from .prompts import PromptFormatter
from .utils import generic_collate

from torch.utils.data import DataLoader


COT = """\
Q: Find a movie similar to Star Wars Episode IV - A New Hope, Indiana Jones and the Last Crusade, Star Wars Episode V - The Empire Strikes Back, The Big Lebowski:
Options:
(A) Tetsuo
(B) the Ironman
(C) The Princess Bride
(D) The Barkley Marathons The Race That Eats Its Young
(E) Bug
A: Let's think step by step.
- Star Wars Episode IV - A New Hope (action, adventure, fantasy; 1977)
- Indiana Jones and the Last Crusade (action, adventure; 1989)
- Star Wars Episode V - The Empire Strikes Back (action, adventure, fantasy; 1980)
- The Big Lebowski (action, drama, comedy; 1998)
These are all famous classic American movies produced before 2000. Amongst all the options, the only movie similar to these ones seems to be The Princess Bride (1987). So the answer is (C).

Q: Find a movie similar to Twister, The Silence of the Lambs, Independence Day, Braveheart:
Options:
(A) They Shoot Horses
(B) Don't They
(C) Forrest Gump
(D) The Salton Sea
(E) Extreme Days
A: Let's think step by step.
- Twister (action, adventure, thriller; 1996)
- The Silence of the Lambs (crime, drama, thriller; 1991)
- Independence Day (action, science-fiction, drama; 1996)
- Braveheart (biography, drama, epic; 1995)
These are all famous Hollywood movies produced around the 1990s. Amongst all the options, the only movie similar to these ones seems to be Forrest Gump (comedy, drama, romance; 1994). So the answer is (C).

Q: Find a movie similar to Minority Report, Total Recall, Inside Out, Forrest Gump:
Options:
(A) Phenomena
(B) Lilting
(C) Catwoman
(D) Edge of Tomorrow
A: Let's think step by step.
- Minority Report (action, crime, mystery; 2002)
- Total Recall (action, adventure, science-fiction; 2012)
- Inside Out (animation, family, comedy; 2015)
- Forrest Gump (comedy, drama, romance; 1994)
These are all famous movies produced in the past few decades. Amongst all the options, the only movie similar to these ones seems to be Edge of Tomorrow (action, adventure, crime, mystery; 2014), as it is also a science-fiction movie and features Tom Cruise. So the answer is (D).\
"""
PROB_HEADER = "Q: "


class MovieDataset(BaseDataset):
    """Date understanding task."""

    description = "Find a movie similar to the given set of movies."
    data_file = "movie_recommendation.json"

    def __init__(self, n=1000):
        super().__init__()
        self.n = n
        self._examples = []
        self._clean_examples = []
        self._corrupted_examples = []
        self._labels = []

    def get_questions(self):
        task = json.load(open(Path(__file__).parent / "data" / self.data_file))
        for ex in task["examples"]:
            self._examples.append({"input": ex["input"], "target": ex["target"]})
        random.shuffle(self._examples)
        self._examples = self._examples[: self.n]

    def format_questions(self, formatter: PromptFormatter):
        global PROB_HEADER, COT
        Qs, As = [PROB_HEADER + f"{v['input']}" for v in self._examples], [
            f"\nA: {v['target']}" for v in self._examples
        ]
        answer_starter = "\nA: "
        if formatter.name == "chain-of-thought":
            answer_starter = "\nA: Let's think step by step."
        self._clean_examples = [
            formatter.format(
                task_description=self.description,
                prompt=v + answer_starter,
                questions=Qs,
                answers=As,
                cot=COT,
            )
            for v in Qs
        ]
        self._labels = [v["target"] for v in self._examples]
        for i in range(len(self._clean_examples)):
            # for each clean example, get example that has different target
            diff = list(
                filter(
                    lambda j: self._labels[j] != self._labels[i],
                    range(len(self._examples)),
                )
            )
            self._corrupted_examples.append(self._clean_examples[random.choice(diff)])

    def to_dataloader(self, model, batch_size: int, collate_fn=None):
        collate_fn = partial(generic_collate, model)
        return DataLoader(self, batch_size=batch_size, collate_fn=collate_fn)

    def __len__(self):
        return len(self._examples)

    def __getitem__(self, idx):
        return (
            self._clean_examples[idx],
            self._corrupted_examples[idx],
            self._labels[idx],
        )
