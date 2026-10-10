from meisenmeister.plan_and_preprocess.plan_experiment import plan_experiment
from meisenmeister.plan_and_preprocess.preprocess import preprocess


def plan_and_preprocess(d: int, num_workers: int = 4) -> None:
    plan_experiment(d)
    preprocess(d, num_workers=num_workers)
