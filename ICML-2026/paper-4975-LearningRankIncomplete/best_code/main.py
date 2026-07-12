import matplotlib
matplotlib.use('Agg')

from experiment.experiment import ExperimentBatch
from configs import movielens, plackett_luce_mcar, plackett_luce_mnar, sushi_mnar

def main():
    batch = ExperimentBatch(name="Experiment Batch", experiments=[
        plackett_luce_mcar.get_experiment(),
        plackett_luce_mnar.get_experiment(),
        movielens.get_experiment(),
        sushi_mnar.get_experiment(),
    ])
    batch.run()
    batch.plot()

if __name__ == '__main__':
    main()
