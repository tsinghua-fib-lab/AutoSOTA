import optuna
from main import train
from opt import args
import yaml

cut_l = {'MUTAG': 5, 'ENZYMES': 20, 'PROTEINS': 50, 'REDDIT-MULTI-5K': 15, 'REDDIT-MULTI-5K': 15, 'NCI1': 20, 'BZR': 13, 'DD': 65,
         'REDDIT-BINARY': 13, 'PTC_MR': 20}
cut_h = {'MUTAG': 16, 'ENZYMES': 37, 'PROTEINS': 64, 'REDDIT-MULTI-5K': 24, 'REDDIT-MULTI-5K': 15, 'NCI1': 45, 'BZR': 19, 'DD': 82,
         'REDDIT-BINARY': 19, 'PTC_MR': 30}


def optuna_train(trial: optuna.trial.Trial):
    with open('config.yml', 'r') as file:
        data = yaml.safe_load(file)
    config = data['dataset'][args.dataset]
    acc = train(
        lr=trial.suggest_float('lr', 1e-6, 5e-1, log=True),
        weight_decay=trial.suggest_float('weight_decay', 1e-6, 5e-1, log=True),
        hid_paths=trial.suggest_int('hid_paths', 100, 800, step=50),
        norm=config['norm'],
        cutoff=trial.suggest_int(
            'cutoff', cut_l[args.dataset], cut_h[args.dataset], step=1),
        # cutoff= None , #IM-B,IM-M,COLLAB
        dropout=trial.suggest_float('dropout', 0.0, 0.4, step=0.05),
        norm_attr=config['norm_attr'])
    return acc


if __name__ == '__main__':
    study: optuna.study.Study = optuna.create_study(study_name='{}-tune'.format(args.dataset),
                                                    storage="sqlite:///optuna/optuna.sqlite3",
                                                    direction="maximize",
                                                    load_if_exists=True,
                                                    sampler=optuna.samplers.RandomSampler())
    study.optimize(optuna_train, n_trials=10000)
