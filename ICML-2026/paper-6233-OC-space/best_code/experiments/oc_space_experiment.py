import gc
import resource
from datetime import datetime

resource.setrlimit(resource.RLIMIT_AS, (100*1024*1024*1024, 100*1024*1024*1024))

import click
import os

import veritas
import numpy as np
import random
import json
import prada
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

import util

from verification import run_fairness_tasks, run_adversarial_robustness_tasks, run_lipschitz_task, run_complex_norm_robustness_tasks
from depth_first_ocenum import enumerate_ocs_to_disk

@click.group()
def cli():
    pass


@cli.command('list_pareto_models')
@click.argument("compressed_file")
@click.argument("save_directory")
@click.option("--seed", default=util.SEED)
@click.option("--silent", is_flag=True, default=True)
def list_pareto_models_cmd(compressed_file, save_directory, seed, silent):
    # Prints a list of commands to enumerate all pareto front models
    np.random.seed(seed)
    random.seed(seed)

    results = {}
    datasets = set()
    with open(compressed_file, "r") as f:
        for line in f:
            if not line.startswith('{'):
                continue
            line_dict = json.loads(line.strip())
            key = f"{line_dict['dname']}_{line_dict['params']['n_estimators']}_{line_dict['params']['max_depth']}_{line_dict['params']['learning_rate']}_{line_dict['fold']}"
            results[key] = line_dict
            datasets.add(line_dict['dname'])

    datasets = sorted(list(datasets))

    pareto_fronts = {}

    refinements = line_dict['refinements']
    idx = [i for i, r in enumerate(refinements) if r['penalty'] in ['lop', 'ours']][0]  # Get index of 'lop' or 'ours' in compressed file

    for ds in datasets:
        df = []
        for depth in [4, 6, 8]:
            for n_estimators in [10, 25, 50, 100]:
                for lr in [0.1, 0.25, 0.5, 1.0]:
                    avg_mtest = 0
                    avg_nleafs = 0
                    for fold in range(5):
                        key = f"{ds}_{n_estimators}_{depth}_{lr}_{fold}"
                        if key in results:
                            assert results[key]['refinements'][idx]['penalty'] in ['lop', 'ours'], f"Expected 'lop' or 'ours' penalty, got {results[key]['refinements'][idx]['penalty']}"
                            avg_mtest += results[key]['refinements'][idx]['mtest']/5
                            avg_nleafs += results[key]['refinements'][idx]['nleafs']/5
                    df.append(np.array([depth, n_estimators, lr, avg_mtest, avg_nleafs]))

        df = pd.DataFrame(np.array(df), columns=['max_depth', 'n_estimators', 'learning_rate', 'mtest', 'nleafs'])
        on_front, on_hull = util.pareto_front_xy(df['nleafs'].to_numpy(), df['mtest'].to_numpy())
        
        df['on_front'] = on_front
        df['on_hull'] = on_hull
        
        pareto_fronts[ds] = df

    for ds in datasets:
        pareto_models = pareto_fronts[ds][pareto_fronts[ds]['on_front'] == True]
        for model in pareto_models.itertuples():
            for fold in range(5):
                print("python3 experiments/oc_space_experiment.py enumerate_model", ds,
                        '--models_file', compressed_file,
                        '--save_directory', save_directory,
                        '--seed', seed,
                        '--fold', fold,
                        '--learning-rate', model.learning_rate,
                        '--max-depth', int(model.max_depth),
                        '--n-estimators', int(model.n_estimators))

@cli.command('enumerate_model')
@click.argument("dname")
@click.option("--models_file", required=True)
@click.option("--save_directory", required=True)
@click.option("--seed", default=util.SEED)
@click.option("--silent", is_flag=True, default=True)
@click.option("--timeout", default=86400)
@click.option("--fold", default=0)
@click.option("--learning-rate", default=0.1, type=float)
@click.option("--max-depth", default=4, type=int)
@click.option("--n-estimators", default=10, type=int)
def verify_pareto_models_cmd(dname, seed, models_file, save_directory, silent, timeout, fold, learning_rate, max_depth, n_estimators):
    np.random.seed(seed)
    random.seed(seed)

    # Now load in the correct model
    with open(models_file, "r") as f:
        for line in f:
            if not line.startswith('{'):
                continue
            line_dict = json.loads(line.strip())
            if line_dict['dname'] != dname or line_dict['fold'] != fold or float(line_dict['params']['learning_rate']) != learning_rate or int(line_dict['params']['max_depth']) != max_depth or int(line_dict['params']['n_estimators']) != n_estimators:
                continue
            else:
                refinements = line_dict['refinements']
                idx = [i for i, r in enumerate(refinements) if r['penalty'] in ['lop', 'ours']][0]
                model = line_dict['refinements'][idx]['model_json']
                assert line_dict['refinements'][idx]['penalty'] == 'ours' or line_dict['refinements'][idx]['penalty'] == 'lop',\
                    f"Expected 'lop' or 'ours' penalty, got {line_dict['refinements'][idx]['penalty']}"
                break
            
    model = veritas.AddTree.from_json(model)
    
    results = {
        'date_time': util.nowstr(),
        'hostname': os.uname()[1],
        'dname': dname,
        'fold': fold,
        'seed': seed,
        'metric_name': line_dict['metric_name'],
        'mtrain': line_dict['mtrain'],
        'mvalid': line_dict['mvalid'],
        'mtest': line_dict['mtest'],
        'ntrees': line_dict['ntrees'],
        'nnodes': line_dict['nnodes'],
        'nleafs': line_dict['nleafs'],
        'params': line_dict['params'],
    }

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    oc_file = f'{save_directory}/OC_enum_{dname}_{line_dict["params"]["n_estimators"]}_{line_dict["params"]["max_depth"]}_{line_dict["params"]["learning_rate"]}_fold{fold}_{run_id}.zarr'
    oc_file = remove_glob_expansion(oc_file) # Removes '[]' characters from filename which can cause issues with some filesystems (e.g. SFTP)

    storage_options = {
            "username": os.environ['SFTP_USERNAME'],
            "password": os.environ['SFTP_PASSWORD'],
            "look_for_keys": False,
            "allow_agent": False,
    } if save_directory.startswith("sftp://") else None
        
    enumeration_results = enumerate_ocs_to_disk(model, buffer_size=1000*8194, filename=oc_file, timeout=timeout, storage_options=storage_options)

    results['enumeration'] = enumeration_results
                
    print(json.dumps(results))

@cli.command('list_enumerated_models')
@click.argument("enumerations_file")
@click.argument("models_file")
def list_enumerated_models_cmd(enumerations_file, models_file):
    # Prints a list of commands to verify (closest adversarial example) all enumerated models

    with open(enumerations_file, "r") as f:
        for line in f:
            if not line.startswith('{'):
                continue
            line_dict = json.loads(line.strip())
            
            params = line_dict['params']

            print("python3 experiments/oc_space_experiment.py verify_model", line_dict['enumeration']['filename'], 
                "--models_file", models_file, 
                "--dname", line_dict['dname'],
                "--fold", line_dict['fold'],
                "--learning-rate", params['learning_rate'],
                "--max-depth", params['max_depth'],
                "--n-estimators", params['n_estimators']
                )
    
        

@cli.command('verify_model')
@click.argument("oc_file")
@click.option("--models_file", required=True)
@click.option("--timeout", default=86400)
@click.option("--dname", required=True)
@click.option("--fold", required=True, type=int)
@click.option("--learning-rate", required=True, type=float)
@click.option("--max-depth", required=True, type=int)
@click.option("--n-estimators", required=True, type=int)
@click.option("--seed", default=util.SEED)
def verify_model_cmd(oc_file, models_file, timeout, dname, fold, learning_rate, max_depth, n_estimators, seed):
    np.random.seed(seed)
    random.seed(seed)
    d, dtrain, dvalid, dtest = util.get_dataset(dname, seed, int(fold), True)

    with open(models_file, "r") as f:
        for line in f:
            # print(line)
            if not line.startswith('{'):
                continue
            line_dict = json.loads(line.strip())
            if line_dict['dname'] != dname or line_dict['fold'] != fold or float(line_dict['params']['learning_rate']) != learning_rate or int(line_dict['params']['max_depth']) != max_depth or int(line_dict['params']['n_estimators']) != n_estimators:
                continue
            else:
                refinements = line_dict['refinements']
                idx = [i for i, r in enumerate(refinements) if r['penalty'] in ['lop', 'ours']][0]
                model = line_dict['refinements'][idx]['model_json']
                assert line_dict['refinements'][idx]['penalty'] == 'ours' or line_dict['refinements'][idx]['penalty'] == 'lop',\
                    f"Expected 'lop' or 'ours' penalty, got {line_dict['refinements'][idx]['penalty']}"
                break

    model = veritas.AddTree.from_json(model)

    results = {
                'date_time': util.nowstr(),
                'hostname': os.uname()[1],
                'dname': dname,
                'fold': fold,
                'seed': seed,
                'metric_name': line_dict['metric_name'],
                'mtrain': line_dict['mtrain'],
                'mvalid': line_dict['mvalid'],
                'mtest': line_dict['mtest'],
                'ntrees': line_dict['ntrees'],
                'nnodes': line_dict['nnodes'],
                'nleafs': line_dict['nleafs'],
                'params': line_dict['params'],
            }
    
    storage_options = {
            "username": os.environ['SFTP_USERNAME'],
            "password": os.environ['SFTP_PASSWORD'],
            "look_for_keys": False,
            "allow_agent": False,
    } if oc_file.startswith("sftp://") else None

    try:
        verification_results = run_adversarial_robustness_tasks(model, dtest.X, dtest.y, oc_file=oc_file, storage_options=storage_options, timeout=timeout, n=500)
    except (MemoryError, OSError) as e:
        verification_results = {'failed_oc': True}
        print(e)
        gc.collect()

    results['verification'] = verification_results

    print(json.dumps(results))


@cli.command('list_robustness_models')
@click.argument("enumerations_file")
@click.argument("models_file")
@click.argument("dname", required=True, nargs=-1)
def list_enumerated_models_cmd(enumerations_file, models_file, dname):
    # Prints a list of commands to verify (hybrid norm robustness). Takes dname as a list of datasets to filter the enumerated models to verify.
    with open(enumerations_file, "r") as f:
        for line in f:
            if not line.startswith('{'):
                continue
            line_dict = json.loads(line.strip())
            if not line_dict['dname'] in dname:
                continue
            params = line_dict['params']

            print("python3 experiments/oc_space_experiment.py verify_complex_norm", line_dict['enumeration']['filename'], 
                "--models_file", models_file, 
                "--dname", line_dict['dname'],
                "--fold", line_dict['fold'],
                "--learning-rate", params['learning_rate'],
                "--max-depth", params['max_depth'],
                "--n-estimators", params['n_estimators']
                )
    
        

@cli.command('verify_complex_norm')
@click.argument("oc_file")
@click.option("--models_file", required=True)
@click.option("--timeout", default=86400)
@click.option("--dname", required=True)
@click.option("--fold", required=True, type=int)
@click.option("--learning-rate", required=True, type=float)
@click.option("--max-depth", required=True, type=int)
@click.option("--n-estimators", required=True, type=int)
@click.option("--seed", default=util.SEED)
@click.option("--linf", default=0.1, type=float)
@click.option("--l1", default=1.0, type=float)
def verify_model_cmd(oc_file, models_file, timeout, dname, fold, learning_rate, max_depth, n_estimators, seed, linf, l1):
    np.random.seed(seed)
    random.seed(seed)
    d, dtrain, dvalid, dtest = util.get_dataset(dname, seed, int(fold), True)

    with open(models_file, "r") as f:
        for line in f:
            if not line.startswith('{'):
                continue
            line_dict = json.loads(line.strip())
            if line_dict['dname'] != dname or line_dict['fold'] != fold or float(line_dict['params']['learning_rate']) != learning_rate or int(line_dict['params']['max_depth']) != max_depth or int(line_dict['params']['n_estimators']) != n_estimators:
                continue
            else:
                refinements = line_dict['refinements']
                idx = [i for i, r in enumerate(refinements) if r['penalty'] in ['lop', 'ours']][0]
                model = line_dict['refinements'][idx]['model_json']
                assert line_dict['refinements'][idx]['penalty'] == 'ours' or line_dict['refinements'][idx]['penalty'] == 'lop',\
                    f"Expected 'lop' or 'ours' penalty, got {line_dict['refinements'][idx]['penalty']}"
                break

    model = veritas.AddTree.from_json(model)

    results = {
                'date_time': util.nowstr(),
                'hostname': os.uname()[1],
                'dname': dname,
                'fold': fold,
                'seed': seed,
                'metric_name': line_dict['metric_name'],
                'mtrain': line_dict['mtrain'],
                'mvalid': line_dict['mvalid'],
                'mtest': line_dict['mtest'],
                'ntrees': line_dict['ntrees'],
                'nnodes': line_dict['nnodes'],
                'nleafs': line_dict['nleafs'],
                'params': line_dict['params'],
            }
    
    storage_options = {
            "username": os.environ['SFTP_USERNAME'],
            "password": os.environ['SFTP_PASSWORD'],
            "look_for_keys": False,
            "allow_agent": False,
    } if oc_file.startswith("sftp://") else None

    try:
        verification_results = run_complex_norm_robustness_tasks(model, dtest.X, dtest.y, 500, linf, l1, oc_file=oc_file, storage_options=storage_options, timeout=timeout)
    except (MemoryError, OSError) as e:
        verification_results = {'failed_oc': True}
        print(e)
        gc.collect()

    results['verification'] = verification_results

    print(json.dumps(results))

@cli.command('list_fairness_models')
@click.argument("enumerations_file")
@click.argument("models_file")
@click.option("--dname", required=True, multiple=True)
@click.option("--protected-attribute", required=True, multiple=True)
def list_fairness_models_cmd(enumerations_file, models_file, dname, protected_attribute):
    # Prints a list of commands to verify (fairness) all enumerated models. Takes dname and protected attributes as lists to filter the enumerated models to verify.
    with open(enumerations_file, "r") as f:
        for line in f:
            if not line.startswith('{'):
                continue
            line_dict = json.loads(line.strip())
            if not line_dict['dname'] in dname:
                continue
            params = line_dict['params']

            print("python3 experiments/oc_space_experiment.py verify_fairness", line_dict['enumeration']['filename'], 
                "--models_file", models_file, 
                "--dname", line_dict['dname'],
                "--fold", line_dict['fold'],
                "--learning-rate", params['learning_rate'],
                "--max-depth", params['max_depth'],
                "--n-estimators", params['n_estimators'],
                "--protected-attribute", protected_attribute[dname.index(line_dict['dname'])]
                )

        

@cli.command('verify_fairness')
@click.argument("oc_file")
@click.option("--models_file", required=True)
@click.option("--timeout", default=86400)
@click.option("--dname", required=True)
@click.option("--fold", required=True, type=int)
@click.option("--learning-rate", required=True, type=float)
@click.option("--max-depth", required=True, type=int)
@click.option("--n-estimators", required=True, type=int)
@click.option("--seed", default=util.SEED)
@click.option("--protected-attribute", required=True)
def verify_fairness_cmd(oc_file, models_file, timeout, dname, fold, learning_rate, max_depth, n_estimators, seed, protected_attribute):
    np.random.seed(seed)
    random.seed(seed)
    d, dtrain, dvalid, dtest = util.get_dataset(dname, seed, int(fold), True)

    with open(models_file, "r") as f:
        for line in f:
            if not line.startswith('{'):
                continue
            line_dict = json.loads(line.strip())
            if line_dict['dname'] != dname or line_dict['fold'] != fold or float(line_dict['params']['learning_rate']) != learning_rate or int(line_dict['params']['max_depth']) != max_depth or int(line_dict['params']['n_estimators']) != n_estimators:
                continue
            else:
                refinements = line_dict['refinements']
                idx = [i for i, r in enumerate(refinements) if r['penalty'] in ['lop', 'ours']][0]
                model = line_dict['refinements'][idx]['model_json']
                assert line_dict['refinements'][idx]['penalty'] == 'ours' or line_dict['refinements'][idx]['penalty'] == 'lop',\
                    f"Expected 'lop' or 'ours' penalty, got {line_dict['refinements'][idx]['penalty']}"
                break

    model = veritas.AddTree.from_json(model)

    results = {
                'date_time': util.nowstr(),
                'hostname': os.uname()[1],
                'dname': dname,
                'fold': fold,
                'seed': seed,
                'metric_name': line_dict['metric_name'],
                'mtrain': line_dict['mtrain'],
                'mvalid': line_dict['mvalid'],
                'mtest': line_dict['mtest'],
                'ntrees': line_dict['ntrees'],
                'nnodes': line_dict['nnodes'],
                'nleafs': line_dict['nleafs'],
                'params': line_dict['params'],
            }
    
    storage_options = {
            "username": os.environ['SFTP_USERNAME'],
            "password": os.environ['SFTP_PASSWORD'],
            "look_for_keys": False,
            "allow_agent": False,
    } if oc_file.startswith("sftp://") else None

    try:
        verification_results = run_fairness_tasks(model, oc_file=oc_file, storage_options=storage_options, columns=dtest.X.columns, non_fixed_columns=[protected_attribute], timeout=timeout)
    except (MemoryError, OSError) as e:
        verification_results = {'failed_oc': True}
        print(e)
        gc.collect()

    results['verification'] = verification_results

    print(json.dumps(results))


@cli.command('lipschitz')
@click.argument("dname")
@click.option("--seed", default=util.SEED) # The paper results were obtained using a different seed before we fixed it.
@click.option("--save_directory", required=True)
@click.option("--silent", is_flag=True, default=True)
@click.option("--timeout", default=86400)
@click.option("--fold", default=0)
def lipschitz_cmd(dname, seed, save_directory, silent, timeout, fold):
    # To get the paper results, use the Vehicle dataset
    np.random.seed(seed)
    random.seed(seed)

    def get_dataset(dname, seed, fold, silent):
        d = prada.get_dataset(dname, seed=seed, silent=silent)
        d.load_dataset()
        d.robust_normalize()
        #d.transform_target()
        d.scale_target()
        d.astype(veritas.FloatT)

        if d.is_binary():
            d.use_balanced_accuracy()

        dtrain, dtest = d.train_and_test_fold(fold, nfolds=4)
        # dtrain, dvalid = dtrain.split(0, nfolds=3)

        return d, dtrain, dtest

    d, dtrain, dtest = get_dataset(dname, seed, fold, silent)

     # Load in the pareto fronts

    for ntrees in [10, 11, 12]:
        
        params = {
        "random_state": seed,
        "n_jobs": 1,
        "n_estimators": ntrees,
        "max_depth": 4,
        "learning_rate": 1.0,
        }
        model_type = "xgb"
        model_class = d.get_model_class(model_type)

        clf, _ = dtrain.train(model_class, params)

        # Transfer to Veritas AddTree
        model = veritas.get_addtree(clf)

        oc_file = f'{save_directory}/OC_enum_lipschitz_{ntrees}.zarr'
        oc_file = remove_glob_expansion(oc_file) # Removes '[]' characters from filename which can cause issues with some filesystems (e.g. SFTP)

        storage_options = {
                "username": os.environ['SFTP_USERNAME'],
                "password": os.environ['SFTP_PASSWORD'],
                "look_for_keys": False,
                "allow_agent": False,
        } if oc_file.startswith("sftp://") else None
        
        enumeration_results = enumerate_ocs_to_disk(model, buffer_size=1000*8194, filename=oc_file, timeout=timeout, storage_options=storage_options)

        assert not enumeration_results['failed'], "OC enumeration failed, cannot run Lipschitz verification"

        print(f"OC space enumerated for model with {ntrees} trees, now running Lipschitz verification")

        results = {
            'date_time': util.nowstr(),
            'hostname': os.uname()[1],
            'dname': dname,
            'fold': fold,
            'seed': seed,
            'params': params,
        }

        for Dx in [0.01, 0.1, 1]:
            try:
                verification_results = run_lipschitz_task(model, oc_file, storage_options, Dx, timeout=timeout)
            except (MemoryError, OSError):
                verification_results = {'failed_oc': True}
                gc.collect()

            results['fairness'] = {
                'verification_results': verification_results,
                'nleafs': model.num_leafs(),
                'nnodes': model.num_nodes(),
                'mtest': dtest.metric(model),
                'mtrain': dtrain.metric(model),
                'enumeration': enumeration_results,
                'lipschitz_bound': verification_results,
            }

            print(json.dumps(results))


def bound_oc_space(at):
    splits = at.get_splits()
    bound_1 = 1
    for _, f_values in splits.items():
        bound_1 *= (len(f_values) + 1)

    bound_2 = 1
    for t in at:
        bound_2 *= (t.num_leaves())
    return bound_1, bound_2

def remove_glob_expansion(path):
    return path.replace("[", "_").replace("]", "_")

if __name__ == "__main__":
    cli()