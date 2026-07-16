import os
import common_utils
import torch
from tqdm.auto import tqdm
import wandb
from pathlib import Path

from common_utils.common import now
from GetParams import get_args
from Main import setup_args, epoch_ce
from CreateData import setup_problem
from CreateModel import create_model


def read_sweep(sweeps_dir, sweep_id, name=None, problem=None, wandb_entity='dataset_extraction', wandb_project_name='Dataset_Extraction'):
    api = wandb.Api({'entity': wandb_entity})
    wandb_sweep = api.sweep(f'{wandb_entity}/{wandb_project_name}/{sweep_id}')

    sweep = lambda: None
    sweep.id = sweep_id
    sweep.config = wandb_sweep.config
    if problem is None:
        sweep.problem = wandb_sweep.config['parameters']['problem']['value']
    else:
        sweep.problem = problem
    sweep.pretrained_model_path = wandb_sweep.config['parameters']['pretrained_model_path']['value']
    sweep.full_name = f'{sweep.problem}_{sweep_id}'
    if name is not None:
        sweep.full_name += f'_{name}'

    sweep.dir = os.path.join(sweeps_dir, sweep.full_name)
    return sweep


def download_sweep_results_from_wandb(sweep, max_runs_to_download, wandb_entity='dataset_extraction'):
    # Download runs results from WANDB
    api = wandb.Api({'entity': wandb_entity})

    os.makedirs(sweep.dir, exist_ok=True)

    runs = api.runs(
        path="dataset_extraction/Dataset_Extraction",
        filters={"sweepName": f"{sweep.id}"},
    )

    df = []
    for run in tqdm(runs):
        try:
            df.append((run.id, run.name, run.summary['extraction score']))
        except:
            pass
    df = sorted(df, key=lambda x: x[2])

    print('saving to:', sweep.dir)
    for i, q in enumerate(tqdm(df[:max_runs_to_download])):
        run_id, run_name, score = q
        run_dir = os.path.join(sweep.dir, 'runs', f'{run_id}_{run_name}')
        try:
            if os.path.exists(run_dir):
                print('EXISTS:', run_dir)
                continue

            run = api.run(f'dataset_extraction/Dataset_Extraction/{run_id}')
            x_path = sorted([i.name for i in list(run.files()) if 'x.pth' in i.name], key=lambda x: len(x))[-1]
            print(run.id, run.name, run.summary['extraction score'], x_path)
            run.file(x_path).download(run_dir, replace=True)

            Path(os.path.join(run_dir, x_path)).rename(os.path.join(run_dir, 'x.pth'))

            common_utils.common.dump_obj_with_dict(run.summary, f"{run_dir}/summary.txt")
            common_utils.common.dump_obj_with_dict(run.config, f"{run_dir}/config.txt")

        except Exception as e:
            print(e)
            pass

    print(now(), 'DONE!')


@torch.no_grad()
def get_all_reconstruction_outputs(sweep, verbose=True):
    if verbose: print('Reading extracted files from folder')
    dnames = os.listdir(os.path.join(sweep.dir, 'runs'))
    xx = []
    for run_dirname in tqdm(dnames):
        run_dir = os.path.join(sweep.dir, 'runs', run_dirname)
        try:
            x = torch.load(os.path.join(run_dir, 'x.pth'))
        except RuntimeError as e:
            print('x path:', os.path.join(run_dir, 'x.pth'))
            raise e
        xx.append(x.data)

    X = torch.cat(xx)
    if verbose: print('X.shape:', X.shape)
    return X


@torch.no_grad()
def sweep_get_data_model(sweep, verbose=True, run_train_test=False, put_in_sweep=True):
    # Get Training data for this sweep
    l = []
    if hasattr(sweep, 'problem'): l.append(f'--problem={sweep.problem}')
    for k, value in sweep.config['parameters'].items():
        if 'value' not in value: continue
        v = value['value']
        if k is None: continue
        l.append(f"--{k}={v}")
    l.append('--wandb_active=False')
    args = get_args(l)
    args = setup_args(args)
    if verbose: print(args)

    train_loader, test_loader, val_loader = setup_problem(args)

    # train set
    Xtrn, Ytrn = next(iter(train_loader))
    ds_mean = Xtrn.mean(dim=0, keepdims=True).data
    Xtrn = Xtrn.data - ds_mean.data

    # test set
    Xtst, Ytst = next(iter(test_loader))
    Xtst = Xtst - ds_mean

    # verify balancedness
    if verbose: print('BALNACENESS TRN:', {c: Ytrn[Ytrn == c].shape[0] for c in range(args.num_classes)})
    if verbose: print('BALNACENESS TST:', {c: Ytst[Ytst == c].shape[0] for c in range(args.num_classes)})

    # get model
    model = create_model(args, extraction=False)
    model = common_utils.common.load_weights(model, args.pretrained_model_path, device=args.device)
    model.eval()
    # get model's weights
    W = model.layers[0].weight

    if run_train_test:
        train_loader = [(Xtrn, Ytrn)]
        test_loader = [(Xtst, Ytst)]
        # compute train/test (reduce mean there..)
        trn_error, trn_loss, trn_vals = epoch_ce(args, train_loader, model, epoch=-1, device=args.device, opt=None)
        if verbose: print('Train Error:', trn_error, trn_loss)
        tst_error, tst_loss, tst_vals = epoch_ce(args, test_loader, model, epoch=-1, device=args.device, opt=None)
        if verbose: print('Test  Error:', tst_error, tst_loss)
        sweep.trn_error = trn_error
        sweep.trn_loss = trn_loss
        sweep.tst_error = tst_error
        sweep.tst_loss = tst_loss

    if put_in_sweep:
        sweep.Xtrn = Xtrn
        sweep.Ytrn = Ytrn
        sweep.Xtst = Xtst
        sweep.Ytst = Ytst
        sweep.ds_mean = ds_mean
        sweep.W = W
        sweep.model = model

    return args, Xtrn, Ytrn, ds_mean, W, model