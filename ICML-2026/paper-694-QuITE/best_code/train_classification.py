"""Entrypoint: train a QuITE-equipped MTS backbone for IMTS classification.

Examples:
    # QuITE-PatchTST on P19 (binary classification → AUROC/AUPRC reported)
    python train_classification.py --dataset P19 --gpu 0 --epoch 100 \
        --batch_size 64 --lr 1e-3 --nhead 1 --nlayer 3 --hid_dim 32 \
        --patch_size 3.75 --stride 3.75 --model patchtst --irr_emb --mode quite

    # QuITE-PatchTST on PAM (8-class activity → Precision/Recall/F1 reported)
    python train_classification.py --dataset PAM --gpu 0 --epoch 100 \
        --batch_size 64 --lr 1e-3 --nhead 2 --nlayer 3 --hid_dim 128 \
        --patch_size 10 --stride 10 --model patchtst --irr_emb --mode quite
"""
import os
import math
import time
import argparse
import warnings
import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
)

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser('QuITE Classification')

# ----- Training -----
parser.add_argument('--epoch', type=int, default=1000)
parser.add_argument('--patience', type=int, default=10,
                    help='Early-stopping patience (no val-loss improvement)')
parser.add_argument('-b', '--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--logmode', type=str, default='a')

# ----- Dataset -----
parser.add_argument('--dataset', type=str, default='P12', choices=['P12', 'P19', 'PAM'])

# ----- Model & Embedding -----
parser.add_argument('--model', type=str, required=True,
                    choices=['patchtst', 'patchmixer', 'tmix', 'itransformer', 's_mamba', 'timexer'],
                    help='MTS backbone')
parser.add_argument('--mode', type=str, default='False',
                    choices=['quite', 'mean', 'mtand', 'add', 'concat', 'False'],
                    help="Embedding mode: 'quite'=QuITE (paper main), 'mean'/'mtand'=baselines (Table 5), "
                         "'add'/'concat'=non-irregular baselines, 'False'=vanilla backbone")
parser.add_argument('--irr_emb', action='store_true',
                    help='Use QuITE-style query-based embedding (paper main method)')
parser.add_argument('-hd', '--hid_dim', type=int, default=64)
parser.add_argument('--nhead', type=int, default=4,
                    help='Paper §6.1 standardizes QuITE-equipped backbones to 4 heads')
parser.add_argument('--nlayer', type=int, default=1)
parser.add_argument('-ps', '--patch_size', type=float, default=6)
parser.add_argument('--stride', type=float, default=6)
parser.add_argument('--dropout', type=float, default=0.1)

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

from models.quite import Model
from utils import (
    get_data_split, getStats, getStats_static,
    tensorize_normalize_extract_feature_patch,
    tensorize_normalize_extract_feature_pam_patch,
    evaluate_model_patch, get_logger, makedirs,
)

args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)

model_path = './models_classification/'
makedirs(model_path)


def layer_of_patches(n_patch):
    if n_patch == 1:
        return 1
    if n_patch % 2 == 0:
        return 1 + layer_of_patches(n_patch / 2)
    return layer_of_patches(n_patch + 1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Dataset-specific config (no split_idx — we use k+1 per fold)
DATASET_CFG = {
    'P12': dict(base_path='../data/P12', d_static=9, variables_num=36, n_class=2, history=48),
    'P19': dict(base_path='../data/P19', d_static=6, variables_num=34, n_class=2, history=60),
    'PAM': dict(base_path='../data/PAM', d_static=0, variables_num=17, n_class=8, history=60),
}
cfg = DATASET_CFG[args.dataset]
base_path = cfg['base_path']
d_static = args.d_static = cfg['d_static']
variables_num = cfg['variables_num']
n_class = args.n_class = cfg['n_class']
args.history = cfg['history']

# Metric arrays
acc_arr, auprc_arr, auroc_arr = [], [], []
precision_arr, recall_arr, f1_arr = [], [], []

makedirs("logs/")
log_path = f"logs/{args.dataset}_{args.model}_{args.mode}_{args.hid_dim}hdims_{args.nlayer}nlayers_{args.nhead}nheads.log"
logger = get_logger(logpath=log_path, filepath=os.path.abspath(__file__), mode=args.logmode)
logger.info(args)

# Run 5 seeds × 5 splits
for k in range(5):
    torch.manual_seed(k); torch.cuda.manual_seed(k); np.random.seed(k)
    split_idx = k + 1
    logger.info(f"Using split: {split_idx}")

    if args.dataset == 'P12':
        split_path = f'/splits/phy12_split{split_idx}.npy'
    elif args.dataset == 'P19':
        split_path = f'/splits/phy19_split{split_idx}_new.npy'
    elif args.dataset == 'PAM':
        split_path = f'/splits/PAM_split_{split_idx}.npy'

    Ptrain, Pval, Ptest, ytrain, yval, ytest = get_data_split(base_path, split_path, dataset=args.dataset)
    print(len(Ptrain), len(Pval), len(Ptest), len(ytrain), len(yval), len(ytest))

    args.ndim = variables_num
    args.npatch = int(math.ceil((args.history - args.patch_size) / args.stride)) + 1
    args.patch_layer = layer_of_patches(args.npatch)
    args.scale_patch_size = args.patch_size / args.history
    args.task = 'classification'

    # Normalize & tensorize
    if args.dataset in ('P12', 'P19'):
        T, F = Ptrain[0]['arr'].shape
        D = len(Ptrain[0]['extended_static'])
        Ptrain_tensor = np.zeros((len(Ptrain), T, F))
        Ptrain_static_tensor = np.zeros((len(Ptrain), D))
        for i in range(len(Ptrain)):
            Ptrain_tensor[i] = Ptrain[i]['arr']
            Ptrain_static_tensor[i] = Ptrain[i]['extended_static']

        mf, stdf = getStats(Ptrain_tensor)
        ms, ss = getStats_static(Ptrain_static_tensor, dataset=args.dataset)

        Ptrain_tensor, Ptrain_mask_tensor, Ptrain_static_tensor, Ptrain_time_tensor, ytrain_tensor, maxlen = \
            tensorize_normalize_extract_feature_patch(Ptrain, ytrain, mf, stdf, ms, ss, args)
        Pval_tensor, Pval_mask_tensor, Pval_static_tensor, Pval_time_tensor, yval_tensor, maxlen = \
            tensorize_normalize_extract_feature_patch(Pval, yval, mf, stdf, ms, ss, args)
        Ptest_tensor, Ptest_mask_tensor, Ptest_static_tensor, Ptest_time_tensor, ytest_tensor, maxlen = \
            tensorize_normalize_extract_feature_patch(Ptest, ytest, mf, stdf, ms, ss, args)

    elif args.dataset == 'PAM':
        mf, stdf = getStats(Ptrain)

        Ptrain_tensor, Ptrain_mask_tensor, Ptrain_static_tensor, Ptrain_time_tensor, ytrain_tensor, maxlen = \
            tensorize_normalize_extract_feature_pam_patch(Ptrain, ytrain, mf, stdf, args)
        Pval_tensor, Pval_mask_tensor, Pval_static_tensor, Pval_time_tensor, yval_tensor, maxlen = \
            tensorize_normalize_extract_feature_pam_patch(Pval, yval, mf, stdf, args)
        Ptest_tensor, Ptest_mask_tensor, Ptest_static_tensor, Ptest_time_tensor, ytest_tensor, maxlen = \
            tensorize_normalize_extract_feature_pam_patch(Ptest, ytest, mf, stdf, args)

    args.maxlen = 0 if args.irr_emb else maxlen
    model = Model(args).to(args.device)
    logger.info(model)
    logger.info(f'parameters: {count_parameters(model)}')

    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Batching strategy
    if args.dataset != 'PAM':
        # Upsample minority class for binary tasks
        idx_0 = np.where(ytrain == 0)[0]
        idx_1 = np.where(ytrain == 1)[0]
        expanded_idx_1 = np.concatenate([idx_1, idx_1, idx_1], axis=0)
        K0 = len(idx_0) // int(args.batch_size / 2)
        K1 = len(expanded_idx_1) // int(args.batch_size / 2)
        n_batches = np.min([K0, K1])
    else:
        all_indices = np.arange(len(ytrain))
        n_batches = len(all_indices) // args.batch_size

    best_val_epoch = 0
    best_loss_val = float('inf')
    save_time = None

    # Training loop
    for epoch in range(args.epoch):
        if epoch - best_val_epoch > args.patience:
            break

        model.train()
        if args.dataset != 'PAM':
            np.random.shuffle(expanded_idx_1); I1 = expanded_idx_1
            np.random.shuffle(idx_0); I0 = idx_0
        else:
            np.random.shuffle(all_indices)

        st = time.time()
        for n in range(n_batches):
            if args.dataset != 'PAM':
                idx0_b = I0[n * int(args.batch_size / 2):(n + 1) * int(args.batch_size / 2)]
                idx1_b = I1[n * int(args.batch_size / 2):(n + 1) * int(args.batch_size / 2)]
                idx = np.concatenate([idx0_b, idx1_b], axis=0)
            else:
                idx = all_indices[n * args.batch_size:(n + 1) * args.batch_size]

            P = Ptrain_tensor[idx].cuda()
            P_mask = Ptrain_mask_tensor[idx].cuda()
            P_static = Ptrain_static_tensor[idx].cuda() if d_static != 0 else None
            P_time = Ptrain_time_tensor[idx].cuda()
            y = ytrain_tensor[idx].cuda()

            outputs = model.classification(P, P_time, P_mask, P_static)
            optimizer.zero_grad()
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        logger.info(f'training time per epoch :{time.time() - st:.3f}s')

        # Validation
        model.eval()
        with torch.no_grad():
            st = time.time()
            out_val = evaluate_model_patch(
                model, Pval_tensor, Pval_mask_tensor, Pval_static_tensor, Pval_time_tensor,
                n_classes=n_class, batch_size=args.batch_size,
            )
            logger.info(f"val time per epoch :{time.time() - st:.2f}s")

            # Val loss on raw logits
            val_loss = criterion(out_val.cuda(), torch.from_numpy(yval.squeeze(1)).long().cuda())

            probs_val = torch.softmax(out_val, dim=1).detach().cpu().numpy()
            y_val_pred = np.argmax(probs_val, axis=1)
            acc_val = np.sum(yval.ravel() == y_val_pred.ravel()) / yval.shape[0]

            if n_class == 2:
                auc_val = roc_auc_score(yval, probs_val[:, 1])
                aupr_val = average_precision_score(yval, probs_val[:, 1])
                logger.info(
                    f"Validation: Epoch {epoch}, train_loss:{loss.item():.4f}, "
                    f"val_loss:{val_loss.item():.4f}, acc_val:{acc_val * 100:.2f}, "
                    f"aupr_val:{aupr_val * 100:.2f}, auc_val:{auc_val * 100:.2f}"
                )
            else:
                prec_val = precision_score(yval.ravel(), y_val_pred, average='macro')
                rec_val = recall_score(yval.ravel(), y_val_pred, average='macro')
                f1_val = f1_score(yval.ravel(), y_val_pred, average='macro')
                logger.info(
                    f"Validation: Epoch {epoch}, train_loss:{loss.item():.4f}, "
                    f"val_loss:{val_loss.item():.4f}, acc_val:{acc_val * 100:.2f}, "
                    f"prec_val:{prec_val * 100:.2f}, rec_val:{rec_val * 100:.2f}, "
                    f"f1_val:{f1_val * 100:.2f}"
                )

            if val_loss.item() < best_loss_val:
                best_val_epoch = epoch
                best_loss_val = val_loss.item()
                save_time = str(int(time.time()))
                torch.save(model.state_dict(), f"{model_path}_{args.dataset}_{save_time}_{k}.pt")

    # Testing
    if save_time is None:
        logger.warning(f"No model saved for fold {k}; skipping testing.")
        continue

    model_filename = f"{model_path}_{args.dataset}_{save_time}_{k}.pt"
    if not os.path.exists(model_filename):
        logger.error(f"Model file missing: {model_filename}")
        continue

    model.eval()
    model.load_state_dict(torch.load(model_filename))
    logger.info(f"Loaded model for testing: {model_filename}")

    with torch.no_grad():
        st = time.time()
        out_test = evaluate_model_patch(
            model, Ptest_tensor, Ptest_mask_tensor, Ptest_static_tensor, Ptest_time_tensor,
            n_classes=n_class, batch_size=args.batch_size,
        )
        logger.info(f"test time per epoch :{time.time() - st:.2f}s")

        probs = torch.softmax(out_test, dim=1).detach().cpu().numpy()
        ypred = np.argmax(probs, axis=1)
        y_test_real = ytest.ravel()
        acc = np.sum(y_test_real == ypred) / y_test_real.shape[0]

        if n_class == 2:
            auc = roc_auc_score(y_test_real, probs[:, 1])
            aupr = average_precision_score(y_test_real, probs[:, 1])
            logger.info(f'Testing: AUROC = {auc * 100:.2f} | AUPRC = {aupr * 100:.2f} | Accuracy = {acc * 100:.2f}')
            auprc_arr.append(aupr * 100)
            auroc_arr.append(auc * 100)
        else:
            prec = precision_score(y_test_real, ypred, average='macro')
            rec = recall_score(y_test_real, ypred, average='macro')
            f1 = f1_score(y_test_real, ypred, average='macro')
            logger.info(f'Testing: PRECISION = {prec * 100:.2f} | RECALL = {rec * 100:.2f} | '
                        f'F1 = {f1 * 100:.2f} | Accuracy = {acc * 100:.2f}')
            precision_arr.append(prec * 100)
            recall_arr.append(rec * 100)
            f1_arr.append(f1 * 100)
        acc_arr.append(acc * 100)

# Aggregate
logger.info('------------------------------------------')
logger.info(f'args.dataset: {args.dataset}')
if n_class == 2:
    logger.info(f'AUPRC    = {np.mean(auprc_arr):.1f}±{np.std(auprc_arr):.1f}')
    logger.info(f'AUROC    = {np.mean(auroc_arr):.1f}±{np.std(auroc_arr):.1f}')
else:
    logger.info(f'PRECISION = {np.mean(precision_arr):.1f}±{np.std(precision_arr):.1f}')
    logger.info(f'RECALL    = {np.mean(recall_arr):.1f}±{np.std(recall_arr):.1f}')
    logger.info(f'F1        = {np.mean(f1_arr):.1f}±{np.std(f1_arr):.1f}')
logger.info(f'Accuracy  = {np.mean(acc_arr):.1f}±{np.std(acc_arr):.1f}')
