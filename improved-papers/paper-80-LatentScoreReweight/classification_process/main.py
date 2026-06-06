import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import warnings
import time
import json
import logging

from classification_process.classification_model import MLP
from classification_process.utils import timestamp, calculate_separate_acc, seed_everything
from classification_process.data_utils.dataset import Weighted_Dataset
from tabsyn.vae.model import Encoder_model
from utils_train import preprocess

warnings.filterwarnings('ignore')



D_TOKEN = 4

N_HEAD = 1
FACTOR = 32
NUM_LAYERS = 2


def train_classification(args): 

    args.log_path += f"/{args.dataname}/train/{args.comment}_{args.mode}_{timestamp()}"
    os.makedirs(f"{args.log_path}", exist_ok=True)
    if args.method == "common_error" or args.method == "error_diff" or args.method == "several_sigma_error_diff":
        args.comment += f"_{args.timestep_weight_criterion}"
        if args.temperature != None:
            args.comment += f"_{args.temperature}"
        if args.selected_several_sigma_indices != None:
            args.comment += f"_{args.selected_several_sigma_indices}"
    elif args.method == "discrete_sigma_error" or args.method == "discrete_sigma_error_diff":
        args.comment += f"_{args.single_sigma_index}T"
    gfile_stream = open(os.path.join(args.log_path, f'{args.comment}.txt'), 'w')
    handler = logging.StreamHandler(gfile_stream)
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel('INFO')
    logging.info(f"{args}")

    device = args.device
    dataname = args.dataname
    dataset_dir = f'/data/my_stored_dataset/{dataname}'
    with open(f'{dataset_dir}/info.json', 'r') as f:
        info = json.load(f)

    X_num, X_cat, categories, d_numerical, y = preprocess(dataset_dir, task_type = info['task_type'], 
                                                        concat=False, return_y=True)

    X_train_num, X_test_num = X_num
    X_train_cat, X_test_cat = X_cat
    y_train, y_test = y["train"], y["test"]

    X_train_num, X_test_num = torch.tensor(X_train_num).float(), torch.tensor(X_test_num).float()
    X_train_cat, X_test_cat = torch.tensor(X_train_cat), torch.tensor(X_test_cat)

    X_train_num = X_train_num.to(device)
    X_train_cat = X_train_cat.to(device)

    X_test_num = X_test_num.to(device)
    X_test_cat = X_test_cat.to(device)

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    encoder_save_path = f"{curr_dir}/ckpt/{dataname}/encoder.pt"
    pre_encoder = Encoder_model(NUM_LAYERS, d_numerical, categories, D_TOKEN, n_head = N_HEAD, factor = FACTOR).to(device)
    pre_encoder.load_state_dict(torch.load(encoder_save_path))
    pre_encoder.eval()

    print('Successfully load the model!')

    save_model_dir = f"{args.log_path}/saved_models"
    os.makedirs(save_model_dir, exist_ok=True)

    train_z = pre_encoder(X_train_num, X_train_cat).detach().cpu().numpy()
    train_z = train_z[:, 1:, :]
    B, num_tokens, token_dim = train_z.shape
    in_dim = num_tokens * token_dim
    
    train_z = train_z.reshape(B, in_dim)

    
    train_dataset = Weighted_Dataset(args, train_z, y_train, dataname=dataname, returned_attr=None,
                        use_weight=args.use_weight, weight_criterion=args.weight_criterion, 
                        temperature=args.temperature, weight_store_path=args.log_path)

    batch_size = 4096
    train_loader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 0,
    )

    test_z = pre_encoder(X_test_num, X_test_cat).detach().cpu().numpy()
    test_z = test_z[:, 1:, :]
    test_B_size = test_z.shape[0]
    test_z = test_z.reshape(test_B_size, -1)

    # Build test datasets with attribute info for worst-group checkpointing
    test_dataset_no_attr = Weighted_Dataset(args, test_z, y_test, dataname=dataname, returned_attr=None,
                        use_weight=False)
    test_loader = DataLoader(test_dataset_no_attr, batch_size=batch_size, shuffle=True)

    # Worst-group checkpointing: build attr-aware test loaders for SEX and race
    wg_attrs = ["SEX", "race"]
    wg_test_loaders = {}
    for wg_attr in wg_attrs:
        wg_ds = Weighted_Dataset(args, test_z, y_test, dataname=dataname, returned_attr=wg_attr,
                                 use_weight=False)
        wg_test_loaders[wg_attr] = DataLoader(wg_ds, batch_size=batch_size, shuffle=False)

    num_epochs = 1000

    classification_model = MLP(train_z.shape[1]).to(device)
    num_params = sum(p.numel() for p in classification_model.parameters())
    logging.info(f"the number of parameters: {num_params}")

    optimizer = torch.optim.Adam(classification_model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=20, verbose=True)
    if args.use_weight or args.non_param_dro:
        train_criterion = nn.CrossEntropyLoss(reduction='none')
    else:
        train_criterion = nn.CrossEntropyLoss()
    classification_model.train()

    best_worst_group_acc = -1.0
    patience = 0
    for epoch in range(num_epochs):
        
        batch_loss = 0.0
        len_input = 0
        for i, batch in enumerate(train_loader):
            if args.use_weight:
                latent, y, weight = batch
                weight = weight.to(device)
            else:
                latent, y = batch
            latent = latent.float().to(device)
            y = y.long().to(device)

            y_hat = classification_model(latent)
            
            if args.method == "error_diff" or args.method == "ERM" \
                or args.method == "common_error" or args.method == "discrete_sigma_error" \
                or args.method == "discrete_sigma_error_diff" or args.method == "several_sigma_error_diff":
                loss = torch.mean(train_criterion(y_hat, y) * weight) if args.use_weight else train_criterion(y_hat, y)
                
                batch_loss += loss.item() * len(latent)
                len_input += len(latent)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        curr_loss = batch_loss / len_input
        scheduler.step(curr_loss)

        if epoch % 10 == 0:
            logging.info(f"Epoch {epoch}, training loss: {curr_loss:.2f}")

            # Compute worst-group accuracy across SEX and race attributes
            classification_model.eval()
            with torch.no_grad():
                worst_accs = []
                for wg_attr, wg_loader in wg_test_loaders.items():
                    all_pred, all_y, all_attr = [], [], []
                    for latent_b, y_b, attr_b in wg_loader:
                        latent_b = latent_b.float().to(device)
                        y_b = y_b.long().to(device)
                        y_hat_b = classification_model(latent_b)
                        _, pred_b = torch.max(y_hat_b, 1)
                        all_pred.append(pred_b.cpu().numpy())
                        all_y.append(y_b.cpu().numpy())
                        all_attr.append(attr_b.numpy())
                    all_pred = np.concatenate(all_pred).reshape(-1)
                    all_y = np.concatenate(all_y).reshape(-1)
                    all_attr = np.concatenate(all_attr).reshape(-1)
                    # Compute worst of 4 subgroup accs
                    subgroup_accs = []
                    for a_val in [0, 1]:
                        for y_val in [0, 1]:
                            idx = np.where((all_attr == a_val) & (all_y == y_val))[0]
                            if len(idx) > 0:
                                subgroup_accs.append((all_pred[idx] == all_y[idx]).mean())
                    worst_accs.append(min(subgroup_accs))
                avg_worst_group_acc = np.mean(worst_accs)
            classification_model.train()

            logging.info(f"Epoch {epoch}, worst_group_acc: {avg_worst_group_acc:.4f}")

            if avg_worst_group_acc > best_worst_group_acc:
                logging.info(f"Update model in epoch {epoch}, worst_group_acc {avg_worst_group_acc:.4f} > {best_worst_group_acc:.4f}")
                best_worst_group_acc = avg_worst_group_acc
                patience = 0
                torch.save(classification_model.state_dict(), f'{save_model_dir}/{args.method}_best.pt')
            else:
                patience += 1
                if patience == 40:
                    logging.info(f'Early stopping in epoch {epoch}')
                    break

def test_classification(args):
    args.log_path += f"/{args.dataname}/test/{args.eval_attribute}/{args.comment}_{args.mode}_{timestamp()}"
    os.makedirs(f"{args.log_path}", exist_ok=True)
    # args.comment += f"_{args.eval_attribute}"
    if args.method == "common_error" or args.method == "error_diff" or args.method == "several_sigma_error_diff":
        args.comment += f"_{args.timestep_weight_criterion}"
        if args.temperature != None:
            args.comment += f"_{args.temperature}"
        if args.selected_several_sigma_indices != None:
            args.comment += f"_{args.selected_several_sigma_indices}"
    elif args.method == "discrete_sigma_error" or args.method == "discrete_sigma_error_diff":
        args.comment += f"_{args.single_sigma_index}T"
    gfile_stream = open(os.path.join(args.log_path, f'{args.comment}.txt'), 'w')
    handler = logging.StreamHandler(gfile_stream)
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel('INFO')
    logging.info(f"{args}")

    device = args.device
    dataname = args.dataname
    dataset_dir = f'/data/my_stored_dataset/{dataname}'
    with open(f'{dataset_dir}/info.json', 'r') as f:
        info = json.load(f)

    X_num, X_cat, categories, d_numerical, y = preprocess(dataset_dir, task_type = info['task_type'], 
                                                        concat=False, return_y=True)

    _, X_test_num = X_num
    _, X_test_cat = X_cat
    y_test = y["test"]

    X_test_num = torch.tensor(X_test_num).float().to(device)
    X_test_cat = torch.tensor(X_test_cat).to(device)

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    encoder_save_path = f"{curr_dir}/ckpt/{dataname if dataname[:3] != 'ACS' else dataname[:13]}/encoder.pt"
    pre_encoder = Encoder_model(NUM_LAYERS, d_numerical, categories, D_TOKEN, n_head = N_HEAD, factor = FACTOR).to(device)
    pre_encoder.load_state_dict(torch.load(encoder_save_path))
    pre_encoder.eval()

    print('Successfully load the encoder!')

    test_z = pre_encoder(X_test_num, X_test_cat).detach().cpu().numpy()
    test_z = test_z[:, 1:, :]
    B, num_tokens, token_dim = test_z.shape
    in_dim = num_tokens * token_dim
    
    # prepare test dataset, which returns attribute
    test_z = test_z.reshape(B, in_dim)
    test_dataset = Weighted_Dataset(args, test_z, y_test, dataname=dataname, returned_attr=args.eval_attribute,
                        use_weight=False)

    batch_size = 4096
    test_loader = DataLoader(
        test_dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 0,
    )

    # load specific model
    classification_model = MLP(test_z.shape[1]).to(device)
    num_params = sum(p.numel() for p in classification_model.parameters())
    logging.info(f"the number of parameters: {num_params}")
    classification_model.load_state_dict(torch.load(args.evaluated_model_path))
    classification_model.eval()
    

    # test, calculate average acc, worst acc, EO
    test_criterion = nn.CrossEntropyLoss()
    total_loss, total_acc = 0.0, 0
    len_input = 0
    all_predict, all_y, all_attr = [], [], []
    with torch.no_grad():
        for i, (latent, y, attr) in enumerate(test_loader):
            latent = latent.float().to(device)
            y = y.long().to(device)

            y_hat = classification_model(latent)
            loss = test_criterion(y_hat, y)

            total_loss += loss.item() * len(latent)
            len_input += len(latent)

            _, predicted_test = torch.max(y_hat, 1)
            total_acc += (predicted_test == y).sum().cpu().numpy()
            all_predict.append(predicted_test.cpu().numpy())
            all_y.append(y.cpu().numpy())
            all_attr.append(attr.numpy())

    avg_loss = total_loss / len_input
    avg_acc = total_acc / len_input

    logging.info(f"Final test loss: {avg_loss:.3f}; Final test acc: {avg_acc:.3f}")
    all_predict = np.concatenate(all_predict, axis=0).reshape(-1)
    all_y = np.concatenate(all_y, axis=0).reshape(-1)
    all_attr = np.concatenate(all_attr, axis=0).reshape(-1)
    calculate_separate_acc(all_attr, all_y, all_predict, avg_acc)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training Classification Model')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
    parser.add_argument('--log_path', type=str, required=True)
    parser.add_argument('--mode', required=True, type=str)
    parser.add_argument("--comment", type=str, required=True)
    
    parser.add_argument('--use_weight', action="store_true")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--weight_criterion", type=str)

    # method
    parser.add_argument('--method', type=str, required=True)
    
    # DRO baseline utils
    parser.add_argument('--non_param_dro', action="store_true")
    parser.add_argument('--chi2_eta', type=float, default=None)
    parser.add_argument('--cvar_alpha', type=float, default=None)
    parser.add_argument('--kappa', type=float, default=None)
    parser.add_argument('--tau', type=float, default=None)
    parser.add_argument('--alpha', type=float, default=1)

    # using when training
    parser.add_argument("--error_reflection", type=str, choices=["linear", "softmax"])
    parser.add_argument('--timestep_weight_criterion', type=str, choices=["EDM", "down", "up", "updown", "downup", "avg"])

    # used for single T training
    parser.add_argument("--single_sigma_index", type=int)
    # used for several T training
    parser.add_argument("--selected_several_sigma_indices", nargs='+', type=int)

    # using when testing
    parser.add_argument("--eval_attribute", type=str)
    parser.add_argument("--evaluated_model_path", type=str)
    parser.add_argument("--seed", type=int, default=42)

    # using for specifying baseline methods
    parser.add_argument("--baseline_method", type=str)

    # used for ablation of K
    parser.add_argument("--K", type=int, default=32)

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'
        
    seed_everything(args.seed)
    args.log_path += f"_{args.seed}"

    if args.mode == "train":
        train_classification(args)
    elif args.mode == "test":
        test_classification(args)