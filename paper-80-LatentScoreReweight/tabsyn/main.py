import os
import torch

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import warnings
import time
import pandas as pd

from tqdm import tqdm
from tabsyn.model import MLPDiffusion, Model
from tabsyn.latent_utils import get_input_train

warnings.filterwarnings('ignore')

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def main(args): 
    device = args.device

    seed_everything(args.seed)

    train_z, _, _, ckpt_path, _ = get_input_train(args)
    if args.train_diffusion_model_class == 0:
        df = pd.read_csv(f"/data/my_stored_dataset/{args.dataname}/train.csv")
        if args.dataname == "adult":
            y = df["income"].values.reshape(-1)
            indices = torch.tensor(y == " <=50K")
        elif args.dataname == "shoppers":
            y = df["Revenue"].values.reshape(-1)
            indices = torch.tensor(y == False)
        elif args.dataname == "default":
            y = df["default payment next month"].values.reshape(-1)
            indices = torch.tensor(y == 0)
        elif args.dataname == "bank":
            y = df["y"].values.reshape(-1)
            indices = torch.tensor(y == "no")
        elif args.dataname[:10] == "ACS_income":
            y = df["PINCP"].values.reshape(-1)
            indices = torch.tensor(y == False)
        elif args.dataname[:4] == "taxi":
            y = df["trip_duration"].values.reshape(-1)
            indices = torch.tensor(y == 0)

        train_z = train_z[indices]
        print(f"train diffusion model with class 0, {train_z.shape[0]} samples")
    elif args.train_diffusion_model_class == 1:
        df = pd.read_csv(f"/data/my_stored_dataset/{args.dataname}/train.csv")
        if args.dataname == "adult":
            y = df["income"].values.reshape(-1)
            indices = torch.tensor(y == " >50K")
        elif args.dataname == "shoppers":
            y = df["Revenue"].values.reshape(-1)
            indices = torch.tensor(y == True)
        elif args.dataname == "default":
            y = df["default payment next month"].values.reshape(-1)
            indices = torch.tensor(y == 1)
        elif args.dataname == "bank":
            y = df["y"].values.reshape(-1)
            indices = torch.tensor(y == "yes")
        elif args.dataname[:10] == "ACS_income":
            y = df["PINCP"].values.reshape(-1)
            indices = torch.tensor(y == True)
        elif args.dataname[:4] == "taxi":
            y = df["trip_duration"].values.reshape(-1)
            indices = torch.tensor(y == 1)

        train_z = train_z[indices]
        print(f"train diffusion model with class 1, {train_z.shape[0]} samples")
    else:
        assert args.train_diffusion_model_class == 2
        print("train diffusion model with all samples")

    print(ckpt_path)

    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    in_dim = train_z.shape[1] 

    mean, std = train_z.mean(0), train_z.std(0)

    train_z = (train_z - mean) / 2
    train_data = train_z


    batch_size = 4096
    train_loader = DataLoader(
        train_data,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 4,
    )

    num_epochs = 10000 + 1

    denoise_fn = MLPDiffusion(in_dim, 1024).to(device)
    print(denoise_fn)

    num_params = sum(p.numel() for p in denoise_fn.parameters())
    print("the number of parameters", num_params)

    model = Model(denoise_fn = denoise_fn, hid_dim = train_z.shape[1]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=20, verbose=True)

    model.train()

    best_loss = float('inf')
    patience = 0
    start_time = time.time()
    for epoch in range(num_epochs):
        
        pbar = tqdm(train_loader, total=len(train_loader))
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}")

        batch_loss = 0.0
        len_input = 0
        for batch in pbar:
            inputs = batch.float().to(device)
            loss = model(inputs)
        
            loss = loss.mean()

            batch_loss += loss.item() * len(inputs)
            len_input += len(inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"Loss": loss.item()})

        curr_loss = batch_loss/len_input
        scheduler.step(curr_loss)

        if curr_loss < best_loss:
            best_loss = curr_loss
            patience = 0
            torch.save(model.state_dict(), f'{ckpt_path}/model_{args.train_diffusion_model_class}.pt')
        else:
            patience += 1
            if patience == 500:
                print('Early stopping')
                break

        if epoch % 1000 == 0:
            torch.save(model.state_dict(), f'{ckpt_path}/model_{args.train_diffusion_model_class}_{epoch}.pt')

    end_time = time.time()
    print('Time: ', end_time - start_time)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training of TabSyn')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'