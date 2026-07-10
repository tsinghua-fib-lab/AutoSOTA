import torch
import torch.nn as nn
from models import MAEModel
from utils.others import set_global_seed, log_mae_recon
from utils.lin_probe import run_linear_probe
from utils.shuffle import patchify, unpatchify
from utils.losses import mae_loss
from loaders_wm import ColoredMNISTLoader, ColoredFMNISTLoader, PatchedCIFARLoader
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse
import yaml
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encdim", type=int, default=192)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--task", type=str, default="mnist")
    args = parser.parse_args()
    encdim = args.encdim
    seed = args.seed

    IMG_SIZE = 32
    PATCH_SIZE = 4
    PATCH_CHANNEL = 3
    BATCH_SIZE = 512
    NUM_CLASSES = 10
    LR = 1e-3
    WEIGHT_DECAY = 0.05
    MASK_RATIO = 0.75
    EPOCHS = 100
    NUM_WORKERS = 6
    ENC_DIM = encdim
    SEED = seed
    os.makedirs(f"./results", exist_ok=True)
    os.makedirs(f"./logs", exist_ok=True)
    os.makedirs(f"./ckpt", exist_ok=True)

    print(f"Running MAE on {args.task} with encoder dimension: {encdim} and seed: {seed}")

    writer = SummaryWriter(log_dir=f'./logs/mae_{args.task.lower()}_wm_logs_{encdim}_{seed}')

    class GrayToRGB:
        def __call__(self, x):
            # x: (1, H, W)
            return x.repeat(3, 1, 1)

    set_global_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if "mnist" in args.task.lower():
        PATCH_CHANNEL = 3
        if args.task.lower() == "mnist":
            mean = [0.1307]
            std = [0.3081]
            Dts = datasets.MNIST
            loader = ColoredMNISTLoader(batch_size=BATCH_SIZE, device=device)
        elif args.task.lower() == "fmnist":
            mean = [0.2860]
            std = [0.3530]
            Dts = datasets.FashionMNIST
            loader = ColoredFMNISTLoader(batch_size=BATCH_SIZE, device=device)
    elif "cifar10" in args.task.lower():
        PATCH_CHANNEL = 3
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470, 0.2435, 0.2616]
        Dts = datasets.CIFAR10
        loader = PatchedCIFARLoader(batch_size=BATCH_SIZE, device=device)
    else:
        raise NotImplementedError(f"Dataset {args.task} not supported")

    

    print(f"Running MAE on {args.task} with encoder dimension: {encdim} and seed: {seed}")


    if "mnist" in args.task.lower():
        transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        GrayToRGB(),
        transforms.Normalize(mean=mean,
                            std=std),
                            ])
    elif "cifar10" in args.task.lower():
        transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,
                            std=std),
                            ])
    ds = Dts(root='../data', train=True, download=True,
                                transform=transform)

    tr_ds = Dts(root='../data', train=True, download=True,
                                transform=transform)
    te_ds = Dts(root='../data', train=False, download=True,
                                transform=transform)
    
    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    te_loader = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = MAEModel(img_size=IMG_SIZE,
                        patch_size=PATCH_SIZE,
                        patch_ch=PATCH_CHANNEL,
                        enc_dim=ENC_DIM,
                        mask_ratio=MASK_RATIO)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_lp_acc = 0.0
    for e in range(EPOCHS):
        model.train()
        losses = []
        for _ in tqdm(range(len(ds) // BATCH_SIZE)):
            img = loader.get_mae_batch()

            pred, mask = model(img)
            

            loss = mae_loss(img, pred, mask, PATCH_SIZE, normalize=False)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        avg_loss = sum(losses) / len(losses)
        print(f"Epoch [{e+1}/{EPOCHS}] Loss: {avg_loss:.4f}")
        writer.add_scalar("Loss/train", avg_loss, e + 1)
        
        torch.save(model.encoder.state_dict(), f'./ckpt/mae_{args.task.lower()}_wm_encoder_{encdim}_{seed}_last.pth')
        torch.save(model.decoder.state_dict(), f'./ckpt/mae_{args.task.lower()}_wm_decoder_{encdim}_{seed}_last.pth')

        # linear probe
        lp_acc = 0.0
        if (e+1) % 10 == 0:
            model.eval()
            lp_acc = run_linear_probe(
            encoder=model.encoder,
            train_loader=tr_loader, 
            val_loader=te_loader,
            num_classes=NUM_CLASSES,
            device=device,
            probe_epochs=5,
            lr=1e-3
            )
            print(f"Linear probe accuracy: {lp_acc:.4f}")
            if lp_acc > best_lp_acc:
                best_lp_acc = lp_acc
                print(f"saving best model with acc {best_lp_acc} at {e+1} epoch!")
                torch.save(model.encoder.state_dict(),
            f'./ckpt/mae_{args.task.lower()}_wm_encoder_{encdim}_{seed}_best.pth')
                torch.save(model.decoder.state_dict(),
            f'./ckpt/mae_{args.task.lower()}_wm_decoder_{encdim}_{seed}_best.pth')
            writer.add_scalar("Accuracy/linear_probe", lp_acc, e + 1)
        
        
        # log reconstructed images
        model.eval()
        with torch.no_grad():
            img = loader.get_mae_batch()[:1]
            pred, mask = model(img)

            tgt = patchify(img, PATCH_SIZE)
            recon_patches = tgt * (1 - mask.unsqueeze(-1)) + pred * mask.unsqueeze(-1)
            recon_img = unpatchify(recon_patches, IMG_SIZE, PATCH_CHANNEL)
            mask_img = unpatchify(mask.repeat(1,1, PATCH_CHANNEL * PATCH_SIZE**2), IMG_SIZE, PATCH_CHANNEL)

        log_mae_recon(writer, e + 1, img.cpu(), recon_img.cpu(), mask_img=mask_img.cpu())

    writer.close()
    
    with open(f"./results/mae_{args.task.lower()}_wm_logs_{encdim}_{seed}.yaml", "w") as f:
        yaml.dump({
            "best_linear_probe_acc": float(best_lp_acc),
            "last_linear_probe_acc": float(lp_acc)
        }, f)
        