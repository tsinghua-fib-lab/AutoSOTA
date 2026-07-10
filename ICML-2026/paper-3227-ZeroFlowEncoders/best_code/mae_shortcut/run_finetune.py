import os
from models import ZFEModel, Classifier
from utils.others import set_global_seed

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import argparse

IMG_SIZE = 96
PATCH_SIZE = 8
PATCH_CHANNEL = 3
BATCH_SIZE = 128
NUM_CLASSES = 10
LR = 1e-3
WEIGHT_DECAY = 0.05
MASK_RATIO = 0.75
EPOCHS = 50
NUM_WORKERS = 6
ENC_DIM = 192
SEED = 42

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=str, default='zfe', help='job type: zfe or mae')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    set_global_seed(SEED)

    model = ZFEModel(
    img_size=IMG_SIZE,
    patch_size=PATCH_SIZE,
    enc_dim=ENC_DIM,
    mask_ratio=MASK_RATIO,
    )
    model.encoder.load_state_dict(torch.load(f'{args.job}_stl10_encoder_192_37_last.pth', map_location='cpu'), strict=True)
    model = Classifier(model.encoder, num_classes=NUM_CLASSES).to(device)

    # MAE-style finetune: use CLS
    model.encoder.global_pool = False


    # ===== data =====
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4467, 0.4398, 0.4066],
            std=[0.2241, 0.2215, 0.2239],
        ),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4467, 0.4398, 0.4066],
            std=[0.2241, 0.2215, 0.2239],
        ),
    ])

    tr_ds = datasets.STL10(root='../data', split='train', transform=train_transform)
    te_ds = datasets.STL10(root='../data', split='test', transform=test_transform)

    tr_loader = DataLoader(
        tr_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS)
    te_loader = DataLoader(
        te_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS)

    optimizer = torch.optim.AdamW(
    [
        {"params": model.encoder.parameters(), "lr": 1e-4},
        {"params": model.classifier.parameters(), "lr": 1e-3},
    ],
    weight_decay=WEIGHT_DECAY
    )

    loss_fn = torch.nn.CrossEntropyLoss()
    acc_fn = lambda logit, label: torch.mean((logit.argmax(dim=-1) == label).float())

    best_val_acc = 0
    optimizer.zero_grad()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        accs = []
        for img, label in tqdm(iter(tr_loader)):
            img = img.to(device)
            label = label.to(device)
            logits = model(img)
            loss = loss_fn(logits, label)
            acc = acc_fn(logits, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
            accs.append(acc.item())
        avg_train_loss = sum(losses) / len(losses)
        avg_train_acc = sum(accs) / len(accs)
        print(f'In epoch {epoch}, average training loss is {avg_train_loss}, average training acc is {avg_train_acc}.')

        model.eval()
        with torch.no_grad():
            losses = []
            accs = []
            for img, label in tqdm(iter(te_loader)):
                img = img.to(device)
                label = label.to(device)
                logits = model(img)
                loss = loss_fn(logits, label)
                acc = acc_fn(logits, label)
                losses.append(loss.item())
                accs.append(acc.item())
            avg_val_loss = sum(losses) / len(losses)
            avg_val_acc = sum(accs) / len(accs)
            print(f'In epoch {epoch}, average validation loss is {avg_val_loss}, average validation acc is {avg_val_acc}.')
            if avg_val_acc > best_val_acc:
                best_val_acc = avg_val_acc
                print(f'saving best model with acc {best_val_acc} at {epoch} epoch!')
                torch.save(model.state_dict(), f'{args.job}_stl10_best_encoder_{ENC_DIM}_{SEED}.pth')

