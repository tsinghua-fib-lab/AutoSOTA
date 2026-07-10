import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import cross_val_score


# def extractor(encoder, dl, device, bn= None):
#     encoder.eval()
#     if bn is not None:
#         bn.eval()
    
#     feats = []
#     labels = []
#     with torch.no_grad():
#         for imgs, y in dl:
#             imgs = imgs.to(device)
#             feat = encoder.forward_features(imgs) # token-level features (B, n_patches, emb_dim)
#             # feat = feat.mean(dim=1)  # global average pooling: img level features for linear probe
#             if bn is not None:
#                 feat = bn(feat)

#             feats.append(feat.cpu().numpy())
#             labels.append(y.numpy())
    
#     feats = np.concatenate(feats, axis=0)
#     labels = np.concatenate(labels, axis=0)
#     return feats, labels
    
# def lin_probe(train_ds, test_ds, encoder, device, bn=None):
#     if bn is not None:
#         print("Using BN layer in linear probe")
#     tr_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
#     te_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

#     imgs,_ = next(iter(tr_loader))
#     with torch.no_grad():
#         tokens = encoder.forward_features(imgs.to(device))
#         D = tokens.shape[-1]

#     if bn is not None:
#         bn = bn(D).to(device)

#     X_train, y_train = extractor(encoder, tr_loader, device, bn)
#     X_test, y_test = extractor(encoder, te_loader, device, bn)

#     clf = LogisticRegression(max_iter=4000)
#     clf.fit(X_train, y_train)

#     acc = clf.score(X_test, y_test)
#     return acc

def run_linear_probe(
    encoder,
    train_loader,
    val_loader,
    num_classes,
    device,
    probe_epochs=5,
    lr=1e-3,
):
    """
    Zero-side-effect linear probe.
    The input encoder is NEVER modified.
    """

    import copy

    # 1) clone encoder (no side effects)
    encoder_lp = copy.deepcopy(encoder).to(device)
    encoder_lp.eval()

    # make sure linear probe uses GAP (MAE style)
    if hasattr(encoder_lp, "global_pool"):
        encoder_lp.global_pool = True
        print("Using global average pooling for linear probe.")

    # freeze cloned encoder
    for p in encoder_lp.parameters():
        p.requires_grad = False

    # 2) build linear head
    classifier = nn.Linear(encoder_lp.emb_dim, num_classes).to(device)

    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=lr,
        weight_decay=0.0
    )
    criterion = nn.CrossEntropyLoss()

    # 3) train linear classifier

    for _ in range(probe_epochs):
        classifier.train()
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                feats = encoder_lp.forward_features(imgs)  # (B, emb_dim)

            logits = classifier(feats)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 4) evaluate
    classifier.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            feats = encoder_lp.forward_features(imgs)
            logits = classifier(feats)
            preds = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


def run_linear_probe_wm(
    encoder,
    train_loader,
    val_loader,
    num_classes,
    device,
    probe_epochs=5,
    lr=1e-3,
):
    """
    Zero-side-effect linear probe.
    The input encoder is NEVER modified.
    """

    import copy

    # 1) clone encoder (no side effects)
    encoder_lp = copy.deepcopy(encoder).to(device)
    encoder_lp.eval()

    # make sure linear probe uses GAP (MAE style)
    if hasattr(encoder_lp, "global_pool"):
        encoder_lp.global_pool = True
        print("Using global average pooling for linear probe.")

    # freeze cloned encoder
    for p in encoder_lp.parameters():
        p.requires_grad = False

    # 2) build linear head
    classifier = nn.Linear(encoder_lp.emb_dim, num_classes).to(device)

    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=lr,
        weight_decay=0.0
    )
    criterion = nn.CrossEntropyLoss()

    # 3) train linear classifier

    for _ in range(probe_epochs):
        classifier.train()
        for i_ in range(len(train_loader.data)//train_loader.batch_size):
            imgs, labels = train_loader.get_mae_wmlp_batch()
            imgs = imgs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                feats = encoder_lp.forward_features(imgs)  # (B, emb_dim)

            logits = classifier(feats)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 4) evaluate
    classifier.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for _ in range(len(val_loader.test_data)//val_loader.batch_size):
            imgs, labels = val_loader.get_mae_test_batch()
            imgs = imgs.to(device)
            labels = labels.to(device)

            feats = encoder_lp.forward_features(imgs)
            logits = classifier(feats)
            preds = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total