import sys, os, time, numpy as np, torch, random
sys.path.insert(0, "/repo")

from config.basic import ConfigBasic
from data.get_datasets_BIQA import get_datasets_BIQA
from networks.util import prepare_model
from utils.loss_util import ConOrdLoss
from utils.util import to_np, make_dir, cal_srocc_plcc, AverageMeter
from utils.comparison_utils import find_kNN
from copy import deepcopy

N_SPLITS = 3
EPOCHS = 30
LR = 5e-7
WD = 1e-4

def build_cfg(split_seed):
    cfg = ConfigBasic()
    cfg.dataset = "bid"
    cfg.data_root = "/tmp/bid_local"
    cfg.logscale = False
    cfg.set_biqa_dataset()
    cfg.device = torch.device("cuda:0")
    cfg.batch_size = 32
    cfg.test_batch_size = 1000
    cfg.num_workers = 0
    cfg.epochs = EPOCHS
    cfg.learning_rate = LR
    cfg.weight_decay = WD
    cfg.temp = 0.07
    cfg.epsilon = 1e-7
    cfg.tau = 0
    cfg.label_diff = "l2"
    cfg.similarity_type = "L2"
    cfg.metric = "L2"
    cfg.k = 10
    cfg.margin = 0.05
    cfg.lr_decay_rate = 0.1
    cfg.warmup_epochs = 2
    cfg.split_seed = split_seed
    cfg.model = "ConOrd"
    cfg.backbone = "vitB16"
    cfg.ref_mode = "flex"
    cfg.ref_point_num = 60
    cfg.fiducial_point_num = 60
    cfg.start_norm = True
    cfg.drct_wieght = 0
    make_dir("/tmp/multi_split_results")
    return cfg

srcc_all = []
plcc_all = []

for split_idx in range(N_SPLITS):
    split_seed = 42 + split_idx
    cfg = build_cfg(split_seed)
    random_seed = split_seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    random.seed(random_seed)

    print("=== Split %d (seed=%d) ===" % (split_idx, split_seed), flush=True)
    loader_dict = get_datasets_BIQA(cfg)
    print("Data: train=%d, test=%d" % (len(loader_dict["train_for_val"].dataset), len(loader_dict["val"].dataset)), flush=True)

    model = prepare_model(cfg)
    model = model.to(cfg.device)

    param_groups = []
    for key, value in dict(model.named_children()).items():
        param_groups += [{"params": value.parameters(), "lr": cfg.learning_rate}]
    param_groups += [{"params": model.ref_points, "lr": cfg.learning_rate * 10}]

    optimizer = torch.optim.AdamW(params=param_groups, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=cfg.warmup_epochs
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, cfg.epochs - cfg.warmup_epochs, eta_min=cfg.learning_rate * cfg.lr_decay_rate
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[cfg.warmup_epochs]
    )
    criterion = ConOrdLoss(label_diff=cfg.label_diff, feature_sim=cfg.similarity_type, temperature=cfg.temp)

    best_srcc = 0.0
    best_plcc = 0.0
    for epoch in range(cfg.epochs):
        model.train()
        losses = AverageMeter()
        for idx, (images, _, ranks, _) in enumerate(loader_dict["train"]):
            images = images.to(cfg.device)
            ranks = ranks.to(cfg.device)
            bsz = ranks.shape[0] // 2
            features = model.encoder(images)
            features = torch.nn.functional.normalize(features, dim=-1)
            # IDEA-11: Gaussian noise regularization (training only)
            if model.training:
                features = features + torch.randn_like(features) * 0.02
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            total_loss = criterion(features, ranks, cfg)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            losses.update(total_loss.item(), ranks.size(0))
        scheduler.step()

        model.eval()
        embs_train, embs_test = [], []
        with torch.no_grad():
            for x, _, _, _ in loader_dict["train_for_val"]:
                embs_train.append(model.encoder(x.to(cfg.device)).cpu())
            for x, _, _, _ in loader_dict["val"]:
                embs_test.append(model.encoder(x.to(cfg.device)).cpu())
        embs_train = torch.cat(embs_train).to(cfg.device)
        embs_test = torch.cat(embs_test).to(cfg.device)

        train_labels = np.array(loader_dict["train_for_val"].dataset.mos)
        test_labels = np.array(loader_dict["val"].dataset.mos)
        vals, inds = find_kNN(embs_test, embs_train, k=cfg.k, metric=cfg.metric)
        inds = np.squeeze(to_np(inds), 0)
        if inds.ndim == 1:
            inds = inds[np.newaxis, :]
        nn_labels = train_labels[inds[:, :cfg.k]]
        preds = np.mean(nn_labels, axis=-1)
        srcc, plcc = cal_srocc_plcc(preds, test_labels)

        if epoch % 10 == 0:
            print("  Epoch %2d: SRCC=%.4f, PLCC=%.4f" % (epoch, srcc, plcc), flush=True)
        if srcc > best_srcc:
            best_srcc = srcc
            best_plcc = plcc

    print("  Split %d best: SRCC=%.4f, PLCC=%.4f" % (split_idx, best_srcc, best_plcc), flush=True)
    srcc_all.append(best_srcc)
    plcc_all.append(best_plcc)

    del model, optimizer, scheduler, criterion, loader_dict
    torch.cuda.empty_cache()

print("=== FINAL ===", flush=True)
print("SRCC: %s" % [round(v,4) for v in srcc_all], flush=True)
print("PLCC: %s" % [round(v,4) for v in plcc_all], flush=True)
print("SRCC median: %.4f" % np.median(srcc_all), flush=True)
print("SRCC mean+/-std: %.4f+/-%.4f" % (np.mean(srcc_all), np.std(srcc_all)), flush=True)
print("PLCC median: %.4f" % np.median(plcc_all), flush=True)
