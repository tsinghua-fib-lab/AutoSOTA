#!/usr/bin/env python3
"""
Apply SOTA optimization ideas to train_decentralized.py.
Usage: python3 apply_idea.py <idea_id> [--undo]
"""
import sys, os, shutil

SRC = "/repo/train_decentralized.py"
BACKUP = "/repo/train_decentralized.py.baseline"

def ensure_backup():
    if not os.path.exists(BACKUP):
        shutil.copy2(SRC, BACKUP)
        print("Backup saved to " + BACKUP)

def restore_baseline():
    if os.path.exists(BACKUP):
        shutil.copy2(BACKUP, SRC)
        print("Restored from " + BACKUP)
    else:
        print("No backup found!")

def read_src():
    with open(SRC) as f:
        return f.read()

def write_src(content):
    with open(SRC, "w") as f:
        f.write(content)

def apply_param01():
    """PARAM-01: LR warmup + cosine decay schedule."""
    content = read_src()

    # Add scheduler creation after optimizers are created
    old = ("    optimizers = [optim.SGD(m.parameters(), lr=args.lr, weight_decay=args.weight_decay)\n"
           "                  for m in models]\n"
           "    criterion = nn.CrossEntropyLoss()\n"
           "    scaler = torch.cuda.amp.GradScaler()")

    new = ("    optimizers = [optim.SGD(m.parameters(), lr=args.lr, weight_decay=args.weight_decay)\n"
           "                  for m in models]\n"
           "    # PARAM-01: LR warmup + cosine decay (proportional warmup)\n"
           "    warmup_rounds = max(10, args.n_rounds // 10)\n"
           "    total_rounds = args.n_rounds\n"
           "    def lr_lambda(r):\n"
           "        if r < warmup_rounds:\n"
           "            return (r + 1) / max(warmup_rounds, 1)\n"
           "        progress = (r - warmup_rounds) / max(total_rounds - warmup_rounds, 1)\n"
           "        return max(0.5 * (1 + math.cos(math.pi * progress)), 1e-5 / args.lr)\n"
           "    schedulers = [optim.lr_scheduler.LambdaLR(opt, lr_lambda) for opt in optimizers]\n"
           "    criterion = nn.CrossEntropyLoss()\n"
           "    scaler = torch.cuda.amp.GradScaler()")

    if old not in content:
        print("ERROR: PARAM-01 target not found!")
        return False
    content = content.replace(old, new)

    # Add scheduler step after mixing
    old2 = "        sim_time += bcd_sec"
    new2 = ("        sim_time += bcd_sec\n"
            "        # PARAM-01: step schedulers\n"
            "        for sched in schedulers:\n"
            "            sched.step()")

    if old2 not in content:
        print("ERROR: PARAM-01 scheduler step target not found!")
        return False
    content = content.replace(old2, new2)

    write_src(content)
    print("PARAM-01 applied: LR warmup + cosine decay")
    return True

def apply_code03():
    """CODE-03: Gradient clipping."""
    content = read_src()

    old = ("                scaler.scale(loss).backward()\n"
           "                scaler.step(opt)\n"
           "                scaler.update()")

    new = ("                scaler.scale(loss).backward()\n"
           "                # CODE-03: gradient clipping\n"
           "                scaler.unscale_(opt)\n"
           "                torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=5.0)\n"
           "                scaler.step(opt)\n"
           "                scaler.update()")

    if old not in content:
        print("ERROR: CODE-03 target not found!")
        return False
    content = content.replace(old, new)

    write_src(content)
    print("CODE-03 applied: gradient clipping (max_norm=5.0)")
    return True

def apply_algo04():
    """ALGO-04: Label smoothing."""
    content = read_src()

    old = "    criterion = nn.CrossEntropyLoss()"
    new = "    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)"

    if old not in content:
        print("ERROR: ALGO-04 target not found!")
        return False
    content = content.replace(old, new)

    write_src(content)
    print("ALGO-04 applied: label smoothing (epsilon=0.1)")
    return True

def apply_param02():
    """PARAM-02: SGD with momentum."""
    content = read_src()

    old = "optimizers = [optim.SGD(m.parameters(), lr=args.lr, weight_decay=args.weight_decay)"
    new = "optimizers = [optim.SGD(m.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)"

    if old not in content:
        print("ERROR: PARAM-02 target not found!")
        return False
    content = content.replace(old, new)

    write_src(content)
    print("PARAM-02 applied: SGD with momentum=0.9")
    return True

def apply_algo01():
    """ALGO-01: Mixup data augmentation."""
    content = read_src()

    old = ("    cropped.sub_(mean).div_(std)\n"
           "\n"
           "    return cropped")

    new = ("    cropped.sub_(mean).div_(std)\n"
           "\n"
           "    # ALGO-01: Mixup augmentation\n"
           "    if B >= 2:\n"
           "        alpha_mix = 0.2\n"
           "        dist = torch.distributions.Beta(alpha_mix, alpha_mix)\n"
           "        lam = dist.sample((B,)).to(device)\n"
           "        lam_img = lam.view(B, 1, 1, 1)\n"
           "        perm = torch.randperm(B, device=device)\n"
           "        cropped_mix = lam_img * cropped + (1 - lam_img) * cropped[perm]\n"
           "        return cropped_mix, lam, perm\n"
           "    else:\n"
           "        return cropped, None, None")

    if old not in content:
        print("ERROR: ALGO-01 target not found!")
        return False
    content = content.replace(old, new)

    # Update the call site
    old2 = ("                imgs = gpu_augment(imgs)\n"
            "\n"
            "                opt.zero_grad()\n"
            "                with torch.cuda.amp.autocast():\n"
            "                    loss = criterion(m(imgs), lbls)")

    new2 = ("                aug_result = gpu_augment(imgs)\n"
            "                if isinstance(aug_result, tuple):\n"
            "                    imgs_aug, mixup_lam, mixup_perm = aug_result\n"
            "                else:\n"
            "                    imgs_aug = aug_result\n"
            "                    mixup_lam = None\n"
            "\n"
            "                opt.zero_grad()\n"
            "                with torch.cuda.amp.autocast():\n"
            "                    logits = m(imgs_aug)\n"
            "                    if mixup_lam is not None:\n"
            "                        lbls_mix = lbls[mixup_perm]\n"
            "                        loss = (mixup_lam.unsqueeze(1) * criterion(logits, lbls) +\n"
            "                                (1 - mixup_lam.unsqueeze(1)) * criterion(logits, lbls_mix))\n"
            "                    else:\n"
            "                        loss = criterion(logits, lbls)")

    if old2 not in content:
        print("ERROR: ALGO-01 call site target not found!")
        return False
    content = content.replace(old2, new2)

    write_src(content)
    print("ALGO-01 applied: Mixup data augmentation")
    return True

def apply_code02():
    """CODE-02: Stratified batch sampling."""
    content = read_src()

    old = ("                # Random batch from this node's data\n"
           "                if n_local >= args.batch_size:\n"
           "                    bidx = idx_t[torch.randint(0, n_local, (args.batch_size,), device=device)]\n"
           "                else:\n"
           "                    # Repeat with replacement for small nodes\n"
           "                    bidx = idx_t[torch.randint(0, n_local, (args.batch_size,), device=device)]")

    new = ("                # CODE-02: stratified batch sampling\n"
           "                node_lbls = train_lbl[idx_t]\n"
           "                unique_classes = node_lbls.unique()\n"
           "                n_classes_present = len(unique_classes)\n"
           "                \n"
           "                if n_classes_present > 0 and n_classes_present < args.batch_size:\n"
           "                    per_class_idx = []\n"
           "                    for c in unique_classes:\n"
           "                        c_mask = (node_lbls == c).nonzero(as_tuple=True)[0]\n"
           "                        if len(c_mask) > 0:\n"
           "                            pick = c_mask[torch.randint(0, len(c_mask), (1,), device=device)]\n"
           "                            per_class_idx.append(pick)\n"
           "                    stratified = torch.cat(per_class_idx)\n"
           "                    remaining = args.batch_size - len(stratified)\n"
           "                    fill = idx_t[torch.randint(0, n_local, (remaining,), device=device)]\n"
           "                    bidx = torch.cat([stratified, fill])\n"
           "                else:\n"
           "                    bidx = idx_t[torch.randint(0, n_local, (args.batch_size,), device=device)]")

    if old not in content:
        print("ERROR: CODE-02 target not found!")
        return False
    content = content.replace(old, new)

    write_src(content)
    print("CODE-02 applied: stratified batch sampling")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 apply_idea.py <idea_id> [--undo]")
        print("Available: PARAM-01, CODE-03, ALGO-04, PARAM-02, ALGO-01, CODE-02")
        sys.exit(1)

    idea = sys.argv[1].upper()
    undo = "--undo" in sys.argv

    if undo:
        restore_baseline()
        sys.exit(0)

    ensure_backup()

    ideas = {
        "PARAM-01": apply_param01,
        "CODE-03": apply_code03,
        "ALGO-04": apply_algo04,
        "PARAM-02": apply_param02,
        "ALGO-01": apply_algo01,
        "CODE-02": apply_code02,
    }

    if idea not in ideas:
        print("Unknown idea: " + idea)
        print("Available: " + ", ".join(sorted(ideas.keys())))
        sys.exit(1)

    ideas[idea]()
