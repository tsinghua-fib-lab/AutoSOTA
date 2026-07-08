#!/usr/bin/env python3
"""Fix Mixup loss computation in train_decentralized.py."""
with open("/repo/train_decentralized.py") as f:
    content = f.read()

old = """                    if mixup_lam is not None:
                        lbls_mix = lbls[mixup_perm]
                        loss = (mixup_lam.unsqueeze(1) * criterion(logits, lbls) +
                                (1 - mixup_lam.unsqueeze(1)) * criterion(logits, lbls_mix))
                    else:
                        loss = criterion(logits, lbls)"""

new = """                    if mixup_lam is not None:
                        lbls_mix = lbls[mixup_perm]
                        loss_per_sample = F.cross_entropy(logits, lbls, reduction="none", label_smoothing=0.1)
                        loss_per_sample_mix = F.cross_entropy(logits, lbls_mix, reduction="none", label_smoothing=0.1)
                        loss = (mixup_lam * loss_per_sample + (1 - mixup_lam) * loss_per_sample_mix).mean()
                    else:
                        loss = criterion(logits, lbls)"""

if old in content:
    content = content.replace(old, new)
    with open("/repo/train_decentralized.py", "w") as f:
        f.write(content)
    print("Mixup fix applied successfully!")
else:
    print("Pattern not found. Searching for mixup section...")
    idx = content.find("if mixup_lam")
    if idx >= 0:
        print("Found at index", idx)
        print(content[idx:idx+600])
    else:
        print("mixup_lam not found in file!")
