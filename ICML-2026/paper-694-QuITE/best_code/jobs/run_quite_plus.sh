#!/bin/bash
# Reproduce QuITE++ forecasting results (paper Table 4).
#
# Per-(dataset, horizon) hyperparameters (HID / NLAYER / NHEAD) are the values
# selected by the QuITE++ grid search (hid_dim ∈ {32, 64}, nlayer ∈ {1, 2, 3},
# nhead ∈ {1, 2, 4, 8} — paper §6.1).
# Patch / stride / batch_size follow paper Table C.1(a).

GPU=${GPU:-0}
SEEDS=${SEEDS:-"1 2 3 4 5"}

run() {
    python train_quite_plus.py \
        --patience 50 --lr 1e-3 --gpu $GPU "$@"
}

for seed in $SEEDS; do
    # ============ Human Activity (bs=32) ============
    # 3000 → 1000 : HID=64, NLAYER=3, NHEAD=8
    run --dataset activity --history 3000 --patch_size 750 --stride 750 --batch_size 32 \
        --hid_dim 64 --nlayer 3 --nhead 8 --seed $seed
    # 2000 → 2000 : HID=64, NLAYER=3, NHEAD=4
    run --dataset activity --history 2000 --patch_size 500 --stride 500 --batch_size 32 \
        --hid_dim 64 --nlayer 3 --nhead 4 --seed $seed
    # 1000 → 3000 : HID=64, NLAYER=2, NHEAD=2
    run --dataset activity --history 1000 --patch_size 250 --stride 250 --batch_size 32 \
        --hid_dim 64 --nlayer 2 --nhead 2 --seed $seed

    # ============ USHCN (bs=128, patch=1.5) ============
    # 24 → 1  : HID=32, NLAYER=1, NHEAD=2
    run --dataset ushcn --history 24 --pred_window 1  --patch_size 1.5 --stride 1.5 --batch_size 128 \
        --hid_dim 32 --nlayer 1 --nhead 2 --seed $seed
    # 24 → 6  : HID=32, NLAYER=1, NHEAD=4
    run --dataset ushcn --history 24 --pred_window 6  --patch_size 1.5 --stride 1.5 --batch_size 128 \
        --hid_dim 32 --nlayer 1 --nhead 4 --seed $seed
    # 24 → 12 : HID=64, NLAYER=2, NHEAD=2
    run --dataset ushcn --history 24 --pred_window 12 --patch_size 1.5 --stride 1.5 --batch_size 128 \
        --hid_dim 64 --nlayer 2 --nhead 2 --seed $seed

    # ============ PhysioNet (bs=64) ============
    # 12 → 36 : HID=64, NLAYER=3, NHEAD=2,  patch=6
    run --dataset physionet --history 12 --patch_size 6 --stride 6 --batch_size 64 \
        --hid_dim 64 --nlayer 3 --nhead 2 --seed $seed
    # 24 → 24 : HID=64, NLAYER=2, NHEAD=4,  patch=6
    run --dataset physionet --history 24 --patch_size 6 --stride 6 --batch_size 64 \
        --hid_dim 64 --nlayer 2 --nhead 4 --seed $seed
    # 36 → 12 : HID=64, NLAYER=1, NHEAD=4,  patch=9
    run --dataset physionet --history 36 --patch_size 9 --stride 9 --batch_size 64 \
        --hid_dim 64 --nlayer 1 --nhead 4 --seed $seed

    # ============ MIMIC-III ============
    # 12 → 36 : HID=32, NLAYER=3, NHEAD=4,  patch=6,   bs=16
    run --dataset mimic --history 12 --patch_size 6   --stride 6   --batch_size 16 \
        --hid_dim 32 --nlayer 3 --nhead 4 --seed $seed
    # 24 → 24 : HID=32, NLAYER=3, NHEAD=4,  patch=12,  bs=8
    run --dataset mimic --history 24 --patch_size 12  --stride 12  --batch_size 8 \
        --hid_dim 32 --nlayer 3 --nhead 4 --seed $seed
    # 36 → 12 : HID=32, NLAYER=1, NHEAD=4,  patch=4.5, bs=8
    run --dataset mimic --history 36 --patch_size 4.5 --stride 4.5 --batch_size 8 \
        --hid_dim 32 --nlayer 1 --nhead 4 --seed $seed
done
