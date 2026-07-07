#!/bin/bash
# Reproduce QuITE-equipped MTS backbones for forecasting (paper Table 2).
#
# Hyperparameters follow:
#   - Paper §6.1 (Implementation Details): hidden=64, 4 attention heads.
#   - Paper Appendix B.1 (per-backbone layers):
#       PatchTST=3, iTransformer=3, TimeXer=3, TMix=2, S-Mamba=2, PatchMixer=1.
#   - Paper Table C.1(a): per-dataset patch_size / stride / batch_size / lr.

GPU=${GPU:-0}
HID=${HID:-64}
NHEAD=${NHEAD:-4}
SEEDS=${SEEDS:-"1 2 3 4 5"}
MODELS=${MODELS:-"patchtst patchmixer tmix itransformer s_mamba timexer"}


nlayer_for() {
    case "$1" in
        patchtst|itransformer|timexer) echo 3 ;;
        tmix|s_mamba)                  echo 2 ;;
        patchmixer)                    echo 1 ;;
    esac
}

run() {
    local model="$1"; shift
    local nlayer
    nlayer=$(nlayer_for "$model")
    python train_forecasting.py \
        --model "$model" --hid_dim $HID --nhead $NHEAD --nlayer "$nlayer" \
        --patience 50 --lr 1e-3 --gpu $GPU --irr_emb --mode quite "$@"
}

for model in $MODELS; do
    for seed in $SEEDS; do
        # ---------- Human Activity (bs=32) ----------
        run $model --dataset activity --history 3000 --patch_size 750 --stride 750 --batch_size 32 --seed $seed
        run $model --dataset activity --history 2000 --patch_size 500 --stride 500 --batch_size 32 --seed $seed
        run $model --dataset activity --history 1000 --patch_size 250 --stride 250 --batch_size 32 --seed $seed

        # ---------- USHCN (bs=128, patch=1.5) ----------
        run $model --dataset ushcn --history 24 --pred_window 1  --patch_size 1.5 --stride 1.5 --batch_size 128 --seed $seed
        run $model --dataset ushcn --history 24 --pred_window 6  --patch_size 1.5 --stride 1.5 --batch_size 128 --seed $seed
        run $model --dataset ushcn --history 24 --pred_window 12 --patch_size 1.5 --stride 1.5 --batch_size 128 --seed $seed

        # ---------- PhysioNet (bs=64) — Table C.1: history→patch 12→6, 24→6, 36→9 ----------
        run $model --dataset physionet --history 12 --patch_size 6 --stride 6 --batch_size 64 --seed $seed
        run $model --dataset physionet --history 24 --patch_size 6 --stride 6 --batch_size 64 --seed $seed
        run $model --dataset physionet --history 36 --patch_size 9 --stride 9 --batch_size 64 --seed $seed

        # ---------- MIMIC-III — Table C.1: history→(patch,bs) 12→(6,16), 24→(12,8), 36→(4.5,8) ----------
        run $model --dataset mimic --history 12 --patch_size 6   --stride 6   --batch_size 16 --seed $seed
        run $model --dataset mimic --history 24 --patch_size 12  --stride 12  --batch_size 8  --seed $seed
        run $model --dataset mimic --history 36 --patch_size 4.5 --stride 4.5 --batch_size 8  --seed $seed
    done
done
