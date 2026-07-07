#!/bin/bash
# Reproduce QuITE-equipped MTS backbones for classification (paper Table 3).
#
# Hyperparameters follow the author-supplied per-backbone settings:
#   hidden=64 (both standalone and QuITE-equipped), nhead=4 (4 attention heads
#   for attention-based backbones), per-backbone nlayer from paper Appendix B.1
#   (PatchTST=3, iTransformer=3, TimeXer=3, TMix=2, S-Mamba=2, PatchMixer=1).
# Patch / stride / bs follow paper Table C.1(b): P19=3.75, P12=6, PAM=10, bs=64.

GPU=${GPU:-0}
HID=${HID:-64}
NHEAD=${NHEAD:-4}
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
    python train_classification.py \
        --model "$model" --hid_dim $HID --nhead $NHEAD --nlayer "$nlayer" \
        --epoch 1000 --batch_size 64 --lr 1e-3 --gpu $GPU --irr_emb --mode quite "$@"
}

for model in $MODELS; do
    # Table C.1(b)
    run $model --dataset P19 --patch_size 3.75 --stride 3.75
    run $model --dataset P12 --patch_size 6    --stride 6
    run $model --dataset PAM --patch_size 10   --stride 10
done
