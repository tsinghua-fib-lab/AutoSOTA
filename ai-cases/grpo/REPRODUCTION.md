# Reproduction guide

## 1. Install the source overlay

The selected implementation is frozen at project commit `5ecb93f74065e7fe94df5cfded51618c39246d63`. The overlay contains every changed or added file relative to upstream UFT commit `1fda767dafdec06e37de4b2d83854c1becf45569`, plus the verifier and license files. It is not a standalone replacement for the upstream framework.

From the package root:

```bash
git clone https://github.com/liumy2010/UFT.git upstream-uft
git -C upstream-uft checkout 1fda767dafdec06e37de4b2d83854c1becf45569
python scripts/apply_overlay.py upstream-uft
```

The installer refuses a different base revision or an already modified checkout. It verifies all overlay hashes before copying files.

## 2. Runtime and external assets

The recorded training runtime uses Python 3.10, CUDA 12.4, PyTorch 2.4.0, vLLM 0.6.3, and a compatible FlashAttention build. `requirements.lock.txt` records the original dependency pins; it is an environment reference, not a guarantee that every binary wheel remains available on every platform. Follow the upstream installation prerequisites, then install the modified framework in that compatible environment with `pip install -e upstream-uft`.

Obtain Qwen2.5-0.5B and the original Countdown split: 10,240 training and 1,024 evaluation examples. The recorded model revision is `060db6499f32faf8b98477b0a26969ef7d8b9987`. Model files, reference solutions, and data parquet files are not bundled. Use the upstream Countdown preparation tools and verify that the resulting split and Hint columns match the recorded training protocol before claiming an exact numerical reproduction.

## 3. Launch the selected recipe

```bash
export WORKSPACE="$PWD/workspace"
export UFT_REPO_ROOT="$PWD/upstream-uft"
export MODEL_PATH="/path/to/Qwen2.5-0.5B"
export DATA_DIR="/path/to/countdown"
export METHOD_VARIANT="mirror-paired-pda-iso"
export RUN_ID="pda-iso-seed0-500"
export TRAIN_STEPS=500
export TRAIN_SEED=0
export HINT_STEPS=300
export ADAPTIVE_HINT_WARMUP_STEPS=40
export TRAIN_BATCH_SIZE=256
export TEST_FREQ=25
export SAVE_FREQ=100
export N_GPUS=4
mkdir -p "$WORKSPACE"
bash scripts/train_overnight_inside_container.sh
```

For matched controls, use `METHOD_VARIANT=mirror-paired-grpo` or `mirror-paired-drgrpo`, with distinct `RUN_ID` values and the same shared settings. Inspect the emitted runtime configuration against `configs/*.recorded.env`. Recorded `/workspace/...` paths are historical locations; supply your own paths rather than executing those `.env` files as shell scripts.

PDA-Iso must use correctness scale 1.8, format scale **0.2**, local projection, tail width 16, mass 16, fixed loss denominator 64, VCC=0, Guard disabled, and length frontier disabled. The library's historical 0.8 format default is not the selected recipe.

## 4. Validate before expensive training

After installing dependencies and applying the overlay:

```bash
cd upstream-uft
python -m unittest discover -s tests -p 'test_drgrpo.py' -v
```

Also run the other top-level `test_*.py` mechanism tests included in the overlay, then a short training smoke run with a separate output directory. A smoke run checks execution, not final performance. The handoff's packaging checks validate Python syntax, source integrity, and result arithmetic; they do not claim a new distributed-training or GPU replay.

## 5. Compare the correct endpoints

Use final validation after 500 steps, not the best intermediate result. Keep no-Hint evaluation, response caps, seeds, and data splits fixed. Record correctness, evaluation output length, format extraction, truncation, training-step time, and activation counts. An independent 250-step run must not be substituted for the step-250 checkpoint of a 500-step run. Warm-start repair includes the cost and state of its parent checkpoint.
