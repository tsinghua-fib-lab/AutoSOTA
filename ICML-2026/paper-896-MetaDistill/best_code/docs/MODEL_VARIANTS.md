# Model variants used for evaluation

This bundle contains two evaluation tracks:

- BBOB (four optimizers: LES / LDE / LGA / POM).
- Control tasks (LunarLander / BipedalWalker / Acrobot).

For each optimizer, the evaluation scripts support:

- a native baseline checkpoint (non-distilled)
- a MetaDistill checkpoint (distilled)
- per-variant SSFT settings (md_j0, md_j1, md_j3, md_j5)

All paths below are relative to `metadistill/`.

The distilled checkpoints can be regenerated with:

- `scripts/collect_trajectories.py`
- `scripts/train_distill.py --config configs/distill_<optimizer>.json`

The baseline checkpoints can be regenerated with:

- `scripts/train_lde_pg.py --config configs/lde_pg_pop200_d10_sample_h64.json`
- `scripts/train_metabbo_les.py`
- `scripts/train_metabbo_lga.py`

## BBOB

LES

- baseline ckpt: `checkpoints/baselines/les_metabbo.pt`
- md ckpt: `checkpoints/metadistill/bbob/les.pt`
- baseline config: `configs/les_config.json`
- md config: `configs/les_d10_pop200.json`

LDE

- baseline ckpt: `checkpoints/baselines/lde_policy_gradient.pt`
- md ckpt: `checkpoints/metadistill/bbob/lde.pt`
- baseline config: `configs/lde_pg_pop200_d10_sample_h64.json`
- md config: `configs/lde_d10_pop200.json`

LGA

- baseline ckpt: `checkpoints/baselines/lga_metabbo.pt`
- md ckpt: `checkpoints/metadistill/bbob/lga.pt`
- baseline config: `configs/lga_config.json`
- md config: `configs/lga_d10_pop200_parentsoft_tau1.json`

POM

- baseline ckpt: `checkpoints/baselines/pom_original.pt`
- md ckpt (BBOB): `checkpoints/metadistill/bbob/pom.pt`
- baseline config: `configs/pom_config.json`
- md config: `configs/pom_d10_pop200.json`

## Control Tasks

Compared to BBOB, the LDE/LGA checkpoints are shared, while LES/POM use different distilled variants.

LES (control tasks)

- baseline ckpt: `checkpoints/baselines/les_metabbo.pt`
- md ckpt: `checkpoints/metadistill/control/les.pt`
- md config: `configs/les_d10_pop200.json`

LDE (control tasks)

- baseline ckpt: `checkpoints/baselines/lde_policy_gradient.pt`
- md ckpt: `checkpoints/metadistill/control/lde.pt`
- baseline config: `configs/lde_pg_pop200_d10_sample_h64.json`
- md config: `configs/lde_d10_pop200.json`

LGA (control tasks)

- baseline ckpt: `checkpoints/baselines/lga_metabbo.pt`
- md ckpt: `checkpoints/metadistill/control/lga.pt`
- baseline config: `configs/lga_config.json`
- md config: `configs/lga_d10_pop200_parentsoft_tau1.json`

POM (control tasks)

- baseline ckpt: `checkpoints/baselines/pom_original.pt`
- md ckpt: `checkpoints/metadistill/control/pom.pt`
- baseline config: `configs/pom_config.json`
- md config: `configs/pom_d10_pop200.json`
