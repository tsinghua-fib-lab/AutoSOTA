#!/usr/bin/env bash
set -euo pipefail

workspace="${WORKSPACE:-/workspace}"
repo_root="${UFT_REPO_ROOT:-${workspace}/src/UFT-bifrontier}"
model_path="${MODEL_PATH:-${workspace}/models/Qwen2.5-0.5B}"
data_dir="${DATA_DIR:-${workspace}/src/UFT/data/countdown}"

variant="${METHOD_VARIANT:-bifrontier}"
source_commit="${SOURCE_COMMIT:-unknown}"
paired_hint_enable=false
posterior_frontier_enable=false
frontier_scaling_enable=false
mirror_view_enable=false
mirror_view_paired=true
mirror_view_schedule="constant"
mirror_view_balanced=false
repetition_quarantine_enable=false
strategy_archive_enable=false
forge_anneal_enable=false
format_constraint_enable=false
format_constraint_feedback_source="train"
format_constraint_apply_scope="all"
deployment_probe_enable=false
deployment_probe_feedback_source="no_hint"
mirror_view_probability_default="0.5"
format_constraint_target_rate_default="0.995"
format_constraint_dual_lr_default="2.0"
format_constraint_max_bonus_default="0.20"
format_constraint_min_bonus_default="0.0"
format_constraint_release_lr_scale_default="1.0"
format_constraint_lcb_z_default="0.0"
format_constraint_ema_alpha_default="0.10"
format_constraint_warmup_steps_default="40"
format_constraint_apply_hint_threshold_default="0.0"
anchor_probe_enable=false
anchor_probe_size_default="128"
anchor_probe_frequency_default="10"
anchor_probe_seed_default="20260908"
actor_lr_default="1e-6"
actor_loss_normalization_default="token_mean"
actor_clip_ratio_high_default="0.2"
pda_correctness_scale_default="1.8"
pda_format_scale_default="0.8"
pda_projection_default="local"
pda_invalid_tail_tokens_default="16"
pda_protocol_mass_default="16.0"
pda_view_contrast_scale_default="0.0"
pda_protocol_guard_enable=false
pda_protocol_guard_multiplier_default="1.0"
pda_protocol_guard_scope_default="all"
pda_protocol_guard_valid_credit_default="1.0"
pda_correct_length_frontier_enable=false
response_target_tokens_default="192"
response_dual_lr_default="0.05"
response_max_coef_default="0.35"
response_ema_alpha_default="0.10"
response_warmup_steps_default="40"
deployment_probe_breach_patience_default="2"
deployment_probe_decay_factor_default="0.25"
format_score_default="0.1"
hint_steps_default="300"
train_file="${TRAIN_FILE:-${data_dir}/train.parquet}"
case "${variant}" in
  pa-baha)
    response_budget_enable=false
    advantage_variant="standard"
    method_name="PA-BAHA-UFT"
    ;;
  bifrontier)
    response_budget_enable=true
    advantage_variant="bifrontier"
    method_name="Bi-Frontier-UFT"
    ;;
  lexi)
    response_budget_enable=false
    advantage_variant="lexi"
    method_name="LEXI-GRPO-UFT"
    ;;
  lexiguard)
    response_budget_enable=true
    advantage_variant="lexiguard"
    method_name="LEXI-Guard-UFT"
    ;;
  pfa)
    response_budget_enable=false
    advantage_variant="standard"
    posterior_frontier_enable=true
    method_name="Posterior-Frontier-GRPO-UFT"
    ;;
  pair)
    response_budget_enable=false
    advantage_variant="standard"
    paired_hint_enable=true
    method_name="PAIR-UFT"
    ;;
  pair-pfa)
    response_budget_enable=false
    advantage_variant="standard"
    paired_hint_enable=true
    posterior_frontier_enable=true
    method_name="PAIR-PFA-UFT"
    ;;
  fsa)
    response_budget_enable=false
    advantage_variant="standard"
    frontier_scaling_enable=true
    method_name="Frontier-Scaled-Advantage-GRPO-UFT"
    ;;
  mirror)
    response_budget_enable=false
    advantage_variant="standard"
    mirror_view_enable=true
    method_name="MIRROR-UFT"
    ;;
  mirror-fsa)
    response_budget_enable=false
    advantage_variant="standard"
    mirror_view_enable=true
    frontier_scaling_enable=true
    method_name="MIRROR-FSA-UFT"
    ;;
  trq)
    response_budget_enable=false
    advantage_variant="standard"
    repetition_quarantine_enable=true
    method_name="TRQ-GRPO-UFT"
    ;;
  mirror-alt-trq)
    response_budget_enable=false
    advantage_variant="standard"
    mirror_view_enable=true
    mirror_view_paired=false
    repetition_quarantine_enable=true
    method_name="MIRROR-ALT-TRQ-UFT"
    ;;
  mirror-alt)
    response_budget_enable=false
    advantage_variant="standard"
    mirror_view_enable=true
    mirror_view_paired=false
    method_name="MIRROR-ALT-UFT"
    ;;
  mirror-alt-far)
    response_budget_enable=false
    advantage_variant="standard"
    mirror_view_enable=true
    mirror_view_paired=false
    # Formatted-but-incorrect responses retain the answer contract without
    # ever outranking a verifier-correct response.
    format_score_default="0.3"
    method_name="MIRROR-ALT-FAR-UFT"
    ;;
  mirror-ramp)
    response_budget_enable=false
    advantage_variant="standard"
    mirror_view_enable=true
    mirror_view_paired=false
    mirror_view_schedule="ramp"
    method_name="MIRROR-RAMP-UFT"
    ;;
  mirror-ramp25)
    response_budget_enable=false
    advantage_variant="standard"
    mirror_view_enable=true
    mirror_view_paired=false
    mirror_view_schedule="ramp"
    mirror_view_probability_default="0.25"
    method_name="MIRROR-RAMP25-UFT"
    ;;
  mirror-ramp-nhfc)
    response_budget_enable=false
    advantage_variant="standard"
    mirror_view_enable=true
    mirror_view_paired=false
    mirror_view_schedule="ramp"
    format_constraint_enable=true
    format_constraint_feedback_source="no_hint"
    format_constraint_apply_scope="no_hint"
    format_constraint_target_rate_default="0.98"
    method_name="MIRROR-RAMP-NHFC-UFT"
    ;;
  mirror-ramp-drgrpo)
    response_budget_enable=false
    advantage_variant="drgrpo"
    mirror_view_enable=true
    mirror_view_paired=false
    mirror_view_schedule="ramp"
    actor_loss_normalization_default="fixed"
    method_name="MIRROR-RAMP-DrGRPO-UFT"
    ;;
  mirror-ramp-pda-iso)
    response_budget_enable=false
    advantage_variant="pda_drgrpo"
    mirror_view_enable=true
    mirror_view_paired=false
    mirror_view_schedule="ramp"
    actor_loss_normalization_default="fixed"
    # Coefficients 1.8 and 0.2 exactly match the semantic/format decomposition
    # of Dr.GRPO's 2 * (reward - group_mean) for 1.0/0.1/0.0 rewards.
    pda_format_scale_default="0.2"
    method_name="MIRROR-RAMP-PDA-IsoScale-UFT"
    ;;
  mirror-ramp-pda)
    response_budget_enable=false
    advantage_variant="pda_drgrpo"
    mirror_view_enable=true
    mirror_view_paired=false
    mirror_view_schedule="ramp"
    actor_loss_normalization_default="fixed"
    method_name="MIRROR-RAMP-PDA-DrGRPO-UFT"
    ;;
  mirror-ramp-pda-global)
    response_budget_enable=false
    advantage_variant="pda_drgrpo"
    mirror_view_enable=true
    mirror_view_paired=false
    mirror_view_schedule="ramp"
    actor_loss_normalization_default="fixed"
    # Same decomposed signals and fixed projection mass as PDA, but format
    # credit is spread globally. This is the locality mechanism ablation.
    pda_projection_default="global"
    method_name="MIRROR-RAMP-PDA-GlobalAblation-UFT"
    ;;
  mirror-paired-pda)
    response_budget_enable=false
    advantage_variant="pda_drgrpo"
    mirror_view_enable=true
    mirror_view_paired=true
    mirror_view_schedule="constant"
    actor_loss_normalization_default="fixed"
    method_name="MIRROR-PAIRED-PDA-DrGRPO-UFT"
    ;;
  mirror-paired-drgrpo)
    response_budget_enable=false
    advantage_variant="drgrpo"
    mirror_view_enable=true
    mirror_view_paired=true
    mirror_view_schedule="constant"
    actor_loss_normalization_default="fixed"
    method_name="MIRROR-PAIRED-DrGRPO-Control-UFT"
    ;;
  mirror-paired-grpo)
    # Matched control for the original GRPO objective used by UFT.  It keeps
    # the current adaptive-hint and paired-view data path, changing only the
    # optimizer back to standard group-standardized advantages and token-mean
    # loss normalization.
    response_budget_enable=false
    advantage_variant="standard"
    mirror_view_enable=true
    mirror_view_paired=true
    mirror_view_schedule="constant"
    actor_loss_normalization_default="token_mean"
    method_name="MIRROR-PAIRED-GRPO-UFT-Control"
    ;;
  mirror-paired-pda-iso)
    response_budget_enable=false
    advantage_variant="pda_drgrpo"
    mirror_view_enable=true
    mirror_view_paired=true
    mirror_view_schedule="constant"
    actor_loss_normalization_default="fixed"
    pda_format_scale_default="0.2"
    method_name="MIRROR-PAIRED-PDA-IsoScale-UFT"
    ;;
  mirror-paired-pda-nstr-barrier|mirror-paired-pda-tail-penalty)
    # Full-training versions of the previously warm-started repair methods.
    # These start from the base model with the normal 300-step hint schedule,
    # 1e-6 learning rate and DrGRPO-matched PDA coefficients.  The controller
    # reads a fixed training-only anchor, never the reported test set.
    response_budget_enable=false
    advantage_variant="pda_drgrpo"
    mirror_view_enable=true
    mirror_view_paired=true
    mirror_view_schedule="constant"
    actor_loss_normalization_default="fixed"
    pda_format_scale_default="0.2"
    pda_protocol_guard_enable=true
    format_constraint_enable=true
    format_constraint_feedback_source="anchor"
    format_constraint_apply_scope="low_hint"
    format_constraint_apply_hint_threshold_default="0.333334"
    format_constraint_target_rate_default="0.95"
    format_constraint_dual_lr_default="0.35"
    format_constraint_max_bonus_default="0.10"
    format_constraint_min_bonus_default="0.02"
    format_constraint_release_lr_scale_default="0.25"
    format_constraint_lcb_z_default="0.0"
    format_constraint_ema_alpha_default="0.75"
    format_constraint_warmup_steps_default="75"
    anchor_probe_enable=true
    anchor_probe_frequency_default="10"
    case "${variant}" in
      *nstr-barrier)
        pda_protocol_guard_scope_default="all_wrong"
        pda_protocol_guard_valid_credit_default="1.0"
        method_name="MIRROR-PAIRED-PDA-NSTR-Barrier-Full"
        ;;
      *)
        pda_protocol_guard_scope_default="all"
        pda_protocol_guard_valid_credit_default="0.0"
        method_name="MIRROR-PAIRED-PDA-TailPenalty-Full"
        ;;
    esac
    ;;
  mirror-paired-pda-vcc05)
    response_budget_enable=false
    advantage_variant="pda_drgrpo"
    mirror_view_enable=true
    mirror_view_paired=true
    mirror_view_schedule="constant"
    actor_loss_normalization_default="fixed"
    pda_format_scale_default="0.2"
    pda_view_contrast_scale_default="0.5"
    method_name="MIRROR-PAIRED-PDA-ViewContrast05-UFT"
    ;;
  mirror-paired-pda-vcc10)
    response_budget_enable=false
    advantage_variant="pda_drgrpo"
    mirror_view_enable=true
    mirror_view_paired=true
    mirror_view_schedule="constant"
    actor_loss_normalization_default="fixed"
    pda_format_scale_default="0.2"
    pda_view_contrast_scale_default="1.0"
    method_name="MIRROR-PAIRED-PDA-ViewContrast10-UFT"
    ;;
  mirror-paired-pda-guard05)
    response_budget_enable=false
    advantage_variant="pda_drgrpo"
    mirror_view_enable=true
    mirror_view_paired=true
    mirror_view_schedule="constant"
    actor_loss_normalization_default="fixed"
    pda_format_scale_default="0.2"
    pda_protocol_guard_enable=true
    pda_protocol_guard_multiplier_default="0.5"
    format_constraint_enable=true
    format_constraint_feedback_source="no_hint"
    format_constraint_apply_scope="no_hint"
    format_constraint_target_rate_default="0.95"
    format_constraint_dual_lr_default="0.5"
    format_constraint_max_bonus_default="0.20"
    format_constraint_warmup_steps_default="75"
    method_name="MIRROR-PAIRED-PDA-DualGuard05-UFT"
    ;;
  mirror-paired-pda-guard10)
    response_budget_enable=false
    advantage_variant="pda_drgrpo"
    mirror_view_enable=true
    mirror_view_paired=true
    mirror_view_schedule="constant"
    actor_loss_normalization_default="fixed"
    pda_format_scale_default="0.2"
    pda_protocol_guard_enable=true
    pda_protocol_guard_multiplier_default="1.0"
    format_constraint_enable=true
    format_constraint_feedback_source="no_hint"
    format_constraint_apply_scope="no_hint"
    format_constraint_target_rate_default="0.95"
    format_constraint_dual_lr_default="0.5"
    format_constraint_max_bonus_default="0.20"
    format_constraint_warmup_steps_default="75"
    method_name="MIRROR-PAIRED-PDA-DualGuard10-UFT"
    ;;
  mirror-paired-pda-guard20)
    response_budget_enable=false
    advantage_variant="pda_drgrpo"
    mirror_view_enable=true
    mirror_view_paired=true
    mirror_view_schedule="constant"
    actor_loss_normalization_default="fixed"
    pda_format_scale_default="0.2"
    pda_protocol_guard_enable=true
    pda_protocol_guard_multiplier_default="2.0"
    format_constraint_enable=true
    format_constraint_feedback_source="no_hint"
    format_constraint_apply_scope="no_hint"
    format_constraint_target_rate_default="0.95"
    format_constraint_dual_lr_default="0.5"
    format_constraint_max_bonus_default="0.20"
    format_constraint_warmup_steps_default="75"
    method_name="MIRROR-PAIRED-PDA-DualGuard20-UFT"
    ;;
  mirror-paired-pda-anchor-soft)
    response_budget_enable=false
    advantage_variant="pda_drgrpo"
    mirror_view_enable=true
    mirror_view_paired=true
    mirror_view_schedule="constant"
    actor_loss_normalization_default="fixed"
    pda_format_scale_default="0.2"
    pda_protocol_guard_enable=true
    format_constraint_enable=true
    format_constraint_feedback_source="anchor"
    format_constraint_apply_scope="low_hint"
    format_constraint_apply_hint_threshold_default="0.333334"
    format_constraint_target_rate_default="0.95"
    format_constraint_dual_lr_default="0.35"
    format_constraint_max_bonus_default="0.10"
    format_constraint_min_bonus_default="0.01"
    format_constraint_release_lr_scale_default="0.25"
    format_constraint_lcb_z_default="0.0"
    format_constraint_ema_alpha_default="0.50"
    format_constraint_warmup_steps_default="75"
    anchor_probe_enable=true
    method_name="MIRROR-PAIRED-PDA-AnchorGuard-Soft-UFT"
    ;;
  mirror-paired-pda-anchor-balanced)
    response_budget_enable=false
    advantage_variant="pda_drgrpo"
    mirror_view_enable=true
    mirror_view_paired=true
    mirror_view_schedule="constant"
    actor_loss_normalization_default="fixed"
    pda_format_scale_default="0.2"
    pda_protocol_guard_enable=true
    format_constraint_enable=true
    format_constraint_feedback_source="anchor"
    format_constraint_apply_scope="low_hint"
    format_constraint_apply_hint_threshold_default="0.333334"
    format_constraint_target_rate_default="0.98"
    format_constraint_dual_lr_default="0.50"
    format_constraint_max_bonus_default="0.20"
    format_constraint_min_bonus_default="0.02"
    format_constraint_release_lr_scale_default="0.25"
    format_constraint_lcb_z_default="0.5"
    format_constraint_ema_alpha_default="0.50"
    format_constraint_warmup_steps_default="50"
    anchor_probe_enable=true
    method_name="MIRROR-PAIRED-PDA-AnchorGuard-Balanced-UFT"
    ;;
  mirror-paired-pda-anchor-robust)
    response_budget_enable=false
    advantage_variant="pda_drgrpo"
    mirror_view_enable=true
    mirror_view_paired=true
    mirror_view_schedule="constant"
    actor_loss_normalization_default="fixed"
    pda_format_scale_default="0.2"
    pda_protocol_guard_enable=true
    pda_protocol_guard_multiplier_default="1.25"
    format_constraint_enable=true
    format_constraint_feedback_source="anchor"
    format_constraint_apply_scope="low_hint"
    format_constraint_apply_hint_threshold_default="0.50"
    format_constraint_target_rate_default="0.99"
    format_constraint_dual_lr_default="0.50"
    format_constraint_max_bonus_default="0.20"
    format_constraint_min_bonus_default="0.03"
    format_constraint_release_lr_scale_default="0.20"
    format_constraint_lcb_z_default="1.0"
    format_constraint_ema_alpha_default="0.50"
    format_constraint_warmup_steps_default="50"
    anchor_probe_enable=true
    method_name="MIRROR-PAIRED-PDA-AnchorGuard-Robust-UFT"
    ;;
  mirror-paired-pda-anchor-repair-soft)
    # A low-drift consolidation phase initialized from the high-correctness
    # 200-step PDA checkpoint. This is reported separately from equal-budget
    # fresh runs because its cumulative training budget is larger.
    model_path="${PDA_REPAIR_MODEL_PATH:-${workspace}/checkpoints/uft/pda-vcc-long-20260907-mirror-paired-pda-iso-s0-pda-vcc-long-20260907-mirror-paired-pda-iso-s0/actor/global_step_200}"
    hint_steps_default="1"
    actor_lr_default="2e-7"
    response_budget_enable=false
    advantage_variant="pda_drgrpo"
    mirror_view_enable=true
    mirror_view_paired=true
    mirror_view_schedule="constant"
    actor_loss_normalization_default="fixed"
    pda_correctness_scale_default="0.9"
    pda_format_scale_default="0.2"
    pda_protocol_guard_enable=true
    format_constraint_enable=true
    format_constraint_feedback_source="anchor"
    format_constraint_apply_scope="low_hint"
    format_constraint_apply_hint_threshold_default="0.333334"
    format_constraint_target_rate_default="0.95"
    format_constraint_dual_lr_default="0.35"
    format_constraint_max_bonus_default="0.10"
    format_constraint_min_bonus_default="0.02"
    format_constraint_release_lr_scale_default="0.25"
    format_constraint_lcb_z_default="0.0"
    format_constraint_ema_alpha_default="0.75"
    format_constraint_warmup_steps_default="0"
    anchor_probe_enable=true
    anchor_probe_frequency_default="1"
    method_name="MIRROR-PAIRED-PDA-AnchorRepair-Soft-UFT"
    ;;
  mirror-paired-pda-anchor-repair-robust)
    model_path="${PDA_REPAIR_MODEL_PATH:-${workspace}/checkpoints/uft/pda-vcc-long-20260907-mirror-paired-pda-iso-s0-pda-vcc-long-20260907-mirror-paired-pda-iso-s0/actor/global_step_200}"
    hint_steps_default="1"
    actor_lr_default="2e-7"
    response_budget_enable=false
    advantage_variant="pda_drgrpo"
    mirror_view_enable=true
    mirror_view_paired=true
    mirror_view_schedule="constant"
    actor_loss_normalization_default="fixed"
    pda_correctness_scale_default="0.9"
    pda_format_scale_default="0.2"
    pda_protocol_guard_enable=true
    pda_protocol_guard_multiplier_default="1.25"
    format_constraint_enable=true
    format_constraint_feedback_source="anchor"
    format_constraint_apply_scope="low_hint"
    format_constraint_apply_hint_threshold_default="0.50"
    format_constraint_target_rate_default="0.99"
    format_constraint_dual_lr_default="0.50"
    format_constraint_max_bonus_default="0.20"
    format_constraint_min_bonus_default="0.03"
    format_constraint_release_lr_scale_default="0.20"
    format_constraint_lcb_z_default="1.0"
    format_constraint_ema_alpha_default="0.75"
    format_constraint_warmup_steps_default="0"
    anchor_probe_enable=true
    anchor_probe_frequency_default="1"
    method_name="MIRROR-PAIRED-PDA-AnchorRepair-Robust-UFT"
    ;;
  mirror-paired-pda-anchor-repair-plain)
    model_path="${PDA_REPAIR_MODEL_PATH:-${workspace}/checkpoints/uft/pda-vcc-long-20260907-mirror-paired-pda-iso-s0-pda-vcc-long-20260907-mirror-paired-pda-iso-s0/actor/global_step_200}"
    hint_steps_default="1"
    actor_lr_default="2e-7"
    response_budget_enable=false
    advantage_variant="pda_drgrpo"
    mirror_view_enable=true
    mirror_view_paired=true
    mirror_view_schedule="constant"
    actor_loss_normalization_default="fixed"
    pda_correctness_scale_default="0.9"
    pda_format_scale_default="0.2"
    # Keep the identical fixed greedy train anchor for compute/diagnostic
    # matching, but do not attach a controller or Guard gradient.
    anchor_probe_enable=true
    anchor_probe_frequency_default="1"
    method_name="MIRROR-PAIRED-PDA-AnchorRepair-Plain-Control"
    ;;
  mirror-paired-pda-anchor-repair-nstr-barrier|mirror-paired-pda-anchor-repair-nstr-penalty|mirror-paired-pda-anchor-repair-tail-penalty)
    # Null-Space Tail Repair (NSTR): preserve PDA's semantic ordering by
    # routing protocol repair away from mixed-correctness groups.  The penalty
    # variants also remove positive guard credit for formatted failures.
    model_path="${PDA_REPAIR_MODEL_PATH:-${workspace}/checkpoints/uft/pda-vcc-long-20260907-mirror-paired-pda-iso-s0-pda-vcc-long-20260907-mirror-paired-pda-iso-s0/actor/global_step_200}"
    hint_steps_default="1"
    actor_lr_default="2e-7"
    response_budget_enable=false
    advantage_variant="pda_drgrpo"
    mirror_view_enable=true
    mirror_view_paired=true
    mirror_view_schedule="constant"
    actor_loss_normalization_default="fixed"
    pda_correctness_scale_default="0.9"
    pda_format_scale_default="0.2"
    pda_protocol_guard_enable=true
    format_constraint_enable=true
    format_constraint_feedback_source="anchor"
    format_constraint_apply_scope="low_hint"
    format_constraint_apply_hint_threshold_default="0.333334"
    format_constraint_target_rate_default="0.95"
    format_constraint_dual_lr_default="0.35"
    format_constraint_max_bonus_default="0.10"
    format_constraint_min_bonus_default="0.02"
    format_constraint_release_lr_scale_default="0.25"
    format_constraint_lcb_z_default="0.0"
    format_constraint_ema_alpha_default="0.75"
    format_constraint_warmup_steps_default="0"
    anchor_probe_enable=true
    anchor_probe_frequency_default="1"
    case "${variant}" in
      *nstr-barrier)
        pda_protocol_guard_scope_default="all_wrong"
        pda_protocol_guard_valid_credit_default="1.0"
        method_name="MIRROR-PAIRED-PDA-NSTR-Barrier"
        ;;
      *nstr-penalty)
        pda_protocol_guard_scope_default="all_wrong"
        pda_protocol_guard_valid_credit_default="0.0"
        method_name="MIRROR-PAIRED-PDA-NSTR-Penalty"
        ;;
      *)
        pda_protocol_guard_scope_default="all"
        pda_protocol_guard_valid_credit_default="0.0"
        method_name="MIRROR-PAIRED-PDA-TailPenalty"
        ;;
    esac
    ;;
  mirror-paired-pda-anchor-repair-frontier192|mirror-paired-pda-anchor-repair-frontier96|mirror-paired-pda-anchor-repair-frontier64)
    model_path="${PDA_REPAIR_MODEL_PATH:-${workspace}/checkpoints/uft/pda-vcc-long-20260907-mirror-paired-pda-iso-s0-pda-vcc-long-20260907-mirror-paired-pda-iso-s0/actor/global_step_200}"
    hint_steps_default="1"
    actor_lr_default="2e-7"
    response_budget_enable=true
    case "${variant}" in
      *frontier96) response_target_tokens_default="96" ;;
      *frontier64) response_target_tokens_default="64" ;;
      *) response_target_tokens_default="192" ;;
    esac
    response_dual_lr_default="0.03"
    response_max_coef_default="0.10"
    response_ema_alpha_default="0.25"
    response_warmup_steps_default="0"
    advantage_variant="pda_drgrpo"
    mirror_view_enable=true
    mirror_view_paired=true
    mirror_view_schedule="constant"
    actor_loss_normalization_default="fixed"
    pda_correctness_scale_default="0.9"
    pda_format_scale_default="0.2"
    pda_protocol_guard_enable=true
    pda_correct_length_frontier_enable=true
    format_constraint_enable=true
    format_constraint_feedback_source="anchor"
    format_constraint_apply_scope="low_hint"
    format_constraint_apply_hint_threshold_default="0.333334"
    format_constraint_target_rate_default="0.95"
    format_constraint_dual_lr_default="0.35"
    format_constraint_max_bonus_default="0.10"
    format_constraint_min_bonus_default="0.02"
    format_constraint_release_lr_scale_default="0.25"
    format_constraint_ema_alpha_default="0.75"
    format_constraint_warmup_steps_default="0"
    anchor_probe_enable=true
    anchor_probe_frequency_default="1"
    method_name="MIRROR-PAIRED-PDA-AnchorRepair-Frontier${response_target_tokens_default}"
    ;;
  mirror-paired-pda-anchor-repair-frontier128)
    model_path="${PDA_REPAIR_MODEL_PATH:-${workspace}/checkpoints/uft/pda-vcc-long-20260907-mirror-paired-pda-iso-s0-pda-vcc-long-20260907-mirror-paired-pda-iso-s0/actor/global_step_200}"
    hint_steps_default="1"
    actor_lr_default="2e-7"
    response_budget_enable=true
    response_target_tokens_default="128"
    response_dual_lr_default="0.03"
    response_max_coef_default="0.10"
    response_ema_alpha_default="0.25"
    response_warmup_steps_default="0"
    advantage_variant="pda_drgrpo"
    mirror_view_enable=true
    mirror_view_paired=true
    mirror_view_schedule="constant"
    actor_loss_normalization_default="fixed"
    pda_correctness_scale_default="0.9"
    pda_format_scale_default="0.2"
    pda_protocol_guard_enable=true
    pda_correct_length_frontier_enable=true
    format_constraint_enable=true
    format_constraint_feedback_source="anchor"
    format_constraint_apply_scope="low_hint"
    format_constraint_apply_hint_threshold_default="0.333334"
    format_constraint_target_rate_default="0.95"
    format_constraint_dual_lr_default="0.35"
    format_constraint_max_bonus_default="0.10"
    format_constraint_min_bonus_default="0.02"
    format_constraint_release_lr_scale_default="0.25"
    format_constraint_ema_alpha_default="0.75"
    format_constraint_warmup_steps_default="0"
    anchor_probe_enable=true
    anchor_probe_frequency_default="1"
    method_name="MIRROR-PAIRED-PDA-AnchorRepair-Frontier128"
    ;;
  mirror-ramp-dr-vfc)
    echo "mirror-ramp-dr-vfc is disabled: data.val_files is the reported test set, so training feedback would leak test information" >&2
    exit 2
    ;;
  mirror-ramp-dr-far20)
    response_budget_enable=false
    advantage_variant="drgrpo"
    mirror_view_enable=true
    mirror_view_paired=false
    mirror_view_schedule="ramp"
    actor_loss_normalization_default="fixed"
    # Preserve the verifier-correct reward at 1.0 while doubling the much
    # smaller format-only residual from 0.1 to 0.2.
    format_score_default="0.2"
    method_name="MIRROR-RAMP-FormatReward20-DrGRPO-UFT"
    ;;
  mirror-ramp-dr-fastdeassist)
    response_budget_enable=false
    advantage_variant="drgrpo"
    mirror_view_enable=true
    mirror_view_paired=false
    mirror_view_schedule="ramp"
    actor_loss_normalization_default="fixed"
    # Reach low-assistance/no-hint samples earlier to reduce the train versus
    # deployment format gap exposed by the 150-step multi-seed experiment.
    hint_steps_default="200"
    method_name="MIRROR-RAMP-FastDeassist-DrGRPO-UFT"
    ;;
  mirror-ramp-drclip)
    response_budget_enable=false
    advantage_variant="drgrpo"
    mirror_view_enable=true
    mirror_view_paired=false
    mirror_view_schedule="ramp"
    actor_loss_normalization_default="fixed"
    actor_clip_ratio_high_default="0.28"
    method_name="MIRROR-RAMP-DrGRPO-ClipHigher-UFT"
    ;;
  mirror-ramp-hcaa)
    response_budget_enable=false
    advantage_variant="hcaa"
    mirror_view_enable=true
    mirror_view_paired=false
    mirror_view_schedule="ramp"
    method_name="MIRROR-RAMP-HCAA-UFT"
    ;;
  mirror-ramp-cliphigher)
    response_budget_enable=false
    advantage_variant="standard"
    mirror_view_enable=true
    mirror_view_paired=false
    mirror_view_schedule="ramp"
    actor_clip_ratio_high_default="0.28"
    method_name="MIRROR-RAMP-ClipHigher-UFT"
    ;;
  pcasa)
    response_budget_enable=false
    advantage_variant="standard"
    mirror_view_enable=true
    mirror_view_paired=false
    mirror_view_schedule="ramp"
    format_constraint_enable=true
    format_constraint_feedback_source="no_hint"
    format_constraint_apply_scope="no_hint"
    format_constraint_target_rate_default="0.98"
    deployment_probe_enable=true
    method_name="PCASA-UFT"
    ;;
  pcasa-dr)
    response_budget_enable=false
    advantage_variant="drgrpo"
    mirror_view_enable=true
    mirror_view_paired=false
    mirror_view_schedule="ramp"
    format_constraint_enable=true
    format_constraint_feedback_source="no_hint"
    format_constraint_apply_scope="no_hint"
    format_constraint_target_rate_default="0.98"
    deployment_probe_enable=true
    actor_loss_normalization_default="fixed"
    method_name="PCASA-DrGRPO-UFT"
    ;;
  pcasa-hcaa)
    response_budget_enable=false
    advantage_variant="hcaa"
    mirror_view_enable=true
    mirror_view_paired=false
    mirror_view_schedule="ramp"
    format_constraint_enable=true
    format_constraint_feedback_source="no_hint"
    format_constraint_apply_scope="no_hint"
    format_constraint_target_rate_default="0.98"
    deployment_probe_enable=true
    method_name="PCASA-HCAA-UFT"
    ;;
  pcasa-cliphigher)
    response_budget_enable=false
    advantage_variant="standard"
    mirror_view_enable=true
    mirror_view_paired=false
    mirror_view_schedule="ramp"
    format_constraint_enable=true
    format_constraint_feedback_source="no_hint"
    format_constraint_apply_scope="no_hint"
    format_constraint_target_rate_default="0.98"
    deployment_probe_enable=true
    actor_clip_ratio_high_default="0.28"
    method_name="PCASA-ClipHigher-UFT"
    ;;
  pcasa-hcaa-cliphigher)
    response_budget_enable=false
    advantage_variant="hcaa"
    mirror_view_enable=true
    mirror_view_paired=false
    mirror_view_schedule="ramp"
    format_constraint_enable=true
    format_constraint_feedback_source="no_hint"
    format_constraint_apply_scope="no_hint"
    format_constraint_target_rate_default="0.98"
    deployment_probe_enable=true
    actor_clip_ratio_high_default="0.28"
    method_name="PCASA-HCAA-ClipHigher-UFT"
    ;;
  pcasa-dr-soft)
    response_budget_enable=false
    advantage_variant="drgrpo"
    mirror_view_enable=true
    mirror_view_paired=false
    mirror_view_schedule="ramp"
    format_constraint_enable=true
    format_constraint_feedback_source="no_hint"
    format_constraint_apply_scope="no_hint"
    format_constraint_target_rate_default="0.98"
    deployment_probe_enable=true
    deployment_probe_breach_patience_default="3"
    deployment_probe_decay_factor_default="0.50"
    actor_loss_normalization_default="fixed"
    method_name="PCASA-DrGRPO-SoftGate-UFT"
    ;;
  mirror-dualformat)
    response_budget_enable=false
    advantage_variant="standard"
    mirror_view_enable=true
    mirror_view_paired=false
    format_constraint_enable=true
    method_name="MIRROR-DualFormat-UFT"
    ;;
  scasa)
    response_budget_enable=false
    advantage_variant="standard"
    mirror_view_enable=true
    mirror_view_paired=false
    mirror_view_schedule="ramp"
    mirror_view_balanced=true
    format_constraint_enable=true
    method_name="SCASA-UFT"
    ;;
  mirror-trq)
    response_budget_enable=false
    advantage_variant="standard"
    mirror_view_enable=true
    repetition_quarantine_enable=true
    method_name="MIRROR-TRQ-UFT"
    ;;
  mirror-qda-trq)
    response_budget_enable=false
    advantage_variant="standard"
    mirror_view_enable=true
    repetition_quarantine_enable=true
    strategy_archive_enable=true
    method_name="MIRROR-QDA-TRQ-UFT"
    ;;
  forge)
    response_budget_enable=false
    advantage_variant="standard"
    train_file="${FORGE_TRAIN_FILE:-${workspace}/data/structural/train_forge_mix.parquet}"
    method_name="FORGE-UFT"
    ;;
  forge-fsa)
    response_budget_enable=false
    advantage_variant="standard"
    frontier_scaling_enable=true
    train_file="${FORGE_TRAIN_FILE:-${workspace}/data/structural/train_forge_mix.parquet}"
    method_name="FORGE-FSA-UFT"
    ;;
  forge-v2-trq)
    response_budget_enable=false
    advantage_variant="standard"
    repetition_quarantine_enable=true
    train_file="${FORGE_V2_TRAIN_FILE:-${workspace}/data/structural/train_forge_v2_mix.parquet}"
    method_name="FORGE-v2-TRQ-UFT"
    ;;
  forge-v2-anneal)
    response_budget_enable=false
    advantage_variant="standard"
    forge_anneal_enable=true
    train_file="${FORGE_V2_TRAIN_FILE:-${workspace}/data/structural/train_forge_v2_mix.parquet}"
    method_name="FORGE-v2-Anneal-UFT"
    ;;
  *)
    echo "unknown METHOD_VARIANT: ${variant}" >&2
    exit 2
    ;;
esac

run_id="${RUN_ID:-${variant}-countdown-$(date -u +%Y%m%dT%H%M%SZ)}"
experiment_name="${EXPERIMENT_NAME:-${variant}-countdown-qwen2.5-0.5b-beta0.001}"
train_steps="${TRAIN_STEPS:-125}"
hint_steps="${HINT_STEPS:-${hint_steps_default}}"
format_score="${FORMAT_SCORE:-${format_score_default}}"
train_batch_size="${TRAIN_BATCH_SIZE:-256}"
val_batch_size="${VAL_BATCH_SIZE:-1312}"
rollout_n="${ROLLOUT_N:-4}"
if [[ "${paired_hint_enable}" == "true" || ( "${mirror_view_enable}" == "true" && "${mirror_view_paired}" == "true" ) ]]; then
  # Two prompt treatments times two samples keeps four responses per logical
  # problem, exactly matching the baseline generation budget.
  rollout_n=2
fi
n_gpus="${N_GPUS:-4}"
train_seed="${TRAIN_SEED:-0}"
sft_loss_coef="${SFT_LOSS_COEF:-0.001}"
actor_lr="${ACTOR_LR:-${actor_lr_default}}"
test_freq="${TEST_FREQ:-25}"
save_freq="${SAVE_FREQ:-125}"
val_before_train="${VAL_BEFORE_TRAIN:-true}"

adaptive_warmup_steps="${ADAPTIVE_HINT_WARMUP_STEPS:-40}"
response_target_tokens="${RESPONSE_TARGET_TOKENS:-${response_target_tokens_default}}"
response_dual_lr="${RESPONSE_DUAL_LR:-${response_dual_lr_default}}"
response_max_coef="${RESPONSE_MAX_COEF:-${response_max_coef_default}}"
response_ema_alpha="${RESPONSE_EMA_ALPHA:-${response_ema_alpha_default}}"
response_warmup_steps="${RESPONSE_WARMUP_STEPS:-${response_warmup_steps_default}}"
lexi_format_coef="${LEXI_FORMAT_COEF:-0.5}"
incorrect_tail_tokens="${INCORRECT_TAIL_TOKENS:-128}"
incorrect_tail_coef="${INCORRECT_TAIL_COEF:-0.05}"
paired_min_hint_chunks="${PAIRED_MIN_HINT_CHUNKS:-1}"
paired_ate_bonus="${PAIRED_ATE_BONUS:-0.5}"
posterior_coefficient="${POSTERIOR_COEFFICIENT:-0.5}"
posterior_min_observations="${POSTERIOR_MIN_OBSERVATIONS:-1}"
frontier_min_weight="${FRONTIER_MIN_WEIGHT:-0.25}"
frontier_max_weight="${FRONTIER_MAX_WEIGHT:-1.75}"
frontier_min_observations="${FRONTIER_MIN_OBSERVATIONS:-1}"
mirror_view_probability="${MIRROR_VIEW_PROBABILITY:-${mirror_view_probability_default}}"
mirror_ramp_start_step="${MIRROR_RAMP_START_STEP:-50}"
mirror_ramp_end_step="${MIRROR_RAMP_END_STEP:-100}"
format_constraint_target_rate="${FORMAT_CONSTRAINT_TARGET_RATE:-${format_constraint_target_rate_default}}"
format_constraint_dual_lr="${FORMAT_CONSTRAINT_DUAL_LR:-${format_constraint_dual_lr_default}}"
format_constraint_max_bonus="${FORMAT_CONSTRAINT_MAX_BONUS:-${format_constraint_max_bonus_default}}"
format_constraint_min_bonus="${FORMAT_CONSTRAINT_MIN_BONUS:-${format_constraint_min_bonus_default}}"
format_constraint_release_lr_scale="${FORMAT_CONSTRAINT_RELEASE_LR_SCALE:-${format_constraint_release_lr_scale_default}}"
format_constraint_lcb_z="${FORMAT_CONSTRAINT_LCB_Z:-${format_constraint_lcb_z_default}}"
format_constraint_ema_alpha="${FORMAT_CONSTRAINT_EMA_ALPHA:-${format_constraint_ema_alpha_default}}"
format_constraint_warmup_steps="${FORMAT_CONSTRAINT_WARMUP_STEPS:-${format_constraint_warmup_steps_default}}"
format_constraint_apply_hint_threshold="${FORMAT_CONSTRAINT_APPLY_HINT_THRESHOLD:-${format_constraint_apply_hint_threshold_default}}"
anchor_probe_size="${ANCHOR_PROBE_SIZE:-${anchor_probe_size_default}}"
anchor_probe_frequency="${ANCHOR_PROBE_FREQUENCY:-${anchor_probe_frequency_default}}"
anchor_probe_seed="${ANCHOR_PROBE_SEED:-${anchor_probe_seed_default}}"
deployment_probe_target_rate="${DEPLOYMENT_PROBE_TARGET_RATE:-0.98}"
deployment_probe_breach_patience="${DEPLOYMENT_PROBE_BREACH_PATIENCE:-${deployment_probe_breach_patience_default}}"
deployment_probe_decay_factor="${DEPLOYMENT_PROBE_DECAY_FACTOR:-${deployment_probe_decay_factor_default}}"
deployment_probe_recovery_step="${DEPLOYMENT_PROBE_RECOVERY_STEP:-0.20}"
deployment_probe_min_scale="${DEPLOYMENT_PROBE_MIN_SCALE:-0.10}"
deployment_probe_warmup_steps="${DEPLOYMENT_PROBE_WARMUP_STEPS:-50}"
actor_loss_normalization="${ACTOR_LOSS_NORMALIZATION:-${actor_loss_normalization_default}}"
fixed_loss_normalizer="${FIXED_LOSS_NORMALIZER:-64.0}"
drgrpo_advantage_scale="${DRGRPO_ADVANTAGE_SCALE:-2.0}"
pda_correctness_scale="${PDA_CORRECTNESS_SCALE:-${pda_correctness_scale_default}}"
pda_format_scale="${PDA_FORMAT_SCALE:-${pda_format_scale_default}}"
pda_projection="${PDA_PROJECTION:-${pda_projection_default}}"
pda_invalid_tail_tokens="${PDA_INVALID_TAIL_TOKENS:-${pda_invalid_tail_tokens_default}}"
pda_protocol_mass="${PDA_PROTOCOL_MASS:-${pda_protocol_mass_default}}"
pda_view_contrast_scale="${PDA_VIEW_CONTRAST_SCALE:-${pda_view_contrast_scale_default}}"
pda_protocol_guard_multiplier="${PDA_PROTOCOL_GUARD_MULTIPLIER:-${pda_protocol_guard_multiplier_default}}"
pda_protocol_guard_scope="${PDA_PROTOCOL_GUARD_SCOPE:-${pda_protocol_guard_scope_default}}"
pda_protocol_guard_valid_credit="${PDA_PROTOCOL_GUARD_VALID_CREDIT:-${pda_protocol_guard_valid_credit_default}}"
hcaa_advantage_scale="${HCAA_ADVANTAGE_SCALE:-2.0}"
hcaa_amplification="${HCAA_AMPLIFICATION:-1.0}"
hcaa_hardness_power="${HCAA_HARDNESS_POWER:-1.0}"
actor_clip_ratio_high="${ACTOR_CLIP_RATIO_HIGH:-${actor_clip_ratio_high_default}}"
repetition_coefficient="${REPETITION_COEFFICIENT:-0.10}"
repetition_ngram_size="${REPETITION_NGRAM_SIZE:-8}"
repetition_min_repeats="${REPETITION_MIN_REPEATS:-3}"
repetition_invalid_tail_tokens="${REPETITION_INVALID_TAIL_TOKENS:-256}"
strategy_archive_coefficient="${STRATEGY_ARCHIVE_COEFFICIENT:-0.15}"
forge_anneal_start_step="${FORGE_ANNEAL_START_STEP:-75}"
forge_anneal_end_step="${FORGE_ANNEAL_END_STEP:-100}"

run_dir="${workspace}/runs/uft/${run_id}"
checkpoint_prefix="${workspace}/checkpoints/uft/${run_id}"
mkdir -p "${run_dir}" "${workspace}/runs/uft/wandb" "${workspace}/cache/huggingface" \
  "${workspace}/cache/ray" "${workspace}/checkpoints/uft"

export PYTHONPATH="${repo_root}:${PYTHONPATH:-}"
export HF_HOME="${workspace}/cache/huggingface"
export RAY_TMPDIR="${workspace}/cache/ray"
export TOKENIZERS_PARALLELISM=false
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_MODE=offline
export WANDB_DIR="${workspace}/runs/uft/wandb"
export WANDB_RUN_ID="${run_id}"
export WANDB_NAME="${run_id}"

cd "${repo_root}"

{
  echo "method=${method_name}"
  echo "variant=${variant}"
  echo "run_id=${run_id}"
  echo "git_commit=${source_commit}"
  echo "model_path=${model_path}"
  echo "data_dir=${data_dir}"
  echo "train_file=${train_file}"
  echo "train_steps=${train_steps}"
  echo "hint_steps=${hint_steps}"
  echo "adaptive_hint=true"
  echo "adaptive_difficulty_mode=least_assisted"
  echo "adaptive_hint_warmup_steps=${adaptive_warmup_steps}"
  echo "advantage_variant=${advantage_variant}"
  echo "paired_hint_enable=${paired_hint_enable}"
  echo "posterior_frontier_enable=${posterior_frontier_enable}"
  echo "frontier_scaling_enable=${frontier_scaling_enable}"
  echo "mirror_view_enable=${mirror_view_enable}"
  echo "mirror_view_paired=${mirror_view_paired}"
  echo "mirror_view_probability=${mirror_view_probability}"
  echo "mirror_view_schedule=${mirror_view_schedule}"
  echo "mirror_view_ramp_start_step=${mirror_ramp_start_step}"
  echo "mirror_view_ramp_end_step=${mirror_ramp_end_step}"
  echo "mirror_view_balanced=${mirror_view_balanced}"
  echo "format_constraint_enable=${format_constraint_enable}"
  echo "format_constraint_target_rate=${format_constraint_target_rate}"
  echo "format_constraint_dual_lr=${format_constraint_dual_lr}"
  echo "format_constraint_max_bonus=${format_constraint_max_bonus}"
  echo "format_constraint_min_bonus=${format_constraint_min_bonus}"
  echo "format_constraint_release_lr_scale=${format_constraint_release_lr_scale}"
  echo "format_constraint_lcb_z=${format_constraint_lcb_z}"
  echo "format_constraint_ema_alpha=${format_constraint_ema_alpha}"
  echo "format_constraint_warmup_steps=${format_constraint_warmup_steps}"
  echo "format_constraint_feedback_source=${format_constraint_feedback_source}"
  echo "format_constraint_apply_scope=${format_constraint_apply_scope}"
  echo "format_constraint_apply_hint_threshold=${format_constraint_apply_hint_threshold}"
  echo "anchor_probe_enable=${anchor_probe_enable}"
  echo "anchor_probe_size=${anchor_probe_size}"
  echo "anchor_probe_frequency=${anchor_probe_frequency}"
  echo "anchor_probe_seed=${anchor_probe_seed}"
  echo "deployment_probe_enable=${deployment_probe_enable}"
  echo "deployment_probe_feedback_source=${deployment_probe_feedback_source}"
  echo "deployment_probe_target_rate=${deployment_probe_target_rate}"
  echo "deployment_probe_breach_patience=${deployment_probe_breach_patience}"
  echo "deployment_probe_decay_factor=${deployment_probe_decay_factor}"
  echo "deployment_probe_recovery_step=${deployment_probe_recovery_step}"
  echo "deployment_probe_min_scale=${deployment_probe_min_scale}"
  echo "deployment_probe_warmup_steps=${deployment_probe_warmup_steps}"
  echo "actor_loss_normalization=${actor_loss_normalization}"
  echo "fixed_loss_normalizer=${fixed_loss_normalizer}"
  echo "drgrpo_advantage_scale=${drgrpo_advantage_scale}"
  echo "pda_correctness_scale=${pda_correctness_scale}"
  echo "pda_format_scale=${pda_format_scale}"
  echo "pda_projection=${pda_projection}"
  echo "pda_invalid_tail_tokens=${pda_invalid_tail_tokens}"
  echo "pda_protocol_mass=${pda_protocol_mass}"
  echo "pda_view_contrast_scale=${pda_view_contrast_scale}"
  echo "pda_protocol_guard_enable=${pda_protocol_guard_enable}"
  echo "pda_protocol_guard_multiplier=${pda_protocol_guard_multiplier}"
  echo "pda_protocol_guard_scope=${pda_protocol_guard_scope}"
  echo "pda_protocol_guard_valid_credit=${pda_protocol_guard_valid_credit}"
  echo "pda_correct_length_frontier_enable=${pda_correct_length_frontier_enable}"
  echo "hcaa_advantage_scale=${hcaa_advantage_scale}"
  echo "hcaa_amplification=${hcaa_amplification}"
  echo "hcaa_hardness_power=${hcaa_hardness_power}"
  echo "actor_clip_ratio_high=${actor_clip_ratio_high}"
  echo "repetition_quarantine_enable=${repetition_quarantine_enable}"
  echo "repetition_coefficient=${repetition_coefficient}"
  echo "strategy_archive_enable=${strategy_archive_enable}"
  echo "strategy_archive_coefficient=${strategy_archive_coefficient}"
  echo "forge_anneal_enable=${forge_anneal_enable}"
  echo "forge_anneal_start_step=${forge_anneal_start_step}"
  echo "forge_anneal_end_step=${forge_anneal_end_step}"
  echo "format_score=${format_score}"
  echo "response_budget_enable=${response_budget_enable}"
  echo "response_target_tokens=${response_target_tokens}"
  echo "response_dual_lr=${response_dual_lr}"
  echo "response_max_coef=${response_max_coef}"
  echo "response_ema_alpha=${response_ema_alpha}"
  echo "response_warmup_steps=${response_warmup_steps}"
  echo "lexi_format_coef=${lexi_format_coef}"
  echo "incorrect_tail_tokens=${incorrect_tail_tokens}"
  echo "incorrect_tail_coef=${incorrect_tail_coef}"
  echo "train_batch_size=${train_batch_size}"
  echo "val_batch_size=${val_batch_size}"
  echo "rollout_n=${rollout_n}"
  echo "logical_rollout_n=4"
  echo "n_gpus=${n_gpus}"
  echo "seed=${train_seed}"
  echo "sft_loss_coef=${sft_loss_coef}"
  echo "actor_lr=${actor_lr}"
  echo "test_freq=${test_freq}"
  echo "save_freq=${save_freq}"
  echo "val_before_train=${val_before_train}"
  echo "checkpoint_prefix=${checkpoint_prefix}"
} | tee "${run_dir}/run.env"

python3.10 -m verl.trainer.main_ppo \
  data.train_files="${train_file}" \
  data.val_files="${data_dir}/test.parquet" \
  data.train_batch_size="${train_batch_size}" \
  data.val_batch_size="${val_batch_size}" \
  data.max_prompt_length=256 \
  data.max_response_length=1024 \
  actor_rollout_ref.model.path="${model_path}" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr="${actor_lr}" \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size=4 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.loss_normalization="${actor_loss_normalization}" \
  actor_rollout_ref.actor.fixed_loss_normalizer="${fixed_loss_normalizer}" \
  actor_rollout_ref.actor.clip_ratio_high="${actor_clip_ratio_high}" \
  +actor_rollout_ref.actor.sft_loss_coef="${sft_loss_coef}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.rollout.n="${rollout_n}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size=4 \
  critic.model.path="${model_path}" \
  critic.ppo_micro_batch_size=4 \
  algorithm.kl_ctrl.kl_coef=0.001 \
  algorithm.adv_estimator=grpo \
  trainer.seed="${train_seed}" \
  'trainer.logger=["console","wandb"]' \
  +trainer.val_before_train="${val_before_train}" \
  trainer.default_hdfs_dir=null \
  trainer.default_local_dir="${checkpoint_prefix}" \
  trainer.n_gpus_per_node="${n_gpus}" \
  trainer.nnodes=1 \
  trainer.save_freq="${save_freq}" \
  trainer.test_freq="${test_freq}" \
  trainer.total_training_steps="${train_steps}" \
  trainer.total_training_steps_hint="${hint_steps}" \
  +trainer.uniform_sampling=False \
  +trainer.stage=False \
  trainer.adaptive_hint.enable=True \
  trainer.adaptive_hint.warmup_steps="${adaptive_warmup_steps}" \
  trainer.adaptive_hint.difficulty_mode=least_assisted \
  trainer.paired_hint.enable="${paired_hint_enable}" \
  trainer.paired_hint.min_hint_chunks="${paired_min_hint_chunks}" \
  trainer.paired_hint.ate_bonus="${paired_ate_bonus}" \
  trainer.posterior_frontier.enable="${posterior_frontier_enable}" \
  trainer.posterior_frontier.coefficient="${posterior_coefficient}" \
  trainer.posterior_frontier.min_observations="${posterior_min_observations}" \
  trainer.frontier_scaling.enable="${frontier_scaling_enable}" \
  trainer.frontier_scaling.min_weight="${frontier_min_weight}" \
  trainer.frontier_scaling.max_weight="${frontier_max_weight}" \
  trainer.frontier_scaling.min_observations="${frontier_min_observations}" \
  trainer.mirror_view.enable="${mirror_view_enable}" \
  trainer.mirror_view.paired="${mirror_view_paired}" \
  trainer.mirror_view.random_probability="${mirror_view_probability}" \
  trainer.mirror_view.schedule="${mirror_view_schedule}" \
  trainer.mirror_view.ramp_start_step="${mirror_ramp_start_step}" \
  trainer.mirror_view.ramp_end_step="${mirror_ramp_end_step}" \
  trainer.mirror_view.balanced="${mirror_view_balanced}" \
  trainer.format_constraint.enable="${format_constraint_enable}" \
  trainer.format_constraint.target_rate="${format_constraint_target_rate}" \
  trainer.format_constraint.dual_lr="${format_constraint_dual_lr}" \
  trainer.format_constraint.max_bonus="${format_constraint_max_bonus}" \
  trainer.format_constraint.min_bonus="${format_constraint_min_bonus}" \
  trainer.format_constraint.release_lr_scale="${format_constraint_release_lr_scale}" \
  trainer.format_constraint.lcb_z="${format_constraint_lcb_z}" \
  trainer.format_constraint.ema_alpha="${format_constraint_ema_alpha}" \
  trainer.format_constraint.warmup_steps="${format_constraint_warmup_steps}" \
  trainer.format_constraint.feedback_source="${format_constraint_feedback_source}" \
  trainer.format_constraint.apply_scope="${format_constraint_apply_scope}" \
  trainer.format_constraint.apply_hint_threshold="${format_constraint_apply_hint_threshold}" \
  trainer.anchor_probe.enable="${anchor_probe_enable}" \
  trainer.anchor_probe.size="${anchor_probe_size}" \
  trainer.anchor_probe.frequency="${anchor_probe_frequency}" \
  trainer.anchor_probe.seed="${anchor_probe_seed}" \
  trainer.deployment_probe.enable="${deployment_probe_enable}" \
  trainer.deployment_probe.feedback_source="${deployment_probe_feedback_source}" \
  trainer.deployment_probe.target_format_rate="${deployment_probe_target_rate}" \
  trainer.deployment_probe.breach_patience="${deployment_probe_breach_patience}" \
  trainer.deployment_probe.decay_factor="${deployment_probe_decay_factor}" \
  trainer.deployment_probe.recovery_step="${deployment_probe_recovery_step}" \
  trainer.deployment_probe.min_scale="${deployment_probe_min_scale}" \
  trainer.deployment_probe.warmup_steps="${deployment_probe_warmup_steps}" \
  trainer.repetition_quarantine.enable="${repetition_quarantine_enable}" \
  trainer.repetition_quarantine.coefficient="${repetition_coefficient}" \
  trainer.repetition_quarantine.ngram_size="${repetition_ngram_size}" \
  trainer.repetition_quarantine.min_repeats="${repetition_min_repeats}" \
  trainer.repetition_quarantine.invalid_tail_tokens="${repetition_invalid_tail_tokens}" \
  trainer.strategy_archive.enable="${strategy_archive_enable}" \
  trainer.strategy_archive.coefficient="${strategy_archive_coefficient}" \
  trainer.forge_anneal.enable="${forge_anneal_enable}" \
  trainer.forge_anneal.start_step="${forge_anneal_start_step}" \
  trainer.forge_anneal.end_step="${forge_anneal_end_step}" \
  trainer.format_score="${format_score}" \
  trainer.advantage_variant="${advantage_variant}" \
  trainer.drgrpo.advantage_scale="${drgrpo_advantage_scale}" \
  trainer.pda.correctness_scale="${pda_correctness_scale}" \
  trainer.pda.format_scale="${pda_format_scale}" \
  trainer.pda.projection="${pda_projection}" \
  trainer.pda.invalid_tail_tokens="${pda_invalid_tail_tokens}" \
  trainer.pda.protocol_mass="${pda_protocol_mass}" \
  trainer.pda.view_contrast_scale="${pda_view_contrast_scale}" \
  trainer.pda.protocol_guard_enable="${pda_protocol_guard_enable}" \
  trainer.pda.protocol_guard_multiplier="${pda_protocol_guard_multiplier}" \
  trainer.pda.protocol_guard_scope="${pda_protocol_guard_scope}" \
  trainer.pda.protocol_guard_valid_credit="${pda_protocol_guard_valid_credit}" \
  trainer.pda.correct_length_frontier_enable="${pda_correct_length_frontier_enable}" \
  trainer.hcaa.advantage_scale="${hcaa_advantage_scale}" \
  trainer.hcaa.amplification="${hcaa_amplification}" \
  trainer.hcaa.hardness_power="${hcaa_hardness_power}" \
  trainer.response_budget.enable="${response_budget_enable}" \
  trainer.response_budget.target_tokens="${response_target_tokens}" \
  trainer.response_budget.dual_lr="${response_dual_lr}" \
  trainer.response_budget.max_coef="${response_max_coef}" \
  trainer.response_budget.ema_alpha="${response_ema_alpha}" \
  trainer.response_budget.warmup_steps="${response_warmup_steps}" \
  trainer.lexiguard.format_coef="${lexi_format_coef}" \
  trainer.lexiguard.incorrect_tail_tokens="${incorrect_tail_tokens}" \
  trainer.lexiguard.incorrect_tail_coef="${incorrect_tail_coef}" \
  trainer.project_name=UFT \
  trainer.experiment_name="${experiment_name}" \
  trainer.lower_prob=0.05 \
  trainer.upper_prob=0.95 \
  trainer.total_epochs=30000 \
  2>&1 | tee "${run_dir}/train.log"

# Ray worker output can arrive after the driver-side tee closes when two Ray
# jobs run sequentially in one container.  Recover the worker log only when the
# canonical log has no step record; this avoids duplicates in the usual case.
if ! grep -aq 'step:[0-9][0-9]* -' "${run_dir}/train.log"; then
  while IFS= read -r worker_log; do
    if grep -aq 'step:[0-9][0-9]* -' "${worker_log}"; then
      sed -n '/epoch [0-9][0-9]*, step/,$p' "${worker_log}" >> "${run_dir}/train.log"
    fi
  done < <(find "${RAY_TMPDIR}/ray/session_latest/logs" -maxdepth 1 -type f -name 'worker-*-out' | sort)
fi
