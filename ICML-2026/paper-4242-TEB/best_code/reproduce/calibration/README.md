# Calibration files

The EB scheduler reads a per-(model, GPU) calibration JSON to initialize the
cost-model coefficients. The runner (`common/common_eb.sh::resolve_calibration`)
looks up files by name:

```
pd_calibration_<MODEL_SHORT>_<GPU_TAG>.json    # preferred (per-GPU)
pd_calibration_<MODEL_SHORT>.json              # legacy fallback (any GPU)
```

`<GPU_TAG>` is one of `H200`, `H100`, `B300`, `B200`, `RTXPRO6000`, `L40S`,
`A6000`, etc. (auto-detected by `detect_gpu_name`).

## Provided sample

- `pd_calibration_Qwen3-8B_H200.json` — measured on NVIDIA H200, Qwen3-8B.

## Generating for your hardware

```bash
python -m vllm.v1.core.sched.calibration \
    --model Qwen/Qwen3-8B \
    --output calibration/pd_calibration_Qwen3-8B_<YOUR_GPU_TAG>.json
```

Calibration is a one-time cost per (model, GPU): a few minutes of GPU time.
The JSON is tiny (~500 bytes); commit yours alongside if you want others to
reproduce on the same hardware.
