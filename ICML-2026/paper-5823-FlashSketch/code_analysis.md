# Code Analysis - FlashSketch (Paper 5823) SOTA Optimization

## Evaluation Path
- Main script: external/GraSS/MLP_MNIST/score.py
- MLP (trained) + MNIST dataset for TRAK attribution
- Uses TRAKAttributor from external/GraSS/_dattri/algorithm/trak.py
- Projection via external/GraSS/_dattri/func/projection.py
- LDS metric via external/GraSS/_dattri/metric/lds

## Key Files
- Kernel: kernels/flashsketch/flashsketch_kernel.cu (block_perm_kernel + split variant)
- C++ binding: kernels/flashsketch/flashsketch_ext.cpp (launch config, affine params)
- Python wrapper: kernels/flashsketch/flashsketch.py (padding, scaling)
- Sketch interface: sketches/flashsketch.py (FlashSketchConfig dataclass)

## Config Path
- Score.py CLI args -> projector_kwargs -> FlashSketchConfig -> flashsketch_cuda_apply -> CUDA
- Key params: proj_dim=1024, kappa=4, s=4, block_rows=128, seed=42
- Template params: Tn=32, Tk=128, threads=256

## Metric Parser
- LDS from results/flashsketch_grass-1024.pt key lds
- Speedup = grass_proj_only_time / flashsketch_proj_only_time
- proj_only_time_ms from result file

## Baseline Metrics
- LDS: 0.371, Geomean Speedup: 3.29, proj_only_time_ms: 49.06
- grass LDS: 0.366, grass proj_only_time_ms: 161.49

## Nsight Compute
- NOT available in container. Cannot run ncu profiling.
- Fallback: cudaEvent-based timing already in TRAKAttributor

## Safe Modification Targets
1. flashsketch_kernel.cu: template params, launch bounds, inner loop
2. flashsketch_ext.cpp: launch config, affine params, bc_tiles logic
3. flashsketch.py (both): config defaults, scale computation
4. score.py: damping grid, seed loop (NOT metric definition)

## DO NOT MODIFY
- external/GraSS/_dattri/metric/lds.py (metric definition)
- external/GraSS/_dattri/benchmark/ (dataset/split definitions)
- external/GraSS/_dattri/algorithm/base.py (evaluation protocol)
