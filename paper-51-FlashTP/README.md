# FlashTP [[paper](https://openreview.net/pdf?id=wiQe95BPaB)]

# Overview
**FlashTP** is a high-performance Tensor-Product library for Machine Learning Interatomic Potentials (MLIPs). 
It is a drop-in replacement for channelwise tensorproduct ("uvu") that uses `e3nn.o3.TensorProduct`.

FlashTP implements fused, sparsity-aware CUDA kernels to accelerate the Tensor-Product layer in MLIPs, leveraging optimizations such as:

- **Kernel fusion**  
- **Sparse computation**  
- **Input reuse**  

For full details on the optimizations, please see our ICML ’25 [paper](https://openreview.net/pdf?id=wiQe95BPaB).

# TODO
- [ ] **Performance Improvements** 
- [ ] **JIT implemenation** 

# Prerequisites
- Python 3.11 (Not a strict requirement)
- CUDA Toolkit 12 or higher  
- PyTorch 2.4.1 or higher  
- [e3nn](https://github.com/e3nn/e3nn)  
- [torch_scatter](https://github.com/rusty1s/pytorch_scatter) (for accurate _e3nn_ performance benchmarks)

# Installation and features
1. Clone this repository:  
   ```bash
   git clone https://github.com/SNU-ARC/flashTP.git
   cd flashTP
   ```
2. Ensure CUDA toolkit 12 or higher is installed on your system.
3. Install the Python dependencies and the package itself: (Compiling the CUDA kernels can take **up to 10 minutes**)
   
   By default, builds optimized kernels for NVIDIA A100 and H100 GPUs.
   
   You can customize `CUDA_ARCH_LIST` to match the Compute Capability of your NVIDIA GPU.
   Please view [this link](https://developer.nvidia.com/cuda-gpus) for Compute Capability.
     
   ```bash
   pip install -r requirements.txt
   CUDA_ARCH_LIST="80;90" pip install . --no-build-isolation 
   ```
   We recommend using Virtualenv or Conda to manage the dependencies.
   

# How to use FlashTP
## Standard `e3nn` TensorProduct 
```python
import torch
import e3nn.o3
from torch_scatter import scatter

# Initalization
tp = e3nn.o3.TensorProduct(i_in1,i_in2,i_out,inst_tuple,shared_weights=False, internal_weights=False)

# Execution
in1 = in1_node[edge_src]
out_large_e3nn = tp(in1,in2,weight)
out_e3nn = scatter(out_large_e3nn, edge_dst.to(torch.int64), dim=0, dim_size=total_node, reduce="sum")
```

## FlashTP replacement
```python
import torch
import flashTP_e3nn

# Initalization
flashtp = flashTP_e3nn.uvu_TP(i_in1,i_in2,i_out,inst_tuple, device="cuda", dtype=used_dtype)

# Execution
out_ours = flashtp(in1,in2,weight, edge_src, edge_dst)
```

# Evaluation
Run ./example/test.sh to run microbenchmarks in ./ir_config.
For detailed information on the input parameters, please refer to `./example/flashTP_test.py`.
```bash
cd example
bash ./test.sh
```
For SevenNet integration please checkout [flash branch of SevenNet](https://github.com/MDIL-SNU/SevenNet/tree/flash).

# Citation
Please cite our paper if you find our work useful.
```
@inproceedings{leeflashtp,
  title={FlashTP: Fused, Sparsity-Aware Tensor Product for Machine Learning Interatomic Potentials},
  author={Lee, Seung Yul and Kim, Hojoon and Park, Yutack and Jeong, Dawoon and Han, Seungwu and Park, Yeonhong and Lee, Jae W},
  booktitle={Forty-second International Conference on Machine Learning}
}
```
