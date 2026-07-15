# model/

### Overview

The **Model** module is the computational core of this toolkit. It handles the loading of Large Language Models (LLMs) and Sparse Autoencoders (SAEs), and executes inference to extract sparse feature activations.

It bridges `transformer_lens` (for model hooking) and `sae_lens` (for SAE encoding), providing adapters for specific SAE weight formats (e.g., BatchTopK).

### Core Components

#### 1. `load.py`: Model Loading & Adaptation
Handles model and SAE initialization with support for offline environments and format conversion.
* **LLM Loading**: Wraps `HookedTransformer` with a monkey-patching mechanism to intercept `AutoConfig`, enabling forced offline loading from local paths.
* **SAE Loading**: Provides `load_custom_batch_topk_as_jumprelu` to convert raw PyTorch weights (e.g., from BatchTopK training) into `sae_lens` native `JumpReLUSAE` instances, automatically handling parameter renaming and transposition.

#### 2. `run.py`: Activation Generation
Manages the forward pass and feature extraction pipeline.
* **Formatting**: Processes input text using prompt templates (User/Assistant) and identifies valid token start/end indices.
* **Caching & Encoding**: Runs the model to cache residual stream activations, encodes them via the SAE, and filters out high-norm outliers.
* **Sparse Storage**: Converts activation results into Sparse COO Tensors to optimize memory usage.

### Usage

This module is typically used in conjunction with the `data_loader` module. Below is a minimal example:

```python
import torch
from sae_analysis.src.model.load import load_hooked_transformer_offline, load_custom_batch_topk_as_jumprelu
from sae_analysis.src.model.run import generate_activations

# 1. Configuration
MODEL_PATH = "path/to/your/hf_model"
SAE_PATH = "path/to/your/sae/ae.pt"
LAYER = 18
DEVICE = "cuda"

# 2. Load Model (LLM)
tokenizer, model = load_hooked_transformer_offline(
    model_name="Qwen/Qwen3-8B",
    model_path=MODEL_PATH,
    device=DEVICE
)

# 3. Load SAE (Auto-converted to JumpReLU)
sae = load_custom_batch_topk_as_jumprelu(
    model_name="Qwen/Qwen3-8B",
    sae_id="custom_sae_id",
    sae_path=SAE_PATH,
    layer=LAYER,
    device=DEVICE
)

# 4. Generate Activations
# Assuming 'dataset' is a HuggingFace Dataset with a "prompt" column
results = generate_activations(
    tokenizer=tokenizer,
    model=model,
    sae=sae,
    layer=LAYER,
    dataset=dataset,
    data_type="prompt", 
    batch_size=1
)

# 5. Save Results
torch.save(results, "activations.pt")

```

### Output Structure

`generate_activations` returns a dictionary containing the necessary data for downstream analysis:

| Key | Type | Description |
| --- | --- | --- |
| `sparse_acts` | `torch.sparse_coo_tensor` | The core sparse activation matrix with shape `[Total_Tokens, d_sae]`, optimized via `coalesce()`. |
| `valid_token_idx` | `torch.Tensor` | Shape `[Batch, 2]`. Stores the start and end indices of valid text for each sample (excluding padding/special tokens). |
| `seq_lens` | `torch.Tensor` | Shape `[Batch]`. Records the sequence length of each sample. |
| `shape` | `torch.Size` | The original shape of the flattened activation tensor `[Batch * Length, d_sae]`. |

### Integration

* **Script**: See `0_generate_activations.py` in the project root for the full batch processing workflow.
* **Dependencies**: Requires `sae_lens` for SAE operations and `transformer_lens` for model instrumentation.