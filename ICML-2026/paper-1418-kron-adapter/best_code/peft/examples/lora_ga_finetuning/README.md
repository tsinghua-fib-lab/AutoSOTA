- [Preparation](#preparation)
- [Quick start](#quick-start)
- [What exactly does the above code do?](#what-exactly-does-the-above-code-do)
- [Why do we need to save the model twice?](#why-do-we-need-to-save-the-model-twice)
- [Examples](#examples)
- [Detail usage of functions and classes](#detail-usage-of-functions-and-classes)
  - [LoraGAConfig](#loragaconfig)
  - [estimate\_gradient](#estimate_gradient)
  - [LoraGAContext](#loragacontext)
  - [save\_loraga\_model\_init](#save_loraga_model_init)
  - [save\_loraga\_model\_final](#save_loraga_model_final)
- [For quantization model](#for-quantization-model)
  - [LoRA](#lora)
  - [LoRA-GA](#lora-ga)
  - [Reason for offload](#reason-for-offload)
  - [Offload method](#offload-method)
- [Citation](#citation)

## Preparation

Clone the LoRA-GA repository and install custom `peft`:

```bash
git clone https://github.com/Outsider565/LoRA-GA.git
cd LoRA-GA
pip install -e peft
```

## Quick start

```python
from peft import PeftModel, get_peft_model, LoraGAConfig,
from peft.utils.lora_ga_utils import estimate_gradient, LoraGAContext, save_loraga_model_init, save_loraga_model_final

peft_config = LoraGAConfig(
    target_modules=find_all_linear_modules(model=model),
)

named_grad = estimate_gradient(
    model=model,
    dataloader=dataloader,
    accelerator=accelerator,
    quant_flag=False,
)

with LoraGAContext(model=model, named_grad=named_grad):
    model = get_peft_model(model=model, peft_config=peft_config, adapter_name="default")

save_loraga_model_init(model, save_dir=save_dir)

"""
train model
"""

save_loraga_model_final(model, save_dir=save_dir)
# after save_loraga_model_final, you can load it just like you load lora model
model = PeftModel.from_pretrained(model, save_dir)
```

## What exactly does the above code do?

1. `LoraGAConfig` is subclass of `LoraConfig`. `LoraGAConfig` will set `peft_type` to `PeftType.LORAGA` and `init_lora_weights` = "lora_ga".

2. `estimate_gradient` will use the data in the dataloader for forward and backward propagation, and return a dictionary named_grad. The key of this dictionary belongs to the submodule name of the model, and the value is the gradient of the weight W of the corresponding module.

3. `LoraGAContext` will attach named_grad to model as an attribute of model. named_grad will pass named_grad to `LoraGAModel` which is a subclass of LoraModel. After using named_grad to initialize `LoraGAModel(LoraModel)`, `LoraGAModel` frees it.

After you use `get_peft_model` to initialize the model, you can fine-tune the PEFT model in the same way you would with a standard LoRA model.

## Why do we need to save the model twice?

![](../resource/pic/lora_ga_algo.png)

When LoRA-GA initialization is performed, the weight W is modified as follows:

$$W_{init}=W_{pre\_trained}-\eta B_{init} A_{init}$$.

Obtain $W_{init}, A_{init}, B_{init}$ after LoRA-GA initialization.

Obtain $W_{init}, A_{final}, B_{final}$ after the train the peft model.

However, PEFT models only save the weights of the adapters. Therefore, to correctly capture the changes in the LoRA adapters during training, we need to save:
$$B_{final}A_{final}-B_{init}A_{init}$$


So you need to save the init adapter after `get_peft_model` (`save_loraga_model_init`), and then save the $final-init$ adapter after fine-tuning (`save_loraga_model_final`).

## Examples

1. [Example of float model](./float_llama2-7b_metamath.py)

2. [Example of quantized model](./quant_llama-2-7b_metamath.py)

## Detail usage of functions and classes

### LoraGAConfig

```python
@dataclass
class LoraGAConfig(LoraConfig):
    """
    Configuration for the LoRA-GA (Low-Rank Adaptation with Gradient Adjustment) model.

    This class extends `LoraConfig` to include additional parameters specific to LoRA-GA.
    It sets the `peft_type` to `PeftType.LORAGA` and initializes the LoRA weights with the "lora_ga" strategy.

    Attributes:
        bsz (int): The batch size used during gradient estimation. Defaults to 2.
        iters (int): The number of iterations (batches) for gradient estimation. Defaults to 2.
        direction (str): The direction for LoRA-GA adaptation. Defaults to "ArB2r".
        max_length (str): The maximum length for input sequences. Defaults to 1024.
        dtype (str): The data type for the model parameters. Defaults to "fp32".
        scale (str): The scaling method for gradients. Defaults to "stable".
        stable_gamma (int): The gamma parameter for stable scaling. Defaults to 16.

    Methods:
        __post_init__: Initializes `peft_type` to `PeftType.LORAGA` and `init_lora_weights` to "lora_ga".
    """
    bsz: int = field(
        default=2,
    )
    iters: int = field(
        default=64,
    )
    direction: str = field(
        default="ArB2r",
    )
    max_length: str = field(
        default=1024,
    )
    dtype: str = field(
        default="fp32",
    )
    scale: str = field(default="stable")
    stable_gamma: int = field(
        default=16,
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.LORAGA
        self.init_lora_weights = "lora_ga"
```

### estimate_gradient

```python
def estimate_gradient(
    model,
    dataloader,
    accelerator: Accelerator,
    quant_flag=False,
    origin_type="bf16",
    quant_type="nf4",
    no_split_module_classes=None,
) -> Dict[str, List[torch.Tensor]]:
    """
    Estimates the gradients of a model's parameters over a dataset.

    Args:
        model (torch.nn.Module): The model whose gradients will be estimated.
        dataloader (torch.utils.data.DataLoader): The dataloader for the dataset.
        accelerator (Accelerator): The accelerator used for training.
        quant_flag (bool, optional): If True, quantizes the model parameters. Defaults to False.
        origin_type (str, optional): The original data type of the model parameters. Defaults to "bf16".
        quant_type (str, optional): The data type for quantizing the model parameters. Defaults to "nf4".
        no_split_module_classes (list of type, optional): List of module classes that should not be split during offloading. Defaults to None.

    Returns:
        Dict[str, List[torch.Tensor]]: A dictionary mapping parameter names to their estimated gradients.
    """
```

Notice that the model should always be a floating-point model.

If you use a floating-point model, you can (and should) set `quant_flag` to False to get faster named gradients. However, you can also set `quant_flag` to False to reduce memory overhead. This approach might increase the time the function runs because it offloads part of the model to the CPU, ensuring that GPU consumption for estimating gradients does not exceed the GPU memory occupied by the quantized model training.

If `quant_flag` is set to False, the three arguments `origin_type`, `quant_type`, and `no_split_module_classes` will not have any effect.

If you want to pass a quantized model to `get_peft_model`, you should specify `origin_type` and `quant_type`. Currently, supported `origin_type` are fp32 and bf16, and supported `quant_type` are bitsandbytes int8 and nf4.

The `no_split_module_classes` argument is used to partition the model. For example, it can include residual blocks. The default value is `["LlamaDecoderLayer", "GPT2TransformerBlock", "T5Block", "GPT2Block", "FlaxGPT2Block"]`. You should set `no_split_module_classes` to your block name if your model is not in the default list.

### LoraGAContext

```python
class LoraGAContext:
    """
    Context manager for attaching and detaching a named gradient dictionary to a model.

    This context manager allows you to temporarily attach a dictionary of named gradients
    to the model as an attribute. Upon entering the context, the `named_grad` dictionary
    is set as an attribute of the model. Upon exiting the context, the attribute is removed.

    Attributes:
        model (torch.nn.Module): The model to which the gradient dictionary will be attached.
        named_grad (dict, optional): A dictionary where keys are parameter names and values are gradients. Defaults to None.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        named_grad: dict = None,
    ) -> None:
        self.model = model
        self.named_grad = named_grad

    def __enter__(self):
        setattr(self.model, "named_grad", self.named_grad)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self.model, "named_grad"):
            delattr(self.model, "named_grad")
```

`LoraGAContext` will attach named_grad to model as an attribute of model. named_grad will pass named_grad to `LoraGAModel` which is a subclass of `LoraModel`. After using named_grad to initialize `LoraGAModel(LoraModel)`, `LoraGAModel` free named_grad.

### save_loraga_model_init

```python
def save_loraga_model_init(model: PeftModel, save_dir: str):
    """
    Saves the initial state of a PEFT model with LoRA (Low-Rank Adaptation) layers to a specified directory.

    Args:
        model (PeftModel): The PEFT model to be saved.
        save_dir (str): The directory where the model will be saved.
    """
```

Save $A_{init}$ and $B_{init}$

### save_loraga_model_final

```python
def save_loraga_model_final(model: PeftModel, save_dir: str):
    """
    Saves the final state of a PEFT (Parameter-Efficient Fine-Tuning) model with LoRA (Low-Rank Adaptation) layers and performs cleanup.

    Args:
        model (PeftModel): The PEFT model to be saved.
        save_dir (str): The directory where the model checkpoints will be saved.

    Notes:
        This function saves the model state with the suffix `_final_lora_checkpoint`,
        loads the model from the directory to apply updates, and then deletes both the
        initial and final checkpoint directories. The model is saved again to the
        provided `save_dir` after cleanup.
    """
```

1. Save $A_{final}$ and $B_{final}$

2. Load $A_{init}$ and $B_{init}$ to init_adapter

3. Load $A_{final}$ and $B_{init}$ to init_adapter

4. Get final_adapter - init_adapter

5. Delete checkpoint of init_adapter and checkpoint of final_adapter

6. Save (final-init) adapter

## For quantization model

For quantized model, `estimated_gradient` will spend more time. The reason is below

### LoRA

$W=W0+\alpha AB$

### LoRA-GA

LoRA-GA needs to obtain the partial derivative of the loss with respect to W (equivalent to W0).

However, the W0 of the quantized model is stored in integer format,

and PyTorch does not support obtaining gradients for integer data.

Therefore, please ensure that the model passed to `estimate_gradient` is always a floating-point model.

### Reason for offload

To obtain the derivative of the loss with respect to W, the original floating-point model needs to be used to estimate the gradient.

To ensure that the GPU consumption for `estimate_gradient` does not exceed the GPU memory occupied by the quantized model during training, it is necessary to offload part of the model to the CPU when estimating the gradient.

### Offload method

```python
Initial: Model on the cpu

Divide the model into K blocks.
for i in range(0,K):
    Load the i-th block to the gpu
    Execute the forward of the i-th block
    if i != K-1:
        # If it is not the last block, offload the i-th block to the cpu
        offload the i-th block to the cpu
    else：
        # If it is the last block, because the last block needs to be back-propagated first, no offload is needed
        do nothing
for i in range(K-1, -1, -1):
    Load the i-th block to the gpu
    Execute the backward of the i-th block
    if i != 0:
        # If it is not the 0th block, offload the i-th block to the cpu
        offload the i-th block to the cpu
    else：
        # If it is block 0, since the forward of the next batch first requires block 0 to be forwarded, no offload is needed
        do nothing
```

## Citation

```
@misc{wang2024loragalowrankadaptationgradient,
      title={LoRA-GA: Low-Rank Adaptation with Gradient Approximation},
      author={Shaowen Wang and Linxi Yu and Jian Li},
      year={2024},
      eprint={2407.05000},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.05000},
}
```
