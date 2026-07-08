from datetime import datetime, timedelta
from functools import wraps
from typing import Dict, List
from accelerate import Accelerator
import torch
from tqdm import tqdm
import torch.distributed as dist
from peft import PeftModel
import os


def timer(data_format="ms"):
    """
    A decorator that prints the execution time of a function.

    Args:
        data_format (str, optional): The format in which to display the execution time. Defaults to "ms".
                                     (Note: The current implementation only prints time in minutes and seconds.)

    Returns:
        function: The decorated function with execution time logging.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            begin_time = datetime.now()
            result = func(*args, **kwargs)
            end_time = datetime.now()
            cost = (end_time - begin_time).seconds
            print(
                func.__name__ + " ran" + f" {cost // 60} min {cost % 60}s",
            )
            return result

        return wrapper

    return decorator


def get_record_gradient_hook(model, record_dict):
    """
    Creates a hook to record the gradients of a model's parameters into a dictionary.

    Args:
        model (torch.nn.Module): The model whose gradients will be recorded.
        record_dict (dict): A dictionary to store the recorded gradients.
    """

    def record_gradient_hook(grad):
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                if n not in record_dict:
                    record_dict[n] = p.grad.cpu()
                else:
                    record_dict[n] += p.grad.cpu()
                p.grad = None
        return grad

    return record_gradient_hook


@timer()
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
    if accelerator and model.device.type != "cuda":
        if not quant_flag:
            model.to(accelerator.device)
        else:
            model.to("cpu")
        model.train()
        dataloader = accelerator.prepare(dataloader)
    named_grads = {}
    num_batch = 0
    from .offload_utils_for_quant import show_gpu_and_cpu_memory
    from .offload_utils_for_quant import OffloadContext

    with OffloadContext(
        model=model,
        named_grads=named_grads,
        quant_flag=quant_flag,
        origin_type=origin_type,
        quant_type=quant_type,
        no_split_module_classes=no_split_module_classes,
    ):
        for batch in tqdm(dataloader, desc="Estimating gradient"):
            print(f"batch_size=", len(batch["input_ids"]))
            print("before forward===========================================================")
            show_gpu_and_cpu_memory()
            num_batch += 1
            batch = {k: v for k, v in batch.items()}
            outputs = model(**batch)
            show_gpu_and_cpu_memory()
            print("before backward===========================================")
            show_gpu_and_cpu_memory()
            outputs.loss.backward()
            print("after backward ===========================================================")
            show_gpu_and_cpu_memory()

            get_record_gradient_hook(model, named_grads)(None)  # get gradient of last layer
            # make sure the gradient is cleared
            for grad_name, param in model.named_parameters():
                if param.grad is not None:
                    param.grad = None
    for grad_name, _ in named_grads.items():
        named_grads[grad_name] /= num_batch
    torch.cuda.empty_cache()
    if accelerator and accelerator.num_processes > 1:
        accelerator.wait_for_everyone()
        accelerator.print("Gradient estimation finished, gathering results")
        for name, processed_gradient in tqdm(named_grads.items(), desc="Gathering gradient"):
            processed_gradient = processed_gradient.to(accelerator.device)
            dist.all_reduce(processed_gradient, op=dist.ReduceOp.AVG)
            named_grads[name] = processed_gradient.to("cpu")
    named_grads = {".".join(k.split(".")[:-1]): v for k, v in named_grads.items()}
    return named_grads


@timer()
def save_loraga_model_init(model: PeftModel, save_dir: str):
    """
    Saves the initial state of a PEFT model with LoRA (Low-Rank Adaptation) layers to a specified directory.

    Args:
        model (PeftModel): The PEFT model to be saved.
        save_dir (str): The directory where the model will be saved.
    """

    init_suffix = "init_lora_checkpoint"
    save_dir = os.path.join(save_dir, init_suffix)
    model.save_pretrained(save_dir)


@timer()
def save_loraga_model_final(model: PeftModel, save_dir: str):
    """
    Saves the final state of a PEFT (Parameter-Efficient Fine-Tuning) model with LoRA (Low-Rank Adaptation) layers and performs cleanup.

    Args:
        model (PeftModel): The PEFT model to be saved.
        save_dir (str): The directory where the model checkpoints will be saved.
    """
    import os
    import shutil

    init_suffix = "init_lora_checkpoint"

    tmp_save_dir = save_dir
    model.save_pretrained(tmp_save_dir, path_initial_model_for_weight_conversion=os.path.join(save_dir, init_suffix))

    tmp_save_dir = os.path.join(save_dir, init_suffix)
    if os.path.exists(tmp_save_dir):
        print(f"delete {tmp_save_dir}")
        shutil.rmtree(tmp_save_dir)


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
