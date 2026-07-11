"""Runtime script for T2T-ViT on the ILSVRC2012 validation set.

This script reproduces the runtime numbers in Table 2 of
    Annabelle Michael Carrell, Albert Gong, Abhishek Shetty, Raaz Dwivedi, Lester Mackey
    Low-Rank Thinning
    https://arxiv.org/pdf/2502.12063

Example usage:
```bash
python runtime.py -m METHOD -bn BATCH_NUMBER
```

To compute runtime with fp16, add the --fp16 flag:
```bash
python runtime.py -m METHOD -bn BATCH_NUMBER --fp16
```

Example usage: display script arguments
```bash
python runtime.py --help
```
"""

import os
from functools import partial
import torch
import pandas as pd
from collections import defaultdict

from util_experiments import get_model, get_base_parser, get_modules
from imagenet import get_imagenet_datamodule

parser = get_base_parser()
parser.add_argument(
    "--method", "-m", default="wildcat", type=str, help="attention method"
)
parser.add_argument("--batch_size", "-bs", default=64, type=int, help="batch size")
parser.add_argument(
    "--batch_number", "-bn", default=1, type=int, help="batch number >= 1"
)
parser.add_argument(
    "--num_runs",
    "-n",
    default=1,
    type=int,
    help="number of runs (excluding warm-up runs)",
)
args, opt = parser.parse_known_args()

method = args.method
device = args.device if args.device else torch.device
batch_size = args.batch_size
batch_number = args.batch_number
ckpt_path = args.ckpt_path
dataset_path = args.dataset_path
output_path = args.output_path
num_runs = args.num_runs
seed = args.seed

# Set seed for reproducibility
print(f"Setting torch random seed to {seed}")
torch.manual_seed(seed)
# In compilation, capture dynamic output shape operators, like
# applying a mask to choose a coreset
#torch._dynamo.config.capture_dynamic_output_shape_ops = True###

print("Loading model...")
dtype = torch.float16 if args.fp16 else torch.float32
model = get_model(method, method, args, ckpt_path, device, dtype)

print("Loading data...")
batch_path = os.path.join(output_path, "batches")
os.makedirs(batch_path, exist_ok=True)
print(f"saving tensors to {batch_path}")
inputs_path = os.path.join(batch_path, f"batch-bs{batch_size}-bn{batch_number}.pt")

if os.path.exists(inputs_path):
    inputs = torch.load(inputs_path)
else:
    # Get validation set iterator for target batch size
    datamodule = get_imagenet_datamodule(dataset_path, batch_size=batch_size)
    loader_val = datamodule.val_dataloader()
    iter_val = iter(loader_val)

    # get the requested batch of inputs
    for ii in range(batch_number):
        inputs, _ = next(iter_val)
    # save the batch
    save_path = os.path.join(batch_path, f"batch-bs{batch_size}-bn{ii + 1}.pt")
    torch.save(inputs, save_path)

# Send to device
inputs = inputs.to(device, dtype=dtype)


# Define hook functions for timing individual modules
# using CUDA events
starts = {}
ends = {}


def time_pre(layer_name: str, module: torch.nn.Module, input: torch.Tensor) -> None:
    """Record the start time of the module.

    Args:
        layer_name (str): name of the module
        module (torch.nn.Module): module to time
        input (torch.Tensor): input to the module

    """
    starts[layer_name].record()


def time_post(
    layer_name: str, module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor
) -> None:
    """Record the end time of the module.

    Args:
        layer_name (str): name of the module
        module (torch.nn.Module): module to time
        input (torch.Tensor): input to the module
        output (torch.Tensor): output from the module

    """
    ends[layer_name].record()


print("Registering hooks...")
modules = get_modules(model)
for name, module in modules.items():
    starts[name] = torch.cuda.Event(enable_timing=True)
    module.register_forward_pre_hook(partial(time_pre, name))
    ends[name] = torch.cuda.Event(enable_timing=True)
    module.register_forward_hook(partial(time_post, name))

print("Performing warm-up runs...")
num_warmup_runs = 10
with torch.no_grad():
    for ii in range(num_warmup_runs):
        _ = model(inputs)
        # Ensure all GPU computation has completed
        torch.cuda.synchronize()
        # Calculate runtimes for each module
        for name, module in modules.items():
            print(f'Warmup {ii}, {name}: {starts[name].elapsed_time(ends[name])}')

times = defaultdict(list)
for ii in range(num_runs):
    # Run model forward pass to collect timings
    with torch.no_grad():
        _ = model(inputs)
    # Ensure all GPU computation has completed
    torch.cuda.synchronize()
    # Calculate runtimes for each module
    for name, module in modules.items():
        times[name].append(starts[name].elapsed_time(ends[name]))

# Write times to disk
times_dir = os.path.join(output_path, "times")
os.makedirs(times_dir, exist_ok=True)
times_file = os.path.join(
    times_dir,
    f"times-n{num_runs}-{method}-{device}-bs{batch_size}-bn{batch_number}.csv",
)
times_df = pd.DataFrame(times)

# print mean and std (but store all times)
mean_times = pd.concat(
    [times_df.mean().transpose(), times_df.std().transpose()], axis=1
)
mean_times.columns = ["mean", "std"]
mean_times["s"] = mean_times.apply(
    lambda x: f"{x['mean']:.2f} ± {x['std']:.2f}", axis=1
)

print(f"Saving times to {times_file}:\n{mean_times['s']}")
times_df.to_csv(times_file)
