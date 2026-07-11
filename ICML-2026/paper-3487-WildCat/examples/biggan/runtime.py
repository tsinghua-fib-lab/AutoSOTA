"""Runtime script for BigGAN on the ILSVRC2012 validation set.

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
import numpy as np
from tqdm import tqdm

from util_experiments import get_model, get_base_parser, get_modules
from biggan_models.utils import one_hot_from_int, truncated_noise_sample
#from imagenet import get_imagenet_datamodule

parser = get_base_parser()
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

method = args.attention
device = args.device if args.device else torch.device
batch_size = args.batch_size
batch_number = args.batch_number
ckpt_path = args.ckpt_path
output_path = args.output_path
num_runs = args.num_runs
model_name = args.model_name
attention = args.attention
truncation = args.truncation
num_classes = args.num_classes
data_per_class = args.data_per_class
g = args.g
r = args.r
bins = args.bins

print("Loading model...")
dtype = torch.float16 if args.fp16 else torch.float32
model = get_model(model_name, attention, g=g, r=r, bins=bins)
print(f'type(model): {type(model)}')

# quit()

print("Loading data...")
# Prepare a input
if args.data_per_class > 0:
    labels = np.repeat(np.arange(num_classes), data_per_class).tolist()
elif args.num_outputs > 0:
    labels = np.random.randint(num_classes, size=(args.num_outputs,))

class_vector = one_hot_from_int(labels, batch_size=len(labels))
noise_vector = truncated_noise_sample(truncation=truncation, batch_size=len(labels), seed=args.seed)

# All in tensors
noise_vector = torch.from_numpy(noise_vector)
class_vector = torch.from_numpy(class_vector)

if torch.cuda.is_available():
    # If you have a GPU, put everything on cuda
    noise_vector = noise_vector.to('cuda')
    class_vector = class_vector.to('cuda')
    model = model.to('cuda')

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
    if name == "attention-matrix":
        starts[name] = torch.cuda.Event(enable_timing=True)
        module.register_forward_pre_hook(partial(time_pre, name))
        ends[name] = torch.cuda.Event(enable_timing=True)
        module.register_forward_hook(partial(time_post, name))

print(f'module attention matrix {modules["attention-matrix"]}')

print("Performing warm-up runs...")
num_warmup_runs = 20
with torch.no_grad():
    for idx in range(num_warmup_runs):
        batch_idx = list(range(idx * batch_size, min(len(labels), (idx+1) * batch_size)))
        if len(batch_idx) == 0:
            continue
        # res_all.append(batch_idx)

        n_vec = noise_vector[batch_idx]
        c_vec = class_vector[batch_idx]

        # Generate an image
        _ = model(n_vec, c_vec, truncation)

        # Ensure all GPU computation has completed
        torch.cuda.synchronize()
        # Calculate runtime for each module
        for name in starts.keys():
            print(f'Warmup {idx}, {name}: {starts[name].elapsed_time(ends[name])}')

print("Performing timed runs...")
times = defaultdict(list)
for idx in range(num_runs+num_warmup_runs):
    batch_idx = list(range(idx * batch_size, min(len(labels), (idx+1) * batch_size)))
    if len(batch_idx) == 0:
        continue
    if idx < num_warmup_runs:
        continue
    # res_all.append(batch_idx)

    n_vec = noise_vector[batch_idx]
    c_vec = class_vector[batch_idx]

    # Generate an image
    with torch.no_grad():
        output = model(n_vec, c_vec, truncation)
    # Ensure all GPU computation has completed
    torch.cuda.synchronize()
    # Calculate runtimes for each module
    for name in starts.keys():
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