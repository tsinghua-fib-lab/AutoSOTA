import os 
from collections import OrderedDict

# --- Identify launcher name ---
def get_launcher_name():
    # Try to get BASH source if exists
    bash_source = os.environ.get('BASH_SOURCE')
    if bash_source:
        return os.path.splitext(os.path.basename(bash_source))[0]
    # If not, fallback to None → means terminal or Python
    return None



# --- Clean metric keys (remove dataset prefix) ---
def clean_metric_keys(metrics_dict, dataset_name=None):
    clean_dict = {}
    for k, v in metrics_dict.items():
        if dataset_name and k.startswith(dataset_name + "_"):
            k = k[len(dataset_name) + 1 :]
        clean_dict[k] = v
    return clean_dict


# --- Filter args to include only simple key–value pairs ---
def flatten_args(args):
    clean = {}
    for k, v in vars(args).items():
        # Skip private/internal attributes or callable objects
        if k.startswith("_") or callable(v):
            continue
        # Skip WandB and parser internals
        if k in ["_dynamic", "_prevent_method_masking", "_actions", "_option_string_actions", "_get_kwargs"]:
            continue
        # Convert complex structures to str (to keep csv valid but readable)
        if isinstance(v, (list, dict, OrderedDict, tuple)):
            clean[k] = str(v)
        else:
            clean[k] = v
    return clean