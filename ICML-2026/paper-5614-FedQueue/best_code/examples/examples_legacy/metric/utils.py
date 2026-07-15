import importlib


def get_metric(metric_path, metric_name=None):
    # Extract the module name from the file path (removing the ".py" extension)
    metric_path = metric_path.replace("/", ".")
    module_name = metric_path[:-3] if metric_path.endswith(".py") else metric_path

    # Import the module dynamically
    module = importlib.import_module(module_name)

    if metric_name is not None:
        # Get the specified class from the module
        metric = getattr(module, metric_name)
    else:
        # If no class name is given, return the first class found in the module
        metric = None
        for _, obj in module.__dict__.items():
            if callable(obj):  # Check if it's a function
                metric = obj
                break
    return metric
