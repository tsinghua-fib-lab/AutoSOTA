import torch
from datetime import datetime

def get_current_datetime():
    # Get the current date and time
    current_datetime = datetime.now()

    # Convert it to a string
    datetime_string = current_datetime.strftime(
        "%Y-%m-%d_%H-%M-%S")  # Format as YYYY-MM-DD HH:MM:SS
    return datetime_string

def load_checkpoint_model(model, model_checkpoint_path):
    print(f"Loading model from {model_checkpoint_path}")
    load_item = torch.load(model_checkpoint_path)
    model.load_state_dict({k.replace('module.', ''): v for k, v in load_item.items()}, strict=False)
    return model 
