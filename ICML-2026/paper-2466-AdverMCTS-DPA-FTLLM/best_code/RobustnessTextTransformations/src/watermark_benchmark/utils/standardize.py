""" Standardization """


def standardize(model, sys, user):
    """Return a standardized version of the prompt for a given model"""

    sys = sys.strip() if sys else None
    user = user.strip()

    model = model.lower().strip()

    if "llama-3" in model or ("llama" in model and "3" in model):
        if sys:
            return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{sys} <|eot_id|> \n<|start_header_id|>user<|end_header_id|>\n{user} <|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
        else:
            return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{user} <|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
    elif "llama" in model:
        if sys:
            return f"<s>[INST] <<SYS>> {sys} <</SYS>> {user} [/INST]"
        else:
            return f"<s>[INST] {user} [/INST]"

    if "vicuna" in model or "koala" in model:
        if sys:
            return f"System: {sys}\nHuman: {user}\nAssistant:"
        else:
            return f"Human: {user}\nAssistant:"

    if "mistral" in model.lower():
        if sys:
            return f"<s>[INST] {sys} Hi [/INST] Hello! how can I help you</s>[INST] {user} [/INST]"
        else:
            return f"<s>[INST] {user} [/INST]"

    else:
        raise NotImplementedError(
            f"No known standardization for model {model}. \
                                  Please add it manually to utils/standardize.py"
        )
