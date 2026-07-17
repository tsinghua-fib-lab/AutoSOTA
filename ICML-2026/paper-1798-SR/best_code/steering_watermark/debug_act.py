import sys, os
sys.path.insert(0, "src")
from llm_wrapper import LLMWrapper
import numpy as np

llm = LLMWrapper(
    hf_token=os.environ.get("HF_TOKEN", ""),
    model_id="/models/Llama-3.2-1B-Instruct",
    load_in_8bit=False,
    torch_dtype="torch.bfloat16",
)
print("Shape type:", llm.get_shape_type())

# Test hook
hooks = llm.register_hooks("gather", [15])
_ = llm.gathering_forward(["Hello world test"], max_new_tokens=1)
act = hooks[0].activations
print("Activation shape:", act.shape)
print("Activation type:", type(act))
print("Activation dtype:", act.dtype)
print("First token shape:", act[0].shape if hasattr(act[0], "shape") else len(act[0]))

# Check what split_data_accoring_to_sentence_id2 does
import pandas as pd
from data_processing import split_data_accoring_to_sentence_id2

test_df = pd.DataFrame([
    {
        "classification_label": 0,
        "input_text": "",
        "input_text_id": 0,
        "output_text": "test",
        "steering_noise": 0,
        "steering_type": "test",
        "steering_layers": [],
        "key_vector": np.zeros(2048),
        "input_token_length": 0,
        "input_token_ids": [],
        "output_token_strings": ["test"],
        "perplexity": 1.0,
        "log_diversity": 0.0,
        "quality": [0.5],
        "activations": {15: act},
    }
])
test_df["params"] = [None]

print("\nTesting split_data_accoring_to_sentence_id2...")
dft, dfv, dfte, sl = split_data_accoring_to_sentence_id2(
    test_df, val_size=0.1, test_size=0.2, seed=0,
    token_aggregation=False, sentence_array=False,
    max_token_seq=512, split_labels=[0],
)
print("Train rows:", len(dft))
if len(dft) > 0:
    fwd = dft["fwd_data"].values[0]
    print("fwd_data shape:", fwd.shape)

for h in hooks:
    h.remove()
del llm
print("Done!")
