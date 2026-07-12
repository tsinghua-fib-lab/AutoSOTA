import os, sys
os.chdir("/repo")
sys.path.insert(0, "/repo")

from utils import Config, StageManager
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("/models/Llama-3.2-1B", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    "/models/Llama-3.2-1B",
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
    trust_remote_code=True,
)

newly_added = tokenizer.add_tokens(["<thinking>"])
thinking_token_id = tokenizer.convert_tokens_to_ids("<thinking>")
if newly_added > 0:
    model.resize_token_embeddings(len(tokenizer))
    embeddings = model.get_input_embeddings()
    embeddings.weight.data[thinking_token_id] = embeddings.weight.data[tokenizer.eos_token_id].clone()

from model import LT_Tuning_Model
lt_model = LT_Tuning_Model(
    base_causallm=model,
    thinking_token_id=thinking_token_id,
    eos_token_id=tokenizer.eos_token_id,
    hidden_state_layer_index=-2,
    stage_mode="common",
    fusion_alpha=0.5,
    fusion_top_p=0.8,
    fusion_temperature=1.0,
)

from dataset import get_dataset, get_cot_latent_dataset, build_thinking_strategy, MyCollator

train_raw = get_dataset("data/gsm8k/train_socratic.jsonl", dataset_name="gsm8k")
print("Raw train data:", len(train_raw), "samples")

class FakeConfig:
    pass

config = FakeConfig()
config.current_stage_mode = "common"
config.current_stage_idx = 0
config.thinking_strategy = "confidence"
config.thinking_token = "<thinking>"
config.use_unk_for_thinking = False
config.reinforce_prob_threshold = [0.0, 0.3, 0.2]
config.reinforce_max_eval_length = 2048
config.thinking_insertion_prob = [0.0, 0.85, 0.95]
config.thinking_secondary_insertion_prob = [0.0, 0.15, 0.2]
config.thinking_operator_regex = "[0-9]+|[+\\-*/=]"
config.tokens_per_stage = 10
config.fusion_alpha = [0.5, 0.5, 0.6]
config.fusion_top_p = 0.8
config.fusion_temperature = 1.0
config.seed = 42
config.thinking_prompt_tokens = 0

train_ds_stage0 = get_cot_latent_dataset(
    stage_type="common",
    base_dataset=train_raw,
    configs=config,
    strategy=None,
    tokenizer=tokenizer,
    shuffle=True,
    debug_num=10,
)
print("Stage0 dataset:", len(train_ds_stage0), "samples")

collator = MyCollator(tokenizer=tokenizer, thinking_id=None, label_pad_token_id=-100)
batch = collator([train_ds_stage0[i] for i in range(min(4, len(train_ds_stage0)))])
print("Batch keys:", list(batch.keys()))
print("input_ids shape:", batch["input_ids"].shape)

batch_gpu = {k: v.to("cuda:0") for k, v in batch.items() if isinstance(v, torch.Tensor)}
with torch.no_grad():
    outputs = lt_model(**batch_gpu)
print("Training forward pass OK, loss:", outputs.loss.item())

print("ALL INTEGRATION TESTS PASSED!")
