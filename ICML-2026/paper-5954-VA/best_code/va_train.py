import os
import json
import logging
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
import torch.optim as optim
import deepspeed
from tqdm import tqdm
from peft import LoraConfig, TaskType, get_peft_model
from datasets import Dataset
from transformers import AutoTokenizer, HfArgumentParser

from arguments_va import ModelArguments, DataTrainingArguments, TrainingArguments
from models import Value_Aggregation_Gather, VAPPT_Gather
from loss_utils import mismatched_sizes_all_gather
use_auth_token = os.getenv("HUGGING_FACE_TOKEN")

# ========= 任务说明 =========
Instructions = {
    'allnli': 'Given a premise, retrieve a hypothesis that is entailed by the premise. Retrieve semantically similar text.',
    'dureader': 'Given a Chinese search query, retrieve web passages that answer the question.',
    'eli5_question_answer': 'Provided a user question, retrieve the highest voted answers on Reddit ELI5 forum.',
    'fever': 'Given a claim, retrieve documents that support or refute the claim.',
    'hotpot_qa': 'Given a multi-hop question, retrieve documents that can help answer the question.',
    'miracl': 'Given a question, retrieve Wikipedia passages that answer the question.',
    'mrtydi': 'Given a question, retrieve Wikipedia passages that answer the question.',
    'msmarco_document': 'Given a web search query, retrieve relevant documents that answer the query.',
    'msmarco_passage': 'Given a web search query, retrieve relevant passages that answer the query.',
    'nq': 'Given a question, retrieve Wikipedia passages that answer the question.',
    'quora_duplicates': [
        "Given a question, retrieve questions that are semantically equivalent to the given question.",
        "Find questions that have the same meaning as the input question.",
    ],
    'squad': 'Retrieve Wikipedia passages that answer the question.',
    't2ranking': 'Given a Chinese search query, retrieve web passages that answer the question.',
    'trivia_qa': 'Retrieve Wikipedia passages that answer the question.'
}

# ========= 工具函数 =========
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 确保 CUDA 操作具有确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"随机种子已设置为 {seed}")


def get_distributed_dataloader(dataset, batch_size: int, shuffle: bool = False):
    """
    dataset: HF Dataset 或任意 map-style Dataset
    """
    sampler = DistributedSampler(
        dataset,
        num_replicas=torch.distributed.get_world_size(),
        rank=torch.distributed.get_rank(),
        shuffle=shuffle,
        drop_last=False,
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


def load_data_and_sampling(file_path: str):
    """
    读取 file_path 下所有 jsonl 文件，构造 query / positive / negative。
    """
    all_files = os.listdir(file_path)
    all_data = []
    idx = 0

    for every_file in tqdm(all_files, desc="Loading raw data"):
        print(every_file)
        now_file = os.path.join(file_path, every_file)
        base_name = every_file[:-6]  # 假设文件名类似 allnli.jsonl

        with open(now_file, "r", encoding="utf-8") as f:
            for line in f:
                idx += 1
                inst_cfg = Instructions[base_name]
                if isinstance(inst_cfg, str):
                    instruction = inst_cfg
                else:
                    # quora_duplicates: 两条描述交替使用
                    instruction = inst_cfg[idx % 2]

                line = line.strip()
                if not line:
                    continue
                a_dict = json.loads(line)

                a_dict["query"] =  a_dict["query"] 
                # 保留原 positive / negative
                a_dict["positive"] = a_dict["positive"]
                a_dict["negative"] = a_dict["negative"]
                all_data.append(a_dict)

    if len(all_data) == 0:
        raise ValueError(f"未在 {file_path} 读取到任何样本")

    # 最多采样 1024000 条，样本不够就全用
    num_samples = min(1024, len(all_data))
    samples = random.sample(all_data, num_samples)

    query = [s["query"] for s in samples]
    positive = [s["positive"] for s in samples]
    negative = [s["negative"] for s in samples]

    return query, positive, negative


def build_dataset(data_args: DataTrainingArguments, training_args: TrainingArguments):
    """
    直接在脚本内做 tokenization，返回 HF Dataset（已经 map 好）。
    使用全局 tokenizer。
    """
    set_seed(2025)

    query, positive, negative = load_data_and_sampling(data_args.path)
    dataset = Dataset.from_dict(
        {"query": query, "positive": positive, "negative": negative}
    )
    dataset = dataset.shuffle(seed=42)

    max_length = data_args.max_length

    def tokenize(examples):
        # 一个 batch 内统一处理
        def process(texts):
            encoded = tokenizer(
                texts,
                add_special_tokens=True,
                truncation=True,
                padding=True,
                max_length=max_length,
            )
            return encoded["input_ids"], encoded["attention_mask"]

        query_input_ids, query_attention_mask = process(examples["query"])
        positive_input_ids, positive_attention_mask = process(examples["positive"])
        negative_input_ids, negative_attention_mask = process(examples["negative"])

        return {
            "query_input_ids": query_input_ids,
            "positive_input_ids": positive_input_ids,
            "negative_input_ids": negative_input_ids,
            "query_attention_mask": query_attention_mask,
            "positive_attention_mask": positive_attention_mask,
            "negative_attention_mask": negative_attention_mask,
        }

    encode_ds = dataset.map(
        tokenize,
        batched=True,
        batch_size=training_args.train_batch_size,
        num_proc=None,
    )

    return encode_ds


def all_gather_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    标准 all_gather，当前脚本实际用的是 loss_utils.mismatched_sizes_all_gather，
    这个函数可以作为备用。
    """
    world_size = torch.distributed.get_world_size()
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=0)


# ========= 解析命令行，初始化 tokenizer =========
parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()

tokenizer = AutoTokenizer.from_pretrained(
    model_args.model_name,
    use_auth_token=use_auth_token,
    add_eos_token=True
)

# 确保有 pad_token
if tokenizer.pad_token is None:
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        # 退一步，新增一个 pad_token
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})


# ========= 日志 =========
logger = logging.getLogger(name="my_logger")
os.makedirs("./wiki_train", exist_ok=True)
logging.basicConfig(
    filename=os.path.join("./wiki_train", "wiki_va_scl.log"),
    level=logging.INFO,
    format="%(name)s - %(levelname)s - %(message)s",
)

# ========= 构建数据集 =========
encode_ds = build_dataset(data_args, training_args)
encode_ds.save_to_disk("va_scl_data")  # 如果不想落盘可以注释掉

# 只保留真实存在的列
encode_ds.set_format(
    type="torch",
    columns=[
        "query_input_ids",
        "positive_input_ids",
        "negative_input_ids",
        "query_attention_mask",
        "positive_attention_mask",
        "negative_attention_mask",
    ],
)

# ========= 模型与 DeepSpeed =========
torch.cuda.set_device(training_args.local_rank)

model = VAPPT_Gather(training_args.local_rank)

lora_config = LoraConfig(
    r=16,
    target_modules=["q_proj", "v_proj", "o_proj", "k_proj"],
    task_type=TaskType.FEATURE_EXTRACTION,
    lora_alpha=32,
    lora_dropout=0.05,
)
# 给 backbone 注入 LoRA
model.model = get_peft_model(model.model, lora_config)
model.model.base_model.model.soft_prompt.weight.requires_grad = True
model.model_wrapper() 
for i in range(32):
    model.model.encoder.base_model.model.model.layers[i].input_layernorm.weight.requires_grad = True
# 打印可训练参数
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"参数名称: {name}\t参数尺寸: {param.size()}")


lora_parameters = []
layer_norm_params = []
for n, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if 'soft_prompt' in n:
        continue
    if 'layernorm' in name.lower(): # 判断名称中是否包含'layernorm'
        layer_norm_params.append(param)
        print(n)
        continue
    lora_parameters.append(p)

optimizer = optim.AdamW([
        {"params": model.model.encoder.base_model.model.soft_prompt.parameters(), "lr": 1e-4, "weight_decay": 0.01},
        {"params": lora_parameters, "lr": 3e-5, "weight_decay": 0.01},
        {"params": layer_norm_params, "lr": 1e-6, "weight_decay": 0.01}
    ], betas=(0.9, 0.95), eps=1e-6)

model_engine, optimizer, _, _ = deepspeed.initialize(
    args=training_args,
    config_params=training_args.deepspeed,
    model=model,
    optimizer=optimizer,
    model_parameters=model.parameters(),
)

# ========= DataLoader =========
# 根据 world_size 计算每卡 batch_size，而不是写死除以 8
world_size = torch.distributed.get_world_size()
per_device_batch_size = max(1, training_args.train_batch_size // world_size)

data_loader = get_distributed_dataloader(
    encode_ds,
    batch_size=per_device_batch_size,
    shuffle=False,  # 数据集已经 shuffle 过一次，如果想每轮再 shuffle 可以改成 True 并在每轮调用 sampler.set_epoch
)

# ========= 训练循环 =========
model_engine.train()
temperature = 0.05

for epoch in range(training_args.train_epoch):
    # 如果想让 DistributedSampler 每轮重排，可以在这里：
    # data_loader.sampler.set_epoch(epoch)

    for idx, batch in enumerate(
        tqdm(data_loader, desc=f"Epoch: {epoch + 1}", total=1000)
    ):
        batch = {k: v.cuda() for k, v in batch.items()}

        # 这里假设 va_model.forward 接受和 batch 键同名的参数
        # 比如 def forward(self, query_input_ids, query_attention_mask, positive_input_ids, ...)
        query_embedding, positive_embedding, negative_embedding = model_engine(**batch)

        full_query_embedding = mismatched_sizes_all_gather(query_embedding)
        full_query_embedding = torch.cat(full_query_embedding, dim=0)

        full_positive_embedding = mismatched_sizes_all_gather(positive_embedding)
        full_positive_embedding = torch.cat(full_positive_embedding, dim=0)

        full_negative_embedding = mismatched_sizes_all_gather(negative_embedding)
        full_negative_embedding = torch.cat(full_negative_embedding, dim=0)

        full_weight_embedding = torch.cat(
            [full_positive_embedding, full_negative_embedding], dim=0
        )

        dot_products = full_query_embedding @ full_weight_embedding.T
        probs = F.log_softmax(dot_products / temperature, dim=1)

        ground_truth = torch.arange(
            probs.shape[0], device=probs.device, dtype=torch.long
        )
        loss = F.nll_loss(probs, ground_truth)

        model_engine.backward(loss)

        current_lr = model_engine.get_lr()[0]
        if training_args.local_rank == 0:
            logger.info(
                f"Epoch: {epoch + 1}, Batch: {idx + 1}, Loss: {loss.item()}, LR: {current_lr}"
            )

        if (idx + 1) % 200 == 0:
            model_engine.save_checkpoint(f"{model_args.save_dir}_epoch_{epoch}_step_{idx}")

        model_engine.step()

    # 每个 epoch 结束再存一份
    model_engine.save_checkpoint(f"{model_args.save_dir}_epoch_{epoch}")


