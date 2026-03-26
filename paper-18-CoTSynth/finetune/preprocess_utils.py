from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset
from typing import List


def sanity_check(tokens: List[int], target: List[int]):
    assert len(tokens) == len(
        target
    ), f"length mismatch: {len(tokens)} vs {len(target)}"


class InputOutputDataset(Dataset):
    def __init__(
        self,
        data: List[dict],
        tokenizer: PreTrainedTokenizer,
        max_source_length: int,
        max_target_length: int,
    ):
        super(InputOutputDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.max_seq_length = max_source_length + max_target_length + 1
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i) -> dict:
        data_item = self.data[i]

        PROMPT_FORMAT = "<|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        prompt = PROMPT_FORMAT.format_map({'instruction': data_item["prompt"]})

        a_ids = self.tokenizer.encode(
            text=prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_source_length,
        )
        PROMPT_FORMAT = "{response}<|eot_id|>"
        response = PROMPT_FORMAT.format_map({'response': data_item["response"]})
        b_ids = self.tokenizer.encode(
            text=response,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_target_length,
        )
        context_length = len(a_ids)
        input_ids = a_ids + b_ids
        labels = (
            [-100] * context_length
            + b_ids
        )

        pad_len = self.max_seq_length - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * pad_len
        input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
        labels = labels + [-100] * pad_len

        assert len(input_ids) == len(
            labels
        ), f"length mismatch: {len(input_ids)} vs {len(labels)}"

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }