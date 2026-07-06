import os
from datasets import load_dataset


class LLaVABench:

    def __init__(self):
        # Try local parquet first (offline/paper_data), fall back to HF Hub
        local_parquet = "/paper_data/llava-bench-in-the-wild/data/train-00000-of-00001.parquet"
        if os.path.exists(local_parquet):
            self.ds = load_dataset("parquet", data_files=local_parquet)
        else:
            self.ds = load_dataset("lmms-lab/llava-bench-in-the-wild")

    def obtain_size(self):
        return len(self.ds["train"])

    def retrieve(self, idx):
        row = self.ds["train"][idx]
        result = {
            "idx": idx,
            "img": row["image"],
            "question": row["question"],
            "gt_ans": row["gpt_answer"],
        }
        return result

if __name__ == "__main__":
    benchmark = LLaVABench()
    print(benchmark.retrieve(0))
