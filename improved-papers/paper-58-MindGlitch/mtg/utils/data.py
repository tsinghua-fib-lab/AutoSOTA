from torch.utils.data import Dataset
from datasets import load_dataset


class HFDataset(Dataset):
    def __init__(self, hf_dataset) -> None:
        super().__init__()
        self.data = hf_dataset
        self.succeeded_cnt = 0

    def __getitem__(self, index):
        return self.data[index], index

    def __len__(self):
        return len(self.data)


def custom_collate(batch):
    data = batch[0][0]
    idx = batch[0][1]
    data["idx"] = idx  # Keep the index of of the origin EvalSubjects dataset for reference
    if data["description"]["category"] == None:
        data["description"]["category"] = ""

    if data["description"]["description_valid"] == None:
        data["description"]["description_valid"] = ""

    return data


def ds_filter_func(item):
    if item.get("collection") != "collection_2":
        return False
    if not item.get("quality_assessment"):
        return False
    return all(
        item["quality_assessment"].get(key, 0) >= 5
        for key in ["compositeStructure", "objectConsistency", "imageQuality"]
    )


def load_dataset_collection():
    ds = load_dataset("Yuanshi/Subjects200K", cache_dir="/datawaha/cggroup/eldesoa/datasets/hf_datasets")

    collection_2_valid = ds["train"].filter(
        ds_filter_func,
        num_proc=16,
        cache_file_name="./cache/dataset/collection_2_valid.arrow",  # Optional
    )
    return collection_2_valid