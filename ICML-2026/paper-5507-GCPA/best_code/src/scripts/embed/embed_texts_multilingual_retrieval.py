import os
import datasets
import pickle
from datasets import DatasetDict, Value
from latentis.data.dataset import Feature, DataType, HFDatasetView
from collections import Counter
import random
from tqdm import tqdm
import torch
from latentis.data.encoding.encode import EncodeTask
from latentis.data.utils import default_collate
from latentis.nn.encoders import TextHFEncoder
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
DATA_DIR = Path(os.environ["DATA_PATH"])
INSTANCE_IDS_PATH = DATA_DIR / "ted_multi_instance_ids.pkl"


def load_ted_multi():
    try:
        return datasets.load_dataset("neulab/ted_multi")
    except RuntimeError as exc:
        if "Dataset scripts are no longer supported" not in str(exc):
            raise

        return datasets.load_dataset(
            "parquet",
            data_files={
                split: f"hf://datasets/neulab/ted_multi/plain_text/ted_multi-{split}*.parquet"
                for split in ("train", "validation", "test")
            },
        )


ds = load_ted_multi()
print(ds["train"][0]["translations"]["language"][4], ds["train"][0]["translations"]["translation"][4])
print(ds["train"][0]["translations"]["language"][5], ds["train"][0]["translations"]["translation"][5])

def add_id_column(d):
    d = d.map(lambda ex, idx: {"id": idx}, with_indices=True)
    d = d.cast_column("id", Value("int64"))
    return d

ds_new = DatasetDict({split: add_id_column(ds[split]) for split in ["train","validation","test"]})

# sanity check
for split in ds_new:
    assert "id" in ds_new[split].column_names, f"'id' missing in {split}"
    assert len(ds_new[split]) == len(set(ds_new[split]["id"])), f"ids not unique in {split}"


features_with_names = [
    Feature(
        name="id",
        data_type=DataType.LONG,
    ),
    Feature(
        name="talk_name",
        data_type=DataType.TEXT,
    ),
    Feature(
        name="translations",
        data_type=DataType.TEXT,  # treat translations as text
    ),
]

hf_ds = HFDatasetView(
    name="ted_multi",
    hf_dataset=ds_new,
    id_column="id",
    features=features_with_names,
)
print(len(hf_ds.hf_dataset["train"]))
lang_set = set(hf_ds.hf_dataset["train"][0]["translations"]["language"])
for instance in hf_ds.hf_dataset["train"]:
    lang_set = lang_set.intersection(set(instance["translations"]["language"]))

print(lang_set)


lang_counter = Counter()
for instance in hf_ds.hf_dataset["train"]:
    for lang in instance["translations"]["language"]:
        lang_counter[lang] += 1

print(lang_counter)
lang_global_set = set(["en", "ar", "he", "ru", "ko", "it", "ja", "es", "zh-cn", "fr"])
instance_ids = []

for instance in tqdm(hf_ds.hf_dataset["train"]):
    inter = lang_global_set.intersection(set(instance["translations"]["language"]))
    if len(inter) != len(lang_global_set):
        # print(f"Instance {instance['id']} does not have all languages: {inter} vs {lang_global_set}")
        continue

    instance_ids.append(instance["id"])

print(f"Number of instances with all languages: {len(instance_ids)}")
print(f"Sample instance ids: {instance_ids[:5]}")
encoder_map = {
    "en": "roberta-base",         # English
    "ar": "aubmindlab/bert-base-arabertv02",               # Arabic
    "he": "avichr/heBERT",                                 # Hebrew
    "ru": "DeepPavlov/rubert-base-cased",                  # Russian
    "ko": "kykim/bert-kor-base",                           # Korean
    "it": "dbmdz/bert-base-italian-uncased",               # Italian
    "ja": "cl-tohoku/bert-base-japanese",                  # Japanese
    "es": "dccuchile/bert-base-spanish-wwm-uncased",       # Spanish
    "zh-cn": "google-bert/bert-base-chinese",          # Simplified Chinese
    "fr": "camembert-base",                                # French
}
# Split instance_ids into train and test randomly.
random.seed(42)

random.shuffle(instance_ids)
split_idx = int(0.8 * len(instance_ids))
train_instance_ids = instance_ids[:split_idx]
test_instance_ids = instance_ids[split_idx:]

print(f"Number of training instances: {len(train_instance_ids)}")
print(f"Number of test instances: {len(test_instance_ids)}")

with open(INSTANCE_IDS_PATH, "wb") as f:
    pickle.dump({
        "train": train_instance_ids,
        "test": test_instance_ids
    }, f)


with open(INSTANCE_IDS_PATH, "rb") as f:
    instance_ids = pickle.load(f)

train_instance_ids = instance_ids["train"]
test_instance_ids = instance_ids["test"]

print(f"Number of training instances: {len(train_instance_ids)}")
print(f"Number of test instances: {len(test_instance_ids)}")

num_samples = None
feature_name = "translations"

for lang, encoder_name in encoder_map.items():
    print(f"Processing language: {lang} with encoder: {encoder_name}")
    for split in ['train', 'test']:
        if split == 'train':
            instance_ids_to_use = set(train_instance_ids)
        else:
            instance_ids_to_use = set(test_instance_ids)

        # Filter the dataset view to only include instances with the specified languages
        data = hf_ds.hf_dataset["train"].filter(lambda x: x["id"] in instance_ids_to_use)


        hf_ds_view = HFDatasetView(
            name=hf_ds.name,
            hf_dataset=DatasetDict({split: data}),
            id_column=hf_ds.id_column,
            features=features_with_names,
        )

        assert len(hf_ds_view.hf_dataset[split]) == len(instance_ids_to_use), f"Filtered dataset does not match expected number of instances for {lang} in {split}"

        encoder = TextHFEncoder(encoder_name, trans_variable_lang=lang)

        target_dir = DATA_DIR / hf_ds.name / "encodings" / encoder_name.replace("/", "-") / split
        if target_dir.exists():
            print(f"Skipping {encoder_name} for split {split} as target directory already exists: {target_dir}")
            continue
        
        task = EncodeTask(
            dataset_view=hf_ds_view,
            split=split,
            feature=feature_name,
            model=encoder,
            collate_fn=default_collate,
            encoding_batch_size=32,
            num_workers=16,
            pooler=None,
            save_source_model=False,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            target_path=target_dir,
            write_every=5,
            only_first_N_samples=num_samples,
        )

        task.run()
