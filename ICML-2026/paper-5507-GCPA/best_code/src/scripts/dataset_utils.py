from datasets import Dataset, Image, DatasetDict
import pandas as pd
from pathlib import Path
from latentis.data.dataset import HFDatasetView
from latentis.data.dataset import DataType, Feature


def get_flickr(root_path: str, name: str) -> HFDatasetView:
    # paths
    img_dir = Path(f"{root_path}/Flicker8k_Dataset")
    txt_file = Path(f"{root_path}/Flickr8k_text/Flickr8k.token.txt")
    aud_dir = Path(f"{root_path}/flickr_audio/wavs")

    train_list = set(
        (
            Path(f"{root_path}/Flickr8k_text/Flickr_8k.trainImages.txt")
            .read_text()
            .splitlines()
        )
    )
    val_list = set(
        (
            Path(f"{root_path}/Flickr8k_text/Flickr_8k.devImages.txt")
            .read_text()
            .splitlines()
        )
    )
    test_list = set(
        (
            Path(f"{root_path}/Flickr8k_text/Flickr_8k.testImages.txt")
            .read_text()
            .splitlines()
        )
    )

    rows = []
    with open(txt_file, "r") as f:
        for line in f:
            key, caption = line.strip().split("\t")
            fname, idx = key.split("#")
            img_path = img_dir / fname
            aud_path = aud_dir / f"{fname.split('.')[0]}_{idx}.wav"

            if not (img_path.exists() and aud_path.exists()):
                continue

            # Assign split
            if fname in train_list:
                split = "train"
            elif fname in val_list:
                split = "validation"
            elif fname in test_list:
                split = "test"
            else:
                continue

            rows.append(
                {
                    "image": str(img_path),
                    "text": caption,
                    "audio": str(aud_path),
                    "id": f"{fname}_{idx}",
                    "img_id": fname,
                    "split": split,
                }
            )

    df = pd.DataFrame(rows)

    ds_dict = DatasetDict(
        {
            split: Dataset.from_pandas(df[df["split"] == split].drop(columns=["split"]))
            for split in ["train", "validation", "test"]
        }
    )

    # Cast columns
    for split in ds_dict:
        ds_dict[split] = ds_dict[split].cast_column("image", Image())

    def rename_columns(ds):
        return ds.rename_columns(
            {
                "id": "sample_id"
            }
        ).remove_columns(["__index_level_0__"])

    ds_dict = ds_dict.map(lambda x: x)  # keep structure
    ds_dict = DatasetDict({k: rename_columns(v) for k, v in ds_dict.items()})

    features = [
        Feature(name="sample_id", data_type=DataType.TEXT),
        Feature(name="img_id", data_type=DataType.TEXT),
        Feature(name="image", data_type=DataType.IMAGE),
        Feature(name="text", data_type=DataType.TEXT),
        Feature(name="audio", data_type=DataType.TEXT),  # path to audio file
    ]

    data = HFDatasetView(
        name=name,
        hf_dataset=ds_dict,
        id_column="sample_id",
        features=features,
    )

    return data