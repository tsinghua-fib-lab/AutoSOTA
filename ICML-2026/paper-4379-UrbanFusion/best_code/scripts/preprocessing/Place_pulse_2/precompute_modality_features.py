#!/usr/bin/env python3
"""
Description: Feature extraction for the Place Pulse 2.0 dataset
using various modalities. It processes raw modalities of PP2-M
dataset though the modality encoders and generates the corresponding
features for each modality, and saves them in an HDF5 file.

Run on local machine:
python scripts/preprocessing/Place_pulse_2/precompute_modality_features.py
"""
import os
import sys

sys.path.insert(0, os.path.expanduser("~/SpatialFoundationModel"))
print("Path from:", sys.path[0], flush=True)
import random
from datetime import datetime

import h5py
import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import AutoTokenizer

from srl.data.place_pulse2 import PlacePulse2DatasetModule
from srl.encoders.coordinate_encoder.coordinate_encoder import (
    CoordinateEncoder,
)
from srl.encoders.OSM_encoder.OSM_encoder import OSMEncoder
from srl.encoders.POI_encoder.text_encoders import TextTransformer
from srl.encoders.RS_encoder.RS_encoder import RSEncoder
from srl.encoders.SVI_encoder.svi_encoder import SVIEncoder

# Configurations
# Cluster settings
cluster = True
# Random seed
SEED = 42
# Reproducibility for inference
os.environ["PYTHONHASHSEED"] = str(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# torch._C._jit_set_nvfuser_enabled(False)

# L hyperparameters for spherical harmonics (SatCLIP)
nr_legendre_polys = 100

# Get current date
current_date = datetime.now().strftime("%d_%m_%Y")

# OSM POI settings
osm_neigbors = 15  # Number of neighbors for OSM encoding
osm_buffer = 200  # Buffer distance for OSM encoding in meters

# Language model settings
text_encoder_name = "BAAI/bge-small-en-v1.5"  # Name of the text encoder model
# text_encoder_name = "BAAI/bge-m3"  # Name of the text encoder model

# Paths (local and cluster)
if cluster:
    locations_path = "/svi_data/place-pulse-2.0/locations.tsv"
    places_path = "/svi_data/place-pulse-2.0/places.tsv"
    available_images_path = "/svi_data/place-pulse-2.0/image_file_list.txt"
    SVI_path = (
        "/5_files/svi_data/place-pulse-2.0/SVI/"
    )
    sentinel2_path = "/5_files/svi_data/place-pulse-2.0/sentinel2/"
    POI_path = "/svi_data/place-pulse-2.0/POIs_Texts/texts_15neigh_200m.csv"
    OSM_basemap_path = "/5_files/svi_data/place-pulse-2.0/OSM_basemap/"
    df_index_path = "/svi_data/place-pulse-2.0/h5_index.csv"
    osm_checkpoint_path = "/svi_data/place-pulse-2.0/encoder_weights/OSM_MAE2.ckpt"
    H5_path = f'/cluster/scratch/dmuehlema/5_files/svi_data/place-pulse-2.0/{current_date}_legendre_polys_{nr_legendre_polys}_{text_encoder_name.split("/")[-1]}_OSM_{osm_neigbors}_buffer_{osm_buffer}.h5'

else:
    pass  # Add local paths if needed


class CreateModalityRepresentations(nn.Module):
    def __init__(
        self,
        coordinate_encoder: nn.Module = None,
        SVI_encoder: nn.Module = None,
        sentinel2_encoder: nn.Module = None,
        OSM_encoder: nn.Module = None,
        POI_encoder: nn.Module = None,
        modalities: list = ["coords", "SVI", "sentinel2", "OSM", "POI"],
    ) -> None:
        """
        Initializes the CreateModalityRepresentations class.
        PyTorch module for creating modality representations using
        all modality encoders.

        Parameters:
        -----------
        coordinate_encoder: nn.Module
            Encoder for coordinate data.
        SVI_encoder: nn.Module
            Encoder for SVI data.
        sentinel2_encoder: nn.Module
            Encoder for Sentinel-2 data.
        OSM_encoder: nn.Module
            Encoder for OpenStreetMap data.
        POI_encoder: nn.Module
            Encoder for Points of Interest data.
        modalities: list
            List of modalities to process. Options include
            "coords", "SVI", "sentinel2", "OSM", and "POI".
        """
        super().__init__()
        self.coordinate_encoder = coordinate_encoder
        self.SVI_encoder = SVI_encoder
        self.sentinel2_encoder = sentinel2_encoder
        self.OSM_encoder = OSM_encoder
        self.POI_encoder = POI_encoder
        self.modalities = modalities

    def forward(self, batch: list) -> dict:
        """
        Forward pass to extract features from the input batch.

        Parameters:
        -----------
        batch: list
            A list containing the input data for each modality.
            The order of the modalities should match the order
            specified in self.modalities.
        """
        output = {}
        count = 0
        for modality in self.modalities:
            if modality == "coords":
                coords = batch[count]
                coords = self.coordinate_encoder.forward_features(coords)
                output["coords"] = coords
            elif modality == "SVI":
                svi = batch[count]
                svi = self.SVI_encoder.forward_features(svi)
                output["SVI"] = svi
            elif modality == "sentinel2":
                sentinel2 = batch[count]
                sentinel2 = self.sentinel2_encoder.forward_features(sentinel2)
                output["sentinel2"] = sentinel2
            elif modality == "OSM":
                osm = batch[count]
                osm = self.OSM_encoder.forward_features(osm)
                output["OSM"] = osm
            elif modality == "POI":
                poi = batch[count]
                poi = self.POI_encoder.forward_features(poi)
                output["POI"] = poi
            count += 1
        return output


def preprocess_batch(batch) -> list:
    """
    Preprocesses a batch of data for feature extraction.

    Parameters:
    -----------
    batch: dict
        A dictionary containing the batch data. Expected keys are:
        - "text": Text data to be tokenized.
        - "coords_original": Original coordinates (not processed).
        - "gsv_img": GSV image paths (not processed).
        - Other keys: Tensors to be moved to the device.

    Returns:
    --------
    list
        A list of processed tensors, where each tensor corresponds
        to a modality in the order specified by the
        CreateModalityRepresentations class.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = None
    processed = []
    tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)
    for key, value in batch.items():
        if key.lower() == "text":
            tokenized = tokenizer(value, padding=True, return_tensors="pt").to(
                device
            )
            processed.append(tokenized)
            if batch_size is None:
                batch_size = tokenized.input_ids.size(0)
        elif key.lower() == "coords_original":
            pass
        elif key.lower() == "gsv_img":
            pass
        else:
            processed.append(value.to(device))
            if batch_size is None:
                batch_size = value.size(0)
    return processed


def extract_features_modalities() -> None:
    """
    Extracts features from the Place Pulse 2.0 dataset using various modalities
    and saves them to an HDF5 file.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Settings for PP2-M Dataset
    dataloader = PlacePulse2DatasetModule(
        dataset_name="place-pulse-2.0",
        locations_path=locations_path,
        places_path=places_path,
        available_images_path=available_images_path,
        SVI_path=SVI_path,
        sentinel2_path=sentinel2_path,
        POI_path=POI_path,
        OSM_basemap_path=OSM_basemap_path,
        batch_size=256,
        num_workers=4,
        zoom_levels=[15, 16, 17],
        buffer_distances=[600, 300, 150],
        out_region_cities=[
            "Cape Town",
            "Singapore",
            "Rio De Janeiro",
            "Paris",
            "Milan",
            "New York",
            "Sydney",
        ],
        in_region_split=0.2,
        pin_memory=True,
        persistent_workers=True,
        seed_split=42,
        return_filename=True,
        return_coords=True,
        filter_available_images=False,
    )
    print("Setting up dataloader...", flush=True)
    dataloader.setup()
    all_locations_dataloader = dataloader.all_locations_dataloader()
    print("Dataloader setup complete.", flush=True)
    feature_generator = CreateModalityRepresentations(
        coordinate_encoder=CoordinateEncoder(
            embed_dim=512, legendre_polys=nr_legendre_polys
        ),
        SVI_encoder=SVIEncoder(embed_dim=512),
        sentinel2_encoder=RSEncoder(embed_dim=512),
        OSM_encoder=OSMEncoder(
            embed_dim=512, checkpoint_path=osm_checkpoint_path
        ),
        POI_encoder=TextTransformer(
            embed_dim=512, model_name=text_encoder_name
        ),
        modalities=["coords", "SVI", "sentinel2", "OSM", "POI"],
    )
    feature_generator.to(device)
    feature_generator.eval()
    print(
        "Feature generator initialized and moved to device:",
        device,
        flush=True,
    )
    print("Counting total number of samples...", flush=True)
    N = len(all_locations_dataloader.dataset)
    print(f"Total samples: {N}", flush=True)
    first_batch = next(iter(all_locations_dataloader))
    input = preprocess_batch(first_batch)
    with torch.no_grad():
        features = feature_generator(input)  # features: dict of tensors
    modality_dims = {m: features[m].shape[1] for m in features}
    modalities = list(modality_dims.keys())
    print("Inferred modality dims:", modality_dims, flush=True)
    all_locations_dataloader = dataloader.all_locations_dataloader()
    gsv_list = []
    index_list = []

    with h5py.File(H5_path, "w") as h5f:
        datasets = {
            modality: h5f.create_dataset(
                modality,
                shape=(N, modality_dims[modality]),
                dtype="float64" if modality == "coords" else "float32",
            )
            for modality in modalities
        }
        gsv_img_ds = h5f.create_dataset(
            "gsv_img", shape=(N,), dtype=h5py.string_dtype("utf-8")
        )
        coords_orig_ds = h5f.create_dataset(
            "coords_original", shape=(N, 2), dtype="float32"
        )
        print("HDF5 datasets created.", flush=True)

        idx = 0
        for batch in all_locations_dataloader:
            gsv_imgs = batch["gsv_img"]
            original_coords = batch["coords_original"]
            batch_size = len(gsv_imgs)
            input = preprocess_batch(batch)
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    features = feature_generator(input)

            # Write batch to HDF5
            for i in range(batch_size):
                gsv_img_ds[idx] = str(gsv_imgs[i])
                gsv_list.append(str(gsv_imgs[i]))
                index_list.append(idx)
                coords_orig_ds[idx, :] = np.array(original_coords[i])
                for modality in modalities:
                    arr = features[modality][i].detach().cpu().numpy()
                    datasets[modality][idx, :] = arr
                idx += 1
            print(f"Processed {idx} / {N} samples...", end="\r", flush=True)
    df = pd.DataFrame({"gsv_img": gsv_list, "h5_index": index_list})
    df.to_csv(df_index_path, index=False)
    print("\nFeature extraction and saving completed.", flush=True)


def main():
    extract_features_modalities()


if __name__ == "__main__":
    main()
