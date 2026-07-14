#!/usr/bin/env python3
"""
Description: Sentinel-2 patch extraction tool.
Downloads and extracts image patches for given locations using
Microsoft Planetary Computer and rasterio.
Slow implementation. Used for creating PP2-M dataset.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import pandas as pd
import planetary_computer
import pystac_client
import stackstac
from rasterio.windows import Window
from tqdm import tqdm

# Configuration
# Place Pulse 2.0 paths
LOCATIONS = "svi_data/Place_pulse_2_0/locations.tsv"
PLACES = "svi_data/Place_pulse_2_0/places.tsv"
SVI_DIR = "svi_data/Place_pulse_2_0/images/"
# Output directory for Sentinel-2 patches
OUT_DIR = "svi_data/Place_pulse_2_0/sentinel2/batch_20k/"
# Indices of Place Pulse 2.0 images to download
LOW = 0
HIGH = 20000
# Date range for filtering
DATE_MIN = "2024-01-01"
DATE_MAX = "2024-12-31"
# Number of threads for parallel processing
N_THREADS = 4
# Sentinel‑2 L2A bands 1–12
BAND_LIST = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B11",
    "B12",
]


def main():
    # --- Load & preprocess just once ---
    df_locations = pd.read_csv(LOCATIONS, sep="\t").rename(
        columns={"_id": "location_id"}
    )
    df_places = pd.read_csv(PLACES, sep="\t").rename(
        columns={"_id": "place_id"}
    )
    df_all = pd.merge(df_locations, df_places, on="place_id", how="left")
    gdf = gpd.GeoDataFrame(
        df_all,
        geometry=gpd.points_from_xy(df_all["loc.1"], df_all["loc.0"]),
        crs="EPSG:4326",
    )
    gdf["image_name"] = (
        gdf["loc.0"].map(lambda x: f"{x:.6f}")
        + "_"
        + gdf["loc.1"].map(lambda x: f"{x:.6f}")
        + "_"
        + gdf["location_id"].astype(str)
        + "_"
        + gdf["place_name"].str.replace(" ", "", regex=False)
        + ".JPG"
    )
    if "image_name" not in gdf.columns:
        raise ValueError("GeoDataFrame must have an 'image_name' column")

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Single STAC client for all threads
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1/",
        modifier=planetary_computer.sign_inplace,
    )

    def process_point(i: int):
        """Download one 256×256 patch and write it as a GeoTIFF."""
        row = gdf.iloc[i]
        name = str(row["image_name"]).split(".JPG")[0]
        lon, lat = row.geometry.x, row.geometry.y

        search = catalog.search(
            intersects={"type": "Point", "coordinates": [lon, lat]},
            collections=["sentinel-2-l2a"],
            datetime=f"{DATE_MIN}/{DATE_MAX}",
            sortby=[{"field": "eo:cloud_cover", "direction": "asc"}],
            max_items=1,  # stop after first match
            limit=1,  # page size = 1
        )

        item = next(search.items(), None)
        if item is None:
            print(f"[{i}] No scene for {name} → skipped")
            return

        stack = stackstac.stack(
            item, assets=BAND_LIST, epsg=4326, chunksize=256
        ).chunk({"x": 256, "y": 256})

        transform = stack.rio.transform()
        col_f, row_f = ~transform * (lon, lat)
        col, row = int(col_f), int(row_f)
        half = 256 // 2
        window = Window(col - half, row - half, 256, 256)

        patch3d = (
            stack.isel(time=0)  # drop time dim
            .rio.isel_window(window)
            .astype("uint16")
            .load()  # compute once
        )

        out_fp = out_dir / f"{name}.tif"
        patch3d.rio.to_raster(
            str(out_fp),
            driver="GTiff",
            dtype="uint16",
            compress="LZW",
            tiled=True,
            blockxsize=256,
            blockysize=256,
            interleave="pixel",
        )

    indices = range(LOW, min(HIGH, len(gdf)))
    with ThreadPoolExecutor(max_workers=N_THREADS) as exe:
        list(
            tqdm(
                exe.map(process_point, indices),
                total=len(indices),
                desc="Extracting patches",
            )
        )

    print(f"✅ Done! GeoTIFF patches are in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
