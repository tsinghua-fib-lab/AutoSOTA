#!/usr/bin/env python3
"""
Description: Script to create raster tiles from OpenStreetMap (OSM) data.
Takes large OSM tiles covering entire cities and splits them into smaller
patches around each location in the Place Pulse 2.0 dataset.
Used for generating PP2-M dataset.
"""

import io
import os
import queue
import tarfile
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.windows import from_bounds

# Configuration
# Path to location.tsv file from Place Pulse 2.0 dataset
LOCATIONS_FILE = "svi_data/Place_pulse_2_0/locations.tsv"
# Path to places.tsv file from Place Pulse 2.0 dataset
PLACES_FILE = "svi_data/Place_pulse_2_0/places.tsv"
# Folder of the input OSM basemap tiles
INPUT_DIR = "F:/OSM_basemap/"
# Folder of the output OSM basemap tiles
OUTPUT_DIR = "F:/OSM_basemap/osm_basemap_compressed/"
# Buffer sizes for the output tiles 150 means 300mx300m
BUFFER_SIZES = [150, 300, 600]  # metres
# OSM basemap zoom levels
ZOOM_LEVELS = [17, 16, 15]  # matching order to BUFFER_SIZES
# Output tile size
OUT_SIZE = 256  # pixels (square)
# Path to the log file
LOG_FILE = "F:/OSM_basemap/osm_basemap_compressed/finished_keys2.log"
# Maximum number of worker threads
MAX_WORKERS = 1
# GDAL configuration
os.environ.setdefault("GDAL_NUM_THREADS", "1")
os.environ.setdefault("GDAL_CACHEMAX", "1024")
# Queue for writing output tiles
WRITE_QUEUE = queue.Queue(maxsize=1024)
# Metadata for the output raster patches
PATCH_META_BASE = dict(
    driver="GTiff",
    dtype="uint8",
    count=3,  # RGB
    height=OUT_SIZE,
    width=OUT_SIZE,
    compress="DEFLATE",  # fast lossless codec
    predictor=2,
    tiled=True,
    blockxsize=256,  # internal 256×256 tiles
    blockysize=256,
    interleave="pixel",  # avoids band‑separate strips
)
# Cities to remove from download (already downloaded)
# Empty list if re-process all
REMOVE_PLACES = {
    "Hong Kong",
    "Kyoto",
    "Valparaiso",
    "Copenhagen",
    "Tel Aviv",
    "Warsaw",
    "Toronto",
    "Taipei",
    "Sydney",
    "Stockholm",
    "Singapore",
    "Seattle",
    "Sao Paulo",
    "San Francisco",
    "Rio De Janeiro",
    "Prague",
    "Portland",
    "Philadelphia",
    "Munich",
    "Houston",
    "Chicago",
    "Cape Town",
    "Barcelona",
    "Amsterdam",
    "Belo Horizonte",
    "Boston",
    "Denver",
    "Atlanta",
    "New York",
}


def load_place_pulse_data(loc_file: str, pl_file: str) -> gpd.GeoDataFrame:
    """
    Load Place Pulse data from TSV files and return a GeoDataFrame.

    Parameters
    ----------
    loc_file : str
        Path to the locations TSV file.
    pl_file : str
        Path to the places TSV file.

    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame containing the merged Place Pulse data.
    """
    df_loc = pd.read_csv(loc_file, sep="\t").rename(
        columns={"_id": "location_id"}
    )
    df_pl = pd.read_csv(pl_file, sep="\t").rename(columns={"_id": "place_id"})

    merged = pd.merge(df_loc, df_pl, on="place_id", how="left")
    merged["image_name"] = (
        merged["loc.0"].map(lambda x: f"{x:.6f}")
        + "_"
        + merged["loc.1"].map(lambda x: f"{x:.6f}")
        + "_"
        + merged["location_id"].astype(str)
        + "_"
        + merged["place_name"].str.replace(" ", "", regex=False)
        + ".JPG"
    )
    geometry = gpd.points_from_xy(merged["loc.1"], merged["loc.0"])
    return gpd.GeoDataFrame(merged, geometry=geometry, crs="EPSG:4326")


def expected_fname(buffer_size, zoom_level, image_name):
    """
    Helper --> expected patch filename relative to OUTPUT_DIR.

    Parameters
    ----------
    buffer_size : int
        The buffer size in meters.
    zoom_level : int
        The zoom level.
    image_name : str
        The name of the image file.

    Returns
    -------
    str
        The expected patch filename.
    """
    return f"{buffer_size}_{zoom_level}_{image_name.replace('.JPG', '.tif')}"


def run_task(args):
    """
    Run a processing task for a specific tile.

    Parameters
    ----------
    args : tuple
        A tuple containing the key and rows for the tile.

    Returns
    -------
    list
        A list of status messages.
    """
    key, rows = args
    messages = list(process_tile(key, rows))
    place, buffer_size, zoom_level = key
    messages.append(
        f"FINISHED {place} @ zoom {zoom_level}, buffer {buffer_size}"
    )
    return messages


def writer_thread():
    """
    Collect (rel_path, blob) pairs and append them to a sequential .tar.
    Rotates to a new file every ~2 GB so an interrupted run never corrupts more
    than one shard.  Reading later can happen in‑place via GDAL's /vsitar/.
    """
    shard_idx = 11
    bytes_in_shard = 0
    MAX_SHARD_SIZE = 2 * 1024**3  # 2 GB

    tar_path = os.path.join(OUTPUT_DIR, f"patches_{shard_idx:04d}.tar")
    tar = tarfile.open(tar_path, mode="w")

    while True:
        item = WRITE_QUEUE.get()
        if item is None:  # Graceful exit
            break

        rel_path, blob = item
        info = tarfile.TarInfo(rel_path)
        info.size = len(blob)
        tar.addfile(info, io.BytesIO(blob))  # single sequential write
        bytes_in_shard += info.size
        WRITE_QUEUE.task_done()

        # Rotate archive to keep shards small and corruption‑proof
        if bytes_in_shard >= MAX_SHARD_SIZE:
            tar.close()
            shard_idx += 1
            bytes_in_shard = 0
            tar_path = os.path.join(OUTPUT_DIR, f"patches_{shard_idx:04d}.tar")
            tar = tarfile.open(tar_path, mode="w")

    tar.close()


# Start the tar writer thread
tar_thread = threading.Thread(target=writer_thread, daemon=True)
tar_thread.start()


def process_tile(key: tuple, rows: list):
    """
    Process a single tile for a specific place.

    Parameters
    ----------
    key : tuple
        (place, buffer_size, zoom_level)
    rows : list
        list[(idx, row)] – all Place Pulse rows that fall into this tile.
    """
    start = time.time()
    print(f"Processing {key} with {len(rows)} points")
    place, buffer_size, zoom_level = key
    tile_path = os.path.join(
        INPUT_DIR,
        place.replace(" ", "_"),
        f"{place.replace(' ', '_')}_raster_{zoom_level}_{buffer_size}.tif",
    )

    if not os.path.exists(tile_path):
        yield (
            f"SKIP {place} @ zoom {zoom_level}, buffer {buffer_size} "
            f"(raster missing)"
        )
        return
    end = time.time()
    print(f"Tile path check took {end - start:.2f} seconds")

    start = time.time()
    try:
        existing_files = set(os.listdir(OUTPUT_DIR))
    except FileNotFoundError:
        existing_files = set()

    wanted_files = {
        expected_fname(buffer_size, zoom_level, r.image_name) for _, r in rows
    }

    missing_files = wanted_files.difference(existing_files)

    if not missing_files:
        yield f"SKIP {place} (all {len(wanted_files)} patches already present)"
        return

    end = time.time()
    print(f"Pre-flight check took {end - start:.2f} seconds")
    start = time.time()
    # Open once, copy into RAM
    with rasterio.open(tile_path) as src:
        full_arr = src.read()  # numpy array (bands, rows, cols)
        mem_meta = src.meta.copy()

        # Hold a complete in‑memory clone of the dataset
        with MemoryFile() as mf:
            with mf.open(**mem_meta) as mem:
                mem.write(full_arr)

                # Re‑project all points for this tile in one shot
                geoms = [r.geometry for _, r in rows]
                batch = gpd.GeoSeries(geoms, crs="EPSG:4326")
                batch_proj = batch.to_crs(mem.crs)
                end = time.time()
                print(
                    f"Tile loading and reprojection took "
                    f"{end - start:.2f} seconds"
                )
                # Iterate point‑by‑point
                for (idx, row), pt in zip(rows, batch_proj):

                    # Output filename & early‑out if it exists
                    fname = (
                        f"{buffer_size}_{zoom_level}_"
                        f"{row['image_name'].replace('.JPG', '.tif')}"
                    )
                    if fname not in missing_files:
                        yield f"SKIP {fname} (already exists)"
                        continue

                    x, y = pt.x, pt.y
                    half = buffer_size
                    bounds = (x - half, y - half, x + half, y + half)
                    window = from_bounds(*bounds, transform=mem.transform)

                    # Read patch (256×256, bilinear) from the in‑memory raster
                    try:
                        patch = mem.read(
                            window=window,
                            out_shape=(mem.count, OUT_SIZE, OUT_SIZE),
                            resampling=Resampling.bilinear,
                        )[
                            :3
                        ]  # Keep RGB only
                    except Exception as e:
                        yield f"ERROR reading window for {fname}: {e}"
                        continue

                    # Patch‑specific metadata
                    meta = PATCH_META_BASE.copy()
                    meta["transform"] = src.window_transform(window)

                    try:
                        with MemoryFile() as mf:
                            with mf.open(**meta) as ds:
                                ds.write(patch)
                            blob = mf.read()
                        WRITE_QUEUE.put((fname, blob))  # Enqueue for writer
                    except Exception as e:
                        yield f"ERROR writing {fname}: {e}"


# Thread-safe write queue
Log_queue = Queue()


def log_writer(logfile: str, queue: Queue):
    """
    Dedicated thread that writes log lines.

    This runs in a separate thread to avoid blocking the main thread.

    Parameters
    ---------
    logfile : str
        Path to the log file.
    queue : Queue
        Thread-safe queue for log lines.

    """
    with open(logfile, "a") as f:
        while True:
            line = queue.get()
            if line is None:
                queue.task_done()
                break
            f.write(line + "\n")
            f.flush()  # ensure immediate write
            queue.task_done()


# Start log writer thread
log_thread = threading.Thread(
    target=log_writer, args=(LOG_FILE, Log_queue), daemon=True
)
log_thread.start()


if __name__ == "__main__":
    # Load Place Pulse point data
    gdf = load_place_pulse_data(LOCATIONS_FILE, PLACES_FILE)

    # Normalize and clean place names
    clean = gdf["place_name"].astype(str).str.normalize("NFKC").str.strip()

    # Filter out cities that have already finished 3 or more tasks
    gdf = gdf[~clean.isin(REMOVE_PLACES)]

    # Bundle points by (place, buffer, zoom) – one tile --> one call
    tasks = defaultdict(list)
    for idx, row in gdf.iterrows():
        place = row.place_name
        for buf, zoom in zip(BUFFER_SIZES, ZOOM_LEVELS):
            tasks[(place, buf, zoom)].append((idx, row))

    print(tasks.keys())
    # Parallel execution
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(run_task, item) for item in tasks.items()]

        for fut in as_completed(futures):
            for line in fut.result():  # Forward worker messages
                print(line)
                Log_queue.put(line)
    # Tell log‐writer to finish, wait for its queue to drain, then join
    Log_queue.put(None)
    Log_queue.join()
    log_thread.join()

    # Then tell tar‐writer to finish, wait for its queue, then join
    WRITE_QUEUE.put(None)
    WRITE_QUEUE.join()
    tar_thread.join()
