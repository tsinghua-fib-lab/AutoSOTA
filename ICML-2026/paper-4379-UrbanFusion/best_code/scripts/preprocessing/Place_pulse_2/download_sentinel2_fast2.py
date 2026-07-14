#!/usr/bin/env python3
"""
Description: Sentinel-2 patch extraction tool.
Downloads and extracts image patches for given locations using
Microsoft Planetary Computer and rasterio.
Fast implementation using concurrent processing. Used for
creating PP2-M dataset.
"""

import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import rasterio
import requests
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject
from tqdm import tqdm

# Global defaults; will be overridden per‐city by initializer
# Configurations
# City name (just for initialization, does not affect processing)
PLACE_NAME: str = "Atlanta"
# List of cities to process (Add all cities here)
CITIES: list[str] = [PLACE_NAME]
# Resolution of output patches
TARGET_RES: int = 10  # meters
PIXELS: int = 256  # output patch size in pixels
# Bands for downloading
BAND_LIST: list[str] = [
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
# Ground resolution (in meters) for each band
BAND_RESOLUTION: dict[str, int] = {
    "B01": 60,
    "B02": 10,
    "B03": 10,
    "B04": 10,
    "B05": 20,
    "B06": 20,
    "B07": 20,
    "B08": 10,
    "B8A": 20,
    "B09": 60,
    "B11": 20,
    "B12": 20,
}
# Date range for filtering
DATE_MIN: str = "2024-01-01"
DATE_MAX: str = "2024-12-31"
# STAC API endpoint
CATALOG_URL: str = "https://planetarycomputer.microsoft.com/api/stac/v1/"
# Place Pulse 2.0 files
LOCATIONS: Path = Path("svi_data/Place_pulse_2_0/locations.tsv")
PLACES: Path = Path("svi_data/Place_pulse_2_0/places.tsv")
# Cleaned place name
PLACE_NAME_CLEAN = PLACE_NAME.replace(" ", "")
# Number of threads for parallel processing
N_THREADS: int = 8
# These will be set by init_worker_globals for each city
SCENE_DIR: Path = Path("F:/Sentinel2/scenes") / PLACE_NAME
OUT_DIR: Path = Path("svi_data/Place_pulse_2_0/sentinel2") / PLACE_NAME


def init_worker_globals(
    place_name: str, scene_dir_str: str, out_dir_str: str
) -> None:
    """
    Initialize globals in each worker process (necessary on Windows).

    Parameters
    ----------
    place_name : str
        The name of the place being processed.
    scene_dir_str : str
        The path to the directory containing scene data.
    out_dir_str : str
        The path to the output directory for processed data.
    """
    global PLACE_NAME, SCENE_DIR, OUT_DIR
    PLACE_NAME = place_name
    SCENE_DIR = Path(scene_dir_str)
    OUT_DIR = Path(out_dir_str)


def compute_patch_bbox(
    lon: float, lat: float
) -> tuple[float, float, float, float]:
    """
    Compute the bounding box for a patch centered at the given coordinates.

    Parameters
    ----------
    lon : float
        The longitude of the patch center.
    lat : float
        The latitude of the patch center.

    Returns
    -------
    tuple[float, float, float, float]
        The bounding box (min_lon, min_lat, max_lon, max_lat) of the patch.
    """
    half_m: float = (PIXELS * TARGET_RES) / 2
    lat_rad: float = math.radians(lat)
    deg_lat: float = half_m / 111000.0
    deg_lon: float = half_m / (111000.0 * math.cos(lat_rad))
    return (lon - deg_lon, lat - deg_lat, lon + deg_lon, lat + deg_lat)


def get_tile_extent(scene_path: Path) -> tuple[float, float, float, float]:
    """
    Get the geographic extent (bounding box) of a Sentinel-2 tile.

    Parameters
    ----------
    scene_path : Path
        The path to the directory containing the scene data.

    Returns
    -------
    tuple[float, float, float, float]
        The bounding box (min_lon, min_lat, max_lon, max_lat) of the tile.
    """
    band_fp = scene_path / f"{BAND_LIST[0]}.tif"
    with rasterio.open(band_fp) as src:
        bounds = src.bounds
        src_crs = src.crs
    transformer = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
    points = [
        transformer.transform(x, y)
        for x in (bounds.left, bounds.right)
        for y in (bounds.bottom, bounds.top)
    ]
    lons, lats = zip(*points)
    return (min(lons), min(lats), max(lons), max(lats))


def covers(
    extent: tuple[float, float, float, float],
    patch_bbox: tuple[float, float, float, float],
) -> bool:
    """
    Check if the given extent covers the patch bounding box.

    Parameters
    ----------
    extent : tuple[float, float, float, float]
        The bounding box (min_lon, min_lat, max_lon, max_lat) of the extent.
    patch_bbox : tuple[float, float, float, float]
        The bounding box (min_lon, min_lat, max_lon, max_lat) of the patch.

    Returns
    -------
    bool
        True if the extent covers the patch, False otherwise.
    """
    e_minx, e_miny, e_maxx, e_maxy = extent
    p_minx, p_miny, p_maxx, p_maxy = patch_bbox
    return (
        e_minx <= p_minx
        and e_miny <= p_miny
        and e_maxx >= p_maxx
        and e_maxy >= p_maxy
    )


def extract_patch(
    lon: float,
    lat: float,
    tile_scene_map: dict[str, str],
    tile_extent_map: dict[str, tuple[float, float, float, float]],
) -> tuple[np.ndarray, dict]:
    """
    Extract a patch from the downloaded Sentinel-2 tiles.

    Parameters
    ----------
    lon : float
        The longitude of the patch center.
    lat : float
        The latitude of the patch center.
    tile_scene_map : dict[str, str]
        A mapping of tile IDs to their scene directory names.
    tile_extent_map : dict[str, tuple[float, float, float, float]]
        A mapping of tile IDs to their geographic extents.

    Returns
    -------
    tuple[np.ndarray, dict]
        The extracted patch as a NumPy array and its metadata.
    """
    patch_bbox = compute_patch_bbox(lon, lat)
    neighbors = [
        tid
        for tid, ext in tile_extent_map.items()
        if not (
            ext[2] < patch_bbox[0]
            or ext[0] > patch_bbox[2]
            or ext[3] < patch_bbox[1]
            or ext[1] > patch_bbox[3]
        )
    ]
    if not neighbors:
        raise RuntimeError(
            f"No downloaded tile intersects patch at {lon},{lat}"
        )

    # Open first band to compute transform & resolution
    first_fp = str(
        SCENE_DIR / tile_scene_map[neighbors[0]] / f"{BAND_LIST[0]}.tif"
    )
    with rasterio.open(first_fp) as ref:
        mosaic_crs = ref.crs
        transformer = Transformer.from_crs(
            "EPSG:4326", mosaic_crs, always_xy=True
        )
        x0, y0 = transformer.transform(patch_bbox[0], patch_bbox[1])
        x1, y1 = transformer.transform(patch_bbox[2], patch_bbox[3])
        xres = (x1 - x0) / PIXELS
        yres = (y1 - y0) / PIXELS
        ref_profile = ref.profile.copy()

    out_arr = np.empty((len(BAND_LIST), PIXELS, PIXELS), dtype=np.uint16)
    patch_bounds = (x0, y0, x1, y1)

    for i, band in enumerate(BAND_LIST):
        paths = [
            str(SCENE_DIR / tile_scene_map[tid] / f"{band}.tif")
            for tid in neighbors
        ]
        arr, out_transform = merge(
            paths,
            bounds=patch_bounds,
            res=(xres, yres),
            nodata=0,
            dtype="uint16",
        )
        out_arr[i] = arr[0]

    profile = ref_profile.copy()
    profile.update(
        {
            "crs": mosaic_crs,
            "transform": out_transform,
            "width": PIXELS,
            "height": PIXELS,
            "count": len(BAND_LIST),
            "dtype": "uint16",
            "compress": "LZW",
            "tiled": True,
            "blockxsize": PIXELS,
            "blockysize": PIXELS,
        }
    )

    return out_arr, profile


def resample_to_10m(input_path: Path, output_path: Path) -> None:
    """
    Resample a Sentinel-2 image to 10m resolution.

    Parameters
    ----------
    input_path : Path
        The file path to the input image.
    output_path : Path
        The file path to the output image.
    """
    with rasterio.open(input_path) as src:
        dst_transform, w, h = calculate_default_transform(
            src.crs,
            src.crs,
            src.width,
            src.height,
            *src.bounds,
            resolution=TARGET_RES,
        )
        kwargs = src.meta.copy()
        kwargs.update({"transform": dst_transform, "width": w, "height": h})
        with rasterio.open(output_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=src.crs,
                    resampling=Resampling.bilinear,
                )


def download_scene_assets(item: pystac_client, scene_id: str) -> None:
    """
    Download Sentinel-2 scene assets.

    Parameters
    ----------
    item : pystac_client
        The STAC item representing the scene.
    scene_id : str
        The ID of the scene to download.
    """
    scene_path = SCENE_DIR / scene_id
    scene_path.mkdir(parents=True, exist_ok=True)
    for band in BAND_LIST:
        asset = planetary_computer.sign(item.assets[band])
        href = asset.href
        orig_fp = scene_path / f"{band}_orig.tif"
        final_fp = scene_path / f"{band}.tif"
        if not orig_fp.exists():
            resp = requests.get(href, stream=True)
            with open(orig_fp, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
        if BAND_RESOLUTION[band] != TARGET_RES:
            resample_to_10m(orig_fp, final_fp)
        else:
            orig_fp.rename(final_fp)


def worker(args: tuple) -> None:
    """
    Process a single Sentinel-2 scene.

    Parameters
    ----------
    args : tuple
        A tuple containing the arguments for processing the scene.
    """
    lon, lat, out_fp, tmap, extmap = args
    stack, prof = extract_patch(lon, lat, tmap, extmap)
    with rasterio.open(out_fp, "w", **prof) as dst:
        for i in range(prof["count"]):
            dst.write(stack[i], i + 1)


def main(place_name: str):
    """
    Main entry point for downloading Sentinel-2 scenes.

    Parameters
    ----------
    place_name : str
        The name of the place to process.
    """
    # Override globals in parent process
    global PLACE_NAME, PLACE_NAME_CLEAN, SCENE_DIR, OUT_DIR
    PLACE_NAME = place_name
    PLACE_NAME_CLEAN = PLACE_NAME.replace(" ", "")
    SCENE_DIR = Path("F:/Sentinel2/scenes") / PLACE_NAME_CLEAN
    OUT_DIR = Path("svi_data/Place_pulse_2_0/sentinel2") / PLACE_NAME_CLEAN
    SCENE_DIR.mkdir(exist_ok=True, parents=True)
    OUT_DIR.mkdir(exist_ok=True, parents=True)

    # Load locations & places
    df1 = pd.read_csv(LOCATIONS, sep="\t").rename(
        columns={"_id": "location_id"}
    )
    df2 = pd.read_csv(PLACES, sep="\t").rename(columns={"_id": "place_id"})
    df = pd.merge(df1, df2, on="place_id", how="left")
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["loc.1"], df["loc.0"]),
        crs="EPSG:4326",
    )
    gdf["image_name"] = (
        gdf["loc.0"].map(lambda x: f"{x:.6f}")
        + "_"
        + gdf["loc.1"].map(lambda x: f"{x:.6f}")
        + "_"
        + gdf["location_id"].astype(str)
        + "_"
        + gdf["place_name"].fillna("").str.replace(" ", "")
        + ".JPG"
    )
    gdf = gdf.sort_values(by="place_name").reset_index(drop=True)
    gdf = gdf[gdf["place_name"] == PLACE_NAME]

    # build STAC catalog
    catalog = pystac_client.Client.open(
        CATALOG_URL, modifier=planetary_computer.sign_inplace
    )

    # maps for downloaded tiles
    tile_scene_map: dict[str, str] = {}
    tile_extent_map: dict[str, tuple[float, float, float, float]] = {}

    # assign and download scenes
    for row in tqdm(gdf.itertuples(index=False), desc="Assigning scenes"):
        lon, lat = row.geometry.x, row.geometry.y
        patch_bbox = compute_patch_bbox(lon, lat)
        corners = [
            (patch_bbox[0], patch_bbox[1]),
            (patch_bbox[0], patch_bbox[3]),
            (patch_bbox[2], patch_bbox[1]),
            (patch_bbox[2], patch_bbox[3]),
        ]
        missing = []
        for x, y in corners:
            covered = any(
                ext[0] <= x <= ext[2] and ext[1] <= y <= ext[3]
                for ext in tile_extent_map.values()
            )
            if not covered:
                missing.append((x, y))
        for x, y in missing:
            # For some reason, the API fails for Copenhagen, take the second
            # search result
            if PLACE_NAME == "Copenhagen":
                search = catalog.search(
                    intersects={"type": "Point", "coordinates": [x, y]},
                    collections=["sentinel-2-l2a"],
                    datetime=f"{DATE_MIN}/{DATE_MAX}",
                    sortby=[{"field": "eo:cloud_cover", "direction": "asc"}],
                    max_items=2,
                )
                # Get the second item
                items = list(search.items())

                # Iterate over items and print max value of tile
                item = items[1]

            else:
                search = catalog.search(
                    intersects={"type": "Point", "coordinates": [x, y]},
                    collections=["sentinel-2-l2a"],
                    datetime=f"{DATE_MIN}/{DATE_MAX}",
                    sortby=[{"field": "eo:cloud_cover", "direction": "asc"}],
                    max_items=1,
                )
                item = next(search.items(), None)
            if not item:
                continue
            sid = item.id
            scene_folder = SCENE_DIR / sid
            if not (
                scene_folder.exists()
                or (scene_folder / f"{BAND_LIST[0]}.tif").exists()
            ):
                download_scene_assets(item, sid)
            tile_scene_map[sid] = sid
            tile_extent_map[sid] = get_tile_extent(scene_folder)

    # Prepare extraction tasks
    tasks = []
    for row in gdf.itertuples(index=False):
        lon, lat = row.geometry.x, row.geometry.y
        out_fp = OUT_DIR / f"{row.image_name.split('.JPG')[0]}.tif"
        tasks.append((lon, lat, out_fp, tile_scene_map, tile_extent_map))

    # Parallel extraction, with initializer to set globals in each worker
    with ProcessPoolExecutor(
        max_workers=N_THREADS,
        initializer=init_worker_globals,
        initargs=(PLACE_NAME, str(SCENE_DIR), str(OUT_DIR)),
    ) as executor:
        for _ in tqdm(
            executor.map(worker, tasks), total=len(tasks), desc="Extracting"
        ):
            pass


if __name__ == "__main__":
    # Windows will use 'spawn' by default, so initializer is required 'Rome',
    for city in CITIES:
        main(city)
