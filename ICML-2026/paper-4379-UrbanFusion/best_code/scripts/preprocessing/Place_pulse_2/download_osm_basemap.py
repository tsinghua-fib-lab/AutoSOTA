#!/usr/bin/env python3
"""
Description: Script to downloading raster tiles from OpenStreetMap (OSM).
Downloads patches for entire cities (faster than downloading individual
patches). Used for generating PP2-M dataset.
"""
import os

import contextily as cx
import contextily.tile as _ct
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, calculate_default_transform, reproject

# Configuration
# Path to location.tsv file from Place Pulse 2.0 dataset
LOCATIONS_FILE = "svi_data/Place_pulse_2_0/locations.tsv"
# Path to places.tsv file from Place Pulse 2.0 dataset
PLACES_FILE = "svi_data/Place_pulse_2_0/places.tsv"
# Folder for the output OSM basemap tiles
OUTPUT_DIR = "F:/OSM_basemap/"
# Buffer sizes for naming of output tiles
BUFFER_SIZES = [150, 300, 600]
# OSM basemap zoom levels
ZOOM_LEVELS = [17, 16, 15]  # matching order to BUFFER_SIZES

# Monkey-patch for handling 404 errors in contextily
# Sometimes tileserver fails if a tile is not available
_orig = _ct._retryer


def soft_retryer(tile_url: str, wait: float, max_retries: int) -> np.ndarray:
    """
    Soft retryer for tile requests.

    Parameters
    ----------
    tile_url : str
        The URL of the tile to request.
    wait : float
        The amount of time to wait before retrying.
    max_retries : int
        The maximum number of retry attempts.

    Returns
    -------
    np.ndarray
        The requested tile image. Or a zero-filled array if the tile is not
        found.
    """
    try:
        return _orig(tile_url, wait, max_retries)
    except Exception as err:
        print(f"Error: {err}")
        if "404" in str(err):
            return np.zeros((256, 256, 4), dtype=np.uint8)
        raise


# Monkey-patch for handling 404 errors in contextily
_ct._retryer = soft_retryer
_ct.USER_AGENT = "PlacePulseDownloader/0.2 (+me@mydomain.com)"


def load_place_pulse_data(loc_file: str, pl_file: str) -> gpd.GeoDataFrame:
    """
    Load and join Place Pulse location and place data.

    Parameters
    ----------
    loc_file : str
        Path to the locations TSV file.
    pl_file : str
        Path to the places TSV file.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with point geometries in EPSG:4326.
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


if __name__ == "__main__":
    # Load all locations once
    gdf = load_place_pulse_data(LOCATIONS_FILE, PLACES_FILE)

    # Process each city individually
    for city in gdf["place_name"].unique():
        print(f"Processing city: {city}")
        gdf_filtered = gdf[gdf["place_name"] == city]
        buffer_meters = max(BUFFER_SIZES) + 400  # Bit more than max buffersize
        gdf_utm = gdf_filtered.to_crs(gdf_filtered.estimate_utm_crs())
        bbox = gdf_utm.union_all().envelope.buffer(buffer_meters)
        bbox_gdf = gpd.GeoDataFrame(geometry=[bbox], crs=gdf_utm.crs)
        bbox_web_mercator = bbox_gdf.to_crs(epsg=3857)
        minx, miny, maxx, maxy = bbox_web_mercator.total_bounds

        # Set user agent for contextily
        _ct.USER_AGENT = "PlacePulseDownloader/0.2 (+me@mydomain.com)"
        for i in range(len(BUFFER_SIZES)):
            try:
                patch_size_meters = BUFFER_SIZES[i]
                zoom = ZOOM_LEVELS[i]
                city = city.replace(" ", "_")
                output_path = os.path.join(
                    OUTPUT_DIR,
                    f"{city}/{city}_raster_{zoom}_{patch_size_meters}.tif",
                )
                if not os.path.exists(output_path):
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    # Download the OSM tile
                    img, extent = cx.bounds2raster(
                        minx,
                        miny,
                        maxx,
                        maxy,
                        output_path,
                        zoom=zoom,
                        source=cx.providers.OpenStreetMap.Mapnik,
                        use_cache=True,
                    )
                    img = None  # Free resources

                    # Reproject the raster to the target CRS (local UTM)
                    with rasterio.open(output_path) as raster:
                        dst_crs = gdf_utm.crs
                        transform, width, height = calculate_default_transform(
                            raster.crs,
                            dst_crs,
                            raster.width,
                            raster.height,
                            *raster.bounds,
                        )
                        kwargs = raster.meta.copy()
                        kwargs.update(
                            {
                                "crs": dst_crs,
                                "transform": transform,
                                "width": width,
                                "height": height,
                            }
                        )
                        data = raster.read()
                    os.remove(output_path)  # Delete the original raster file
                    # Write the reprojected raster
                    with rasterio.open(output_path, "w", **kwargs) as dst:
                        for i in range(1, data.shape[0] + 1):
                            reproject(
                                source=data[i - 1],
                                destination=rasterio.band(dst, i),
                                src_transform=raster.transform,
                                src_crs=raster.crs,
                                dst_transform=transform,
                                dst_crs=dst_crs,
                                resampling=Resampling.nearest,
                            )
            except Exception as e:
                print(f"Error processing {city}: {e}")
                continue
