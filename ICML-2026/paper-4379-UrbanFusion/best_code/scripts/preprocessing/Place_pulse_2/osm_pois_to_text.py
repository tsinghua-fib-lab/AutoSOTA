#!/usr/bin/env python3
"""
Description: Script to extract and process OpenStreetMap (OSM) points of
interest (POIs) and convert them into text descriptions for locations.
Used for generating PP2-M dataset.
"""

import os

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

# Define paths
pois_folder = r"svi_data/Place-pulse-2.0/POIs/"
locations_folder = r"svi_data/Place-pulse-2.0/Locations/"
output_folder = r"svi_data/Place-pulse-2.0/POIs_Texts/"
location_file = r"svi_data/Place-pulse-2.0/locations.tsv"
places_file = r"svi_data/Place-pulse-2.0/places.tsv"


def get_nearest(
    src_points: np.ndarray,
    candidates: np.ndarray,
    k_neighbors: int = 10,
    remove_first: bool = True,
    max_distance: float = None,
) -> tuple:
    """
    Find nearest neighbors for all source points from a set of candidate
    points, optionally filtering out neighbors above a specified distance
    threshold.

    Parameters
    ----------
    src_points : array-like of shape (n_samples, n_features)
        Array of source points.
    candidates : array-like of shape (n_candidates, n_features)
        Array of candidate points.
    k_neighbors : int, optional
        Number of nearest neighbors to retrieve, default is 10.
    remove_first : bool, optional
        Whether to remove the first neighbor (e.g., the point itself), default
        is True.
    max_distance : float, optional
        Maximum allowable distance for neighbors. Neighbors with distances
        above this threshold are removed. If None, no threshold filtering is
        applied. Default is None.

    Returns
    -------
    tuple of lists of arrays
        A tuple containing two lists:
          - A list of arrays of neighbor indices for each source point.
          - A list of arrays of corresponding distances.
    """
    num_neighbors = k_neighbors + int(remove_first)
    tree = BallTree(candidates, leaf_size=15, metric="euclidean")
    distances, indices = tree.query(src_points, k=num_neighbors)

    # Optionally remove the first neighbor (the point itself)
    if remove_first:
        distances = distances[:, 1:]
        indices = indices[:, 1:]

    # Apply maximum distance filtering if threshold is provided
    if max_distance is not None:
        filtered_indices = []
        filtered_distances = []
        for d_row, i_row in zip(distances, indices):
            mask = d_row <= max_distance
            filtered_indices.append(i_row[mask])
            filtered_distances.append(d_row[mask])
        return filtered_indices, filtered_distances

    return indices, distances


def convert_to_text(
    pois: gpd.GeoDataFrame, closest_pois: list, distance_of_closest: list
) -> list:
    """
    Convert a list of POIs to a texts for each location.

    Parameters
    ----------
    pois : GeoDataFrame
        GeoDataFrame containing POIs with their attributes.
    closest_pois : list of arrays
        List of arrays containing indices of the closest POIs for each
        location.
    distance_of_closest : list of arrays
        List of arrays containing distances to the closest POIs for each
        location.

    Returns
    -------
    list of str
        List of text descriptions for each location, including nearby POIs and
        their distances.
    """
    text_per_location = []
    for i in range(len(closest_pois)):
        text_for_location = "Nearby OSM points of interests are:"
        # print(f"=== Point labelled as {row_of_activity['label']}  === ")
        # # (coords: {row_of_activity['geometry']})
        for k in range(len(closest_pois[i])):
            close_poi = closest_pois[i][k]
            poi_type = pois.iloc[close_poi]["poi_type"]
            name = pois.iloc[close_poi]["name"]
            # print(pois.iloc[close_poi]["address"])
            dist = distance_of_closest[i][k]
            text_for_location += (
                f"\n{name} (type: {poi_type}) with distance {round(dist)}m,"
            )
        text_per_location.append(text_for_location)
    return text_per_location


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
    geopandas.GeoDataFrame
        GeoDataFrame with point geometries in EPSG:4326.
    """
    df_loc = pd.read_csv(loc_file, sep="\t").rename(
        columns={"_id": "location_id"}
    )
    df_pl = pd.read_csv(pl_file, sep="\t").rename(columns={"_id": "place_id"})
    merged = pd.merge(df_loc, df_pl, on="place_id", how="left")
    geopoints = gpd.points_from_xy(merged["loc.1"], merged["loc.0"])
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
    return gpd.GeoDataFrame(merged, geometry=geopoints, crs="EPSG:4326")


if __name__ == "__main__":
    # Load all locations once
    location_gdf = load_place_pulse_data(location_file, places_file)

    os.makedirs(output_folder, exist_ok=True)

    # Settings
    buffer_around_locations = 200  # meters
    num_neighbors = 15  # number of POIs to consider

    # List all POI files
    pois_texts = []

    # Iterate over all POI files
    for poi_filename in os.listdir(pois_folder):
        if poi_filename.endswith(".geojson"):
            city_name = poi_filename.replace("pois_", "").replace(
                "_overpass.geojson", ""
            )
            print(f"Processing city: {city_name}")

            # Paths
            in_path_pois = os.path.join(pois_folder, poi_filename)
            out_path = os.path.join(
                output_folder,
                (
                    f"pois_{city_name}_text_{num_neighbors}neigh_"
                    f"{buffer_around_locations}m.csv"
                ),
            )

            # Load POIs
            pois = gpd.read_file(in_path_pois)

            # Filter only locations for current city
            locations_city = location_gdf[
                location_gdf["place_name"].str.lower().str.replace(" ", "_")
                == city_name
            ].copy()

            if locations_city.empty:
                print(f"⚠️ No locations found for {city_name}, skipping.")
                continue

            # Estimate a local UTM CRS based on the POIs
            local_crs = pois.estimate_utm_crs()

            # Project both POIs and locations to local CRS
            pois_proj = pois.to_crs(local_crs)
            locations_proj = locations_city.to_crs(local_crs)

            # Get coordinates
            poi_coord_arr = np.stack(
                [pois_proj.geometry.x, pois_proj.geometry.y]
            ).swapaxes(1, 0)
            data_coord_arr = np.stack(
                [locations_proj.geometry.x, locations_proj.geometry.y]
            ).swapaxes(1, 0)

            # Find nearest POIs
            closest_pois, distance_of_closest = get_nearest(
                data_coord_arr,
                poi_coord_arr,
                k_neighbors=num_neighbors,
                remove_first=False,
                max_distance=buffer_around_locations,
            )

            # Generate text descriptions
            text_per_location = convert_to_text(
                pois, closest_pois, distance_of_closest
            )
            locations_proj["text"] = text_per_location

            # Save output
            locations_proj.to_csv(
                out_path, index=False, sep=",", encoding="utf-8"
            )
            pois_texts.append(locations_proj[["image_name", "text"]])
            print(f"✅ Saved text descriptions for {city_name} → {out_path}")

    # Combine all texts into one DataFrame
    all_texts_df = pd.concat(pois_texts, ignore_index=True)
    print(all_texts_df.head())
    all_texts_df.to_csv(
        os.path.join(
            output_folder,
            f"texts_{num_neighbors}neigh_{buffer_around_locations}m.csv",
        ),
        index=False,
        sep=",",
        encoding="utf-8",
    )
