#!/usr/bin/env python3
"""
Description: Script to download OpenStreetMap (OSM) points of interest (POIs)
for locations in the Place Pulse 2.0 dataset.
"""
import gc
import os
from typing import Any, Dict, List

import geopandas as gpd
import overpy
import pandas as pd
from shapely.geometry import Point, box

# Configuration
# Path to Place Pulse 2.0 dataset files
LOCATIONS_FILE = "svi_data/Place_pulse_2_0/locations.tsv"
PLACES_FILE = "svi_data/Place_pulse_2_0/places.tsv"
# Path to output directory for POIs
OUTPUT_DIR = "svi_data/Place_pulse_2_0/POIs/"
# Buffer distance for POI extraction
BUFFER_DISTANCE_M = 1000  # Buffer radius in meters

# Custom filters for POI extraction
CUSTOM_FILTER: Dict[str, Any] = {
    "healthcare": True,
    "shop": True,
    "leisure": True,
    "amenity": True,
    "tourism": True,
    "building": ["religious", "transportation"],
    "public_transport": ["station"],
    "theatre": True,
    "cinema": True,
}


def ensure_output_dir(path: str) -> None:
    """
    Create the output directory if it does not exist.

    Parameters
    ----------
    path : str
        Directory path to create.
    """
    os.makedirs(path, exist_ok=True)


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
    return gpd.GeoDataFrame(merged, geometry=geopoints, crs="EPSG:4326")


def compute_bbox(points: gpd.GeoSeries) -> List[float]:
    """
    Compute WGS84 bounding box for buffered points.

    Parameters
    ----------
    points : geopandas.GeoSeries
        Series of point geometries.

    Returns
    -------
    List[float]
        [south, west, north, east] bounds in degrees.
    """
    utm_crs = points.estimate_utm_crs()
    buffered = points.to_crs(utm_crs).buffer(BUFFER_DISTANCE_M)
    unioned = buffered.union_all()
    minx, miny, maxx, maxy = unioned.bounds
    bounds_box = gpd.GeoSeries([box(minx, miny, maxx, maxy)], crs=utm_crs)
    south, west, north, east = bounds_box.to_crs(epsg=4326).total_bounds[
        [1, 0, 3, 2]
    ]
    return [south, west, north, east]


def build_overpass_query(
    bounds: List[float], tag_filters: Dict[str, Any]
) -> str:
    """
    Construct an Overpass QL query for nodes matching tag filters.

    Parameters
    ----------
    bounds : list of float
        [south, west, north, east] in degrees.
    tag_filters : dict
        Tag keys mapped to True or list of allowed values.

    Returns
    -------
    str
        Overpass QL query string.
    """
    south, west, north, east = bounds
    clauses = []

    for key, vals in tag_filters.items():
        if vals is True:
            clause = f'  node["{key}"]({south},{west},' f"{north},{east});"
        else:
            pattern = "|".join(vals)
            clause = (
                f'  node["{key}"~"^({pattern})$"]('
                f"{south},{west},{north},{east});"
            )
        clauses.append(clause)

    query_body = "\n".join(clauses)
    return "[out:json][timeout:120];\n" "(\n" + query_body + "\n);\nout body;"


def fetch_poi_nodes(
    query: str, client: overpy.Overpass
) -> List[Dict[str, Any]]:
    """
    Execute Overpass query and collect node information.

    Parameters
    ----------
    query : str
        Overpass QL query.
    client : overpy.Overpass
        Overpass API client instance.

    Returns
    -------
    list of dict
        Node records with tags and geometry.
    """
    result = client.query(query)
    records: List[Dict[str, Any]] = []

    for node in result.nodes:
        tags = node.tags or {}
        record = {
            "id": node.id,
            "lon": node.lon,
            "lat": node.lat,
            "geometry": Point(node.lon, node.lat),
            "name": tags.get("name"),
        }
        for attr in [
            "amenity",
            "shop",
            "leisure",
            "tourism",
            "building",
            "public_transport",
            "theatre",
            "cinema",
            "religion",
        ]:
            record[attr] = tags.get(attr)
        records.append(record)

    return records


def filter_and_classify_pois(pois: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Remove unwanted amenities and classify POIs.

    Parameters
    ----------
    pois : geopandas.GeoDataFrame
        GeoDataFrame of raw POIs.

    Returns
    -------
    geopandas.GeoDataFrame
        Filtered with 'poi_type' and 'name'.
    """
    exclude = [
        "parking",
        "parking_space",
        "bench",
        "bicycle_parking",
        "motorcycle_parking",
        "post_box",
        "toilets",
    ]
    pois = pois[~pois["amenity"].isin(exclude)].copy()

    pois["poi_type"] = pois["amenity"].fillna(pois["leisure"])

    health_mask = pois.apply(
        lambda row: any(
            "healthcare" in (row[col] or "")
            for col in ["amenity", "shop", "leisure", "tourism"]
        ),
        axis=1,
    )
    mask = health_mask & pois["poi_type"].isna()
    pois.loc[mask, "poi_type"] = "healthcare"

    museum_mask = pois["name"].str.contains("museum", case=False, na=False)
    mask = museum_mask & pois["poi_type"].isna()
    pois.loc[mask, "poi_type"] = "museum"

    for col in ["religion", "public_transport", "shop", "tourism"]:
        pois["poi_type"] = pois["poi_type"].fillna(pois[col])

    final_cols = ["id", "lon", "lat", "geometry", "poi_type", "name"]
    pois = pois[final_cols].dropna()
    return pois.reset_index(drop=True)


def main() -> None:
    """
    Main workflow: load data, query Overpass, process, save.

    Returns
    -------
    None
    """
    ensure_output_dir(OUTPUT_DIR)
    api_client = overpy.Overpass()
    geo_df = load_place_pulse_data(LOCATIONS_FILE, PLACES_FILE)

    for place_name, group in geo_df.groupby("place_name"):
        bounds = compute_bbox(group.geometry)
        print(
            f"{place_name}: bbox = {bounds[0]:.6f}, "
            f"{bounds[1]:.6f}, {bounds[2]:.6f}, "
            f"{bounds[3]:.6f}"
        )

        query_str = build_overpass_query(bounds, CUSTOM_FILTER)

        nodes = fetch_poi_nodes(query_str, api_client)
        if not nodes:
            print(f"{place_name}: no POIs found, skipping.")
            continue

        poi_gdf = gpd.GeoDataFrame(nodes, geometry="geometry", crs="EPSG:4326")
        processed = filter_and_classify_pois(poi_gdf)

        safe_name = place_name.lower().replace(" ", "_")
        out_path = os.path.join(
            OUTPUT_DIR, f"pois_{safe_name}_overpass.geojson"
        )
        processed.to_file(out_path, driver="GeoJSON")
        print(f"{place_name}: saved {len(processed)} POIs " f"→ {out_path}")

        del nodes, poi_gdf, processed
        gc.collect()


if __name__ == "__main__":
    main()
