"""
dashboard/components/map_panel.py
Builds GeoPandas GeoDataFrames for wellbore sticks and bottom-hole markers.
These are converted to GeoJSON for dash-leaflet dl.GeoJSON layers.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point


def build_trajectory_geodataframe(
    df_lateral: pd.DataFrame,
    df_header: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """
    Build a GeoDataFrame of wellbore sticks from the directional survey.

    Each well = one LineString (lon, lat per MD station, sorted by MD).
    Last station = bottom-hole location.
    Joined with header for well_name, bench, first_prod_date, operator.

    Args:
        df_lateral: Lateral-only directional survey. Requires: uwi, md, latitude, longitude.
        df_header: Well header. Requires: uwi. Optional: well_name, bench, first_prod_date, operator.

    Returns:
        GeoDataFrame (EPSG:4326) with one row per well.
    """
    rows = []
    for uwi, group in df_lateral.groupby("uwi"):
        group = group.sort_values("md")
        # GeoJSON / Leaflet convention: (longitude, latitude)
        coords = list(zip(group["longitude"], group["latitude"]))
        if len(coords) < 2:
            continue
        rows.append({
            "uwi":    uwi,
            "geometry": LineString(coords),
            "bh_lon": coords[-1][0],
            "bh_lat": coords[-1][1],
        })

    if not rows:
        return gpd.GeoDataFrame(columns=["uwi", "geometry"], crs="EPSG:4326")

    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")

    keep_cols = ["uwi", "well_name", "bench", "first_prod_date", "operator",
                 "hole_direction", "spud_date", "rsv_cat", "lateral_length_ft"]
    header_cols = [c for c in keep_cols if c in df_header.columns]
    gdf = gdf.merge(df_header[header_cols], on="uwi", how="left")

    # Ensure uwi is string for proper GeoJSON serialization / tooltip display
    # (should already be str if loaded via WellDataLoader, but defensive)
    if gdf["uwi"].dtype != object:
        gdf["uwi"] = gdf["uwi"].astype(str)

    # Add spud_year for color-by-year support
    if "spud_date" in gdf.columns:
        try:
            gdf["spud_year"] = pd.to_datetime(gdf["spud_date"], errors="coerce").dt.year.astype("Int64").astype(str)
            gdf["spud_year"] = gdf["spud_year"].replace("<NA>", "Unknown")
        except Exception:
            gdf["spud_year"] = "Unknown"

    # Convert all date/timestamp columns to strings for JSON serialization.
    # GeoJSON properties must be JSON-serializable; Timestamp objects are not.
    for col in gdf.columns:
        if col == "geometry":
            continue
        if hasattr(gdf[col].dtype, "tz") or pd.api.types.is_datetime64_any_dtype(gdf[col]):
            gdf[col] = gdf[col].astype(str).replace("NaT", "")
        elif gdf[col].dtype == "object":
            # Clean up any remaining non-string objects (e.g. Timestamp stored as object)
            gdf[col] = gdf[col].apply(lambda v: str(v) if not pd.isna(v) else "")

    return gdf


def inject_feature_colors(
    geojson: dict,
    color_by: str,
    color_map: dict[str, str],
    default_color: str = "#3388ff",
) -> dict:
    """Inject a '_color' property into each GeoJSON feature based on color_by column.

    This avoids JS function references — Leaflet's GeoJSON style function
    can read feature.properties.style.color directly via the 'style' key
    in each feature's properties.
    """
    for feat in geojson.get("features", []):
        props = feat.get("properties", {})
        val = str(props.get(color_by, "")) if color_by else ""
        color = color_map.get(val, default_color)
        # Embed style directly in the feature for Leaflet to pick up
        props["_color"] = color
    return geojson


def build_bottomhole_geodataframe(gdf_trajectories: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Separate GeoDataFrame of bottom-hole Point markers.
    Derived from the last coordinate of each trajectory LineString.
    """
    gdf_bh = gdf_trajectories.copy()
    gdf_bh["geometry"] = gdf_bh.apply(
        lambda r: Point(r["bh_lon"], r["bh_lat"]), axis=1
    )
    return gdf_bh
