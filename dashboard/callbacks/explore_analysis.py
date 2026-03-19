"""
dashboard/callbacks/explore_analysis.py
Callbacks for the Analysis tab on the Explore page.

Handles on-demand execution of:
  - DirectionalBenchNeighbors
  - AvgSpacingCalculator
  - FloatingSectionWPS

Also handles overrides CSV upload / toggle.
"""

import base64
import io
import logging

import dash
import pandas as pd
from dash import Input, Output, State, callback, html, dash_table

from dashboard.pipeline import (
    load_cached_pipeline,
    run_directional_bench_neighbors,
    run_avg_spacing,
    run_floating_wps,
)

logger = logging.getLogger("dashboard")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_csv_upload(contents: str) -> pd.DataFrame:
    """Decode a dcc.Upload CSV contents string into a DataFrame."""
    _, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    return pd.read_csv(io.StringIO(decoded.decode("utf-8")))


def _safe_float(value) -> float | None:
    """Convert input value to float, returning None if empty/invalid."""
    if value is None or value == "" or value == "None":
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _df_to_table_props(df: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    """Convert DataFrame to (columns, data) for dash_table.DataTable."""
    # Round floats for display
    for col in df.select_dtypes(include=["float64", "float32"]).columns:
        df[col] = df[col].round(2)
    columns = [{"name": c, "id": c} for c in df.columns]
    data = df.to_dict("records")
    return columns, data


# ---------------------------------------------------------------------------
# Overrides: toggle shared vs separate
# ---------------------------------------------------------------------------

@callback(
    Output("overrides-shared-div", "style"),
    Output("overrides-separate-div", "style"),
    Input("overrides-mode", "value"),
    prevent_initial_call=True,
)
def toggle_overrides_mode(mode):
    if mode == "separate":
        return {"display": "none"}, {"display": "block"}
    return {"display": "block"}, {"display": "none"}


# ---------------------------------------------------------------------------
# Overrides: parse uploaded CSV into editable table
# ---------------------------------------------------------------------------

@callback(
    Output("overrides-table", "data"),
    Output("overrides-table", "columns"),
    Input("overrides-upload", "contents"),
    Input("overrides-dbn-upload", "contents"),
    Input("overrides-avg-upload", "contents"),
    prevent_initial_call=True,
)
def parse_overrides_csv(shared_contents, dbn_contents, avg_contents):
    """Parse whichever CSV was uploaded and display in the editable table."""
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    contents = None
    if trigger_id == "overrides-upload" and shared_contents:
        contents = shared_contents
    elif trigger_id == "overrides-dbn-upload" and dbn_contents:
        contents = dbn_contents
    elif trigger_id == "overrides-avg-upload" and avg_contents:
        contents = avg_contents

    if not contents:
        return dash.no_update, dash.no_update

    try:
        df = _parse_csv_upload(contents)
        # Rename 'uwi' column if needed
        if "well_i" in df.columns and "uwi" not in df.columns:
            df = df.rename(columns={"well_i": "uwi"})
        # Keep only expected columns
        expected = ["uwi", "cutoff_ft", "vertical_cutoff_ft", "overlap_pct_k_min"]
        keep = [c for c in expected if c in df.columns]
        if not keep or "uwi" not in keep:
            return [], [{"name": c, "id": c} for c in expected]
        df = df[keep]
        cols = [{"name": c, "id": c, "type": "numeric" if c != "uwi" else "text"} for c in df.columns]
        return df.to_dict("records"), cols
    except Exception as exc:
        logger.warning("Failed to parse overrides CSV: %s", exc)
        return dash.no_update, dash.no_update


def _get_overrides_df(table_data: list[dict] | None) -> pd.DataFrame | None:
    """Build overrides DataFrame from the editable table data."""
    if not table_data:
        return None
    df = pd.DataFrame(table_data)
    if df.empty or "uwi" not in df.columns:
        return None
    # Convert numeric columns
    for col in ["cutoff_ft", "vertical_cutoff_ft", "overlap_pct_k_min"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Run DirectionalBenchNeighbors
# ---------------------------------------------------------------------------

@callback(
    Output("dbn-result-store", "data"),
    Output("dbn-result-table", "columns"),
    Output("dbn-result-table", "data"),
    Output("dbn-status", "children"),
    Input("btn-run-dbn", "n_clicks"),
    State("dbn-cutoff-ft", "value"),
    State("dbn-vertical-cutoff-ft", "value"),
    State("dbn-overlap-pct-min", "value"),
    State("dbn-axis-mode", "value"),
    State("dbn-prefer-axis", "value"),
    State("overrides-table", "data"),
    State("pipeline-result-store", "data"),
    prevent_initial_call=True,
)
def run_dbn_callback(n_clicks, cutoff_ft, vertical_cutoff_ft, overlap_pct_min,
                     axis_mode, prefer_axis, overrides_data, pipeline_result):
    if not pipeline_result or not pipeline_result.get("cache_path"):
        return dash.no_update, [], [], "No pipeline data. Run Calculate first."

    cutoff = _safe_float(cutoff_ft)
    if cutoff is None or cutoff <= 0:
        return dash.no_update, [], [], "Cutoff (ft) is required and must be > 0."

    try:
        data = load_cached_pipeline(pipeline_result["cache_path"])
        df_ik = data.get("df_ik_pairs")
        if df_ik is None or df_ik.empty:
            return dash.no_update, [], [], "No IK pairs found in cache."

        header_df = data["header_df"]
        overrides_df = _get_overrides_df(overrides_data)

        result = run_directional_bench_neighbors(
            df_ik, header_df,
            cutoff_ft=cutoff,
            vertical_cutoff_ft=_safe_float(vertical_cutoff_ft),
            overlap_pct_k_min=_safe_float(overlap_pct_min),
            axis_mode=axis_mode or "any",
            prefer_axis=prefer_axis if prefer_axis else None,
            overrides_df=overrides_df,
        )
        cols, rows = _df_to_table_props(result)
        status = f"Done: {len(result):,} wells"
        # Store as JSON records for other callbacks
        return result.to_dict("records"), cols, rows, status

    except Exception as exc:
        logger.exception("DBN run failed: %s", exc)
        return dash.no_update, [], [], f"Error: {exc}"


# ---------------------------------------------------------------------------
# Run AvgSpacingCalculator
# ---------------------------------------------------------------------------

@callback(
    Output("avg-spacing-result-store", "data"),
    Output("avg-result-table", "columns"),
    Output("avg-result-table", "data"),
    Output("avg-status", "children"),
    Input("btn-run-avg", "n_clicks"),
    State("dbn-cutoff-ft", "value"),
    State("dbn-vertical-cutoff-ft", "value"),
    State("dbn-overlap-pct-min", "value"),
    State("dbn-axis-mode", "value"),
    State("avg-neighborhood-mode", "value"),
    State("avg-chain-sort-mode", "value"),
    State("avg-edge-pick", "value"),
    State("overrides-table", "data"),
    State("pipeline-result-store", "data"),
    prevent_initial_call=True,
)
def run_avg_callback(n_clicks, cutoff_ft, vertical_cutoff_ft, overlap_pct_min,
                     axis_mode, neighborhood_mode, chain_sort_mode, edge_pick,
                     overrides_data, pipeline_result):
    if not pipeline_result or not pipeline_result.get("cache_path"):
        return dash.no_update, [], [], "No pipeline data. Run Calculate first."

    cutoff = _safe_float(cutoff_ft)
    if cutoff is None or cutoff <= 0:
        return dash.no_update, [], [], "Cutoff (ft) is required and must be > 0."

    try:
        data = load_cached_pipeline(pipeline_result["cache_path"])
        df_ik = data.get("df_ik_pairs")
        if df_ik is None or df_ik.empty:
            return dash.no_update, [], [], "No IK pairs found in cache."

        lateral_df = data["lateral_df"]
        overrides_df = _get_overrides_df(overrides_data)

        result = run_avg_spacing(
            df_ik, lateral_df,
            cutoff_ft=cutoff,
            vertical_cutoff_ft=_safe_float(vertical_cutoff_ft),
            overlap_pct_k_min=_safe_float(overlap_pct_min),
            overrides_df=overrides_df,
            axis_mode=axis_mode or "any",
            neighborhood_mode=neighborhood_mode or "chain",
            chain_sort_mode=chain_sort_mode or "pca",
            edge_pick=edge_pick or "min",
        )
        cols, rows = _df_to_table_props(result)
        status = f"Done: {len(result):,} wells"
        return result.to_dict("records"), cols, rows, status

    except Exception as exc:
        logger.exception("AvgSpacing run failed: %s", exc)
        return dash.no_update, [], [], f"Error: {exc}"


# ---------------------------------------------------------------------------
# Run FloatingSectionWPS
# ---------------------------------------------------------------------------

@callback(
    Output("wps-result-store", "data"),
    Output("wps-result-table", "columns"),
    Output("wps-result-table", "data"),
    Output("wps-status", "children"),
    Input("btn-run-wps", "n_clicks"),
    State("wps-box-hw", "value"),
    State("wps-box-hh", "value"),
    State("wps-corr-hw", "value"),
    State("wps-corr-ea", "value"),
    State("wps-min-inside", "value"),
    State("wps-exclude-self", "value"),
    State("pipeline-result-store", "data"),
    prevent_initial_call=True,
)
def run_wps_callback(n_clicks, box_hw, box_hh, corr_hw, corr_ea,
                     min_inside, exclude_self, pipeline_result):
    if not pipeline_result or not pipeline_result.get("cache_path"):
        return dash.no_update, [], [], "No pipeline data. Run Calculate first."

    try:
        data = load_cached_pipeline(pipeline_result["cache_path"])
        lateral_df = data["lateral_df"]

        result = run_floating_wps(
            lateral_df,
            box_half_width_ft=float(box_hw or 2640),
            box_half_height_ft=float(box_hh or 2640),
            corridor_half_width_ft=float(corr_hw or 2640),
            corridor_extra_along_ft=float(corr_ea or 0),
            min_inside_ft=float(min_inside or 660),
            exclude_self=bool(exclude_self),
        )
        cols, rows = _df_to_table_props(result)
        status = f"Done: {len(result):,} wells"
        return result.to_dict("records"), cols, rows, status

    except Exception as exc:
        logger.exception("FloatingWPS run failed: %s", exc)
        return dash.no_update, [], [], f"Error: {exc}"
