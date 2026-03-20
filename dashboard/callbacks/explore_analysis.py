"""
dashboard/callbacks/explore_analysis.py
Callbacks for the Analysis tab on the Explore page.

Handles on-demand execution of:
  - DirectionalBenchNeighbors
  - AvgSpacingCalculator
  - FloatingSectionWPS
  - Debug pair spacing visualization (Phase B)
  - FloatingSectionWPS visualization (Phase B)
  - AvgSpacing neighborhood plot (Phase B)

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
    run_debug_pair_spacing,
    run_wps_visualize,
    run_avg_spacing_plot,
)
from dashboard.components.matplotlib_render import fig_to_base64_img, capture_all_open_figs

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


_FIGSIZE_MAP = {
    # Debug pair spacing
    ("debug", "small"): (6, 4),
    ("debug", "medium"): (8, 5),
    ("debug", "large"): (12, 7),
    # WPS visualization (square)
    ("wps", "small"): (7, 7),
    ("wps", "medium"): (10, 10),
    ("wps", "large"): (14, 14),
    # Avg neighborhood
    ("avg", "small"): (8, 5),
    ("avg", "medium"): (12.5, 8),
    ("avg", "large"): (16, 10),
}


def _resolve_figsize(category: str, key: str | None) -> tuple[float, float]:
    """Look up figsize tuple from category + size key."""
    return _FIGSIZE_MAP.get((category, key or "medium"), _FIGSIZE_MAP[(category, "medium")])


def _resolve_dpi(dpi_str: str | None) -> int:
    try:
        return int(dpi_str or 150)
    except (ValueError, TypeError):
        return 150


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
    Output("dbn-col-selector", "options"),
    Output("dbn-col-selector", "value"),
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
        return dash.no_update, [], [], "No pipeline data. Run Calculate first.", [], []

    cutoff = _safe_float(cutoff_ft)
    if cutoff is None or cutoff <= 0:
        return dash.no_update, [], [], "Cutoff (ft) is required and must be > 0.", [], []

    try:
        data = load_cached_pipeline(pipeline_result["cache_path"])
        df_ik = data.get("df_ik_pairs")
        if df_ik is None or df_ik.empty:
            return dash.no_update, [], [], "No IK pairs found in cache.", [], []

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
        all_cols = [c["id"] for c in cols]
        options = [{"label": c, "value": c} for c in all_cols]
        status = f"Done: {len(result):,} wells"
        return result.to_dict("records"), cols, rows, status, options, all_cols

    except Exception as exc:
        logger.exception("DBN run failed: %s", exc)
        return dash.no_update, [], [], f"Error: {exc}", [], []


# ---------------------------------------------------------------------------
# Run AvgSpacingCalculator
# ---------------------------------------------------------------------------

@callback(
    Output("avg-spacing-result-store", "data"),
    Output("avg-result-table", "columns"),
    Output("avg-result-table", "data"),
    Output("avg-status", "children"),
    Output("avg-col-selector", "options"),
    Output("avg-col-selector", "value"),
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
        return dash.no_update, [], [], "No pipeline data. Run Calculate first.", [], []

    cutoff = _safe_float(cutoff_ft)
    if cutoff is None or cutoff <= 0:
        return dash.no_update, [], [], "Cutoff (ft) is required and must be > 0.", [], []

    try:
        data = load_cached_pipeline(pipeline_result["cache_path"])
        df_ik = data.get("df_ik_pairs")
        if df_ik is None or df_ik.empty:
            return dash.no_update, [], [], "No IK pairs found in cache.", [], []

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
        all_cols = [c["id"] for c in cols]
        options = [{"label": c, "value": c} for c in all_cols]
        status = f"Done: {len(result):,} wells"
        return result.to_dict("records"), cols, rows, status, options, all_cols

    except Exception as exc:
        logger.exception("AvgSpacing run failed: %s", exc)
        return dash.no_update, [], [], f"Error: {exc}", [], []


# ---------------------------------------------------------------------------
# Run FloatingSectionWPS
# ---------------------------------------------------------------------------

@callback(
    Output("wps-result-store", "data"),
    Output("wps-result-table", "columns"),
    Output("wps-result-table", "data"),
    Output("wps-status", "children"),
    Output("wps-col-selector", "options"),
    Output("wps-col-selector", "value"),
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
        return dash.no_update, [], [], "No pipeline data. Run Calculate first.", [], []

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
        all_cols = [c["id"] for c in cols]
        options = [{"label": c, "value": c} for c in all_cols]
        status = f"Done: {len(result):,} wells"
        return result.to_dict("records"), cols, rows, status, options, all_cols

    except Exception as exc:
        logger.exception("FloatingWPS run failed: %s", exc)
        return dash.no_update, [], [], f"Error: {exc}", [], []


# ---------------------------------------------------------------------------
# Column selector callbacks for analysis tables
# ---------------------------------------------------------------------------

@callback(
    Output("dbn-result-table", "columns", allow_duplicate=True),
    Input("dbn-col-selector", "value"),
    State("dbn-result-table", "data"),
    prevent_initial_call=True,
)
def filter_dbn_columns(selected_cols, data):
    if not selected_cols or not data:
        return dash.no_update
    return [{"name": c, "id": c} for c in selected_cols]


@callback(
    Output("avg-result-table", "columns", allow_duplicate=True),
    Input("avg-col-selector", "value"),
    State("avg-result-table", "data"),
    prevent_initial_call=True,
)
def filter_avg_columns(selected_cols, data):
    if not selected_cols or not data:
        return dash.no_update
    return [{"name": c, "id": c} for c in selected_cols]


@callback(
    Output("wps-result-table", "columns", allow_duplicate=True),
    Input("wps-col-selector", "value"),
    State("wps-result-table", "data"),
    prevent_initial_call=True,
)
def filter_wps_columns(selected_cols, data):
    if not selected_cols or not data:
        return dash.no_update
    return [{"name": c, "id": c} for c in selected_cols]


# ---------------------------------------------------------------------------
# Fullscreen modal toggles for diagnostic plots
# ---------------------------------------------------------------------------

@callback(
    Output("modal-debug-pair", "is_open"),
    Output("modal-debug-pair-body", "children"),
    Input("btn-expand-debug-pair", "n_clicks"),
    State("debug-pair-plots", "children"),
    prevent_initial_call=True,
)
def expand_debug_pair(n_clicks, plot_children):
    if not plot_children:
        return False, "No plots generated yet. Run Debug Pair Spacing first."
    return True, plot_children


@callback(
    Output("modal-wps-viz", "is_open"),
    Output("modal-wps-viz-body", "children"),
    Input("btn-expand-wps-viz", "n_clicks"),
    State("wps-viz-plot", "children"),
    prevent_initial_call=True,
)
def expand_wps_viz(n_clicks, plot_children):
    if not plot_children:
        return False, "No plot generated yet. Run WPS Visualization first."
    return True, plot_children


@callback(
    Output("modal-avg-viz", "is_open"),
    Output("modal-avg-viz-body", "children"),
    Input("btn-expand-avg-viz", "n_clicks"),
    State("avg-viz-plot", "children"),
    prevent_initial_call=True,
)
def expand_avg_viz(n_clicks, plot_children):
    if not plot_children:
        return False, "No plot generated yet. Run Neighborhood Plot first."
    return True, plot_children


# ===========================================================================
# Phase B — Diagnostic Visualizations
# ===========================================================================

# ---------------------------------------------------------------------------
# Auto-fill WPS viz + Avg viz UWI from map click
# ---------------------------------------------------------------------------

@callback(
    Output("wps-viz-uwi", "value"),
    Output("avg-viz-uwi", "value"),
    Input("selected-wells-store", "data"),
    prevent_initial_call=True,
)
def autofill_viz_uwi(selected):
    """When a well is clicked on the map, auto-fill the viz UWI fields."""
    if not selected or not selected.get("clicked_uwi"):
        return dash.no_update, dash.no_update
    uwi = str(selected["clicked_uwi"])
    return uwi, uwi


# ---------------------------------------------------------------------------
# Auto-fill debug pair UWIs from IK table row click
# ---------------------------------------------------------------------------

@callback(
    Output("debug-uwi-i", "value"),
    Output("debug-uwi-k", "value"),
    Input("ik-pairs-table", "active_cell"),
    State("ik-pairs-table", "data"),
    prevent_initial_call=True,
)
def autofill_debug_pair_from_ik(active_cell, table_data):
    """When user clicks a row in the IK Pairs table, auto-fill debug UWIs."""
    if not active_cell or not table_data:
        return dash.no_update, dash.no_update
    row = table_data[active_cell["row"]]
    uwi_i = str(row.get("well_i", ""))
    uwi_k = str(row.get("well_k", ""))
    if uwi_i and uwi_k:
        return uwi_i, uwi_k
    return dash.no_update, dash.no_update


# ---------------------------------------------------------------------------
# Run debug_pair_spacing
# ---------------------------------------------------------------------------

@callback(
    Output("debug-pair-metrics-table", "columns"),
    Output("debug-pair-metrics-table", "data"),
    Output("debug-pair-plots", "children"),
    Output("debug-pair-status", "children"),
    Input("btn-debug-pair", "n_clicks"),
    State("debug-uwi-i", "value"),
    State("debug-uwi-k", "value"),
    State("debug-step-ft", "value"),
    State("debug-contact-ft", "value"),
    State("debug-pca-axis", "value"),
    State("debug-theta-par", "value"),
    State("debug-theta-perp", "value"),
    State("debug-figsize", "value"),
    State("debug-dpi", "value"),
    State("pipeline-result-store", "data"),
    prevent_initial_call=True,
)
def run_debug_pair_callback(
    n_clicks, uwi_i, uwi_k, step_ft, contact_ft, pca_axis,
    theta_par, theta_perp, figsize_key, dpi_str, pipeline_result,
):
    if not pipeline_result or not pipeline_result.get("cache_path"):
        return [], [], [], "No pipeline data. Run Calculate first."

    if not uwi_i or not uwi_k:
        return [], [], [], "Both Well I and Well K UWIs are required."

    try:
        data = load_cached_pipeline(pipeline_result["cache_path"])
        lateral_df = data["lateral_df"]
        header_df = data["header_df"]

        # Validate both UWIs exist in the lateral data
        available = set(lateral_df["uwi"].astype(str).unique())
        if str(uwi_i) not in available:
            return [], [], [], f"Well I '{uwi_i}' not found in lateral data."
        if str(uwi_k) not in available:
            return [], [], [], f"Well K '{uwi_k}' not found in lateral data."

        metrics, figs = run_debug_pair_spacing(
            lateral_df, header_df,
            uwi_i=str(uwi_i),
            uwi_k=str(uwi_k),
            step_ft=int(step_ft or 100),
            use_pca_axis=bool(pca_axis),
            theta_parallel_deg=float(theta_par or 25),
            theta_perp_deg=float(theta_perp or 65),
            contact_threshold_ft=float(contact_ft or 300),
        )

        # Build metrics summary table (key-value pairs, excluding 'paths')
        display_keys = [k for k in metrics if k != "paths"]
        metrics_rows = [{"metric": k, "value": _fmt_metric(metrics[k])} for k in display_keys]
        metrics_cols = [{"name": "Metric", "id": "metric"}, {"name": "Value", "id": "value"}]

        # Apply user-selected figure size and render to inline images
        dpi = _resolve_dpi(dpi_str)
        w, h = _resolve_figsize("debug", figsize_key)
        for fig in figs:
            fig.set_size_inches(w, h)
        images = [fig_to_base64_img(fig, dpi=dpi) for fig in figs]

        alignment = metrics.get("pair_alignment", "?")
        hz = metrics.get("horizontal_dist", "?")
        status = f"Done: {alignment} | horizontal_dist={hz} ft | {len(figs)} plots"
        return metrics_cols, metrics_rows, images, status

    except Exception as exc:
        logger.exception("debug_pair_spacing failed: %s", exc)
        return [], [], [], f"Error: {exc}"


def _fmt_metric(value) -> str:
    """Format a metric value for display."""
    if isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, dict):
        return str(value)
    return str(value)


# ---------------------------------------------------------------------------
# Run FloatingSectionWPS visualization
# ---------------------------------------------------------------------------

@callback(
    Output("wps-viz-plot", "children"),
    Output("wps-viz-status", "children"),
    Input("btn-wps-viz", "n_clicks"),
    State("wps-viz-uwi", "value"),
    State("wps-viz-orientation", "value"),
    State("wps-box-hw", "value"),
    State("wps-box-hh", "value"),
    State("wps-corr-hw", "value"),
    State("wps-corr-ea", "value"),
    State("wps-min-inside", "value"),
    State("wps-exclude-self", "value"),
    State("wps-viz-figsize", "value"),
    State("wps-viz-dpi", "value"),
    State("pipeline-result-store", "data"),
    prevent_initial_call=True,
)
def run_wps_viz_callback(
    n_clicks, uwi_ref, orientation, box_hw, box_hh, corr_hw, corr_ea,
    min_inside, exclude_self, figsize_key, dpi_str, pipeline_result,
):
    if not pipeline_result or not pipeline_result.get("cache_path"):
        return [], "No pipeline data. Run Calculate first."

    if not uwi_ref:
        return [], "Enter a UWI or click a well on the map."

    try:
        data = load_cached_pipeline(pipeline_result["cache_path"])
        lateral_df = data["lateral_df"]

        # Validate UWI
        available = set(lateral_df["uwi"].astype(str).unique())
        if str(uwi_ref) not in available:
            return [], f"UWI '{uwi_ref}' not found in lateral data."

        fs = _resolve_figsize("wps", figsize_key)
        dpi = _resolve_dpi(dpi_str)
        fig = run_wps_visualize(
            lateral_df,
            uwi_ref=str(uwi_ref),
            orientation=orientation or "cardinal",
            box_half_width_ft=float(box_hw or 2640),
            box_half_height_ft=float(box_hh or 2640),
            corridor_half_width_ft=float(corr_hw or 2640),
            corridor_extra_along_ft=float(corr_ea or 0),
            min_inside_ft=float(min_inside or 660),
            exclude_self=bool(exclude_self),
            figsize=fs,
        )
        img = fig_to_base64_img(fig, dpi=dpi)
        return [img], f"Done: {orientation} view for {uwi_ref}"

    except Exception as exc:
        logger.exception("WPS visualize failed: %s", exc)
        return [], f"Error: {exc}"


# ---------------------------------------------------------------------------
# Run AvgSpacing neighborhood plot
# ---------------------------------------------------------------------------

@callback(
    Output("avg-viz-plot", "children"),
    Output("avg-viz-status", "children"),
    Input("btn-avg-viz", "n_clicks"),
    State("avg-viz-uwi", "value"),
    State("avg-viz-mode", "value"),
    State("dbn-cutoff-ft", "value"),
    State("dbn-vertical-cutoff-ft", "value"),
    State("dbn-overlap-pct-min", "value"),
    State("dbn-axis-mode", "value"),
    State("avg-neighborhood-mode", "value"),
    State("avg-chain-sort-mode", "value"),
    State("avg-edge-pick", "value"),
    State("overrides-table", "data"),
    State("avg-viz-figsize", "value"),
    State("avg-viz-dpi", "value"),
    State("pipeline-result-store", "data"),
    prevent_initial_call=True,
)
def run_avg_viz_callback(
    n_clicks, well_uwi, plot_mode, cutoff_ft, vertical_cutoff_ft,
    overlap_pct_min, axis_mode, neighborhood_mode, chain_sort_mode,
    edge_pick, overrides_data, figsize_key, dpi_str, pipeline_result,
):
    if not pipeline_result or not pipeline_result.get("cache_path"):
        return [], "No pipeline data. Run Calculate first."

    if not well_uwi:
        return [], "Enter a UWI or click a well on the map."

    cutoff = _safe_float(cutoff_ft)
    if cutoff is None or cutoff <= 0:
        return [], "Cutoff (ft) is required and must be > 0."

    try:
        data = load_cached_pipeline(pipeline_result["cache_path"])
        df_ik = data.get("df_ik_pairs")
        lateral_df = data["lateral_df"]

        if df_ik is None or df_ik.empty:
            return [], "No IK pairs found in cache."

        # Validate UWI
        available = set(lateral_df["uwi"].astype(str).unique())
        if str(well_uwi) not in available:
            return [], f"UWI '{well_uwi}' not found in lateral data."

        overrides_df = _get_overrides_df(overrides_data)

        fs = _resolve_figsize("avg", figsize_key)
        dpi = _resolve_dpi(dpi_str)
        fig = run_avg_spacing_plot(
            df_ik, lateral_df,
            well_i=str(well_uwi),
            cutoff_ft=cutoff,
            vertical_cutoff_ft=_safe_float(vertical_cutoff_ft),
            overlap_pct_k_min=_safe_float(overlap_pct_min),
            overrides_df=overrides_df,
            axis_mode=axis_mode or "any",
            neighborhood_mode=neighborhood_mode or "chain",
            chain_sort_mode=chain_sort_mode or "pca",
            edge_pick=edge_pick or "min",
            plot_mode=plot_mode or "both",
            figsize=fs,
        )
        img = fig_to_base64_img(fig, dpi=dpi)
        return [img], f"Done: {plot_mode} neighborhood for {well_uwi}"

    except Exception as exc:
        logger.exception("AvgSpacing plot failed: %s", exc)
        return [], f"Error: {exc}"
