"""
Step 6 — Export
Download pipeline results as CSV or Excel.
Offers multiple datasets: raw IK pairs, valid IK pairs, neighbor summary, header, production.
"""

import io
import json
import logging
import zipfile

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, State, callback, dcc, html

from dashboard.pipeline import load_cached_pipeline, load_ui_state

logger = logging.getLogger("dashboard")

dash.register_page(__name__, path="/export", name="6 Export", order=6)

layout = dbc.Container(
    [
        html.H3("Step 6 — Export Results", className="mb-1"),
        html.P("Download the spacing results for use in Spotfire, Excel, or other tools.", className="text-muted mb-4"),

        dbc.Card(
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label("Export format"),
                                    dbc.RadioItems(
                                        id="export-format",
                                        options=[
                                            {"label": "CSV (one file per dataset)",  "value": "csv"},
                                            {"label": "Excel (all in one workbook)", "value": "xlsx"},
                                        ],
                                        value="xlsx",
                                        inline=True,
                                    ),
                                ],
                                md=6,
                            ),
                        ],
                        className="mb-3",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label("Datasets to include"),
                                    dbc.Checklist(
                                        id="export-include",
                                        options=[
                                            {"label": "Valid IK pairs (reject-filtered)",       "value": "ik_valid"},
                                            {"label": "IK pairs — all, including rejected",     "value": "ik_raw"},
                                            {"label": "Neighbor summary (1 row/well)",          "value": "neighbor_summary"},
                                            {"label": "Average spacing (1 row/well)",           "value": "avg_spacing"},
                                            {"label": "WPS floating section (1 row/well)",      "value": "wps"},
                                            {"label": "Well header (processed)",                "value": "header"},
                                            {"label": "Production data",                        "value": "production"},
                                        ],
                                        value=["ik_valid", "neighbor_summary", "header"],
                                        input_class_name="me-1",
                                        label_class_name="small",
                                    ),
                                ],
                                md=6,
                            ),
                            dbc.Col(
                                [
                                    dcc.Loading(
                                        id="export-loading",
                                        type="circle",
                                        children=dbc.Button(
                                            [html.I(className="bi bi-download me-2"), "Download"],
                                            id="btn-export",
                                            color="primary",
                                            size="lg",
                                            className="mt-3",
                                        ),
                                    ),
                                    html.Div(id="export-status", className="small text-muted mt-1"),
                                ],
                                md=4,
                                className="d-flex flex-column align-items-start",
                            ),
                        ]
                    ),
                    html.Hr(className="my-3"),
                    html.Div(id="export-preview", className="small text-muted"),
                ]
            )
        ),

        dbc.Alert(id="export-error", color="danger", is_open=False, dismissable=True, className="mt-3"),
        dcc.Download(id="export-download"),

        # ── Session Package Export ──
        dbc.Card(
            dbc.CardBody(
                [
                    html.H6("Session Package", className="mb-2"),
                    html.P(
                        "Export all results as a portable ZIP of Parquet files. "
                        "This package can be re-imported on the Upload page to skip "
                        "Steps 2–4 and go straight to Explore.",
                        className="text-muted small mb-3",
                    ),
                    dcc.Loading(
                        id="session-export-loading",
                        type="circle",
                        children=dbc.Button(
                            [html.I(className="bi bi-box-arrow-up me-2"), "Export Session Package (.zip)"],
                            id="btn-export-session",
                            color="success",
                            size="lg",
                        ),
                    ),
                ]
            ),
            className="mt-3",
        ),
        dcc.Download(id="session-download"),

        html.Hr(),
        dbc.Row(
            dbc.Col(dbc.Button("← Back to Explore", href="/explore", color="secondary", outline=True), width="auto")
        ),
    ],
    fluid=True,
    className="py-4",
)


@callback(
    Output("export-preview", "children"),
    Input("pipeline-result-store", "data"),
    prevent_initial_call=False,
)
def show_preview(pipeline_result):
    """Show row counts for each available dataset."""
    if not pipeline_result or not pipeline_result.get("cache_path"):
        return "No pipeline results available. Run Step 4 first."

    try:
        data = load_cached_pipeline(pipeline_result["cache_path"])
    except Exception:
        return "Could not load cached results."

    lines = []
    ik = data.get("df_ik_pairs")
    if ik is not None:
        lines.append(f"Valid IK pairs: {len(ik):,} rows × {len(ik.columns)} columns")
    ik_raw = data.get("df_ik_pairs_raw")
    if ik_raw is not None:
        lines.append(f"IK pairs (all, incl. rejected): {len(ik_raw):,} rows × {len(ik_raw.columns)} columns")
    summary = data.get("df_spacing")
    if summary is not None:
        lines.append(f"Neighbor summary: {len(summary):,} rows × {len(summary.columns)} columns")
    header = data.get("header_df")
    if header is not None:
        lines.append(f"Header: {len(header):,} wells × {len(header.columns)} columns")
    prod = data.get("production_df")
    if prod is not None:
        lines.append(f"Production: {len(prod):,} rows × {len(prod.columns)} columns")
    else:
        lines.append("Production: not loaded")

    return html.Ul([html.Li(l) for l in lines])


@callback(
    Output("export-download", "data"),
    Output("export-error", "children"),
    Output("export-error", "is_open"),
    Output("export-status", "children"),
    Input("btn-export", "n_clicks"),
    State("pipeline-result-store", "data"),
    State("export-format", "value"),
    State("export-include", "value"),
    prevent_initial_call=True,
)
def do_export(n_clicks, pipeline_result, fmt, include):
    if not pipeline_result or not pipeline_result.get("cache_path"):
        return dash.no_update, "No pipeline results available. Run Step 4 first.", True, ""

    try:
        result = _do_export_inner(pipeline_result, fmt, include)
        total_rows = sum(len(df) for df in _last_export_sheets.values()) if _last_export_sheets else 0
        status = f"Downloaded ({total_rows:,} rows total)."
        return result[0], result[1], result[2], status
    except Exception as exc:
        logger.exception("Export failed: %s", exc)
        return dash.no_update, str(exc), True, ""


# Track last export for status message
_last_export_sheets: dict[str, pd.DataFrame] = {}


def _do_export_inner(pipeline_result, fmt, include):
    global _last_export_sheets
    data = load_cached_pipeline(pipeline_result["cache_path"])

    # Build dict of selected datasets
    sheets: dict[str, pd.DataFrame] = {}
    if "ik_valid" in include and data.get("df_ik_pairs") is not None:
        sheets["IK_Pairs_Valid"] = data["df_ik_pairs"]
    if "ik_raw" in include and data.get("df_ik_pairs_raw") is not None:
        sheets["IK_Pairs_All"] = data["df_ik_pairs_raw"]
    if "neighbor_summary" in include and data.get("df_spacing") is not None:
        sheets["Neighbor_Summary"] = data["df_spacing"]
    if "avg_spacing" in include and data.get("df_avg_spacing") is not None:
        sheets["Avg_Spacing"] = data["df_avg_spacing"]
    if "wps" in include and data.get("df_wps") is not None:
        sheets["WPS_FloatingSection"] = data["df_wps"]
    if "header" in include and data.get("header_df") is not None:
        sheets["Header"] = data["header_df"]
    if "production" in include and data.get("production_df") is not None:
        sheets["Production"] = data["production_df"]

    # Strip timezone info from datetime columns — Excel doesn't support tz-aware datetimes
    for name, df in sheets.items():
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                sheets[name][col] = df[col].dt.tz_localize(None) if hasattr(df[col].dt, 'tz') and df[col].dt.tz else df[col]

    if not sheets:
        return dash.no_update, "No datasets selected.", True

    _last_export_sheets = sheets
    run_id = pipeline_result.get("run_id", "export")

    if fmt == "csv":
        if len(sheets) == 1:
            name, df = next(iter(sheets.items()))
            filename = f"{name}_{run_id}.csv"
            return dcc.send_data_frame(df.to_csv, filename, index=False), "", False
        # Multiple datasets selected → zip one CSV per dataset rather than
        # silently dropping all but the first.
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for sheet_name, df in sheets.items():
                zf.writestr(f"{sheet_name}.csv", df.to_csv(index=False))
        buf.seek(0)
        filename = f"spacing_results_csv_{run_id}.zip"
        return dcc.send_bytes(buf.getvalue(), filename), "", False

    else:
        # Excel: all selected datasets as separate sheets
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            for sheet_name, df in sheets.items():
                if not df.empty:
                    df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
        buf.seek(0)
        filename = f"spacing_results_{run_id}.xlsx"
        return dcc.send_bytes(buf.getvalue(), filename), "", False


# ---------------------------------------------------------------------------
# Session package export (ZIP of Parquet files)
# ---------------------------------------------------------------------------

@callback(
    Output("session-download", "data"),
    Output("export-error", "children", allow_duplicate=True),
    Output("export-error", "is_open", allow_duplicate=True),
    Input("btn-export-session", "n_clicks"),
    State("pipeline-result-store", "data"),
    State("config-store", "data"),
    prevent_initial_call=True,
)
def export_session_package(n_clicks, pipeline_result, config_store):
    if not pipeline_result or not pipeline_result.get("cache_path"):
        return dash.no_update, "No pipeline results available.", True

    try:
        data = load_cached_pipeline(pipeline_result["cache_path"])
    except Exception as exc:
        logger.exception("Session export failed: %s", exc)
        return dash.no_update, str(exc), True

    run_id = pipeline_result.get("run_id", "session")
    buf = io.BytesIO()

    try:
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            # Metadata
            metadata = {
                "run_id": run_id,
                "crs_used": pipeline_result.get("crs_used"),
                "datasets": [],
            }

            # Write each DataFrame as parquet
            dataset_keys = [
                ("df_ik_pairs", "ik_pairs"),
                ("df_ik_pairs_raw", "ik_pairs_raw"),
                ("df_spacing", "neighbor_summary"),
                ("df_avg_spacing", "avg_spacing"),
                ("df_wps", "wps"),
                ("header_df", "header"),
                ("lateral_df", "lateral"),
                ("directional_df", "directional"),
                ("production_df", "production"),
            ]
            for cache_key, file_name in dataset_keys:
                df = data.get(cache_key)
                if df is not None and not df.empty:
                    pq_buf = io.BytesIO()
                    df.to_parquet(pq_buf, index=False, engine="pyarrow")
                    zf.writestr(f"{file_name}.parquet", pq_buf.getvalue())
                    metadata["datasets"].append({
                        "name": file_name,
                        "rows": len(df),
                        "columns": len(df.columns),
                    })

            # Write stats if available
            stats = data.get("stats")
            if stats:
                zf.writestr("stats.json", json.dumps(stats, indent=2))

            zf.writestr("metadata.json", json.dumps(metadata, indent=2))

            # Save UI state from companion JSON (auto-saved by Explore page)
            ui_state = load_ui_state(pipeline_result["cache_path"])
            if config_store:
                ui_state["config_store"] = config_store
            zf.writestr("ui_state.json", json.dumps(ui_state, indent=2))

        buf.seek(0)
        filename = f"spacing_session_{run_id}.zip"
        logger.info("Session package exported: %s (%.1f MB)", filename, buf.tell() / 1024 / 1024)
        return dcc.send_bytes(buf.getvalue(), filename), "", False

    except Exception as exc:
        logger.exception("Session export failed: %s", exc)
        return dash.no_update, str(exc), True
