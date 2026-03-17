"""
Step 6 — Export
Download pipeline results as CSV or Excel.
Offers multiple datasets: raw IK pairs, valid IK pairs, neighbor summary, header, production.
"""

import io

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, State, callback, dcc, html

from dashboard.pipeline import load_cached_pipeline

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
                                            {"label": "Valid IK pairs (reject-filtered)",  "value": "ik_valid"},
                                            {"label": "Neighbor summary (1 row/well)",     "value": "neighbor_summary"},
                                            {"label": "Well header (processed)",           "value": "header"},
                                            {"label": "Production data",                   "value": "production"},
                                        ],
                                        value=["ik_valid", "neighbor_summary", "header"],
                                        input_class_name="me-1",
                                        label_class_name="small",
                                    ),
                                ],
                                md=6,
                            ),
                            dbc.Col(
                                dbc.Button(
                                    [html.I(className="bi bi-download me-2"), "Download"],
                                    id="btn-export",
                                    color="primary",
                                    size="lg",
                                    className="mt-3",
                                ),
                                md=4,
                                className="d-flex align-items-end",
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
    Input("btn-export", "n_clicks"),
    State("pipeline-result-store", "data"),
    State("export-format", "value"),
    State("export-include", "value"),
    prevent_initial_call=True,
)
def do_export(n_clicks, pipeline_result, fmt, include):
    if not pipeline_result or not pipeline_result.get("cache_path"):
        return dash.no_update, "No pipeline results available. Run Step 4 first.", True

    try:
        data = load_cached_pipeline(pipeline_result["cache_path"])
    except Exception as exc:
        return dash.no_update, str(exc), True

    # Build dict of selected datasets
    sheets: dict[str, pd.DataFrame] = {}
    if "ik_valid" in include and data.get("df_ik_pairs") is not None:
        sheets["IK_Pairs_Valid"] = data["df_ik_pairs"]
    if "neighbor_summary" in include and data.get("df_spacing") is not None:
        sheets["Neighbor_Summary"] = data["df_spacing"]
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

    run_id = pipeline_result.get("run_id", "export")

    if fmt == "csv":
        # For CSV: export the first selected dataset (most important)
        # For multi-dataset CSV export, use Excel instead
        name = list(sheets.keys())[0]
        df = sheets[name]
        filename = f"{name}_{run_id}.csv"
        return dcc.send_data_frame(df.to_csv, filename, index=False), "", False

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
