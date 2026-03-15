"""
Step 6 — Export
Download filtered pipeline results as CSV or Excel.
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
                                            {"label": "CSV",   "value": "csv"},
                                            {"label": "Excel", "value": "xlsx"},
                                        ],
                                        value="csv",
                                        inline=True,
                                    ),
                                ],
                                md=4,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Include"),
                                    dbc.Checklist(
                                        id="export-include",
                                        options=[
                                            {"label": "Spacing pairs (IK)",  "value": "spacing"},
                                            {"label": "Well header",          "value": "header"},
                                            {"label": "Production data",      "value": "production"},
                                        ],
                                        value=["spacing", "header"],
                                    ),
                                ],
                                md=4,
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

    sheets = {}
    if "spacing" in include:
        sheets["spacing"] = data.get("df_spacing", pd.DataFrame())
    if "header" in include:
        sheets["header"] = data.get("header_df", pd.DataFrame())
    if "production" in include and data.get("production_df") is not None:
        sheets["production"] = data["production_df"]

    run_id = pipeline_result.get("run_id", "export")

    if fmt == "csv":
        # Export spacing pairs as primary CSV
        df = sheets.get("spacing", pd.DataFrame())
        return (
            dcc.send_data_frame(df.to_csv, f"spacing_{run_id}.csv", index=False),
            "", False,
        )
    else:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            for sheet_name, df in sheets.items():
                if not df.empty:
                    df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
        buf.seek(0)
        return (
            dcc.send_bytes(buf.read, f"spacing_{run_id}.xlsx"),
            "", False,
        )
