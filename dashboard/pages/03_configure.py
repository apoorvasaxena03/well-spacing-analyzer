"""
Step 3 — Configure
Set spacing engine parameters before calculation.
UTM zone is auto-detected; all params stored in config-store.
"""

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dcc, html

dash.register_page(__name__, path="/configure", name="3 Configure", order=3)

layout = dbc.Container(
    [
        html.H3("Step 3 — Configure", className="mb-1"),
        html.P("Adjust spacing engine parameters. Defaults work for most Midland Basin datasets.", className="text-muted mb-4"),

        dbc.Card(
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            # UTM Zone
                            dbc.Col(
                                [
                                    dbc.Label("UTM Zone (auto-detected)"),
                                    dbc.Input(
                                        id="cfg-utm-zone",
                                        placeholder="e.g. EPSG:32613",
                                        value="",
                                        disabled=True,
                                    ),
                                    dbc.FormText("Leave blank to auto-detect from data centroid."),
                                    dbc.Checklist(
                                        id="cfg-utm-override",
                                        options=[{"label": "Override UTM zone", "value": "override"}],
                                        value=[],
                                        className="mt-1",
                                    ),
                                ],
                                md=4,
                            ),
                            # Max distance
                            dbc.Col(
                                [
                                    dbc.Label("Max search radius (miles)"),
                                    dbc.Input(
                                        id="cfg-max-distance",
                                        type="number",
                                        value=4.0,
                                        min=0.1,
                                        max=20.0,
                                        step=0.5,
                                    ),
                                    dbc.FormText("Wells farther apart than this are excluded from pairing."),
                                ],
                                md=4,
                            ),
                            # Cutoff ft
                            dbc.Col(
                                [
                                    dbc.Label("Spacing cutoff (ft)"),
                                    dbc.Input(
                                        id="cfg-cutoff-ft",
                                        type="number",
                                        value=5280,
                                        min=100,
                                        max=50000,
                                        step=100,
                                    ),
                                    dbc.FormText("Pairs with horizontal spacing > cutoff are excluded."),
                                ],
                                md=4,
                            ),
                        ],
                        className="mb-3",
                    ),
                    dbc.Row(
                        [
                            # Batch size
                            dbc.Col(
                                [
                                    dbc.Label("Batch size (pairs)"),
                                    dbc.Select(
                                        id="cfg-batch-size",
                                        options=[
                                            {"label": "50,000  (low memory)",    "value": 50_000},
                                            {"label": "200,000 (default)",        "value": 200_000},
                                            {"label": "500,000 (high memory)",    "value": 500_000},
                                        ],
                                        value=200_000,
                                    ),
                                    dbc.FormText("Reduce if you see memory errors on large datasets."),
                                ],
                                md=4,
                            ),
                            # Bench filter
                            dbc.Col(
                                [
                                    dbc.Label("Bench filter (optional)"),
                                    dbc.Input(
                                        id="cfg-bench-filter",
                                        placeholder="e.g. WOLFCAMP A, SPRABERRY",
                                        value="",
                                    ),
                                    dbc.FormText("Comma-separated list. Leave blank to include all benches."),
                                ],
                                md=8,
                            ),
                        ],
                    ),
                ]
            )
        ),

        html.Hr(),
        dbc.Row(
            [
                dbc.Col(dbc.Button("← Back", href="/column-map", color="secondary", outline=True), width="auto"),
                dbc.Col(
                    dbc.Button("Save & Next →", id="btn-save-config", color="primary"),
                    className="text-end",
                ),
            ],
            justify="between",
        ),
        dcc.Location(id="cfg-redirect", refresh=True),
    ],
    fluid=True,
    className="py-4",
)


@callback(
    Output("cfg-utm-zone", "disabled"),
    Input("cfg-utm-override", "value"),
    prevent_initial_call=False,
)
def toggle_utm_input(override):
    return "override" not in (override or [])


@callback(
    Output("config-store", "data"),
    Output("cfg-redirect", "href"),
    Input("btn-save-config", "n_clicks"),
    State("cfg-utm-zone", "value"),
    State("cfg-utm-override", "value"),
    State("cfg-max-distance", "value"),
    State("cfg-cutoff-ft", "value"),
    State("cfg-batch-size", "value"),
    State("cfg-bench-filter", "value"),
    prevent_initial_call=True,
)
def save_config(n_clicks, utm_zone, utm_override, max_dist, cutoff_ft, batch_size, bench_filter):
    cfg = {
        "utm_zone": utm_zone if "override" in (utm_override or []) else None,
        "max_distance_miles": float(max_dist or 4.0),
        "cutoff_ft": float(cutoff_ft or 5280),
        "batch_size": int(batch_size or 200_000),
        "bench_filter": [b.strip() for b in (bench_filter or "").split(",") if b.strip()],
    }
    return cfg, "/calculate"
