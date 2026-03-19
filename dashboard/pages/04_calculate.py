"""
Step 4 — Calculate
Runs the spacing pipeline as a background job (dash.long_callback + DiskcacheManager).
Shows live progress bar. On completion → stores cache path in pipeline-result-store
and displays a funnel report showing well counts at each pipeline step.
"""

import uuid

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dcc, html

from dashboard.pipeline import (
    PipelineStats,
    load_from_files,
    project_and_extract_laterals,
    run_spacing_calculation,
)

dash.register_page(__name__, path="/calculate", name="4 Calculate", order=4)

layout = dbc.Container(
    [
        html.H3("Step 4 — Calculate Spacing", className="mb-1"),
        html.P("The spacing engine will process all well pairs. This may take several minutes for large datasets.", className="text-muted mb-4"),

        dbc.Card(
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            dbc.Col(html.Div(id="calc-summary"), md=8),
                            dbc.Col(
                                dbc.Button(
                                    [html.I(className="bi bi-play-fill me-2"), "Calculate Spacing"],
                                    id="btn-calculate",
                                    color="success",
                                    size="lg",
                                    className="w-100",
                                ),
                                md=4,
                            ),
                        ],
                        align="center",
                    ),
                ]
            ),
            className="mb-4",
        ),

        # Progress area
        html.Div(
            [
                dbc.Progress(id="calc-progress", value=0, striped=True, animated=True, className="mb-2"),
                html.Div(id="calc-status-text", className="text-muted small"),
            ],
            id="calc-progress-area",
            style={"display": "none"},
        ),
        # Spinner shown immediately on click (before background job updates progress)
        dcc.Loading(
            id="calc-loading-spinner",
            type="default",
            children=html.Div(id="calc-loading-target"),
            fullscreen=False,
        ),

        dbc.Alert(id="calc-error", color="danger", is_open=False, dismissable=True, className="mt-3"),

        # Pipeline funnel report — shown after successful calculation
        html.Div(id="calc-funnel-report"),

        dbc.Alert(id="calc-success", color="success", is_open=False, className="mt-3"),

        html.Hr(),
        dbc.Row(
            [
                dbc.Col(dbc.Button("← Back", href="/configure", color="secondary", outline=True), width="auto"),
                dbc.Col(
                    dbc.Button(
                        "Explore Results →",
                        id="btn-to-explore",
                        color="primary",
                        disabled=True,
                        href="/explore",
                    ),
                    className="text-end",
                ),
            ],
            justify="between",
        ),

        dcc.Interval(id="calc-poll", interval=2000, disabled=True),
        dcc.Store(id="calc-job-store"),
    ],
    fluid=True,
    className="py-4",
)


def _build_funnel_report(steps: list[dict]) -> dbc.Card:
    """Build a visual funnel report card from pipeline stats."""
    rows = []
    for step in steps:
        label = step["label"]
        count = step["count"]
        delta = step["delta"]
        unit = step.get("unit", "wells")

        # Format delta badge
        if delta is not None and delta < 0:
            badge = dbc.Badge(f"{delta:,}", color="danger", className="ms-2")
        elif delta is not None and delta == 0:
            badge = dbc.Badge("no change", color="secondary", className="ms-2")
        else:
            badge = None

        row = html.Tr([
            html.Td(label, style={"fontSize": "0.85rem"}),
            html.Td(
                [
                    html.Strong(f"{count:,}"),
                    html.Small(f" {unit}", className="text-muted ms-1"),
                    badge,
                ],
                style={"fontSize": "0.85rem", "textAlign": "right"},
            ),
        ])
        rows.append(row)

    return dbc.Card(
        dbc.CardBody([
            html.H6("Pipeline Summary", className="mb-2"),
            html.P(
                "Well and pair counts at each processing step. "
                "Red badges show how many were removed by each filter.",
                className="text-muted small mb-2",
            ),
            dbc.Table(
                [html.Tbody(rows)],
                bordered=True,
                size="sm",
                hover=True,
                className="mb-0",
            ),
        ]),
        className="mt-3",
    )


# Show progress area immediately on button click (before background job starts)
dash.clientside_callback(
    """
    function(n_clicks) {
        if (!n_clicks) return [window.dash_clientside.no_update,
                               window.dash_clientside.no_update,
                               window.dash_clientside.no_update];
        return [{"display": "block"}, true, "Starting..."];
    }
    """,
    Output("calc-progress-area", "style", allow_duplicate=True),
    Output("btn-calculate", "disabled", allow_duplicate=True),
    Output("calc-status-text", "children", allow_duplicate=True),
    Input("btn-calculate", "n_clicks"),
    prevent_initial_call=True,
)


@callback(
    Output("calc-summary", "children"),
    Output("btn-calculate", "disabled", allow_duplicate=True),
    Output("calc-error", "is_open", allow_duplicate=True),
    Output("calc-error", "children", allow_duplicate=True),
    Output("calc-funnel-report", "children", allow_duplicate=True),
    Output("calc-success", "is_open", allow_duplicate=True),
    Output("calc-progress-area", "style", allow_duplicate=True),
    Input("upload-store", "data"),
    Input("config-store", "data"),
    Input("column-map-store", "data"),
    prevent_initial_call="initial_duplicate",
)
def show_summary(upload_store, config_store, col_map_store):
    # Clear all stale results/errors from previous runs
    clear = (False, "", None, False, {"display": "none"})

    if not upload_store or not config_store:
        return (
            dbc.Alert(
                "Complete Steps 1–3 before calculating. "
                "Go to Configure (Step 3) and click 'Save & Next →'.",
                color="warning",
            ),
            True,
            *clear,
        )

    # Check version: config must have been saved AFTER the current mapping was confirmed
    map_version = (col_map_store or {}).get("_version")
    cfg_version = config_store.get("_mapping_version")
    if cfg_version is None or (map_version and cfg_version != map_version):
        return (
            dbc.Alert(
                "Go to Configure (Step 3) and click 'Save & Next →' before calculating.",
                color="warning",
            ),
            True,
            *clear,
        )

    items = [
        f"Header: {upload_store.get('header_filename', 'n/a')}",
        f"Directional: {upload_store.get('directional_filename', 'n/a')}",
        f"Max distance: {config_store.get('max_distance_miles', 4.0)} miles",
        f"Cutoff: {config_store.get('cutoff_ft', 5280):,} ft",
        f"Batch size: {int(config_store.get('batch_size', 200_000)):,} pairs",
    ]
    if config_store.get("rsv_categories"):
        items.append(f"RSV filter: {', '.join(config_store['rsv_categories'])}")
    if config_store.get("bench_filter"):
        items.append(f"Benches: {', '.join(config_store['bench_filter'])}")

    return html.Ul([html.Li(i, className="small") for i in items]), False, *clear


@callback(
    Output("calc-progress-area", "style"),
    Output("calc-progress", "value"),
    Output("calc-status-text", "children"),
    Output("calc-error", "children"),
    Output("calc-error", "is_open"),
    Output("calc-funnel-report", "children"),
    Output("calc-success", "children"),
    Output("calc-success", "is_open"),
    Output("pipeline-result-store", "data"),
    Output("btn-to-explore", "disabled"),
    Output("btn-calculate", "disabled"),
    Input("btn-calculate", "n_clicks"),
    State("upload-store", "data"),
    State("column-map-store", "data"),
    State("config-store", "data"),
    prevent_initial_call=True,
    background=True,
    progress=[Output("calc-progress", "value"), Output("calc-status-text", "children")],
)
def run_calculation(set_progress, n_clicks, upload_store, col_map_store, config_store):
    """Background spacing calculation job."""
    no_show = {"display": "none"}

    # Hard guard: NEVER run without an explicit button click.
    # background=True callbacks with DiskcacheManager can spuriously re-trigger
    # on page load even with prevent_initial_call=True.
    if not n_clicks:
        return (no_show, 0, "", "", False, None, "", False, dash.no_update, True, False)

    if not upload_store or not col_map_store or not config_store:
        return (no_show, 0, "", "Missing data — complete Steps 1–3 first.", True,
                None, "", False, dash.no_update, True, False)

    stats = PipelineStats()

    set_progress((5, "Loading and pre-processing data..."))
    try:
        dtypes = col_map_store.get("_dtypes", {})
        data = load_from_files(
            header_path=upload_store["header_path"],
            directional_path=upload_store["directional_path"],
            column_map_header=col_map_store["header"],
            column_map_directional=col_map_store["directional"],
            production_path=upload_store.get("production_path"),
            column_map_production=col_map_store.get("production"),
            dtype_map_header=dtypes.get("header"),
            dtype_map_directional=dtypes.get("directional"),
            dtype_map_production=dtypes.get("production"),
            directional_source=config_store.get("directional_source"),
            rsv_categories=config_store.get("rsv_categories"),
            prod_cutoff_months=config_store.get("prod_cutoff_months", 6),
            duc_age_years=config_store.get("duc_age_years", 3),
            permit_window_years=config_store.get("permit_window_years", 2),
            stats=stats,
        )
    except Exception as exc:
        return (no_show, 0, "", str(exc), True, None, "", False, dash.no_update, True, False)

    set_progress((20, "Projecting coordinates to UTM and extracting laterals..."))
    try:
        header_df, lateral_df, crs_used = project_and_extract_laterals(
            header_df=data["header_df"],
            directional_df=data["directional_df"],
            crs_to=config_store.get("utm_zone"),
            directional_source=data["directional_source"],
            stats=stats,
        )
    except Exception as exc:
        return (no_show, 0, "", str(exc), True, None, "", False, dash.no_update, True, False)

    set_progress((35, "Running spacing calculation (this is the slow step)..."))
    run_id = uuid.uuid4().hex[:8]
    try:
        cache_path = run_spacing_calculation(
            header_df=header_df,
            lateral_df=lateral_df,
            max_distance_miles=config_store.get("max_distance_miles", 4.0),
            cutoff_ft=config_store.get("cutoff_ft", 5280),
            batch_size=config_store.get("batch_size", 200_000),
            run_id=run_id,
            stats=stats,
            production_df=data.get("production_df"),
            directional_df=data.get("directional_df"),
        )
    except Exception as exc:
        return (no_show, 0, "", str(exc), True, None, "", False, dash.no_update, True, False)

    set_progress((100, "Done."))
    result_store = {"cache_path": cache_path, "crs_used": crs_used, "run_id": run_id}
    funnel = _build_funnel_report(stats.to_list())
    success_msg = f"Calculation complete. Run ID: {run_id}."
    return (
        {"display": "block"}, 100, "Done.", "", False,
        funnel,
        success_msg, True,
        result_store, False, True,
    )
