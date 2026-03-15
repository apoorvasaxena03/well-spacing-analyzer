"""
Step 5 — Explore
Main visualisation page.

Layout: map panel (left) + tabbed chart panel (right).
- Click a well on the map → gun barrel updates automatically
- All chart panels read from pipeline-result-store and selected-wells-store

GeoJSON IDs are static in the layout; their `data` property is updated by
load_map_layers(). This lets on_well_click reference them without needing
suppress_callback_exceptions for these specific components.
"""

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dcc, html
import dash_leaflet as dl
import plotly.graph_objects as go

from dashboard.pipeline import (
    compute_gun_barrel,
    load_cached_pipeline,
    load_cached_ik_heeltoe,
)
from dashboard.components.gun_barrel import build_gun_barrel_figure, empty_figure
from dashboard.components.map_panel import build_trajectory_geodataframe, build_bottomhole_geodataframe

dash.register_page(__name__, path="/explore", name="5 Explore", order=5)

_EMPTY_GEOJSON = {"type": "FeatureCollection", "features": []}

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

layout = dbc.Container(
    [
        html.H3("Step 5 — Explore Results", className="mb-1"),
        html.P(
            "Click any well on the map to populate the gun barrel and production charts.",
            className="text-muted mb-3",
        ),

        dbc.Row(
            [
                # ── Left: Map ───────────────────────────────────────────────
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader(
                                dbc.Row(
                                    [
                                        dbc.Col(html.Strong("Map"), width="auto"),
                                        dbc.Col(
                                            dbc.Select(
                                                id="map-color-by",
                                                options=[
                                                    {"label": "Color by Bench",    "value": "bench"},
                                                    {"label": "Color by Year",     "value": "year"},
                                                    {"label": "Color by Operator", "value": "operator"},
                                                ],
                                                value="bench",
                                                size="sm",
                                            ),
                                            width=4,
                                        ),
                                    ],
                                    align="center",
                                )
                            ),
                            dbc.CardBody(
                                dl.Map(
                                    [
                                        dl.TileLayer(
                                            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
                                            attribution="© OpenStreetMap contributors",
                                        ),
                                        # Static IDs — data updated by load_map_layers callback
                                        dl.GeoJSON(
                                            id="geojson-trajectories",
                                            data=_EMPTY_GEOJSON,
                                            options={
                                                "style": {
                                                    "color": "#3388ff",
                                                    "weight": 2,
                                                    "opacity": 0.8,
                                                }
                                            },
                                        ),
                                        dl.GeoJSON(
                                            id="geojson-bottomholes",
                                            data=_EMPTY_GEOJSON,
                                        ),
                                        dl.ScaleControl(position="bottomleft"),
                                    ],
                                    id="main-map",
                                    center=[31.5, -101.9],
                                    zoom=10,
                                    style={"height": "65vh"},
                                ),
                                className="p-0",
                            ),
                        ]
                    ),
                    md=5,
                ),

                # ── Right: Charts ───────────────────────────────────────────
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader(
                                dbc.Row(
                                    [
                                        dbc.Col(html.Strong("Charts"), width="auto"),
                                        dbc.Col(
                                            dbc.RadioItems(
                                                id="gb-xaxis-mode",
                                                options=[
                                                    {"label": "Centered",       "value": "sectionDist"},
                                                    {"label": "From reference", "value": "cum_dist"},
                                                ],
                                                value="sectionDist",
                                                inline=True,
                                                className="small",
                                            ),
                                            width="auto",
                                        ),
                                    ],
                                    align="center",
                                )
                            ),
                            dbc.CardBody(
                                dbc.Tabs(
                                    [
                                        dbc.Tab(
                                            dcc.Graph(
                                                id="gun-barrel-chart",
                                                style={"height": "60vh"},
                                                figure=empty_figure("Click a well on the map."),
                                            ),
                                            label="Gun Barrel",
                                            tab_id="tab-gb",
                                        ),
                                        dbc.Tab(
                                            dcc.Graph(
                                                id="cum-oil-chart",
                                                style={"height": "60vh"},
                                                figure=empty_figure("Click a well on the map."),
                                            ),
                                            label="Cum Oil",
                                            tab_id="tab-cum-oil",
                                        ),
                                        dbc.Tab(
                                            dcc.Graph(
                                                id="daily-oil-chart",
                                                style={"height": "60vh"},
                                                figure=empty_figure("Click a well on the map."),
                                            ),
                                            label="Daily Oil",
                                            tab_id="tab-daily-oil",
                                        ),
                                    ],
                                    active_tab="tab-gb",
                                ),
                            ),
                        ]
                    ),
                    md=7,
                ),
            ],
        ),
    ],
    fluid=True,
    className="py-4",
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@callback(
    Output("geojson-trajectories", "data"),
    Output("geojson-bottomholes", "data"),
    Output("main-map", "center"),
    Output("main-map", "zoom"),
    Input("pipeline-result-store", "data"),
    prevent_initial_call=True,
)
def load_map_layers(pipeline_result):
    """Populate map with wellbore sticks and bottom-hole markers."""
    if not pipeline_result or not pipeline_result.get("cache_path"):
        return _EMPTY_GEOJSON, _EMPTY_GEOJSON, [31.5, -101.9], 10

    data = load_cached_pipeline(pipeline_result["cache_path"])
    header_df = data["header_df"]
    lateral_df = data["lateral_df"]

    gdf_traj = build_trajectory_geodataframe(lateral_df, header_df)
    gdf_bh   = build_bottomhole_geodataframe(gdf_traj)

    traj_data = gdf_traj.__geo_interface__
    bh_data   = gdf_bh.__geo_interface__

    # Centre map — header uses surface_lat/surface_lon (canonical name)
    lat_col = "surface_lat" if "surface_lat" in header_df.columns else "latitude"
    lon_col = "surface_lon" if "surface_lon" in header_df.columns else "longitude"
    lat_centre = float(header_df[lat_col].dropna().mean()) if lat_col in header_df.columns else 31.5
    lon_centre = float(header_df[lon_col].dropna().mean()) if lon_col in header_df.columns else -101.9

    return traj_data, bh_data, [lat_centre, lon_centre], 11


@callback(
    Output("selected-wells-store", "data"),
    Input("geojson-trajectories", "n_clicks"),
    Input("geojson-bottomholes", "n_clicks"),
    State("geojson-trajectories", "clickData"),
    State("geojson-bottomholes", "clickData"),
    State("pipeline-result-store", "data"),
    prevent_initial_call=True,
)
def on_well_click(traj_clicks, bh_clicks, traj_click_data, bh_click_data, pipeline_result):
    """Capture clicked well UWI and build its spacing neighborhood."""
    click_data = traj_click_data or bh_click_data
    if not click_data or not pipeline_result:
        return dash.no_update

    props = click_data.get("properties") or {}
    clicked_uwi = props.get("uwi")
    if not clicked_uwi:
        return dash.no_update

    IK, _ = load_cached_ik_heeltoe(pipeline_result)
    if IK.empty:
        return {"clicked_uwi": clicked_uwi, "neighborhood_uwis": [clicked_uwi]}

    # All wells that share a spacing pair with the clicked well
    neighborhood = set(
        IK.loc[
            (IK["well_i"] == clicked_uwi) | (IK["well_k"] == clicked_uwi),
            ["well_i", "well_k"],
        ].values.flatten()
    )
    neighborhood.add(clicked_uwi)

    return {
        "clicked_uwi": clicked_uwi,
        "neighborhood_uwis": sorted(neighborhood),
    }


@callback(
    Output("gun-barrel-chart", "figure"),
    Input("selected-wells-store", "data"),
    Input("gb-xaxis-mode", "value"),
    State("pipeline-result-store", "data"),
    prevent_initial_call=True,
)
def update_gun_barrel(selected, x_col, pipeline_result):
    if not selected or not selected.get("neighborhood_uwis"):
        return empty_figure("Click a well on the map to populate the gun barrel.")

    uwis = selected["neighborhood_uwis"]
    IK, HeelToe = load_cached_ik_heeltoe(pipeline_result)

    if IK.empty:
        return empty_figure("Pipeline results not loaded.")

    # Filter to intra-neighborhood pairs only — never pass full IK to compute_gun_barrel
    IK_filtered      = IK[IK["well_i"].isin(uwis) & IK["well_k"].isin(uwis)]
    HeelToe_filtered = HeelToe[HeelToe["uwi"].isin(uwis)]

    if IK_filtered.empty:
        return empty_figure("No spacing pairs found for selected well.")

    GB = compute_gun_barrel(IK_filtered, HeelToe_filtered)
    return build_gun_barrel_figure(GB, IK_filtered, x_col=x_col)


@callback(
    Output("cum-oil-chart", "figure"),
    Input("selected-wells-store", "data"),
    State("pipeline-result-store", "data"),
    prevent_initial_call=True,
)
def update_cum_oil(selected, pipeline_result):
    if not selected or not pipeline_result:
        return empty_figure("Click a well on the map.")

    data = load_cached_pipeline(pipeline_result["cache_path"])
    prod = data.get("production_df")

    if prod is None or prod.empty:
        return empty_figure("No production data loaded.")

    uwis = selected.get("neighborhood_uwis", [])
    prod_sel = prod[prod["uwi"].isin(uwis)].copy()
    if prod_sel.empty:
        return empty_figure("No production data for selected wells.")

    prod_sel = prod_sel.sort_values(["uwi", "prod_date"])
    prod_sel["cum_oil"] = prod_sel.groupby("uwi")["oil"].cumsum()
    prod_sel["months"] = prod_sel.groupby("uwi")["prod_date"].transform(
        lambda s: (s - s.min()).dt.days / 30.44
    )

    fig = go.Figure()
    for uwi, grp in prod_sel.groupby("uwi"):
        fig.add_trace(go.Scatter(
            x=grp["months"], y=grp["cum_oil"],
            mode="lines", name=str(uwi),
        ))
    fig.update_layout(
        xaxis_title="Months since first production",
        yaxis_title="Cumulative Oil (BBL)",
        template="plotly_white",
        hovermode="x unified",
        margin=dict(t=30, b=50, l=60, r=20),
    )
    return fig


@callback(
    Output("daily-oil-chart", "figure"),
    Input("selected-wells-store", "data"),
    State("pipeline-result-store", "data"),
    prevent_initial_call=True,
)
def update_daily_oil(selected, pipeline_result):
    if not selected or not pipeline_result:
        return empty_figure("Click a well on the map.")

    data = load_cached_pipeline(pipeline_result["cache_path"])
    prod = data.get("production_df")

    if prod is None or prod.empty:
        return empty_figure("No production data loaded.")

    uwis = selected.get("neighborhood_uwis", [])
    prod_sel = prod[prod["uwi"].isin(uwis)].sort_values(["uwi", "prod_date"])

    fig = go.Figure()
    for uwi, grp in prod_sel.groupby("uwi"):
        oil_col = "daily_oil" if "daily_oil" in grp.columns else "oil"
        fig.add_trace(go.Scatter(
            x=grp["prod_date"], y=grp[oil_col],
            mode="lines", name=str(uwi),
        ))
    fig.update_layout(
        xaxis_title="Production Date",
        yaxis_title="Daily Oil (BOPD)",
        template="plotly_white",
        hovermode="x unified",
        margin=dict(t=30, b=50, l=60, r=20),
    )
    return fig
