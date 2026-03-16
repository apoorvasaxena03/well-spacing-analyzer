"""
Step 5 — Explore
Main visualisation page.

Layout: map panel (left) + tabbed chart panel (right).
- Click a well on the map → gun barrel updates automatically
- All chart panels read from pipeline-result-store and selected-wells-store
- Map layer styles (color-by, thickness, opacity) controlled via a
  collapsible settings panel; colors passed to JS via dl.GeoJSON hideout.

GeoJSON IDs are static in the layout; their `data` property is updated by
load_map_layers(). This lets on_well_click reference them without needing
suppress_callback_exceptions for these specific components.
"""

import dash
import pandas as pd
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dcc, html
import dash_leaflet as dl
import plotly.graph_objects as go
import plotly.express as px
from shapely.geometry import shape, Point

from dashboard.pipeline import (
    compute_gun_barrel,
    load_cached_pipeline,
    load_cached_ik_heeltoe,
)
from dashboard.components.gun_barrel import build_gun_barrel_figure, empty_figure
from dashboard.components.map_panel import build_trajectory_geodataframe, build_bottomhole_geodataframe

dash.register_page(__name__, path="/explore", name="5 Explore", order=5)

# ---------------------------------------------------------------------------
# Color palette — 20 distinct colors for categorical variables
# ---------------------------------------------------------------------------
_PALETTE = px.colors.qualitative.Alphabet


def _build_color_map(values: list[str]) -> dict[str, str]:
    """Map each unique value to a hex color from the palette."""
    unique = sorted(set(str(v) for v in values if v and str(v).strip()))
    return {v: _PALETTE[i % len(_PALETTE)] for i, v in enumerate(unique)}

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
                    [
                        # ── Style Controls (collapsible) ──
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    dbc.Row([
                                        dbc.Col(html.Strong("Map"), width="auto"),
                                        dbc.Col(
                                            dbc.Button(
                                                "Style Settings",
                                                id="btn-toggle-style",
                                                color="link",
                                                size="sm",
                                                className="p-0",
                                            ),
                                            width="auto",
                                            className="ms-auto",
                                        ),
                                    ], align="center"),
                                ),
                                dbc.Collapse(
                                    dbc.CardBody(
                                        [
                                            # -- Trajectories row --
                                            html.H6("Trajectories", className="mb-2", style={"fontSize": "0.82rem"}),
                                            dbc.Row([
                                                dbc.Col([
                                                    dbc.Label("Color by", className="small mb-0"),
                                                    dbc.Select(
                                                        id="traj-color-by",
                                                        options=[
                                                            {"label": "Bench",    "value": "bench"},
                                                            {"label": "Spud Year","value": "spud_year"},
                                                            {"label": "Operator", "value": "operator"},
                                                            {"label": "RSV Cat",  "value": "rsv_cat"},
                                                            {"label": "Uniform",  "value": "_uniform"},
                                                        ],
                                                        value="bench",
                                                        size="sm",
                                                    ),
                                                ], md=4),
                                                dbc.Col([
                                                    dbc.Label("Thickness", className="small mb-0"),
                                                    dcc.Slider(id="traj-weight", min=1, max=8, step=1, value=3,
                                                               marks={1: "1", 4: "4", 8: "8"}),
                                                ], md=4),
                                                dbc.Col([
                                                    dbc.Label("Opacity", className="small mb-0"),
                                                    dcc.Slider(id="traj-opacity", min=0.1, max=1.0, step=0.1, value=0.9,
                                                               marks={0.1: ".1", 0.5: ".5", 1.0: "1"}),
                                                ], md=4),
                                            ], className="mb-2"),

                                            html.Hr(className="my-2"),

                                            # -- Bottomholes row --
                                            html.H6("Bottom Holes", className="mb-2", style={"fontSize": "0.82rem"}),
                                            dbc.Row([
                                                dbc.Col([
                                                    dbc.Label("Color by", className="small mb-0"),
                                                    dbc.Select(
                                                        id="bh-color-by",
                                                        options=[
                                                            {"label": "Bench",    "value": "bench"},
                                                            {"label": "Spud Year","value": "spud_year"},
                                                            {"label": "Operator", "value": "operator"},
                                                            {"label": "RSV Cat",  "value": "rsv_cat"},
                                                            {"label": "Uniform",  "value": "_uniform"},
                                                        ],
                                                        value="bench",
                                                        size="sm",
                                                    ),
                                                ], md=4),
                                                dbc.Col([
                                                    dbc.Label("Size", className="small mb-0"),
                                                    dcc.Slider(id="bh-radius", min=2, max=12, step=1, value=4,
                                                               marks={2: "2", 6: "6", 12: "12"}),
                                                ], md=4),
                                                dbc.Col([
                                                    dbc.Label("Opacity", className="small mb-0"),
                                                    dcc.Slider(id="bh-opacity", min=0.1, max=1.0, step=0.1, value=0.8,
                                                               marks={0.1: ".1", 0.5: ".5", 1.0: "1"}),
                                                ], md=4),
                                            ]),
                                        ],
                                        className="py-2",
                                    ),
                                    id="style-collapse",
                                    is_open=False,
                                ),
                            ],
                            className="mb-1",
                        ),

                        # ── Map ──
                        dbc.Card(
                            dbc.CardBody(
                                dl.Map(
                                    [
                                        # ── Base layers (radio toggle) ──
                                        dl.LayersControl(
                                            [
                                                dl.BaseLayer(
                                                    dl.TileLayer(
                                                        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
                                                        attribution="© OpenStreetMap contributors",
                                                    ),
                                                    name="Street",
                                                    checked=True,
                                                ),
                                                dl.BaseLayer(
                                                    dl.TileLayer(
                                                        url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                                                        attribution="© Esri",
                                                    ),
                                                    name="Satellite",
                                                ),
                                                dl.BaseLayer(
                                                    dl.TileLayer(
                                                        url="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
                                                        attribution="© OpenTopoMap",
                                                    ),
                                                    name="Topo",
                                                ),
                                                # ── Overlay layers (checkbox toggle) ──
                                                dl.Overlay(
                                                    dl.GeoJSON(
                                                        id="geojson-trajectories",
                                                        data=_EMPTY_GEOJSON,
                                                        style="dashExtensions.trajectoryStyle",
                                                        hoverStyle={"weight": 6, "color": "#ff7800"},
                                                        hideout={
                                                            "colorMap": {},
                                                            "colorProp": "bench",
                                                            "weight": 3,
                                                            "opacity": 0.9,
                                                            "defaultColor": "#3388ff",
                                                        },
                                                    ),
                                                    name="Trajectories",
                                                    checked=True,
                                                ),
                                                dl.Overlay(
                                                    dl.GeoJSON(
                                                        id="geojson-bottomholes",
                                                        data=_EMPTY_GEOJSON,
                                                        pointToLayer="dashExtensions.bottomholePointToLayer",
                                                        hideout={
                                                            "colorMap": {},
                                                            "colorProp": "bench",
                                                            "radius": 4,
                                                            "opacity": 0.8,
                                                            "defaultColor": "#e74c3c",
                                                        },
                                                    ),
                                                    name="Bottom Holes",
                                                    checked=True,
                                                ),
                                            ],
                                            position="topright",
                                        ),
                                        # ── Tools ──
                                        dl.FullScreenControl(position="topleft"),
                                        dl.ScaleControl(position="bottomleft"),
                                        dl.MeasureControl(
                                            position="topleft",
                                            primaryLengthUnit="feet",
                                            secondaryLengthUnit="miles",
                                            primaryAreaUnit="acres",
                                            activeColor="#ff7800",
                                            completedColor="#00C853",
                                        ),
                                        # ── Draw tools (polygon / rectangle selection) ──
                                        dl.FeatureGroup(
                                            dl.EditControl(
                                                id="draw-control",
                                                position="topleft",
                                                draw={
                                                    "polyline": False,
                                                    "circle": False,
                                                    "circlemarker": False,
                                                    "marker": False,
                                                    "polygon": {"shapeOptions": {"color": "#ff7800"}},
                                                    "rectangle": {"shapeOptions": {"color": "#ff7800"}},
                                                },
                                                edit={"edit": False},
                                            ),
                                        ),
                                    ],
                                    id="main-map",
                                    center=[31.5, -101.9],
                                    zoom=10,
                                    style={"height": "60vh"},
                                ),
                                className="p-0",
                            ),
                        ),

                        # ── Legend ──
                        html.Div(id="map-legend", className="mt-1"),
                    ],
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

# -- Toggle style settings panel --
@callback(
    Output("style-collapse", "is_open"),
    Input("btn-toggle-style", "n_clicks"),
    State("style-collapse", "is_open"),
    prevent_initial_call=True,
)
def toggle_style_panel(n, is_open):
    return not is_open


# -- Trajectory style → hideout --
@callback(
    Output("geojson-trajectories", "hideout"),
    Output("map-legend", "children"),
    Input("traj-color-by", "value"),
    Input("traj-weight", "value"),
    Input("traj-opacity", "value"),
    Input("bh-color-by", "value"),
    State("pipeline-result-store", "data"),
    prevent_initial_call=False,
)
def update_trajectory_style(traj_color_by, weight, opacity, bh_color_by, pipeline_result):
    color_map = {}
    if pipeline_result and pipeline_result.get("cache_path"):
        data = load_cached_pipeline(pipeline_result["cache_path"])
        header_df = data["header_df"]
        if traj_color_by == "_uniform":
            color_map = {}
        elif traj_color_by == "spud_year" and "spud_date" in header_df.columns:
            try:
                years = pd.to_datetime(header_df["spud_date"], errors="coerce").dt.year.dropna().astype(int).astype(str).tolist()
                color_map = _build_color_map(years)
            except Exception:
                color_map = {}
        elif traj_color_by in header_df.columns:
            color_map = _build_color_map(header_df[traj_color_by].dropna().astype(str).tolist())

    hideout = {
        "colorMap": color_map,
        "colorProp": traj_color_by if traj_color_by != "_uniform" else "bench",
        "weight": weight or 3,
        "opacity": opacity or 0.9,
        "defaultColor": "#3388ff",
    }

    # Build legend from trajectory color map (primary layer)
    legend = _build_legend(color_map, traj_color_by)

    return hideout, legend


# -- Bottomhole style → hideout --
@callback(
    Output("geojson-bottomholes", "hideout"),
    Input("bh-color-by", "value"),
    Input("bh-radius", "value"),
    Input("bh-opacity", "value"),
    State("pipeline-result-store", "data"),
    prevent_initial_call=False,
)
def update_bottomhole_style(bh_color_by, radius, opacity, pipeline_result):
    color_map = {}
    if pipeline_result and pipeline_result.get("cache_path"):
        data = load_cached_pipeline(pipeline_result["cache_path"])
        header_df = data["header_df"]
        if bh_color_by == "_uniform":
            color_map = {}
        elif bh_color_by == "spud_year" and "spud_date" in header_df.columns:
            try:
                years = pd.to_datetime(header_df["spud_date"], errors="coerce").dt.year.dropna().astype(int).astype(str).tolist()
                color_map = _build_color_map(years)
            except Exception:
                color_map = {}
        elif bh_color_by in header_df.columns:
            color_map = _build_color_map(header_df[bh_color_by].dropna().astype(str).tolist())

    return {
        "colorMap": color_map,
        "colorProp": bh_color_by if bh_color_by != "_uniform" else "bench",
        "radius": radius or 4,
        "opacity": opacity or 0.8,
        "defaultColor": "#e74c3c",
    }


def _build_legend(color_map: dict, label: str) -> dbc.Card | None:
    """Build a compact color legend from a color map."""
    if not color_map:
        return None
    items = []
    for val, color in sorted(color_map.items()):
        items.append(
            html.Div(
                [
                    html.Span(
                        style={
                            "display": "inline-block",
                            "width": "12px",
                            "height": "12px",
                            "backgroundColor": color,
                            "borderRadius": "2px",
                            "marginRight": "4px",
                            "verticalAlign": "middle",
                        }
                    ),
                    html.Span(val, style={"fontSize": "0.72rem", "verticalAlign": "middle"}),
                ],
                style={"lineHeight": "1.4"},
            )
        )
    return dbc.Card(
        dbc.CardBody(
            [html.H6(label.replace("_", " ").title(), className="mb-1", style={"fontSize": "0.78rem"})]
            + items,
            className="py-1 px-2",
        ),
        style={"maxHeight": "200px", "overflowY": "auto"},
    )


@callback(
    Output("geojson-trajectories", "data"),
    Output("geojson-bottomholes", "data"),
    Output("main-map", "center"),
    Output("main-map", "zoom"),
    Input("pipeline-result-store", "data"),
    prevent_initial_call=False,
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


# ---------------------------------------------------------------------------
# Draw / Lasso selection — select wells within drawn polygon or rectangle
# ---------------------------------------------------------------------------

@callback(
    Output("selected-wells-store", "data", allow_duplicate=True),
    Input("draw-control", "geojson"),
    State("pipeline-result-store", "data"),
    prevent_initial_call=True,
)
def select_wells_by_shape(geojson, pipeline_result):
    """Select all wells whose bottom-hole falls within a drawn polygon/rectangle."""
    if not geojson or not pipeline_result:
        return dash.no_update

    features = geojson.get("features", [])
    if not features:
        return dash.no_update

    # Use the last drawn shape
    drawn = features[-1]
    drawn_geom = shape(drawn["geometry"])

    data = load_cached_pipeline(pipeline_result["cache_path"])
    header_df = data["header_df"]

    # Find the lat/lon columns
    lat_col = "surface_lat" if "surface_lat" in header_df.columns else "latitude"
    lon_col = "surface_lon" if "surface_lon" in header_df.columns else "longitude"

    if lat_col not in header_df.columns or lon_col not in header_df.columns:
        return dash.no_update

    # Find wells within the drawn shape
    selected_uwis = []
    for _, row in header_df.iterrows():
        try:
            pt = Point(float(row[lon_col]), float(row[lat_col]))
            if drawn_geom.contains(pt):
                selected_uwis.append(str(row["uwi"]))
        except (ValueError, TypeError):
            continue

    if not selected_uwis:
        return dash.no_update

    return {
        "clicked_uwi": selected_uwis[0],
        "neighborhood_uwis": sorted(selected_uwis),
    }
