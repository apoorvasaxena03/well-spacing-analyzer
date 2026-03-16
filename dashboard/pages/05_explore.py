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
        dbc.Row([
            dbc.Col(
                html.P(
                    "Click any well on the map to populate the gun barrel and production charts.",
                    className="text-muted mb-0",
                ),
            ),
            dbc.Col(
                [
                    dbc.Button(
                        "Clear Selection",
                        id="btn-clear-selection",
                        color="outline-danger",
                        size="sm",
                        className="me-2",
                    ),
                    dbc.Button(
                        "Filters",
                        id="btn-toggle-filters",
                        color="outline-primary",
                        size="sm",
                    ),
                ],
                width="auto",
                className="d-flex",
            ),
        ], align="center", className="mb-2"),

        # Store for active filter state (list of UWIs that pass filters)
        dcc.Store(id="filter-uwis-store"),

        dbc.Row(
            [
                # ── Filter Panel (collapsible sidebar) ──
                dbc.Col(
                    dbc.Collapse(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H6("Filters", className="mb-2"),
                                    html.P(
                                        "Filter wells across map & charts. "
                                        "Uncheck items to exclude them.",
                                        className="text-muted small mb-2",
                                    ),

                                    # -- Well search --
                                    dbc.Label("Search well", className="small mb-0 fw-bold"),
                                    dbc.Input(
                                        id="filter-search",
                                        placeholder="UWI or well name...",
                                        size="sm",
                                        debounce=True,
                                        className="mb-2",
                                    ),

                                    # -- Bench --
                                    dbc.Label("Bench", className="small mb-0 fw-bold"),
                                    dcc.Dropdown(
                                        id="filter-bench",
                                        multi=True,
                                        placeholder="All benches",
                                        style={"fontSize": "0.8rem"},
                                        className="mb-2",
                                    ),

                                    # -- Operator --
                                    dbc.Label("Operator", className="small mb-0 fw-bold"),
                                    dcc.Dropdown(
                                        id="filter-operator",
                                        multi=True,
                                        placeholder="All operators",
                                        style={"fontSize": "0.8rem"},
                                        className="mb-2",
                                    ),

                                    # -- RSV Category --
                                    dbc.Label("RSV Category", className="small mb-0 fw-bold"),
                                    dbc.Checklist(
                                        id="filter-rsv-cat",
                                        options=[],
                                        value=[],
                                        inline=False,
                                        input_class_name="me-1",
                                        label_class_name="small",
                                        className="mb-2",
                                    ),

                                    # -- Well Status --
                                    dbc.Label("Well Status", className="small mb-0 fw-bold"),
                                    dbc.Checklist(
                                        id="filter-well-status",
                                        options=[],
                                        value=[],
                                        inline=False,
                                        input_class_name="me-1",
                                        label_class_name="small",
                                        className="mb-2",
                                    ),

                                    # -- Spud Year Range --
                                    dbc.Label("Spud Year", className="small mb-0 fw-bold"),
                                    dcc.RangeSlider(
                                        id="filter-spud-year",
                                        min=2000, max=2026,
                                        step=1,
                                        value=[2000, 2026],
                                        marks={2000: "2000", 2010: "2010", 2020: "2020", 2026: "2026"},
                                        tooltip={"placement": "bottom"},
                                        className="mb-2",
                                    ),

                                    # -- Lateral Length Range --
                                    dbc.Label("Lateral Length (ft)", className="small mb-0 fw-bold"),
                                    dcc.RangeSlider(
                                        id="filter-lateral-length",
                                        min=0, max=15000,
                                        step=500,
                                        value=[0, 15000],
                                        marks={0: "0", 5000: "5k", 10000: "10k", 15000: "15k"},
                                        tooltip={"placement": "bottom"},
                                        className="mb-2",
                                    ),

                                    html.Hr(className="my-2"),
                                    html.Div(id="filter-count", className="text-muted small"),
                                ],
                                style={"maxHeight": "75vh", "overflowY": "auto"},
                                className="py-2 px-2",
                            ),
                        ),
                        id="filter-collapse",
                        is_open=False,
                    ),
                    md=2,
                    id="filter-col",
                    style={"display": "none"},
                ),

                # ── Map ───────────────────────────────────────────────
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
                                                        options={
                                                            "style": {"weight": 3, "opacity": 0.9},
                                                        },
                                                        hoverStyle={"weight": 6, "color": "#ff7800"},
                                                    ),
                                                    name="Trajectories",
                                                    checked=True,
                                                ),
                                                dl.Overlay(
                                                    dl.GeoJSON(
                                                        id="geojson-bottomholes",
                                                        data=_EMPTY_GEOJSON,
                                                        pointToLayer={"variable": "dashExtensions.default.ptl0"},
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
                                        # ── Draw tools (line / polygon / rectangle / circle selection) ──
                                        dl.FeatureGroup(
                                            dl.EditControl(
                                                id="draw-control",
                                                position="topleft",
                                                draw={
                                                    "polyline": {"shapeOptions": {"color": "#ff7800", "weight": 3}},
                                                    "circle": {"shapeOptions": {"color": "#ff7800"}},
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
                                    scrollWheelZoom=True,
                                    wheelDebounceTime=80,
                                    wheelPxPerZoomLevel=120,
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
                                [
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
                                    ),
                                    # GB controls row
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                dbc.Select(
                                                    id="gb-color-by",
                                                    options=[
                                                        {"label": "Bench",    "value": "bench"},
                                                        {"label": "Operator", "value": "operator"},
                                                        {"label": "Year",     "value": "spud_year"},
                                                    ],
                                                    value="bench",
                                                    size="sm",
                                                ),
                                                width=3,
                                            ),
                                            dbc.Col(
                                                dbc.Checklist(
                                                    id="gb-toggles",
                                                    options=[
                                                        {"label": "Lines", "value": "lines"},
                                                        {"label": "Labels", "value": "labels"},
                                                    ],
                                                    value=["lines", "labels"],
                                                    inline=True,
                                                    input_class_name="me-1",
                                                    label_class_name="small me-3",
                                                ),
                                                width="auto",
                                            ),
                                        ],
                                        align="center",
                                        className="mt-1",
                                    ),
                                ]
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
    """Populate map with wellbore trajectory lines and bottomhole circle markers."""
    import logging
    logger = logging.getLogger("dashboard")

    if not pipeline_result or not pipeline_result.get("cache_path"):
        logger.warning("load_map_layers: no pipeline result")
        return _EMPTY_GEOJSON, _EMPTY_GEOJSON, [31.5, -101.9], 10

    logger.info("load_map_layers: loading from %s", pipeline_result["cache_path"])
    data = load_cached_pipeline(pipeline_result["cache_path"])
    header_df = data["header_df"]
    lateral_df = data["lateral_df"]

    gdf_traj = build_trajectory_geodataframe(lateral_df, header_df)
    gdf_bh = build_bottomhole_geodataframe(gdf_traj)

    traj_data = gdf_traj.__geo_interface__
    bh_data = gdf_bh.__geo_interface__
    logger.info("load_map_layers: %d trajectories, %d bottomholes", len(gdf_traj), len(gdf_bh))

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
    no_change = dash.no_update
    triggered = dash.ctx.triggered_id

    import logging
    _log = logging.getLogger("dashboard")

    click_data = traj_click_data or bh_click_data
    if not click_data or not pipeline_result:
        return no_change

    props = click_data.get("properties") or {}
    clicked_uwi = props.get("uwi")
    _log.info("on_well_click: triggered=%s uwi=%s type=%s", triggered, clicked_uwi, type(clicked_uwi).__name__)
    if not clicked_uwi:
        return no_change
    # Ensure string
    clicked_uwi = str(clicked_uwi)

    IK, _ = load_cached_ik_heeltoe(pipeline_result)
    if IK.empty:
        return {"clicked_uwi": clicked_uwi, "neighborhood_uwis": [clicked_uwi]}

    # All wells that share a spacing pair with the clicked well
    # Raw IK pairs have well_i/well_k; enriched summary has well_i + uwi_same/near columns
    neighborhood = set()
    if "well_k" in IK.columns:
        neighborhood = set(
            IK.loc[
                (IK["well_i"] == clicked_uwi) | (IK["well_k"] == clicked_uwi),
                ["well_i", "well_k"],
            ].values.flatten()
        )
    elif "well_i" in IK.columns:
        # Enriched summary: collect neighbor UWIs from uwi_same_*/uwi_near_* columns
        row = IK[IK["well_i"] == clicked_uwi]
        if not row.empty:
            for col in IK.columns:
                if col.startswith("uwi_same_") or col.startswith("uwi_near_"):
                    vals = row[col].dropna().tolist()
                    neighborhood.update(str(v) for v in vals if v)
    neighborhood.add(clicked_uwi)
    _log.info("on_well_click: uwi=%s neighborhood=%d wells", clicked_uwi, len(neighborhood))

    return {"clicked_uwi": clicked_uwi, "neighborhood_uwis": sorted(neighborhood)}


@callback(
    Output("gun-barrel-chart", "figure"),
    Input("selected-wells-store", "data"),
    Input("gb-xaxis-mode", "value"),
    Input("gb-color-by", "value"),
    Input("gb-toggles", "value"),
    State("pipeline-result-store", "data"),
    prevent_initial_call=True,
)
def update_gun_barrel(selected, x_col, color_by, toggles, pipeline_result):
    if not selected or not selected.get("neighborhood_uwis"):
        return empty_figure("Click a well on the map to populate the gun barrel.")

    uwis = selected["neighborhood_uwis"]
    IK, HeelToe = load_cached_ik_heeltoe(pipeline_result)

    if IK.empty:
        return empty_figure("Pipeline results not loaded.")

    # Filter to intra-neighborhood pairs only — never pass full IK to compute_gun_barrel
    IK_filtered      = IK[IK["well_i"].isin(uwis) & IK["well_k"].isin(uwis)].copy()
    HeelToe_filtered = HeelToe[HeelToe["uwi"].isin(uwis)]

    if IK_filtered.empty:
        return empty_figure("No spacing pairs found for selected well.")

    # elevation = tvd * -1 (depth below surface → elevation above datum)
    if "tvd_i" in IK_filtered.columns and "elevation_i" not in IK_filtered.columns:
        IK_filtered["elevation_i"] = IK_filtered["tvd_i"] * -1
    if "tvd_k" in IK_filtered.columns and "elevation_k" not in IK_filtered.columns:
        IK_filtered["elevation_k"] = IK_filtered["tvd_k"] * -1

    GB = compute_gun_barrel(IK_filtered, HeelToe_filtered)
    show_lines = "lines" in (toggles or [])
    show_labels = "labels" in (toggles or [])
    return build_gun_barrel_figure(
        GB, IK_filtered,
        x_col=x_col,
        color_by=color_by or "bench",
        show_lines=show_lines,
        show_labels=show_labels,
    )


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
    """Select all wells within a drawn shape (polygon, rectangle, circle, or line buffer).

    For polylines: buffers the line by ~0.5 miles (~0.008 degrees) and selects
    all wells whose surface location falls within the buffer corridor.
    For circles: uses the radius from the drawn feature properties.
    For polygons/rectangles: standard containment check.
    """
    if not geojson or not pipeline_result:
        return dash.no_update

    features = geojson.get("features", [])
    if not features:
        return dash.no_update

    # Use the last drawn shape
    drawn = features[-1]
    drawn_geom = shape(drawn["geometry"])

    # For polylines: buffer to create a corridor (~0.5 mile ≈ 0.008 degrees)
    if drawn_geom.geom_type in ("LineString", "MultiLineString"):
        BUFFER_DEG = 0.008  # ~0.5 miles at Texas latitudes
        drawn_geom = drawn_geom.buffer(BUFFER_DEG)

    # For circles: Leaflet stores radius in meters in properties
    if drawn_geom.geom_type == "Point" and "radius" in drawn.get("properties", {}):
        radius_m = drawn["properties"]["radius"]
        # Convert meters to degrees (approximate)
        radius_deg = radius_m / 111_320
        drawn_geom = drawn_geom.buffer(radius_deg)

    data = load_cached_pipeline(pipeline_result["cache_path"])
    header_df = data["header_df"]

    lat_col = "surface_lat" if "surface_lat" in header_df.columns else "latitude"
    lon_col = "surface_lon" if "surface_lon" in header_df.columns else "longitude"

    if lat_col not in header_df.columns or lon_col not in header_df.columns:
        return dash.no_update

    # Find wells within the shape/buffer
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


# ---------------------------------------------------------------------------
# Clear selection
# ---------------------------------------------------------------------------

@callback(
    Output("selected-wells-store", "data", allow_duplicate=True),
    Output("gun-barrel-chart", "figure", allow_duplicate=True),
    Output("cum-oil-chart", "figure", allow_duplicate=True),
    Output("daily-oil-chart", "figure", allow_duplicate=True),
    Input("btn-clear-selection", "n_clicks"),
    prevent_initial_call=True,
)
def clear_selection(n):
    return (
        None,
        empty_figure("Click a well on the map."),
        empty_figure("Click a well on the map."),
        empty_figure("Click a well on the map."),
    )


# ---------------------------------------------------------------------------
# Filter panel callbacks
# ---------------------------------------------------------------------------

@callback(
    Output("filter-collapse", "is_open"),
    Output("filter-col", "style"),
    Input("btn-toggle-filters", "n_clicks"),
    State("filter-collapse", "is_open"),
    prevent_initial_call=True,
)
def toggle_filter_panel(n, is_open):
    new_state = not is_open
    style = {} if new_state else {"display": "none"}
    return new_state, style


@callback(
    Output("filter-bench", "options"),
    Output("filter-operator", "options"),
    Output("filter-rsv-cat", "options"),
    Output("filter-rsv-cat", "value"),
    Output("filter-well-status", "options"),
    Output("filter-well-status", "value"),
    Output("filter-spud-year", "min"),
    Output("filter-spud-year", "max"),
    Output("filter-spud-year", "value"),
    Output("filter-spud-year", "marks"),
    Output("filter-lateral-length", "min"),
    Output("filter-lateral-length", "max"),
    Output("filter-lateral-length", "value"),
    Output("filter-lateral-length", "marks"),
    Input("pipeline-result-store", "data"),
    prevent_initial_call=False,
)
def populate_filter_options(pipeline_result):
    """Populate filter controls from pipeline header data."""
    defaults = ([], [], [], [], [], [], 2000, 2026, [2000, 2026],
                {2000: "2000", 2026: "2026"}, 0, 15000, [0, 15000],
                {0: "0", 15000: "15k"})
    if not pipeline_result or not pipeline_result.get("cache_path"):
        return defaults

    data = load_cached_pipeline(pipeline_result["cache_path"])
    header_df = data["header_df"]

    # Bench
    bench_opts = []
    if "bench" in header_df.columns:
        bench_opts = [{"label": b, "value": b} for b in sorted(header_df["bench"].dropna().unique())]

    # Operator
    op_opts = []
    if "operator" in header_df.columns:
        op_opts = [{"label": o, "value": o} for o in sorted(header_df["operator"].dropna().unique())]

    # RSV Cat
    rsv_opts, rsv_vals = [], []
    if "rsv_cat" in header_df.columns:
        rsv_vals = sorted(header_df["rsv_cat"].dropna().unique().tolist())
        rsv_opts = [{"label": r, "value": r} for r in rsv_vals]

    # Well Status
    ws_opts, ws_vals = [], []
    if "well_status" in header_df.columns:
        ws_vals = sorted(header_df["well_status"].dropna().unique().tolist())
        ws_opts = [{"label": s, "value": s} for s in ws_vals]

    # Spud Year
    yr_min, yr_max = 2000, 2026
    if "spud_date" in header_df.columns:
        try:
            years = pd.to_datetime(header_df["spud_date"], errors="coerce").dt.year.dropna()
            if len(years):
                yr_min, yr_max = int(years.min()), int(years.max())
        except Exception:
            pass
    yr_marks = {}
    for y in range(yr_min, yr_max + 1, max(1, (yr_max - yr_min) // 4)):
        yr_marks[y] = str(y)
    yr_marks[yr_max] = str(yr_max)

    # Lateral Length
    ll_min, ll_max = 0, 15000
    if "lateral_length_ft" in header_df.columns:
        ll = header_df["lateral_length_ft"].dropna()
        if len(ll):
            ll_min = int(ll.min() // 500 * 500)
            ll_max = int((ll.max() // 500 + 1) * 500)
    ll_marks = {}
    step = max(500, (ll_max - ll_min) // 4)
    for v in range(ll_min, ll_max + 1, step):
        ll_marks[v] = f"{v // 1000}k" if v >= 1000 else str(v)
    ll_marks[ll_max] = f"{ll_max // 1000}k" if ll_max >= 1000 else str(ll_max)

    return (bench_opts, op_opts, rsv_opts, rsv_vals, ws_opts, ws_vals,
            yr_min, yr_max, [yr_min, yr_max], yr_marks,
            ll_min, ll_max, [ll_min, ll_max], ll_marks)


@callback(
    Output("filter-uwis-store", "data"),
    Output("filter-count", "children"),
    Output("geojson-trajectories", "data", allow_duplicate=True),
    Output("geojson-bottomholes", "data", allow_duplicate=True),
    Input("filter-search", "value"),
    Input("filter-bench", "value"),
    Input("filter-operator", "value"),
    Input("filter-rsv-cat", "value"),
    Input("filter-well-status", "value"),
    Input("filter-spud-year", "value"),
    Input("filter-lateral-length", "value"),
    State("pipeline-result-store", "data"),
    prevent_initial_call="initial_duplicate",
)
def apply_filters(search, benches, operators, rsv_cats, statuses,
                  year_range, ll_range, pipeline_result):
    """Filter header data and update map GeoJSON to show only matching wells."""
    import logging
    logger = logging.getLogger("dashboard")

    if not pipeline_result or not pipeline_result.get("cache_path"):
        logger.warning("apply_filters: no pipeline result — skipping")
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    logger.info("apply_filters: search=%s benches=%s operators=%s rsv=%s statuses=%s",
                search, benches, operators, rsv_cats, statuses)

    data = load_cached_pipeline(pipeline_result["cache_path"])
    header_df = data["header_df"]
    lateral_df = data["lateral_df"]
    mask = pd.Series(True, index=header_df.index)

    # Text search
    if search and search.strip():
        s = search.strip().lower()
        text_mask = pd.Series(False, index=header_df.index)
        if "uwi" in header_df.columns:
            text_mask |= header_df["uwi"].astype(str).str.lower().str.contains(s, na=False)
        if "well_name" in header_df.columns:
            text_mask |= header_df["well_name"].astype(str).str.lower().str.contains(s, na=False)
        if "lease_name" in header_df.columns:
            text_mask |= header_df["lease_name"].astype(str).str.lower().str.contains(s, na=False)
        mask &= text_mask

    # Categorical filters (empty = all)
    if benches and "bench" in header_df.columns:
        mask &= header_df["bench"].isin(benches)
    if operators and "operator" in header_df.columns:
        mask &= header_df["operator"].isin(operators)
    if rsv_cats and "rsv_cat" in header_df.columns:
        mask &= header_df["rsv_cat"].isin(rsv_cats)
    if statuses and "well_status" in header_df.columns:
        mask &= header_df["well_status"].isin(statuses)

    # Year range
    if year_range and "spud_date" in header_df.columns:
        try:
            years = pd.to_datetime(header_df["spud_date"], errors="coerce").dt.year
            mask &= (years >= year_range[0]) & (years <= year_range[1]) | years.isna()
        except Exception:
            pass

    # Lateral length range
    if ll_range and "lateral_length_ft" in header_df.columns:
        mask &= (
            (header_df["lateral_length_ft"] >= ll_range[0])
            & (header_df["lateral_length_ft"] <= ll_range[1])
        ) | header_df["lateral_length_ft"].isna()

    filtered = header_df[mask]
    filtered_uwis = filtered["uwi"].astype(str).tolist()
    total = len(header_df)
    shown = len(filtered)
    logger.info("apply_filters: %d/%d wells pass filters", shown, total)

    # Rebuild GeoJSON with only filtered wells
    filtered_lateral = lateral_df[lateral_df["uwi"].isin(filtered["uwi"])]
    gdf_traj = build_trajectory_geodataframe(filtered_lateral, filtered)
    gdf_bh = build_bottomhole_geodataframe(gdf_traj)
    logger.info("apply_filters: %d trajectories, %d bottomholes rebuilt", len(gdf_traj), len(gdf_bh))

    count_text = f"Showing {shown:,} of {total:,} wells"

    return (
        filtered_uwis,
        count_text,
        gdf_traj.__geo_interface__ if not gdf_traj.empty else _EMPTY_GEOJSON,
        gdf_bh.__geo_interface__ if not gdf_bh.empty else _EMPTY_GEOJSON,
    )
