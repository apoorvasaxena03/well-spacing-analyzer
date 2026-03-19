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
from dash import Input, Output, State, callback, dcc, html, dash_table
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


def _expand_neighborhood(IK: pd.DataFrame, uwis: list[str]) -> set[str]:
    """
    Expand selected wells to include all direct IK neighbors.

    Mirrors the Spotfire 'data limiting' behaviour: when a well is selected,
    ALL wells it has spacing pairs with are included in the neighborhood.

    Returns the full set of UWIs (selected + their IK neighbours).
    """
    uwi_set = set(str(u) for u in uwis)
    # Find all pairs involving selected wells (either as well_i or well_k)
    mask = IK["well_i"].astype(str).isin(uwi_set) | IK["well_k"].astype(str).isin(uwi_set)
    involved = IK.loc[mask]
    # Expand: union of all well_i and well_k from those pairs
    return uwi_set | set(involved["well_i"].astype(str)) | set(involved["well_k"].astype(str))

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
                        "Zoom to Wells",
                        id="btn-zoom-to-wells",
                        color="outline-secondary",
                        size="sm",
                        className="me-2",
                    ),
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
        dcc.Store(id="clear-draw-trigger", data=0),
        html.Div(id="_draw-clear-dummy", style={"display": "none"}),
        # Track last GeoJSON n_clicks to distinguish well clicks from empty map clicks
        dcc.Store(id="last-geojson-clicks", data={"traj": 0, "bh": 0}),

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
                                                        style={"variable": "dashExtensions.default.style0"},
                                                        hideout={
                                                            "colorMap": {},
                                                            "colorProp": "bench",
                                                            "weight": 3,
                                                            "opacity": 0.9,
                                                            "defaultColor": "#3388ff",
                                                        },
                                                        hoverStyle={"weight": 6, "color": "#ff7800"},
                                                        onEachFeature={"variable": "dashExtensions.default.oef0"},
                                                    ),
                                                    name="Trajectories",
                                                    checked=True,
                                                ),
                                                dl.Overlay(
                                                    dl.GeoJSON(
                                                        id="geojson-bottomholes",
                                                        data=_EMPTY_GEOJSON,
                                                        pointToLayer={"variable": "dashExtensions.default.ptl_colored"},
                                                        hideout={
                                                            "colorMap": {},
                                                            "colorProp": "spud_year",
                                                            "radius": 4,
                                                            "opacity": 0.8,
                                                            "defaultColor": "#e74c3c",
                                                        },
                                                        onEachFeature={"variable": "dashExtensions.default.oef1"},
                                                    ),
                                                    name="Bottom Holes",
                                                    checked=True,
                                                ),
                                                dl.Overlay(
                                                    dl.GeoJSON(
                                                        id="geojson-selected",
                                                        data=_EMPTY_GEOJSON,
                                                        options={
                                                            "style": {"color": "#ff7800", "weight": 5, "opacity": 1.0},
                                                        },
                                                    ),
                                                    name="Selected Wells",
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
                                            ),
                                            id="draw-feature-group",
                                        ),
                                    ],
                                    id="main-map",
                                    center=[31.5, -101.9],
                                    zoom=10,
                                    scrollWheelZoom=True,
                                    wheelDebounceTime=200,
                                    wheelPxPerZoomLevel=300,
                                    style={"height": "60vh"},
                                ),
                                className="p-0",
                            ),
                        ),

                        # ── Legends ──
                        html.Div(id="map-legend", className="mt-1"),
                        html.Div(id="bh-legend", className="mt-1"),
                    ],
                    md=5,
                ),

                # ── Right: Charts ───────────────────────────────────────────
                dbc.Col([
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
                                                    id="gb-toggle-lines",
                                                    options=[{"label": "Lines", "value": "lines"}],
                                                    value=["lines"],
                                                    inline=True,
                                                    input_class_name="me-1",
                                                    label_class_name="small me-3",
                                                ),
                                                width="auto",
                                            ),
                                            dbc.Col(
                                                dbc.Checklist(
                                                    id="gb-toggle-labels",
                                                    options=[{"label": "Labels", "value": "labels"}],
                                                    value=["labels"],
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
                                [
                                    # Gun Barrel (standalone)
                                    dcc.Graph(
                                        id="gun-barrel-chart",
                                        style={"height": "40vh"},
                                        figure=empty_figure("Click a well on the map."),
                                    ),
                                    html.Hr(className="my-2"),

                                    # Cumulative Production
                                    dbc.Row(
                                        [
                                            dbc.Col(html.Strong("Cumulative Production", className="small"), width="auto"),
                                            dbc.Col(
                                                dbc.RadioItems(
                                                    id="cum-prod-product",
                                                    options=[
                                                        {"label": "Oil",   "value": "oil"},
                                                        {"label": "Gas",   "value": "gas"},
                                                        {"label": "Water", "value": "water"},
                                                    ],
                                                    value="oil",
                                                    inline=True,
                                                    className="small",
                                                ),
                                                width="auto",
                                            ),
                                        ],
                                        align="center",
                                        className="mb-1",
                                    ),
                                    dcc.Graph(
                                        id="cum-oil-chart",
                                        style={"height": "28vh"},
                                        figure=empty_figure("Click a well on the map."),
                                    ),
                                    html.Hr(className="my-2"),

                                    # Daily Production
                                    dbc.Row(
                                        [
                                            dbc.Col(html.Strong("Daily Production", className="small"), width="auto"),
                                            dbc.Col(
                                                dbc.RadioItems(
                                                    id="daily-prod-product",
                                                    options=[
                                                        {"label": "Oil",   "value": "oil"},
                                                        {"label": "Gas",   "value": "gas"},
                                                        {"label": "Water", "value": "water"},
                                                    ],
                                                    value="oil",
                                                    inline=True,
                                                    className="small",
                                                ),
                                                width="auto",
                                            ),
                                        ],
                                        align="center",
                                        className="mb-1",
                                    ),
                                    dcc.Graph(
                                        id="daily-oil-chart",
                                        style={"height": "28vh"},
                                        figure=empty_figure("Click a well on the map."),
                                    ),
                                ],
                                style={"maxHeight": "90vh", "overflowY": "auto"},
                            ),
                        ]
                    ),
                    # ── Data Tables ────────────────────────────────────────
                    dbc.Card(
                        dbc.CardBody(
                            dbc.Accordion(
                                [
                                    dbc.AccordionItem(
                                        [
                                            dcc.Dropdown(
                                                id="ik-col-selector",
                                                multi=True,
                                                placeholder="Select columns to display...",
                                                className="mb-2",
                                                style={"fontSize": "12px"},
                                            ),
                                            dash_table.DataTable(
                                                id="ik-pairs-table",
                                                columns=[],
                                                data=[],
                                                filter_action="native",
                                                sort_action="native",
                                                sort_mode="multi",
                                                page_size=20,
                                                style_table={"overflowX": "auto", "maxHeight": "40vh"},
                                                style_cell={
                                                    "textAlign": "left",
                                                    "fontSize": "12px",
                                                    "padding": "4px 8px",
                                                    "minWidth": "80px",
                                                    "maxWidth": "250px",
                                                    "overflow": "hidden",
                                                    "textOverflow": "ellipsis",
                                                },
                                                style_header={
                                                    "fontWeight": "bold",
                                                    "backgroundColor": "#f8f9fa",
                                                },
                                                tooltip_duration=None,
                                                fixed_rows={"headers": True},
                                            ),
                                        ],
                                        title="IK Spacing Pairs",
                                        item_id="acc-ik",
                                    ),
                                    dbc.AccordionItem(
                                        [
                                            dcc.Dropdown(
                                                id="gb-col-selector",
                                                multi=True,
                                                placeholder="Select columns to display...",
                                                className="mb-2",
                                                style={"fontSize": "12px"},
                                            ),
                                            dash_table.DataTable(
                                                id="gb-data-table",
                                                columns=[],
                                                data=[],
                                                filter_action="native",
                                                sort_action="native",
                                                sort_mode="multi",
                                                page_size=15,
                                                style_table={"overflowX": "auto", "maxHeight": "35vh"},
                                                style_cell={
                                                    "textAlign": "left",
                                                    "fontSize": "12px",
                                                    "padding": "4px 8px",
                                                    "minWidth": "80px",
                                                    "maxWidth": "250px",
                                                    "overflow": "hidden",
                                                    "textOverflow": "ellipsis",
                                                },
                                                style_header={
                                                    "fontWeight": "bold",
                                                    "backgroundColor": "#f8f9fa",
                                                },
                                                tooltip_duration=None,
                                                fixed_rows={"headers": True},
                                            ),
                                        ],
                                        title="Gun Barrel Data",
                                        item_id="acc-gb",
                                    ),
                                    dbc.AccordionItem(
                                        [
                                            dcc.Dropdown(
                                                id="header-col-selector",
                                                multi=True,
                                                placeholder="Select columns to display...",
                                                className="mb-2",
                                                style={"fontSize": "12px"},
                                            ),
                                            dash_table.DataTable(
                                                id="header-data-table",
                                                columns=[],
                                                data=[],
                                                filter_action="native",
                                                sort_action="native",
                                                sort_mode="multi",
                                                page_size=20,
                                                style_table={"overflowX": "auto", "maxHeight": "40vh"},
                                                style_cell={
                                                    "textAlign": "left",
                                                    "fontSize": "12px",
                                                    "padding": "4px 8px",
                                                    "minWidth": "80px",
                                                    "maxWidth": "250px",
                                                    "overflow": "hidden",
                                                    "textOverflow": "ellipsis",
                                                },
                                                style_header={
                                                    "fontWeight": "bold",
                                                    "backgroundColor": "#f8f9fa",
                                                },
                                                tooltip_duration=None,
                                                fixed_rows={"headers": True},
                                            ),
                                        ],
                                        title="Well Header",
                                        item_id="acc-header",
                                    ),
                                ],
                                start_collapsed=True,
                                always_open=True,
                            ),
                        ),
                        className="mt-2",
                    ),
                    ],
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

# -- Clientside: clear drawn shapes from Leaflet map --
dash.clientside_callback(
    """
    function(trigger) {
        (window._leafletMaps || []).forEach(function(map) {
            map.eachLayer(function(layer) {
                if (
                    layer instanceof L.FeatureGroup &&
                    !(layer instanceof L.GeoJSON) &&
                    typeof layer.clearLayers === "function"
                ) {
                    layer.clearLayers();
                }
            });
        });
        return null;
    }
    """,
    Output("_draw-clear-dummy", "children"),
    Input("clear-draw-trigger", "data"),
    prevent_initial_call=True,
)


# -- Toggle style settings panel --
@callback(
    Output("style-collapse", "is_open"),
    Input("btn-toggle-style", "n_clicks"),
    State("style-collapse", "is_open"),
    prevent_initial_call=True,
)
def toggle_style_panel(n, is_open):
    return not is_open


# -- Labels disabled when Lines unchecked --
@callback(
    Output("gb-toggle-labels", "options"),
    Input("gb-toggle-lines", "value"),
    prevent_initial_call=True,
)
def toggle_labels_enabled(lines_value):
    disabled = "lines" not in (lines_value or [])
    return [{"label": "Labels", "value": "labels", "disabled": disabled}]


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

    legend = _build_legend(color_map, traj_color_by)

    return hideout, legend


# -- Bottomhole style → hideout + legend --
@callback(
    Output("geojson-bottomholes", "hideout"),
    Output("bh-legend", "children"),
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

    hideout = {
        "colorMap": color_map,
        "colorProp": bh_color_by if bh_color_by != "_uniform" else "bench",
        "radius": radius or 4,
        "opacity": opacity or 0.8,
        "defaultColor": "#e74c3c",
    }

    legend = _build_legend(color_map, f"Bottom Holes: {bh_color_by or 'spud_year'}")

    return hideout, legend


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
    # Prefer full directional survey for map (shows vertical + build sections);
    # fall back to lateral-only for caches created before this change.
    # Filter to wells present in header_df (only wells used in spacing calc).
    survey_df = data.get("directional_df")
    if survey_df is None or survey_df.empty:
        survey_df = data["lateral_df"]
    elif "uwi" in survey_df.columns and "uwi" in header_df.columns:
        valid_uwis = set(header_df["uwi"].astype(str).unique())
        survey_df = survey_df[survey_df["uwi"].astype(str).isin(valid_uwis)]

    gdf_traj = build_trajectory_geodataframe(survey_df, header_df)
    gdf_bh = build_bottomhole_geodataframe(gdf_traj)

    traj_data = gdf_traj.__geo_interface__
    bh_data = gdf_bh.__geo_interface__
    logger.info("load_map_layers: %d trajectories, %d bottomholes", len(gdf_traj), len(gdf_bh))

    import math

    lat_col = "surface_lat" if "surface_lat" in header_df.columns else "latitude"
    lon_col = "surface_lon" if "surface_lon" in header_df.columns else "longitude"

    lats = header_df[lat_col].dropna() if lat_col in header_df.columns else pd.Series(dtype=float)
    lons = header_df[lon_col].dropna() if lon_col in header_df.columns else pd.Series(dtype=float)

    if lats.empty or lons.empty:
        return traj_data, bh_data, [31.5, -101.9], 10

    lat_min, lat_max = float(lats.min()), float(lats.max())
    lon_min, lon_max = float(lons.min()), float(lons.max())
    center = [(lat_min + lat_max) / 2, (lon_min + lon_max) / 2]

    lat_span = lat_max - lat_min
    lon_span = lon_max - lon_min
    span = max(lat_span, lon_span, 0.001)
    zoom = max(1, min(18, int(math.log2(360 / span)) - 1))

    return traj_data, bh_data, center, zoom


@callback(
    Output("selected-wells-store", "data"),
    Output("last-geojson-clicks", "data"),
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

    # Save current click counts so map-click callback can detect well clicks
    click_counts = {"traj": traj_clicks or 0, "bh": bh_clicks or 0}

    click_data = traj_click_data or bh_click_data
    if not click_data or not pipeline_result:
        return no_change, click_counts

    props = click_data.get("properties") or {}
    clicked_uwi = props.get("uwi")
    _log.info("on_well_click: triggered=%s uwi=%s type=%s", triggered, clicked_uwi, type(clicked_uwi).__name__)
    if not clicked_uwi:
        return no_change, click_counts
    # Ensure string
    clicked_uwi = str(clicked_uwi)

    # Only return the clicked well — no neighborhood expansion.
    # The GB uses Spotfire data-limiting: selected wells = well_i set.
    # Shape selection (polygon/line/circle) selects multiple wells.
    _log.info("on_well_click: uwi=%s selected=1 well", clicked_uwi)
    return {"clicked_uwi": clicked_uwi, "neighborhood_uwis": [clicked_uwi]}, click_counts


@callback(
    Output("gun-barrel-chart", "figure"),
    Input("selected-wells-store", "data"),
    Input("gb-xaxis-mode", "value"),
    Input("gb-color-by", "value"),
    Input("gb-toggle-lines", "value"),
    Input("gb-toggle-labels", "value"),
    State("pipeline-result-store", "data"),
    prevent_initial_call=True,
)
def update_gun_barrel(selected, x_col, color_by, lines_toggle, labels_toggle, pipeline_result):
    import logging
    _log = logging.getLogger("dashboard")
    try:
        return _update_gun_barrel_inner(selected, x_col, color_by, lines_toggle, labels_toggle, pipeline_result)
    except Exception as exc:
        _log.exception("Gun barrel error: %s", exc)
        return empty_figure(f"Error: {exc}")


def _update_gun_barrel_inner(selected, x_col, color_by, lines_toggle, labels_toggle, pipeline_result):
    if not selected or not selected.get("neighborhood_uwis"):
        return empty_figure("Click a well on the map to populate the gun barrel.")

    uwis = selected["neighborhood_uwis"]
    IK, HeelToe = load_cached_ik_heeltoe(pipeline_result)

    if IK.empty:
        return empty_figure("Pipeline results not loaded.")

    # Spotfire data-limiting: IK filtered to well_i in selected wells only.
    # GB will contain only the selected wells — NOT their neighbours.
    uwi_set = set(str(u) for u in uwis)
    IK_filtered = IK[IK["well_i"].astype(str).isin(uwi_set)].copy()
    HeelToe_filtered = HeelToe[HeelToe["uwi"].astype(str).isin(uwi_set)]

    if IK_filtered.empty:
        return empty_figure("No spacing pairs found for selected well. Try selecting multiple wells.")

    # elevation = tvd * -1
    if "tvd_i" in IK_filtered.columns and "elevation_i" not in IK_filtered.columns:
        IK_filtered["elevation_i"] = IK_filtered["tvd_i"] * -1
    if "tvd_k" in IK_filtered.columns and "elevation_k" not in IK_filtered.columns:
        IK_filtered["elevation_k"] = IK_filtered["tvd_k"] * -1

    # Enrich GB with header data (bench, operator, etc.)
    data = load_cached_pipeline(pipeline_result["cache_path"])
    header_df = data["header_df"]

    GB = compute_gun_barrel(IK_filtered, HeelToe_filtered, header_df=header_df)
    show_lines = "lines" in (lines_toggle or [])
    show_labels = show_lines and "labels" in (labels_toggle or [])
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
    Input("cum-prod-product", "value"),
    State("pipeline-result-store", "data"),
    prevent_initial_call=True,
)
def update_cum_production(selected, product, pipeline_result):
    import logging
    _log = logging.getLogger("dashboard")
    try:
        product = product or "oil"
        if not selected or not pipeline_result:
            return empty_figure("Click a well on the map.")

        data = load_cached_pipeline(pipeline_result["cache_path"])
        prod = data.get("production_df")

        if prod is None or prod.empty:
            return empty_figure("No production data loaded.")

        # Find the column for the selected product
        col = product if product in prod.columns else None
        if col is None:
            return empty_figure(f"No '{product}' column in production data.")

        uwis = selected.get("neighborhood_uwis", [])
        prod_sel = prod[prod["uwi"].isin(uwis)].copy()
        if prod_sel.empty:
            return empty_figure("No production data for selected wells.")

        prod_sel = prod_sel.sort_values(["uwi", "prod_date"])
        cum_col = f"cum_{product}"
        prod_sel[cum_col] = prod_sel.groupby("uwi")[col].cumsum()
        prod_sel["months"] = prod_sel.groupby("uwi")["prod_date"].transform(
            lambda s: (s - s.min()).dt.days / 30.44
        )

        unit_map = {"oil": "BBL", "gas": "MCF", "water": "BBL"}
        unit = unit_map.get(product, "")

        fig = go.Figure()
        for uwi, grp in prod_sel.groupby("uwi"):
            fig.add_trace(go.Scatter(
                x=grp["months"], y=grp[cum_col],
                mode="lines", name=str(uwi),
            ))
        fig.update_layout(
            xaxis_title="Months since first production",
            yaxis_title=f"Cumulative {product.title()} ({unit})",
            template="plotly_white",
            hovermode="x unified",
            margin=dict(t=30, b=50, l=60, r=20),
            yaxis=dict(tickformat=","),
        )
        return fig
    except Exception as exc:
        _log.exception("Cum production chart error: %s", exc)
        return empty_figure(f"Error: {exc}")


@callback(
    Output("daily-oil-chart", "figure"),
    Input("selected-wells-store", "data"),
    Input("daily-prod-product", "value"),
    State("pipeline-result-store", "data"),
    prevent_initial_call=True,
)
def update_daily_production(selected, product, pipeline_result):
    import logging
    _log = logging.getLogger("dashboard")
    try:
        product = product or "oil"
        if not selected or not pipeline_result:
            return empty_figure("Click a well on the map.")

        data = load_cached_pipeline(pipeline_result["cache_path"])
        prod = data.get("production_df")

        if prod is None or prod.empty:
            return empty_figure("No production data loaded.")

        # Try daily_<product> first, then fall back to <product>
        daily_col = f"daily_{product}"
        col = daily_col if daily_col in prod.columns else (product if product in prod.columns else None)
        if col is None:
            return empty_figure(f"No '{product}' column in production data.")

        unit_map = {"oil": "BOPD", "gas": "MCFD", "water": "BWPD"}
        unit = unit_map.get(product, "")

        uwis = selected.get("neighborhood_uwis", [])
        prod_sel = prod[prod["uwi"].isin(uwis)].sort_values(["uwi", "prod_date"])
        if prod_sel.empty:
            return empty_figure("No production data for selected wells.")

        fig = go.Figure()
        for uwi, grp in prod_sel.groupby("uwi"):
            fig.add_trace(go.Scatter(
                x=grp["prod_date"], y=grp[col],
                mode="lines", name=str(uwi),
            ))
        fig.update_layout(
            xaxis_title="Production Date",
            yaxis_title=f"Daily {product.title()} ({unit})",
            template="plotly_white",
            hovermode="x unified",
            margin=dict(t=30, b=50, l=60, r=20),
            yaxis=dict(tickformat=","),
        )
        return fig
    except Exception as exc:
        _log.exception("Daily production chart error: %s", exc)
        return empty_figure(f"Error: {exc}")


# ---------------------------------------------------------------------------
# Draw / Lasso selection — select wells within drawn polygon or rectangle
# ---------------------------------------------------------------------------

@callback(
    Output("selected-wells-store", "data", allow_duplicate=True),
    Output("draw-control", "geojson"),
    Output("clear-draw-trigger", "data", allow_duplicate=True),
    Input("draw-control", "geojson"),
    State("pipeline-result-store", "data"),
    prevent_initial_call=True,
)
def select_wells_by_shape(geojson, pipeline_result):
    """Select all wells within a drawn shape (polygon, rectangle, circle, or line buffer)."""
    import logging
    import time
    _log = logging.getLogger("dashboard")

    empty_geojson = {"type": "FeatureCollection", "features": []}

    if not geojson or not pipeline_result:
        return dash.no_update, dash.no_update, dash.no_update

    features = geojson.get("features", [])
    if not features:
        return dash.no_update, dash.no_update, dash.no_update

    # Use the last drawn shape
    drawn = features[-1]
    props = drawn.get("properties", {}) or {}
    geom_type = drawn.get("geometry", {}).get("type", "")
    _log.info("select_wells_by_shape: geom=%s props=%s", geom_type, props)
    drawn_geom = shape(drawn["geometry"])

    # For circles: Leaflet stores center as Point + radius in properties.
    # Buffer the point by the radius to create a circular polygon.
    if drawn_geom.geom_type == "Point":
        radius_m = props.get("radius", 800)
        drawn_geom = drawn_geom.buffer(radius_m / 111_320)
    # LineString, Polygon, Rectangle → use as-is for direct intersection
    # No buffer needed — we check if the shape touches/crosses the trajectory

    data = load_cached_pipeline(pipeline_result["cache_path"])
    # Use full directional survey (matches what the map displays);
    # fall back to lateral-only for old caches.
    survey_df = data.get("directional_df")
    if survey_df is None or survey_df.empty:
        survey_df = data["lateral_df"]
    elif "uwi" in survey_df.columns and "uwi" in data["header_df"].columns:
        valid_uwis = set(data["header_df"]["uwi"].astype(str).unique())
        survey_df = survey_df[survey_df["uwi"].astype(str).isin(valid_uwis)]

    # Build trajectory LineStrings per well and check intersection with drawn shape
    from shapely.geometry import LineString as ShapelyLineString
    _log.info("select_wells_by_shape: checking trajectories against %s (bounds=%s)",
              drawn_geom.geom_type, drawn_geom.bounds)
    selected_uwis = []
    for uwi, grp in survey_df.groupby("uwi"):
        if "longitude" not in grp.columns or "latitude" not in grp.columns:
            continue
        coords = list(zip(grp["longitude"].astype(float), grp["latitude"].astype(float)))
        if len(coords) < 2:
            continue
        try:
            traj = ShapelyLineString(coords)
            if drawn_geom.intersects(traj):
                selected_uwis.append(str(uwi))
        except Exception:
            continue

    _log.info("select_wells_by_shape: %d wells found in shape (geom_type=%s)",
              len(selected_uwis), drawn_geom.geom_type)
    if not selected_uwis:
        return dash.no_update, empty_geojson, time.time()

    # Return selection + clear drawn shapes from map
    return {
        "clicked_uwi": selected_uwis[0],
        "neighborhood_uwis": sorted(selected_uwis),
    }, empty_geojson, time.time()


# ---------------------------------------------------------------------------
# Click empty map space → clear selection
# ---------------------------------------------------------------------------

@callback(
    Output("selected-wells-store", "data", allow_duplicate=True),
    Output("gun-barrel-chart", "figure", allow_duplicate=True),
    Output("cum-oil-chart", "figure", allow_duplicate=True),
    Output("daily-oil-chart", "figure", allow_duplicate=True),
    Output("geojson-selected", "data", allow_duplicate=True),
    Output("ik-pairs-table", "data", allow_duplicate=True),
    Output("ik-pairs-table", "columns", allow_duplicate=True),
    Output("gb-data-table", "data", allow_duplicate=True),
    Output("gb-data-table", "columns", allow_duplicate=True),
    Output("header-data-table", "data", allow_duplicate=True),
    Output("header-data-table", "columns", allow_duplicate=True),
    Output("ik-col-selector", "options", allow_duplicate=True),
    Output("ik-col-selector", "value", allow_duplicate=True),
    Output("gb-col-selector", "options", allow_duplicate=True),
    Output("gb-col-selector", "value", allow_duplicate=True),
    Output("header-col-selector", "options", allow_duplicate=True),
    Output("header-col-selector", "value", allow_duplicate=True),
    Output("clear-draw-trigger", "data", allow_duplicate=True),
    Input("main-map", "click_lat_lng"),
    State("geojson-trajectories", "n_clicks"),
    State("geojson-bottomholes", "n_clicks"),
    State("last-geojson-clicks", "data"),
    prevent_initial_call=True,
)
def on_map_background_click(click_lat_lng, traj_clicks, bh_clicks, last_clicks):
    """Full clear (same as Clear Selection button) when clicking empty map space."""
    import time

    no_update_all = (dash.no_update,) * 18

    if not click_lat_lng:
        return no_update_all

    last = last_clicks or {"traj": 0, "bh": 0}
    if (traj_clicks or 0) != last.get("traj", 0) or (bh_clicks or 0) != last.get("bh", 0):
        return no_update_all

    return (
        None,
        empty_figure("Click a well on the map."),
        empty_figure("Click a well on the map."),
        empty_figure("Click a well on the map."),
        _EMPTY_GEOJSON,
        [], [], [], [],
        [], [],
        [], [], [], [], [], [],
        time.time(),
    )


# ---------------------------------------------------------------------------
# Highlight selected wells on map
# ---------------------------------------------------------------------------

@callback(
    Output("geojson-selected", "data"),
    Input("selected-wells-store", "data"),
    State("pipeline-result-store", "data"),
    prevent_initial_call=True,
)
def highlight_selected_wells(selected, pipeline_result):
    """Render selected wells as highlighted trajectories on the map."""
    if not selected or not selected.get("neighborhood_uwis") or not pipeline_result:
        return _EMPTY_GEOJSON

    uwis = selected["neighborhood_uwis"]
    try:
        data = load_cached_pipeline(pipeline_result["cache_path"])
        header_df = data["header_df"]
        # Use full directional survey for highlight (matches map display)
        survey_df = data.get("directional_df")
        if survey_df is None or survey_df.empty:
            survey_df = data["lateral_df"]
        elif "uwi" in survey_df.columns and "uwi" in header_df.columns:
            valid_uwis = set(header_df["uwi"].astype(str).unique())
            survey_df = survey_df[survey_df["uwi"].astype(str).isin(valid_uwis)]

        sel_survey = survey_df[survey_df["uwi"].astype(str).isin([str(u) for u in uwis])]
        sel_header = header_df[header_df["uwi"].isin(uwis)]

        if sel_survey.empty:
            return _EMPTY_GEOJSON

        gdf = build_trajectory_geodataframe(sel_survey, sel_header)
        return gdf.__geo_interface__ if not gdf.empty else _EMPTY_GEOJSON
    except Exception:
        return _EMPTY_GEOJSON


# ---------------------------------------------------------------------------
# Data tables — IK Pairs + Gun Barrel + Header
# ---------------------------------------------------------------------------

# Default columns shown when data first loads (user can change via dropdown)
_IK_DEFAULT_COLS = [
    "well_i", "well_k", "well_name_i", "well_name_k",
    "horizontal_dist", "vertical_dist", "dist3d",
    "alignment_type", "overlap_pct",
    "bench_i", "bench_k", "operator_i", "operator_k",
]
_GB_DEFAULT_COLS = [
    "well_i", "well_name", "bench", "operator", "rsv_cat",
    "elevation_i", "cum_dist", "sectionDist",
    "horizontal_dist", "vertical_dist",
    "first_prod_date",
]
_HDR_DEFAULT_COLS = [
    "uwi", "well_name", "operator", "bench", "rsv_cat",
    "hole_direction", "spud_date", "first_prod_date",
    "lateral_length_ft",
]


@callback(
    Output("ik-pairs-table", "columns"),
    Output("ik-pairs-table", "data"),
    Output("ik-col-selector", "options"),
    Output("ik-col-selector", "value"),
    Input("selected-wells-store", "data"),
    State("pipeline-result-store", "data"),
    State("ik-col-selector", "value"),
    prevent_initial_call=True,
)
def update_ik_table(selected, pipeline_result, current_selection):
    """Populate IK pairs table — well_i in selected wells (Spotfire data-limiting)."""
    if not selected or not pipeline_result:
        return [], [], [], []
    uwis = selected.get("neighborhood_uwis", [])
    IK, _ = load_cached_ik_heeltoe(pipeline_result)
    if IK.empty:
        return [], [], [], []
    uwi_set = set(str(u) for u in uwis)
    ik_sel = IK[IK["well_i"].astype(str).isin(uwi_set)]
    if ik_sel.empty:
        return [], [], [], []
    # Round numeric columns
    all_cols = list(ik_sel.columns)
    ik_display = ik_sel.copy()
    for col in ik_display.select_dtypes(include="number").columns:
        ik_display[col] = ik_display[col].round(2)
    options = [{"label": c, "value": c} for c in all_cols]
    # Keep user's current selection if valid; otherwise use defaults
    if current_selection:
        visible = [c for c in current_selection if c in all_cols]
    else:
        visible = [c for c in _IK_DEFAULT_COLS if c in all_cols]
    columns = [{"name": c, "id": c} for c in visible]
    return columns, ik_display.to_dict("records"), options, visible


@callback(
    Output("gb-data-table", "columns"),
    Output("gb-data-table", "data"),
    Output("gb-col-selector", "options"),
    Output("gb-col-selector", "value"),
    Input("selected-wells-store", "data"),
    State("pipeline-result-store", "data"),
    State("gb-col-selector", "value"),
    prevent_initial_call=True,
)
def update_gb_table(selected, pipeline_result, current_selection):
    """Populate gun barrel data table with header info for selected wells."""
    if not selected or not pipeline_result:
        return [], [], [], []
    uwis = selected.get("neighborhood_uwis", [])
    IK, HeelToe = load_cached_ik_heeltoe(pipeline_result)
    if IK.empty:
        return [], [], [], []
    uwi_set = set(str(u) for u in uwis)
    IK_filtered = IK[IK["well_i"].astype(str).isin(uwi_set)].copy()
    HeelToe_filtered = HeelToe[HeelToe["uwi"].astype(str).isin(uwi_set)]
    if IK_filtered.empty:
        return [], [], [], []
    if "tvd_i" in IK_filtered.columns and "elevation_i" not in IK_filtered.columns:
        IK_filtered["elevation_i"] = IK_filtered["tvd_i"] * -1
    data = load_cached_pipeline(pipeline_result["cache_path"])
    header_df = data["header_df"]
    GB = compute_gun_barrel(IK_filtered, HeelToe_filtered, header_df=header_df)
    if GB.empty:
        return [], [], [], []
    # Round numeric columns
    all_cols = list(GB.columns)
    for col in GB.select_dtypes(include="number").columns:
        GB[col] = GB[col].round(2)
    options = [{"label": c, "value": c} for c in all_cols]
    if current_selection:
        visible = [c for c in current_selection if c in all_cols]
    else:
        visible = [c for c in _GB_DEFAULT_COLS if c in all_cols]
    columns = [{"name": c, "id": c} for c in visible]
    return columns, GB.to_dict("records"), options, visible


@callback(
    Output("header-data-table", "columns"),
    Output("header-data-table", "data"),
    Output("header-col-selector", "options"),
    Output("header-col-selector", "value"),
    Input("selected-wells-store", "data"),
    State("pipeline-result-store", "data"),
    State("header-col-selector", "value"),
    prevent_initial_call=True,
)
def update_header_table(selected, pipeline_result, current_selection):
    """Populate header data table for selected wells (all columns)."""
    if not selected or not pipeline_result:
        return [], [], [], []
    uwis = selected.get("neighborhood_uwis", [])
    data = load_cached_pipeline(pipeline_result["cache_path"])
    header_df = data["header_df"]
    if header_df.empty:
        return [], [], [], []
    uwi_set = set(str(u) for u in uwis)
    hdr_sel = header_df[header_df["uwi"].astype(str).isin(uwi_set)]
    if hdr_sel.empty:
        return [], [], [], []
    # Expose ALL columns; user picks which to show via dropdown
    all_cols = list(hdr_sel.columns)
    hdr_display = hdr_sel.copy()
    for col in hdr_display.select_dtypes(include="number").columns:
        hdr_display[col] = hdr_display[col].round(2)
    options = [{"label": c, "value": c} for c in all_cols]
    if current_selection:
        visible = [c for c in current_selection if c in all_cols]
    else:
        visible = [c for c in _HDR_DEFAULT_COLS if c in all_cols]
    columns = [{"name": c, "id": c} for c in visible]
    return columns, hdr_display.to_dict("records"), options, visible


# -- Column selector callbacks (update visible columns without reloading data) --

@callback(
    Output("ik-pairs-table", "columns", allow_duplicate=True),
    Input("ik-col-selector", "value"),
    State("ik-pairs-table", "data"),
    prevent_initial_call=True,
)
def filter_ik_columns(selected_cols, data):
    if not selected_cols or not data:
        return dash.no_update
    return [{"name": c, "id": c} for c in selected_cols]


@callback(
    Output("gb-data-table", "columns", allow_duplicate=True),
    Input("gb-col-selector", "value"),
    State("gb-data-table", "data"),
    prevent_initial_call=True,
)
def filter_gb_columns(selected_cols, data):
    if not selected_cols or not data:
        return dash.no_update
    return [{"name": c, "id": c} for c in selected_cols]


@callback(
    Output("header-data-table", "columns", allow_duplicate=True),
    Input("header-col-selector", "value"),
    State("header-data-table", "data"),
    prevent_initial_call=True,
)
def filter_header_columns(selected_cols, data):
    if not selected_cols or not data:
        return dash.no_update
    return [{"name": c, "id": c} for c in selected_cols]


# ---------------------------------------------------------------------------
# Clear selection
# ---------------------------------------------------------------------------

@callback(
    Output("selected-wells-store", "data", allow_duplicate=True),
    Output("gun-barrel-chart", "figure", allow_duplicate=True),
    Output("cum-oil-chart", "figure", allow_duplicate=True),
    Output("daily-oil-chart", "figure", allow_duplicate=True),
    Output("geojson-selected", "data", allow_duplicate=True),
    Output("ik-pairs-table", "data", allow_duplicate=True),
    Output("ik-pairs-table", "columns", allow_duplicate=True),
    Output("gb-data-table", "data", allow_duplicate=True),
    Output("gb-data-table", "columns", allow_duplicate=True),
    Output("header-data-table", "data", allow_duplicate=True),
    Output("header-data-table", "columns", allow_duplicate=True),
    Output("ik-col-selector", "options", allow_duplicate=True),
    Output("ik-col-selector", "value", allow_duplicate=True),
    Output("gb-col-selector", "options", allow_duplicate=True),
    Output("gb-col-selector", "value", allow_duplicate=True),
    Output("header-col-selector", "options", allow_duplicate=True),
    Output("header-col-selector", "value", allow_duplicate=True),
    Output("clear-draw-trigger", "data", allow_duplicate=True),
    Input("btn-clear-selection", "n_clicks"),
    prevent_initial_call=True,
)
def clear_selection(n):
    import time
    return (
        None,
        empty_figure("Click a well on the map."),
        empty_figure("Click a well on the map."),
        empty_figure("Click a well on the map."),
        _EMPTY_GEOJSON,
        [], [], [], [],
        [], [],
        [], [], [], [], [], [],
        time.time(),
    )


# ---------------------------------------------------------------------------
# Zoom to wells (fit bounds to all visible wells)
# ---------------------------------------------------------------------------

@callback(
    Output("main-map", "center", allow_duplicate=True),
    Output("main-map", "zoom", allow_duplicate=True),
    Input("btn-zoom-to-wells", "n_clicks"),
    State("pipeline-result-store", "data"),
    State("filter-uwis-store", "data"),
    prevent_initial_call=True,
)
def zoom_to_wells(n, pipeline_result, filter_uwis):
    """Fit map center and zoom to show all currently visible wells."""
    import math

    if not pipeline_result or not pipeline_result.get("cache_path"):
        return dash.no_update, dash.no_update

    data = load_cached_pipeline(pipeline_result["cache_path"])
    header_df = data["header_df"]

    if filter_uwis:
        header_df = header_df[header_df["uwi"].astype(str).isin(filter_uwis)]

    lat_col = "surface_lat" if "surface_lat" in header_df.columns else "latitude"
    lon_col = "surface_lon" if "surface_lon" in header_df.columns else "longitude"

    if lat_col not in header_df.columns or lon_col not in header_df.columns:
        return dash.no_update, dash.no_update

    lats = header_df[lat_col].dropna()
    lons = header_df[lon_col].dropna()
    if lats.empty or lons.empty:
        return dash.no_update, dash.no_update

    lat_min, lat_max = float(lats.min()), float(lats.max())
    lon_min, lon_max = float(lons.min()), float(lons.max())
    center = [(lat_min + lat_max) / 2, (lon_min + lon_max) / 2]

    # Estimate zoom level from bounding box span
    lat_span = lat_max - lat_min
    lon_span = lon_max - lon_min
    span = max(lat_span, lon_span, 0.001)
    zoom = max(1, min(18, int(math.log2(360 / span)) - 1))

    return center, zoom


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
    # Prefer full directional survey for map (shows vertical + build sections);
    # fall back to lateral-only for caches created before this change.
    survey_df = data.get("directional_df")
    if survey_df is None or survey_df.empty:
        survey_df = data["lateral_df"]
    elif "uwi" in survey_df.columns and "uwi" in header_df.columns:
        valid_uwis = set(header_df["uwi"].astype(str).unique())
        survey_df = survey_df[survey_df["uwi"].astype(str).isin(valid_uwis)]
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
    filtered_survey = survey_df[survey_df["uwi"].isin(filtered["uwi"])]
    gdf_traj = build_trajectory_geodataframe(filtered_survey, filtered)
    gdf_bh = build_bottomhole_geodataframe(gdf_traj)
    logger.info("apply_filters: %d trajectories, %d bottomholes rebuilt", len(gdf_traj), len(gdf_bh))

    count_text = f"Showing {shown:,} of {total:,} wells"

    return (
        filtered_uwis,
        count_text,
        gdf_traj.__geo_interface__ if not gdf_traj.empty else _EMPTY_GEOJSON,
        gdf_bh.__geo_interface__ if not gdf_bh.empty else _EMPTY_GEOJSON,
    )
