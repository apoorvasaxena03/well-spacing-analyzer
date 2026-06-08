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
from dash import Input, Output, State, ALL, callback, dcc, html, dash_table
import dash_leaflet as dl
import plotly.graph_objects as go
import plotly.express as px
from shapely.geometry import shape, Point

from dashboard.pipeline import (
    compute_gun_barrel,
    load_cached_pipeline,
    load_cached_ik_heeltoe,
    save_ui_state,
    load_ui_state,
)
from dashboard.components.gun_barrel import build_gun_barrel_figure, empty_figure
from dashboard.components.map_panel import build_trajectory_geodataframe, build_bottomhole_geodataframe

# Import analysis callbacks so Dash registers them (they reference component IDs in this layout)
import dashboard.callbacks.explore_analysis  # noqa: F401

dash.register_page(__name__, path="/explore", name="5 Explore", order=5)

# ---------------------------------------------------------------------------
# Color palette — 20 distinct colors for categorical variables
# ---------------------------------------------------------------------------
_PALETTE = px.colors.qualitative.Alphabet


_ROLE_COLORS: dict[str, str] = {
    "parent":              "#2196F3",  # blue
    "child":               "#FF9800",  # orange
    "infill_candidate":    "#F44336",  # red
    "no_eligible_neighbor": "#9E9E9E", # grey
}


def _build_color_map(values: list[str], prop: str = "") -> dict[str, str]:
    """Map each unique value to a hex color from the palette.

    For the 'role' property, use fixed colors consistent with notebooks.
    """
    unique = sorted(set(str(v) for v in values if v and str(v).strip()))
    if prop == "role":
        return {v: _ROLE_COLORS.get(v, "#607D8B") for v in unique}
    return {v: _PALETTE[i % len(_PALETTE)] for i, v in enumerate(unique)}

_EMPTY_GEOJSON = {"type": "FeatureCollection", "features": []}


def _clean_dates_for_table(df: pd.DataFrame) -> pd.DataFrame:
    """Convert datetime columns to YYYY-MM-DD strings for DataTable display."""
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            # Strip timezone, format as date only
            if hasattr(df[col].dt, "tz") and df[col].dt.tz is not None:
                df[col] = df[col].dt.tz_localize(None)
            df[col] = df[col].dt.strftime("%Y-%m-%d").replace("NaT", "")
    return df


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
                    dbc.Button(
                        "Save Settings",
                        id="btn-save-ui",
                        color="success",
                        size="sm",
                        className="ms-2",
                        disabled=True,
                    ),
                ],
                width="auto",
                className="d-flex",
            ),
        ], align="center", className="mb-2"),

        # Store for active filter state (list of UWIs that pass filters)
        dcc.Store(id="filter-uwis-store"),
        dcc.Store(id="clear-draw-trigger", data=0),
        dcc.Store(id="dbn-result-store"),
        dcc.Store(id="avg-spacing-result-store"),
        dcc.Store(id="wps-result-store"),
        dcc.Store(id="user-color-overrides", data={}),
        dcc.Store(id="ui-restore-timestamp", data=0),
        dcc.Store(id="ui-dirty", data=False),
        html.Div(id="_draw-clear-dummy", style={"display": "none"}),

        # Fullscreen modals for diagnostic plots
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Debug Pair Spacing"), close_button=True),
            dbc.ModalBody(id="modal-debug-pair-body"),
        ], id="modal-debug-pair", size="xl", fullscreen="xl-down", is_open=False,
           scrollable=True),
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("WPS Visualization"), close_button=True),
            dbc.ModalBody(id="modal-wps-viz-body"),
        ], id="modal-wps-viz", size="xl", fullscreen="xl-down", is_open=False,
           scrollable=True),
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Neighborhood Plot"), close_button=True),
            dbc.ModalBody(id="modal-avg-viz-body"),
        ], id="modal-avg-viz", size="xl", fullscreen="xl-down", is_open=False,
           scrollable=True),

        # Fullscreen modals for Plotly charts
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Gun Barrel — Cross-Section View"), close_button=True),
            dbc.ModalBody(dcc.Graph(id="modal-gb-chart", style={"height": "85vh"})),
        ], id="modal-gb", size="xl", fullscreen="xl-down", is_open=False),
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Cumulative Production"), close_button=True),
            dbc.ModalBody(dcc.Graph(id="modal-cum-chart", style={"height": "85vh"})),
        ], id="modal-cum", size="xl", fullscreen="xl-down", is_open=False),
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Daily Production"), close_button=True),
            dbc.ModalBody(dcc.Graph(id="modal-daily-chart", style={"height": "85vh"})),
        ], id="modal-daily", size="xl", fullscreen="xl-down", is_open=False),
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Production by Role"), close_button=True),
            dbc.ModalBody(dcc.Graph(id="modal-role-prod-chart", style={"height": "85vh"})),
        ], id="modal-role-prod", size="xl", fullscreen="xl-down", is_open=False),

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
                                                    dcc.Dropdown(
                                                        id="traj-color-by",
                                                        options=[
                                                            {"label": "Bench",    "value": "bench"},
                                                            {"label": "Role",     "value": "role"},
                                                            {"label": "Spud Year","value": "spud_year"},
                                                            {"label": "Operator", "value": "operator"},
                                                            {"label": "RSV Cat",  "value": "rsv_cat"},
                                                            {"label": "Uniform",  "value": "_uniform"},
                                                        ],
                                                        value="bench",
                                                        clearable=False,
                                                        style={"fontSize": "0.8rem"},
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
                                                    dcc.Dropdown(
                                                        id="bh-color-by",
                                                        options=[
                                                            {"label": "Bench",    "value": "bench"},
                                                            {"label": "Role",     "value": "role"},
                                                            {"label": "Spud Year","value": "spud_year"},
                                                            {"label": "Operator", "value": "operator"},
                                                            {"label": "RSV Cat",  "value": "rsv_cat"},
                                                            {"label": "Uniform",  "value": "_uniform"},
                                                        ],
                                                        value="bench",
                                                        clearable=False,
                                                        style={"fontSize": "0.8rem"},
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
                                            # -- Tooltip fields row --
                                            html.Hr(className="my-2"),
                                            html.H6("Tooltip Fields", className="mb-2", style={"fontSize": "0.82rem"}),
                                            html.P(
                                                "Select which fields appear when hovering over wells on the map.",
                                                className="text-muted small mb-1",
                                            ),
                                            dcc.Dropdown(
                                                id="tooltip-fields",
                                                options=[],  # populated dynamically
                                                value=["well_name", "uwi", "bench", "operator", "role"],
                                                multi=True,
                                                placeholder="Select fields to show on hover...",
                                                style={"fontSize": "0.8rem"},
                                            ),
                                            # -- Neighbor edges row --
                                            html.Hr(className="my-2"),
                                            html.H6("Neighbor Edges", className="mb-2", style={"fontSize": "0.82rem"}),
                                            dbc.Switch(
                                                id="edge-toggle",
                                                label="Show neighbor edge lines when wells are selected",
                                                value=True,
                                                className="small",
                                            ),
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
                                                        onEachFeature={"variable": "dashExtensions.default.oef0"},
                                                    ),
                                                    name="Selected Wells",
                                                    checked=True,
                                                ),
                                                dl.Overlay(
                                                    dl.GeoJSON(
                                                        id="geojson-edges",
                                                        data=_EMPTY_GEOJSON,
                                                        style={"variable": "dashExtensions.default.edgeStyle"},
                                                        onEachFeature={"variable": "dashExtensions.default.edgeTooltip"},
                                                    ),
                                                    name="Neighbor Edges",
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
                                            secondaryLengthUnit="kilometers",
                                            primaryAreaUnit="acres",
                                            secondaryAreaUnit="sqmeters",
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
                                        # ── Legend overlay (inside dl.Map so it survives fullscreen) ──
                                        html.Div(
                                            [
                                                html.Button(
                                                    "Legend",
                                                    id="btn-toggle-legend",
                                                    style={
                                                        "fontSize": "0.7rem",
                                                        "padding": "2px 8px",
                                                        "backgroundColor": "rgba(255,255,255,0.9)",
                                                        "border": "1px solid #ccc",
                                                        "borderRadius": "3px",
                                                        "cursor": "pointer",
                                                        "marginBottom": "4px",
                                                    },
                                                ),
                                                html.Div(
                                                    [
                                                        html.Div(id="map-legend"),
                                                        html.Div(id="bh-legend", className="mt-1"),
                                                        html.Div(id="edge-legend", className="mt-1"),
                                                    ],
                                                    id="legend-content",
                                                ),
                                            ],
                                            style={
                                                "position": "absolute",
                                                "top": "10px",
                                                "right": "55px",
                                                "zIndex": "1000",
                                                "maxHeight": "50vh",
                                                "overflowY": "auto",
                                                "pointerEvents": "auto",
                                            },
                                        ),
                                    ],
                                    id="main-map",
                                    center=[31.5, -101.9],
                                    zoom=10,
                                    scrollWheelZoom=True,
                                    style={"height": "60vh"},
                                ),
                                className="p-0",
                            ),
                        ),
                    ],
                    md=5,
                ),

                # ── Right: Charts + Analysis ───────────────────────────────
                dbc.Col([
                  dbc.Tabs([
                    dbc.Tab(label="Charts", tab_id="tab-charts", children=[
                    dbc.Card([
                        dbc.CardHeader([
                            dbc.Tabs([
                                dbc.Tab(label="Neighborhood", tab_id="subtab-neighborhood"),
                                dbc.Tab(label="Statistics", tab_id="subtab-statistics"),
                            ], id="charts-subtabs", active_tab="subtab-neighborhood",
                               className="card-header-tabs"),
                        ]),

                        # ── Sub-tab: Neighborhood ─────────────────────────
                        html.Div(id="subtab-neighborhood-content", children=[
                            # GB controls header
                            dbc.CardHeader([
                                dbc.Row([
                                    dbc.Col(html.Strong("Gun Barrel"), width="auto"),
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
                                ], align="center"),
                                dbc.Row([
                                    dbc.Col(
                                        dbc.Select(
                                            id="gb-color-by",
                                            options=[
                                                {"label": "Bench",    "value": "bench"},
                                                {"label": "Role",     "value": "role"},
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
                                    dbc.Col([
                                        dbc.Label("Dot", className="small mb-0"),
                                        dcc.Slider(
                                            id="gb-marker-size", min=4, max=24, step=2,
                                            value=12,
                                            marks={4: "4", 12: "12", 24: "24"},
                                        ),
                                    ], md=2),
                                    dbc.Col([
                                        dbc.Label("Line", className="small mb-0"),
                                        dcc.Slider(
                                            id="gb-line-width", min=1, max=6, step=0.5,
                                            value=1.5,
                                            marks={1: "1", 3: "3", 6: "6"},
                                        ),
                                    ], md=2),
                                    dbc.Col([
                                        dbc.Label("Label", className="small mb-0"),
                                        dcc.Slider(
                                            id="gb-label-size", min=6, max=16, step=1,
                                            value=9,
                                            marks={6: "6", 10: "10", 16: "16"},
                                        ),
                                    ], md=2),
                                ], align="center", className="mt-1"),
                                dbc.Row(
                                    dbc.Col(
                                        dcc.Dropdown(
                                            id="chart-hover-fields",
                                            options=[],
                                            value=["well_name", "bench", "role", "operator"],
                                            multi=True,
                                            placeholder="Select fields for chart hover...",
                                            style={"fontSize": "0.75rem"},
                                        ),
                                    ),
                                    className="mt-1",
                                ),
                            ]),
                            dbc.CardBody([
                                # Gun Barrel
                                dbc.Button("⛶", id="btn-expand-gb",
                                           color="link", size="sm",
                                           className="float-end p-0",
                                           title="Fullscreen"),
                                dcc.Graph(
                                    id="gun-barrel-chart",
                                    style={"height": "40vh"},
                                    figure=empty_figure("Click a well on the map."),
                                ),
                                html.Hr(className="my-2"),

                                # Cumulative Production
                                dbc.Row([
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
                                    dbc.Col(
                                        dbc.Button("⛶", id="btn-expand-cum",
                                                   color="link", size="sm",
                                                   className="p-0", title="Fullscreen"),
                                        width="auto",
                                    ),
                                ], align="center", className="mb-1"),
                                dcc.Graph(
                                    id="cum-oil-chart",
                                    style={"height": "28vh"},
                                    figure=empty_figure("Click a well on the map."),
                                ),
                                html.Hr(className="my-2"),

                                # Daily Production
                                dbc.Row([
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
                                    dbc.Col(
                                        dbc.Button("⛶", id="btn-expand-daily",
                                                   color="link", size="sm",
                                                   className="p-0", title="Fullscreen"),
                                        width="auto",
                                    ),
                                ], align="center", className="mb-1"),
                                dcc.Graph(
                                    id="daily-oil-chart",
                                    style={"height": "28vh"},
                                    figure=empty_figure("Click a well on the map."),
                                ),
                            ], style={"maxHeight": "90vh", "overflowY": "auto"}),
                        ]),

                        # ── Sub-tab: Statistics ───────────────────────────
                        html.Div(id="subtab-statistics-content", style={"display": "none"}, children=[
                            dbc.CardBody([
                                # Production by Role (box plot)
                                dbc.Row([
                                    dbc.Col(html.Strong("Production by Role", className="small"), width="auto"),
                                    dbc.Col(
                                        dbc.Select(
                                            id="role-prod-metric",
                                            options=[
                                                {"label": "Cum Oil 180d/ft",   "value": "cum_oil_180d_per_ft"},
                                                {"label": "Cum Oil 365d/ft",   "value": "cum_oil_365d_per_ft"},
                                                {"label": "Cum Gas 180d/ft",   "value": "cum_gas_180d_per_ft"},
                                                {"label": "Cum Gas 365d/ft",   "value": "cum_gas_365d_per_ft"},
                                                {"label": "Cum Water 180d/ft", "value": "cum_water_180d_per_ft"},
                                                {"label": "Cum Water 365d/ft", "value": "cum_water_365d_per_ft"},
                                            ],
                                            value="cum_oil_180d_per_ft",
                                            size="sm",
                                        ),
                                        width=4,
                                    ),
                                    dbc.Col(
                                        dbc.Checklist(
                                            id="role-prod-facet-bench",
                                            options=[{"label": "Facet by Bench", "value": "facet"}],
                                            value=["facet"],
                                            inline=True,
                                            input_class_name="me-1",
                                            label_class_name="small me-3",
                                        ),
                                        width="auto",
                                    ),
                                    dbc.Col(
                                        dbc.Button("⛶", id="btn-expand-role-prod",
                                                   color="link", size="sm",
                                                   className="p-0", title="Fullscreen"),
                                        width="auto",
                                    ),
                                ], align="center", className="mb-1"),
                                dcc.Graph(
                                    id="role-prod-chart",
                                    style={"height": "75vh"},
                                    figure=empty_figure("Run pipeline with role assignment to see production by role."),
                                ),
                            ]),
                        ]),
                    ]),
                    ]),  # end Charts tab

                    # ── Analysis Tab ─────────────────────────────────────
                    dbc.Tab(label="Analysis", tab_id="tab-analysis", children=[
                        dbc.Card(dbc.CardBody([
                            # -- Shared Parameters --
                            html.H6("Shared Parameters", className="mb-1"),
                            html.P(
                                "These thresholds are shared by Directional Bench Neighbors and "
                                "Average Spacing. They control which IK pairs qualify as neighbors.",
                                className="text-muted small mb-2",
                            ),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Cutoff (ft)", size="sm"),
                                    dbc.Input(id="dbn-cutoff-ft", type="number",
                                              value=1800, min=0, step=100, size="sm"),
                                    dbc.FormText(
                                        "Maximum horizontal spacing (ft) to consider a well a neighbor. "
                                        "Pairs with horizontal_dist > cutoff are excluded. "
                                        "Typical range: 800-2,500 ft."
                                    ),
                                ], md=4),
                                dbc.Col([
                                    dbc.Label("Vertical cutoff (ft)", size="sm"),
                                    dbc.Input(id="dbn-vertical-cutoff-ft", type="number",
                                              placeholder="None", min=0, step=50, size="sm"),
                                    dbc.FormText(
                                        "Optional. If set, pairs must also satisfy vertical_dist <= this. "
                                        "Useful for multi-bench developments where you want to exclude "
                                        "wells in different benches that are laterally close but vertically far."
                                    ),
                                ], md=4),
                                dbc.Col([
                                    dbc.Label("Overlap % min", size="sm"),
                                    dbc.Input(id="dbn-overlap-pct-min", type="number",
                                              placeholder="None", min=0, max=1, step=0.05, size="sm"),
                                    dbc.FormText(
                                        "Optional. Minimum lateral overlap fraction (0-1) required. "
                                        "E.g. 0.40 = well_k must overlap at least 40% of well_i's lateral. "
                                        "Filters out pairs that barely overlap."
                                    ),
                                ], md=4),
                            ], className="mb-2"),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Axis mode", size="sm"),
                                    dbc.Select(id="dbn-axis-mode", size="sm", value="any",
                                               options=[
                                                   {"label": "Any", "value": "any"},
                                                   {"label": "EW only", "value": "EW"},
                                                   {"label": "NS only", "value": "NS"},
                                               ]),
                                    dbc.FormText(
                                        "Direction eligibility. 'Any' = neighbors from all directions. "
                                        "'EW' = only east/west neighbors. 'NS' = only north/south. "
                                        "Use when well pattern has a dominant drilling direction."
                                    ),
                                ], md=4),
                                dbc.Col([
                                    dbc.Label("Prefer axis", size="sm"),
                                    dbc.Select(id="dbn-prefer-axis", size="sm", value="",
                                               options=[
                                                   {"label": "None", "value": ""},
                                                   {"label": "EW", "value": "EW"},
                                                   {"label": "NS", "value": "NS"},
                                               ]),
                                    dbc.FormText(
                                        "Optional tie-breaker. When two candidates tie on distance, "
                                        "prefer the one along this axis. Leave 'None' for no preference."
                                    ),
                                ], md=4),
                            ], className="mb-3"),

                            html.Hr(),

                            # -- Overrides --
                            dbc.Row([
                                dbc.Col([
                                    html.H6("Per-Well Overrides", className="mb-1"),
                                    html.P(
                                        "Upload a CSV to set per-well cutoffs that override the shared values above. "
                                        "Required column: 'uwi'. Optional columns: 'cutoff_ft', 'vertical_cutoff_ft', "
                                        "'overlap_pct_k_min'. Leave cells blank to fall back to the shared value. "
                                        "Applies to DBN and Avg Spacing only (WPS has no per-well overrides).",
                                        className="text-muted small mb-2",
                                    ),
                                    dbc.RadioItems(
                                        id="overrides-mode",
                                        options=[
                                            {"label": "Shared (DBN + Avg)", "value": "shared"},
                                            {"label": "Separate per class", "value": "separate"},
                                        ],
                                        value="shared",
                                        inline=True,
                                        className="small mb-2",
                                    ),
                                ]),
                            ]),
                            html.Div(id="overrides-shared-div", children=[
                                dcc.Upload(
                                    id="overrides-upload",
                                    children=dbc.Button("Upload overrides CSV", size="sm",
                                                        color="secondary", outline=True),
                                    accept=".csv",
                                ),
                            ]),
                            html.Div(id="overrides-separate-div", style={"display": "none"}, children=[
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Label("DBN overrides", size="sm"),
                                        dcc.Upload(
                                            id="overrides-dbn-upload",
                                            children=dbc.Button("Upload CSV", size="sm",
                                                                color="secondary", outline=True),
                                            accept=".csv",
                                        ),
                                    ], md=6),
                                    dbc.Col([
                                        dbc.Label("Avg overrides", size="sm"),
                                        dcc.Upload(
                                            id="overrides-avg-upload",
                                            children=dbc.Button("Upload CSV", size="sm",
                                                                color="secondary", outline=True),
                                            accept=".csv",
                                        ),
                                    ], md=6),
                                ]),
                            ]),
                            dash_table.DataTable(
                                id="overrides-table",
                                columns=[
                                    {"name": "uwi", "id": "uwi"},
                                    {"name": "cutoff_ft", "id": "cutoff_ft", "type": "numeric"},
                                    {"name": "vertical_cutoff_ft", "id": "vertical_cutoff_ft", "type": "numeric"},
                                    {"name": "overlap_pct_k_min", "id": "overlap_pct_k_min", "type": "numeric"},
                                ],
                                data=[],
                                editable=True,
                                row_deletable=True,
                                page_size=5,
                                style_table={"overflowX": "auto", "maxHeight": "15vh"},
                                style_cell={"fontSize": "11px", "padding": "2px 6px"},
                                style_header={"fontWeight": "bold", "backgroundColor": "#f8f9fa"},
                            ),

                            html.Hr(),

                            # ── Directional Bench Neighbors ──
                            dbc.Accordion([
                                dbc.AccordionItem([
                                    html.P(
                                        "Reduces the pairwise IK spacing table to one row per well. "
                                        "For each well_i, picks up to 4 nearest neighbors: "
                                        "same-bench closest (same_1), same-bench opposite direction (same_2), "
                                        "different-bench closest (near_1), different-bench opposite (near_2). "
                                        "Also classifies each well as Unbounded / Partially / Mostly / Fully Bounded.",
                                        className="text-muted small mb-2",
                                    ),
                                    html.P(
                                        "Uses the shared cutoff, vertical cutoff, overlap, and axis parameters above.",
                                        className="text-muted small mb-2 fst-italic",
                                    ),
                                    dbc.Button("Run Neighbor Summary", id="btn-run-dbn",
                                               color="primary", size="sm", className="mb-2"),
                                    dcc.Loading(
                                        html.Div(id="dbn-status", className="small text-muted mb-2"),
                                        type="dot",
                                    ),
                                    dcc.Dropdown(
                                        id="dbn-col-selector",
                                        multi=True,
                                        placeholder="Select columns to display...",
                                        className="mb-2",
                                        style={"fontSize": "12px"},
                                    ),
                                    dash_table.DataTable(
                                        id="dbn-result-table",
                                        columns=[], data=[],
                                        filter_action="native", sort_action="native",
                                        sort_mode="multi", page_size=15,
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
                                        style_header={"fontWeight": "bold", "backgroundColor": "#f8f9fa"},
                                        tooltip_duration=None,
                                        fixed_rows={"headers": True},
                                    ),
                                ], title="Directional Bench Neighbors", item_id="acc-dbn"),

                                # ── Average Spacing ──
                                dbc.AccordionItem([
                                    html.P(
                                        "Computes the average well spacing for each well by building a "
                                        "neighborhood of eligible neighbors and computing chain or dense "
                                        "spacing metrics. Output: one row per well with avg_hz_spacing_ft "
                                        "and avg_vt_spacing_ft.",
                                        className="text-muted small mb-2",
                                    ),
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Label("Neighborhood mode", size="sm"),
                                            dbc.Select(id="avg-neighborhood-mode", size="sm", value="chain",
                                                       options=[
                                                           {"label": "Chain", "value": "chain"},
                                                           {"label": "i2nbr", "value": "i2nbr"},
                                                           {"label": "Dense", "value": "dense"},
                                                       ]),
                                            dbc.FormText(
                                                "Chain (default): orders neighbors A-B-C-D along a line and "
                                                "averages consecutive edge distances. Best for parallel wells. "
                                                "i2nbr: simple mean of all i-to-neighbor distances. "
                                                "Dense: mean of all unique member-member pairs."
                                            ),
                                        ], md=4),
                                        dbc.Col([
                                            dbc.Label("Chain sort", size="sm"),
                                            dbc.Select(id="avg-chain-sort-mode", size="sm", value="pca",
                                                       options=[
                                                           {"label": "PCA", "value": "pca"},
                                                           {"label": "X", "value": "x"},
                                                       ]),
                                            dbc.FormText(
                                                "How to order members along the chain. "
                                                "PCA (default): project onto first principal component "
                                                "(handles angled well groups). X: sort by UTM x-coordinate "
                                                "(faster, fine for E-W wells)."
                                            ),
                                        ], md=4),
                                        dbc.Col([
                                            dbc.Label("Edge pick", size="sm"),
                                            dbc.Select(id="avg-edge-pick", size="sm", value="min",
                                                       options=[
                                                           {"label": "Min", "value": "min"},
                                                           {"label": "Mean", "value": "mean"},
                                                           {"label": "Forward", "value": "forward"},
                                                       ]),
                                            dbc.FormText(
                                                "How to pick the spacing value for each chain edge (A-B). "
                                                "Min: use min(A->B, B->A). Mean: average of both directions. "
                                                "Forward: use A->B only (directional)."
                                            ),
                                        ], md=4),
                                    ], className="mb-2"),
                                    dbc.Button("Run Avg Spacing", id="btn-run-avg",
                                               color="primary", size="sm", className="mb-2"),
                                    dcc.Loading(
                                        html.Div(id="avg-status", className="small text-muted mb-2"),
                                        type="dot",
                                    ),
                                    dcc.Dropdown(
                                        id="avg-col-selector",
                                        multi=True,
                                        placeholder="Select columns to display...",
                                        className="mb-2",
                                        style={"fontSize": "12px"},
                                    ),
                                    dash_table.DataTable(
                                        id="avg-result-table",
                                        columns=[], data=[],
                                        filter_action="native", sort_action="native",
                                        sort_mode="multi", page_size=15,
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
                                        style_header={"fontWeight": "bold", "backgroundColor": "#f8f9fa"},
                                        tooltip_duration=None,
                                        fixed_rows={"headers": True},
                                    ),
                                ], title="Average Spacing", item_id="acc-avg"),

                                # ── Floating Section WPS ──
                                dbc.AccordionItem([
                                    html.P(
                                        "Counts how many wells have lateral inside a floating rectangular "
                                        "section centered on each well (Wells Per Section). Computes counts "
                                        "in three orientations: Cardinal (north-up box), I-Frame (box aligned "
                                        "to well azimuth), and Corridor (lateral-following rectangle). "
                                        "Also reports anisotropy ratio (i-frame / cardinal). "
                                        "No per-well overrides \u2014 all parameters are global.",
                                        className="text-muted small mb-2",
                                    ),
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Label("Box half-width (ft)", size="sm"),
                                            dbc.Input(id="wps-box-hw", type="number",
                                                      value=2640, min=0, step=100, size="sm"),
                                            dbc.FormText(
                                                "East-west half-extent of the floating section. "
                                                "2,640 ft = half mile (standard DSU). The full box is "
                                                "2 x half-width wide."
                                            ),
                                        ], md=4),
                                        dbc.Col([
                                            dbc.Label("Box half-height (ft)", size="sm"),
                                            dbc.Input(id="wps-box-hh", type="number",
                                                      value=2640, min=0, step=100, size="sm"),
                                            dbc.FormText(
                                                "North-south half-extent. 2,640 ft = half mile. "
                                                "Together with half-width, defines a 1-mile x 1-mile "
                                                "section (standard DSU)."
                                            ),
                                        ], md=4),
                                        dbc.Col([
                                            dbc.Label("Min inside (ft)", size="sm"),
                                            dbc.Input(id="wps-min-inside", type="number",
                                                      value=660, min=0, step=50, size="sm"),
                                            dbc.FormText(
                                                "Minimum lateral length inside the box to count a well. "
                                                "660 ft = 1/8 mile. Filters out wells that barely clip "
                                                "the section edge."
                                            ),
                                        ], md=4),
                                    ], className="mb-2"),
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Label("Corridor half-width (ft)", size="sm"),
                                            dbc.Input(id="wps-corr-hw", type="number",
                                                      value=2640, min=0, step=100, size="sm"),
                                            dbc.FormText(
                                                "Cross-well half-width for corridor mode. The corridor "
                                                "follows the reference lateral and extends this far on "
                                                "each side perpendicular to it."
                                            ),
                                        ], md=4),
                                        dbc.Col([
                                            dbc.Label("Corridor extra along (ft)", size="sm"),
                                            dbc.Input(id="wps-corr-ea", type="number",
                                                      value=0, min=0, step=100, size="sm"),
                                            dbc.FormText(
                                                "Extra margin beyond heel and toe along the lateral "
                                                "direction. 0 = corridor ends at heel/toe. Positive "
                                                "values extend the corridor."
                                            ),
                                        ], md=4),
                                        dbc.Col([
                                            dbc.Label("Exclude self", size="sm"),
                                            dbc.Switch(id="wps-exclude-self", value=True,
                                                       className="mt-1"),
                                            dbc.FormText(
                                                "When ON (default), the reference well is not counted "
                                                "in its own WPS. Turn OFF to include the well itself "
                                                "in the count."
                                            ),
                                        ], md=4),
                                    ], className="mb-2"),
                                    dbc.Button("Run WPS", id="btn-run-wps",
                                               color="primary", size="sm", className="mb-2"),
                                    dcc.Loading(
                                        html.Div(id="wps-status", className="small text-muted mb-2"),
                                        type="dot",
                                    ),
                                    dcc.Dropdown(
                                        id="wps-col-selector",
                                        multi=True,
                                        placeholder="Select columns to display...",
                                        className="mb-2",
                                        style={"fontSize": "12px"},
                                    ),
                                    dash_table.DataTable(
                                        id="wps-result-table",
                                        columns=[], data=[],
                                        filter_action="native", sort_action="native",
                                        sort_mode="multi", page_size=15,
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
                                        style_header={"fontWeight": "bold", "backgroundColor": "#f8f9fa"},
                                        tooltip_duration=None,
                                        fixed_rows={"headers": True},
                                    ),
                                ], title="Floating Section WPS", item_id="acc-wps"),

                                # ── Debug Pair Spacing (Phase B) ──
                                dbc.AccordionItem([
                                    html.P(
                                        "Generate detailed diagnostic plots for a specific well pair (i, k). "
                                        "Shows UTM map view, projected i-frame, overlap band, crossline distance "
                                        "series, direction rose, and contact analysis. The plots depend on alignment "
                                        "type: parallel pairs get crossline series + overlap band; oblique/perpendicular "
                                        "pairs get nearest-projection series + UTM segment overlay.",
                                        className="text-muted small mb-2",
                                    ),
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Label("Well I (UWI)", size="sm"),
                                            dbc.Input(id="debug-uwi-i", type="text",
                                                      placeholder="e.g. 42003488930000", size="sm"),
                                            dbc.FormText(
                                                "Reference well UWI. The local coordinate frame is built "
                                                "from this well's lateral. Type or paste the full 14-digit API."
                                            ),
                                        ], md=6),
                                        dbc.Col([
                                            dbc.Label("Well K (UWI)", size="sm"),
                                            dbc.Input(id="debug-uwi-k", type="text",
                                                      placeholder="e.g. 42003488900000", size="sm"),
                                            dbc.FormText(
                                                "Neighbor well UWI. Spacing is computed from well_i to well_k. "
                                                "You can also click a row in the IK Pairs table to auto-fill."
                                            ),
                                        ], md=6),
                                    ], className="mb-2"),
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Label("Step (ft)", size="sm"),
                                            dbc.Input(id="debug-step-ft", type="number",
                                                      value=100, min=10, step=10, size="sm"),
                                            dbc.FormText(
                                                "Sampling interval along the lateral for computing crossline "
                                                "distances. Smaller = more samples = smoother curves but slower. "
                                                "Default 100 ft works well."
                                            ),
                                        ], md=3),
                                        dbc.Col([
                                            dbc.Label("Contact threshold (ft)", size="sm"),
                                            dbc.Input(id="debug-contact-ft", type="number",
                                                      value=300, min=0, step=50, size="sm"),
                                            dbc.FormText(
                                                "Distance threshold for 'contact' classification. "
                                                "Segments closer than this are counted as in contact. "
                                                "Used for oblique/perpendicular pairs."
                                            ),
                                        ], md=3),
                                        dbc.Col([
                                            dbc.Label("PCA axis", size="sm"),
                                            dbc.Switch(id="debug-pca-axis", value=True,
                                                       className="mt-1"),
                                            dbc.FormText(
                                                "Use PCA to determine the along-well axis (recommended). "
                                                "OFF = use first-to-last point as axis direction."
                                            ),
                                        ], md=3),
                                        dbc.Col([
                                            dbc.Label("Theta parallel / perp (°)", size="sm"),
                                            dbc.InputGroup([
                                                dbc.Input(id="debug-theta-par", type="number",
                                                          value=25, min=0, max=90, step=5, size="sm"),
                                                dbc.InputGroupText("/", className="px-1"),
                                                dbc.Input(id="debug-theta-perp", type="number",
                                                          value=65, min=0, max=90, step=5, size="sm"),
                                            ], size="sm"),
                                            dbc.FormText(
                                                "Alignment boundaries. Angle ≤ parallel° → parallel_like. "
                                                "Angle ≥ perp° → perpendicular. Between → oblique."
                                            ),
                                        ], md=3),
                                    ], className="mb-2"),
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Label("Figure size", size="sm"),
                                            dbc.Select(id="debug-figsize", size="sm", value="medium",
                                                       options=[
                                                           {"label": "Small (6x4)", "value": "small"},
                                                           {"label": "Medium (8x5)", "value": "medium"},
                                                           {"label": "Large (12x7)", "value": "large"},
                                                       ]),
                                        ], md=4),
                                        dbc.Col([
                                            dbc.Label("DPI", size="sm"),
                                            dbc.Select(id="debug-dpi", size="sm", value="150",
                                                       options=[
                                                           {"label": "100 (fast)", "value": "100"},
                                                           {"label": "150 (default)", "value": "150"},
                                                           {"label": "200 (high)", "value": "200"},
                                                           {"label": "300 (print)", "value": "300"},
                                                       ]),
                                        ], md=4),
                                    ], className="mb-2"),
                                    dbc.Button("Generate Pair Diagnostics", id="btn-debug-pair",
                                               color="primary", size="sm", className="mb-2"),
                                    dcc.Loading(
                                        html.Div(id="debug-pair-status", className="small text-muted mb-2"),
                                        type="dot",
                                    ),
                                    # Metrics summary table
                                    dash_table.DataTable(
                                        id="debug-pair-metrics-table",
                                        columns=[], data=[],
                                        style_table={"overflowX": "auto", "maxHeight": "20vh"},
                                        style_cell={"textAlign": "left", "fontSize": "11px",
                                                    "padding": "3px 6px"},
                                        style_header={"fontWeight": "bold", "backgroundColor": "#f8f9fa"},
                                    ),
                                    # Container for matplotlib figure images
                                    html.Div([
                                        dbc.Button("Expand", id="btn-expand-debug-pair",
                                                   color="outline-secondary", size="sm",
                                                   className="mb-2"),
                                        html.Div(id="debug-pair-plots"),
                                    ], className="mt-2"),
                                ], title="Debug Pair Spacing", item_id="acc-debug-pair"),

                                # ── WPS Visualization (Phase B) ──
                                dbc.AccordionItem([
                                    html.P(
                                        "Visualize the floating section box/corridor for a single reference well. "
                                        "Shows the reference lateral (bold), the bounding region, and all neighbors "
                                        "color-coded by how they intersect the region. Green = passes min_inside "
                                        "threshold. Orange = intersects but below threshold. Gray = midpoint only.",
                                        className="text-muted small mb-2",
                                    ),
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Label("Reference UWI", size="sm"),
                                            dbc.Input(id="wps-viz-uwi", type="text",
                                                      placeholder="Click a well on map or type UWI",
                                                      size="sm"),
                                            dbc.FormText(
                                                "The well to center the floating section on. "
                                                "Auto-filled when you click a well on the map."
                                            ),
                                        ], md=6),
                                        dbc.Col([
                                            dbc.Label("Orientation", size="sm"),
                                            dbc.Select(id="wps-viz-orientation", size="sm",
                                                       value="cardinal",
                                                       options=[
                                                           {"label": "Cardinal (N-up box)", "value": "cardinal"},
                                                           {"label": "I-Frame (aligned to lateral)", "value": "i_frame"},
                                                           {"label": "Corridor (follows lateral)", "value": "corridor"},
                                                       ]),
                                            dbc.FormText(
                                                "Cardinal: axis-aligned rectangle (standard DSU). "
                                                "I-Frame: box rotated to match the well's lateral direction. "
                                                "Corridor: rectangle that follows the lateral path."
                                            ),
                                        ], md=6),
                                    ], className="mb-2"),
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Label("Figure size", size="sm"),
                                            dbc.Select(id="wps-viz-figsize", size="sm", value="medium",
                                                       options=[
                                                           {"label": "Small (7x7)", "value": "small"},
                                                           {"label": "Medium (10x10)", "value": "medium"},
                                                           {"label": "Large (14x14)", "value": "large"},
                                                       ]),
                                        ], md=4),
                                        dbc.Col([
                                            dbc.Label("DPI", size="sm"),
                                            dbc.Select(id="wps-viz-dpi", size="sm", value="150",
                                                       options=[
                                                           {"label": "100 (fast)", "value": "100"},
                                                           {"label": "150 (default)", "value": "150"},
                                                           {"label": "200 (high)", "value": "200"},
                                                           {"label": "300 (print)", "value": "300"},
                                                       ]),
                                        ], md=4),
                                    ], className="mb-2"),
                                    dbc.Button("Visualize Section", id="btn-wps-viz",
                                               color="primary", size="sm", className="mb-2"),
                                    dcc.Loading(
                                        html.Div(id="wps-viz-status", className="small text-muted mb-2"),
                                        type="dot",
                                    ),
                                    html.Div([
                                        dbc.Button("Expand", id="btn-expand-wps-viz",
                                                   color="outline-secondary", size="sm",
                                                   className="mb-2"),
                                        html.Div(id="wps-viz-plot"),
                                    ], className="mt-2"),
                                ], title="WPS Visualization", item_id="acc-wps-viz"),

                                # ── Avg Spacing Neighborhood Plot (Phase B) ──
                                dbc.AccordionItem([
                                    html.P(
                                        "Diagnostic plot showing a well's neighborhood: lateral trajectories, "
                                        "midpoints, cutoff circle, convex hull, and chain/dense edge overlays. "
                                        "Well_i is drawn bold black, survivors (kept neighbors) in color, "
                                        "non-survivors lighter. Requires running Avg Spacing first.",
                                        className="text-muted small mb-2",
                                    ),
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Label("Well UWI", size="sm"),
                                            dbc.Input(id="avg-viz-uwi", type="text",
                                                      placeholder="Click a well on map or type UWI",
                                                      size="sm"),
                                            dbc.FormText(
                                                "The well to visualize the neighborhood for. "
                                                "Auto-filled when you click a well on the map."
                                            ),
                                        ], md=6),
                                        dbc.Col([
                                            dbc.Label("Plot mode", size="sm"),
                                            dbc.Select(id="avg-viz-mode", size="sm",
                                                       value="both",
                                                       options=[
                                                           {"label": "Both (chain + dense)", "value": "both"},
                                                           {"label": "Chain only", "value": "chain"},
                                                           {"label": "Dense only", "value": "dense"},
                                                       ]),
                                            dbc.FormText(
                                                "Chain: shows A→B→C→D ordered edges with distance labels. "
                                                "Dense: shows all member-member pair lines. "
                                                "Both: overlays both (default)."
                                            ),
                                        ], md=6),
                                    ], className="mb-2"),
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Label("Figure size", size="sm"),
                                            dbc.Select(id="avg-viz-figsize", size="sm", value="medium",
                                                       options=[
                                                           {"label": "Small (8x5)", "value": "small"},
                                                           {"label": "Medium (12.5x8)", "value": "medium"},
                                                           {"label": "Large (16x10)", "value": "large"},
                                                       ]),
                                        ], md=4),
                                        dbc.Col([
                                            dbc.Label("DPI", size="sm"),
                                            dbc.Select(id="avg-viz-dpi", size="sm", value="150",
                                                       options=[
                                                           {"label": "100 (fast)", "value": "100"},
                                                           {"label": "150 (default)", "value": "150"},
                                                           {"label": "200 (high)", "value": "200"},
                                                           {"label": "300 (print)", "value": "300"},
                                                       ]),
                                        ], md=4),
                                    ], className="mb-2"),
                                    dbc.Button("Plot Neighborhood", id="btn-avg-viz",
                                               color="primary", size="sm", className="mb-2"),
                                    dcc.Loading(
                                        html.Div(id="avg-viz-status", className="small text-muted mb-2"),
                                        type="dot",
                                    ),
                                    html.Div([
                                        dbc.Button("Expand", id="btn-expand-avg-viz",
                                                   color="outline-secondary", size="sm",
                                                   className="mb-2"),
                                        html.Div(id="avg-viz-plot"),
                                    ], className="mt-2"),
                                ], title="Avg Spacing Neighborhood Plot", item_id="acc-avg-viz"),

                            ], start_collapsed=False, always_open=True),
                        ]), className="mt-0"),
                    ]),  # end Analysis tab
                  ], id="right-panel-tabs", active_tab="tab-charts"),
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

# -- Sync tooltip fields to JS global (no server round-trip needed) --
dash.clientside_callback(
    """
    function(fields) {
        window._tooltipFields = fields || ["well_name", "uwi", "bench"];
        return null;
    }
    """,
    Output("tooltip-fields", "title"),  # dummy output (title attr, unused)
    Input("tooltip-fields", "value"),
    prevent_initial_call=False,
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
    Input("user-color-overrides", "data"),
    State("pipeline-result-store", "data"),
    prevent_initial_call=False,
)
def update_trajectory_style(traj_color_by, weight, opacity, bh_color_by, user_overrides, pipeline_result):
    color_map = {}
    if pipeline_result and pipeline_result.get("cache_path"):
        data = load_cached_pipeline(pipeline_result["cache_path"])
        header_df = data["header_df"]
        if traj_color_by == "_uniform":
            color_map = {}
        elif traj_color_by == "spud_year" and "spud_date" in header_df.columns:
            try:
                years = pd.to_datetime(header_df["spud_date"], errors="coerce").dt.year.dropna().astype(int).astype(str).tolist()
                color_map = _build_color_map(years, prop="spud_year")
            except Exception:
                color_map = {}
        elif traj_color_by in header_df.columns:
            color_map = _build_color_map(header_df[traj_color_by].dropna().astype(str).tolist(), prop=traj_color_by)

    # Apply user color overrides on top of auto-generated palette
    if user_overrides and color_map:
        for val, custom_color in user_overrides.items():
            if val in color_map:
                color_map[val] = custom_color

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
    Input("user-color-overrides", "data"),
    State("pipeline-result-store", "data"),
    prevent_initial_call=False,
)
def update_bottomhole_style(bh_color_by, radius, opacity, user_overrides, pipeline_result):
    color_map = {}
    if pipeline_result and pipeline_result.get("cache_path"):
        data = load_cached_pipeline(pipeline_result["cache_path"])
        header_df = data["header_df"]
        if bh_color_by == "_uniform":
            color_map = {}
        elif bh_color_by == "spud_year" and "spud_date" in header_df.columns:
            try:
                years = pd.to_datetime(header_df["spud_date"], errors="coerce").dt.year.dropna().astype(int).astype(str).tolist()
                color_map = _build_color_map(years, prop="spud_year")
            except Exception:
                color_map = {}
        elif bh_color_by in header_df.columns:
            color_map = _build_color_map(header_df[bh_color_by].dropna().astype(str).tolist(), prop=bh_color_by)

    # Apply user color overrides
    if user_overrides and color_map:
        for val, custom_color in user_overrides.items():
            if val in color_map:
                color_map[val] = custom_color

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
    """Build a compact color legend with clickable color swatches.

    Each swatch is an <input type='color'> so the user can pick custom colors
    (Spotfire-style). The color-picker inputs share a pattern-matching ID
    so a single callback can capture changes.
    """
    if not color_map:
        return None
    items = []
    for val, color in sorted(color_map.items()):
        items.append(
            html.Div(
                [
                    dbc.Input(
                        type="color",
                        value=color,
                        id={"type": "legend-color-picker", "label": label, "value": val},
                        style={
                            "width": "16px",
                            "height": "16px",
                            "padding": "0",
                            "border": "1px solid #ccc",
                            "borderRadius": "2px",
                            "marginRight": "4px",
                            "verticalAlign": "middle",
                            "cursor": "pointer",
                            "display": "inline-block",
                        },
                    ),
                    html.Span(val, style={"fontSize": "0.72rem", "verticalAlign": "middle"}),
                ],
                style={"lineHeight": "1.6", "display": "flex", "alignItems": "center"},
            )
        )
    return dbc.Card(
        dbc.CardBody(
            [html.H6(label.replace("_", " ").title(), className="mb-1", style={"fontSize": "0.78rem"})]
            + items,
            className="py-1 px-2",
        ),
        style={
            "maxHeight": "200px",
            "overflowY": "auto",
            "backgroundColor": "rgba(255, 255, 255, 0.88)",
            "backdropFilter": "blur(4px)",
            "boxShadow": "0 1px 4px rgba(0,0,0,0.2)",
        },
    )


# -- Color picker callback: capture user overrides from legend swatches --
@callback(
    Output("user-color-overrides", "data", allow_duplicate=True),
    Input({"type": "legend-color-picker", "label": ALL, "value": ALL}, "value"),
    State({"type": "legend-color-picker", "label": ALL, "value": ALL}, "id"),
    State("user-color-overrides", "data"),
    prevent_initial_call=True,
)
def on_color_picker_change(colors, ids, existing_overrides):
    """Capture color picker changes into the user-color-overrides store."""
    overrides = dict(existing_overrides or {})
    for color_val, id_dict in zip(colors, ids):
        if color_val:
            key = id_dict["value"]  # the legend value (e.g., "parent", "Wolfcamp A")
            overrides[key] = color_val
    return overrides


# -- Toggle legend visibility --
@callback(
    Output("legend-content", "style"),
    Input("btn-toggle-legend", "n_clicks"),
    State("legend-content", "style"),
    prevent_initial_call=True,
)
def toggle_legend(n_clicks, current_style):
    current_style = current_style or {}
    if current_style.get("display") == "none":
        return {k: v for k, v in current_style.items() if k != "display"}
    return {**current_style, "display": "none"}


# -- Dynamically populate color-by dropdowns with all header columns --
_PRIORITY_COLOR_COLS = ["bench", "role", "spud_year", "operator", "rsv_cat"]
_SKIP_COLS = {"uwi", "surface_lat", "surface_lon", "latitude", "longitude", "geometry"}


@callback(
    Output("traj-color-by", "options"),
    Output("traj-color-by", "value"),
    Output("bh-color-by", "options"),
    Output("bh-color-by", "value"),
    Output("gb-color-by", "options"),
    Output("gb-color-by", "value"),
    Output("tooltip-fields", "options"),
    Output("tooltip-fields", "value"),
    Output("chart-hover-fields", "options"),
    Output("chart-hover-fields", "value"),
    # Non-dropdown values (sliders, toggles, stores)
    Output("traj-weight", "value"),
    Output("traj-opacity", "value"),
    Output("bh-radius", "value"),
    Output("bh-opacity", "value"),
    Output("gb-xaxis-mode", "value"),
    Output("gb-toggle-lines", "value"),
    Output("gb-toggle-labels", "value"),
    Output("gb-marker-size", "value"),
    Output("gb-line-width", "value"),
    Output("gb-label-size", "value"),
    Output("user-color-overrides", "data"),
    Output("ui-restore-timestamp", "data"),
    Input("pipeline-result-store", "data"),
    prevent_initial_call=False,
)
def populate_and_restore(pipeline_result):
    """Build color-by dropdown options from all header columns + spud_year."""
    # Base options always available
    base = [{"label": "Uniform", "value": "_uniform"}]

    # 22 outputs total: 5 options + 5 values (dropdowns) + 10 values (sliders/toggles) + 2 stores
    _noop = dash.no_update
    if not pipeline_result or not pipeline_result.get("cache_path"):
        # options, value pairs for 5 dropdowns + 12 remaining
        return (base, _noop, base, _noop, [], _noop, [], _noop, [], _noop,
                *([_noop] * 12))

    data = load_cached_pipeline(pipeline_result["cache_path"])
    header_df = data["header_df"]

    # Gather all string/categorical columns + spud_year (derived)
    candidates = set()
    for col in header_df.columns:
        if col in _SKIP_COLS:
            continue
        if header_df[col].dtype == "object" or col in _PRIORITY_COLOR_COLS:
            candidates.add(col)
    # spud_year is derived from spud_date — always offer it if spud_date exists
    if "spud_date" in header_df.columns:
        candidates.add("spud_year")

    # Build options: priority cols first, then rest alphabetically
    priority = [c for c in _PRIORITY_COLOR_COLS if c in candidates]
    rest = sorted(candidates - set(priority))
    all_cols = priority + rest

    options = [
        {"label": col.replace("_", " ").title(), "value": col}
        for col in all_cols
    ]

    map_options = options + base  # map dropdowns get Uniform at the end
    gb_options = options           # gun barrel doesn't need Uniform

    # Tooltip fields: ALL header columns (including numeric) — user picks what to show
    tooltip_skip = {"geometry"}
    tooltip_priority = ["well_name", "uwi", "bench", "operator", "role", "rsv_cat"]
    tooltip_all = [c for c in tooltip_priority if c in header_df.columns]
    tooltip_rest = sorted([
        c for c in header_df.columns
        if c not in tooltip_skip and c not in tooltip_all
    ])
    tooltip_all += tooltip_rest
    if "spud_year" not in tooltip_all and "spud_date" in header_df.columns:
        tooltip_all.append("spud_year")
    tooltip_options = [
        {"label": col.replace("_", " ").title(), "value": col}
        for col in tooltip_all
    ]

    # Restore saved UI state (options + values set atomically in one callback)
    import time as _time
    ui = load_ui_state(pipeline_result["cache_path"])
    _noop = dash.no_update

    if ui:
        import logging
        logging.getLogger("dashboard").debug(
            "populate_and_restore: traj_color_by=%s, gb_marker_size=%s",
            ui.get("traj_color_by"), ui.get("gb_marker_size"),
        )
        g = lambda k: ui.get(k, _noop)  # noqa: E731
    else:
        g = lambda k: _noop  # noqa: E731

    # Return: (options, value) pairs for 5 dropdowns, then 10 sliders/toggles, then 2 stores
    return (
        map_options,      g("traj_color_by"),       # traj-color-by
        map_options,      g("bh_color_by"),         # bh-color-by
        gb_options,       g("gb_color_by"),          # gb-color-by
        tooltip_options,  g("tooltip_fields"),       # tooltip-fields
        tooltip_options,  g("chart_hover_fields"),   # chart-hover-fields
        # Sliders / toggles
        g("traj_weight"), g("traj_opacity"),
        g("bh_radius"),   g("bh_opacity"),
        g("gb_xaxis_mode"),
        g("gb_toggle_lines"), g("gb_toggle_labels"),
        g("gb_marker_size"), g("gb_line_width"), g("gb_label_size"),
        # Stores
        g("user_color_overrides"),
        _time.time() if ui else _noop,               # ui-restore-timestamp
    )


# ---------------------------------------------------------------------------
# Auto-save UI state to cache on any settings change
# ---------------------------------------------------------------------------
_UI_SAVE_KEYS = [
    "traj_color_by", "bh_color_by", "traj_weight", "traj_opacity",
    "bh_radius", "bh_opacity", "tooltip_fields",
    "gb_xaxis_mode", "gb_color_by", "gb_toggle_lines", "gb_toggle_labels",
    "gb_marker_size", "gb_line_width", "gb_label_size",
    "chart_hover_fields", "user_color_overrides",
]


# -- Mark dirty when any UI setting changes (enables Save button) --
@callback(
    Output("btn-save-ui", "disabled"),
    Output("btn-save-ui", "color"),
    Output("btn-save-ui", "children"),
    Input("traj-color-by", "value"),
    Input("bh-color-by", "value"),
    Input("traj-weight", "value"),
    Input("traj-opacity", "value"),
    Input("bh-radius", "value"),
    Input("bh-opacity", "value"),
    Input("tooltip-fields", "value"),
    Input("gb-xaxis-mode", "value"),
    Input("gb-color-by", "value"),
    Input("gb-toggle-lines", "value"),
    Input("gb-toggle-labels", "value"),
    Input("gb-marker-size", "value"),
    Input("gb-line-width", "value"),
    Input("gb-label-size", "value"),
    Input("chart-hover-fields", "value"),
    Input("user-color-overrides", "data"),
    State("ui-restore-timestamp", "data"),
    prevent_initial_call=True,
)
def mark_ui_dirty(*args):
    """Enable Save button when any setting changes. Skip during restore cascade."""
    import time as _time
    restore_ts = args[-1]
    if restore_ts and (_time.time() - restore_ts) < 5.0:
        return dash.no_update, dash.no_update, dash.no_update
    return False, "success", "Save Settings"


# -- Save button: persist UI state to companion JSON on click --
@callback(
    Output("btn-save-ui", "disabled", allow_duplicate=True),
    Output("btn-save-ui", "color", allow_duplicate=True),
    Output("btn-save-ui", "children", allow_duplicate=True),
    Input("btn-save-ui", "n_clicks"),
    State("traj-color-by", "value"),
    State("bh-color-by", "value"),
    State("traj-weight", "value"),
    State("traj-opacity", "value"),
    State("bh-radius", "value"),
    State("bh-opacity", "value"),
    State("tooltip-fields", "value"),
    State("gb-xaxis-mode", "value"),
    State("gb-color-by", "value"),
    State("gb-toggle-lines", "value"),
    State("gb-toggle-labels", "value"),
    State("gb-marker-size", "value"),
    State("gb-line-width", "value"),
    State("gb-label-size", "value"),
    State("chart-hover-fields", "value"),
    State("user-color-overrides", "data"),
    State("pipeline-result-store", "data"),
    prevent_initial_call=True,
)
def save_ui_on_click(
    n_clicks,
    traj_color_by, bh_color_by, traj_weight, traj_opacity,
    bh_radius, bh_opacity, tooltip_fields,
    gb_xaxis_mode, gb_color_by, gb_toggle_lines, gb_toggle_labels,
    gb_marker_size, gb_line_width, gb_label_size,
    chart_hover_fields, user_color_overrides,
    pipeline_result,
):
    """Save all UI settings to companion JSON when user clicks Save."""
    if not pipeline_result or not pipeline_result.get("cache_path"):
        return True, "secondary", "Save Settings"

    ui = dict(zip(_UI_SAVE_KEYS, [
        traj_color_by, bh_color_by, traj_weight, traj_opacity,
        bh_radius, bh_opacity, tooltip_fields,
        gb_xaxis_mode, gb_color_by, gb_toggle_lines, gb_toggle_labels,
        gb_marker_size, gb_line_width, gb_label_size,
        chart_hover_fields, user_color_overrides,
    ]))
    save_ui_state(pipeline_result["cache_path"], ui)
    import logging
    logging.getLogger("dashboard").info("UI settings saved by user click")
    return True, "outline-success", "Saved!"


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

    logger.debug("load_map_layers: loading from %s", pipeline_result["cache_path"])
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
    logger.debug("load_map_layers: %d trajectories, %d bottomholes", len(gdf_traj), len(gdf_bh))

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
    _log.debug("on_well_click: triggered=%s uwi=%s type=%s", triggered, clicked_uwi, type(clicked_uwi).__name__)
    if not clicked_uwi:
        return no_change, click_counts
    # Ensure string
    clicked_uwi = str(clicked_uwi)

    # Only return the clicked well — no neighborhood expansion.
    # The GB uses Spotfire data-limiting: selected wells = well_i set.
    # Shape selection (polygon/line/circle) selects multiple wells.
    _log.debug("on_well_click: uwi=%s selected=1 well", clicked_uwi)
    return {"clicked_uwi": clicked_uwi, "neighborhood_uwis": [clicked_uwi]}, click_counts


@callback(
    Output("gun-barrel-chart", "figure"),
    Input("selected-wells-store", "data"),
    Input("gb-xaxis-mode", "value"),
    Input("gb-color-by", "value"),
    Input("gb-toggle-lines", "value"),
    Input("gb-toggle-labels", "value"),
    Input("gb-marker-size", "value"),
    Input("gb-line-width", "value"),
    Input("gb-label-size", "value"),
    Input("chart-hover-fields", "value"),
    Input("user-color-overrides", "data"),
    State("pipeline-result-store", "data"),
    prevent_initial_call=True,
)
def update_gun_barrel(selected, x_col, color_by, lines_toggle, labels_toggle, marker_size, line_width, label_size, hover_fields, user_overrides, pipeline_result):
    import logging
    _log = logging.getLogger("dashboard")
    try:
        return _update_gun_barrel_inner(selected, x_col, color_by, lines_toggle, labels_toggle, marker_size, line_width, label_size, hover_fields, user_overrides, pipeline_result)
    except Exception as exc:
        _log.exception("Gun barrel error: %s", exc)
        return empty_figure(f"Error: {exc}")


def _update_gun_barrel_inner(selected, x_col, color_by, lines_toggle, labels_toggle, marker_size, line_width, label_size, hover_fields, user_overrides, pipeline_result):
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
        marker_size=int(marker_size or 12),
        line_width=float(line_width or 1.5),
        label_size=int(label_size or 9),
        hover_fields=hover_fields,
        user_color_overrides=user_overrides,
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

        # Production legend: always UWI + well_name (concise)
        header_df = data.get("header_df")
        _name_lookup = {}
        if header_df is not None and "uwi" in header_df.columns and "well_name" in header_df.columns:
            _name_lookup = header_df.set_index("uwi")["well_name"].astype(str).to_dict()

        fig = go.Figure()
        for uwi, grp in prod_sel.groupby("uwi"):
            wname = _name_lookup.get(str(uwi), "")
            trace_name = f"{uwi} | {wname}" if wname and wname not in ("", "nan") else str(uwi)
            fig.add_trace(go.Scatter(
                x=grp["months"], y=grp[cum_col],
                mode="lines", name=trace_name,
            ))
        fig.update_layout(
            xaxis_title="Months since first production",
            yaxis_title=f"Cumulative {product.title()} ({unit})",
            template="plotly_white",
            hovermode="x unified",
            margin=dict(t=30, b=50, l=60, r=20),
            legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="left", x=0, font=dict(size=10)),
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

        # Production legend: always UWI + well_name (concise)
        header_df = data.get("header_df")
        _name_lookup = {}
        if header_df is not None and "uwi" in header_df.columns and "well_name" in header_df.columns:
            _name_lookup = header_df.set_index("uwi")["well_name"].astype(str).to_dict()

        fig = go.Figure()
        for uwi, grp in prod_sel.groupby("uwi"):
            wname = _name_lookup.get(str(uwi), "")
            trace_name = f"{uwi} | {wname}" if wname and wname not in ("", "nan") else str(uwi)
            fig.add_trace(go.Scatter(
                x=grp["prod_date"], y=grp[col],
                mode="lines", name=trace_name,
            ))
        fig.update_layout(
            xaxis_title="Production Date",
            yaxis_title=f"Daily {product.title()} ({unit})",
            template="plotly_white",
            hovermode="x unified",
            margin=dict(t=30, b=50, l=60, r=20),
            legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="left", x=0, font=dict(size=10)),
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
    _log.debug("select_wells_by_shape: geom=%s props=%s", geom_type, props)
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
    _log.debug("select_wells_by_shape: checking trajectories against %s (bounds=%s)",
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

    _log.debug("select_wells_by_shape: %d wells found in shape (geom_type=%s)",
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
    Output("geojson-edges", "data", allow_duplicate=True),
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

    no_update_all = (dash.no_update,) * 19  # 19 outputs now (added edges)

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
        _EMPTY_GEOJSON,  # clear edges
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
# Neighborhood edge lines on map
# ---------------------------------------------------------------------------

@callback(
    Output("geojson-edges", "data"),
    Output("edge-legend", "children"),
    Input("selected-wells-store", "data"),
    Input("edge-toggle", "value"),
    State("pipeline-result-store", "data"),
    prevent_initial_call=True,
)
def build_neighborhood_edges(selected, edge_enabled, pipeline_result):
    """Draw lines from selected wells to their spacing neighbors on the map."""
    if not edge_enabled or not selected or not selected.get("neighborhood_uwis") or not pipeline_result:
        return _EMPTY_GEOJSON, None

    uwis = [str(u) for u in selected["neighborhood_uwis"]]

    try:
        data = load_cached_pipeline(pipeline_result["cache_path"])
        IK = data.get("df_ik_pairs")
        if IK is None or IK.empty:
            return _EMPTY_GEOJSON, None

        # Find all pairs involving selected wells
        IK = IK.copy()
        IK["well_i"] = IK["well_i"].astype(str)
        IK["well_k"] = IK["well_k"].astype(str)
        mask = IK["well_i"].isin(uwis) | IK["well_k"].isin(uwis)
        pairs = IK[mask]

        if pairs.empty:
            return _EMPTY_GEOJSON, None

        # Build UWI → (mid_lon, mid_lat) lookup from lateral midpoint
        # Midpoint = mean of all lateral survey stations per well
        lateral_df = data.get("lateral_df")
        survey_df = data.get("directional_df")
        # Prefer lateral_df (lateral-only) for midpoints; fall back to full survey
        mid_src = lateral_df if lateral_df is not None and not lateral_df.empty else survey_df
        if mid_src is None or mid_src.empty:
            return _EMPTY_GEOJSON, None

        mid_src = mid_src.copy()
        mid_src["uwi"] = mid_src["uwi"].astype(str)
        midpoints = (
            mid_src.groupby("uwi")[["longitude", "latitude"]]
            .mean()
            .rename(columns={"longitude": "mid_lon", "latitude": "mid_lat"})
        )
        mid_lookup = midpoints.to_dict("index")

        # Determine pair_alignment column name (varies by pipeline version)
        align_col = "pair_alignment" if "pair_alignment" in pairs.columns else "alignment_type"
        dist_col = "horizontal_dist" if "horizontal_dist" in pairs.columns else "hz_effective"

        # Build GeoJSON features — one LineString per pair
        features = []
        alignment_types_seen = set()
        for _, row in pairs.iterrows():
            wi, wk = str(row["well_i"]), str(row["well_k"])
            mid_i = mid_lookup.get(wi)
            mid_k = mid_lookup.get(wk)
            if not mid_i or not mid_k:
                continue

            coords = [
                [mid_i["mid_lon"], mid_i["mid_lat"]],
                [mid_k["mid_lon"], mid_k["mid_lat"]],
            ]

            alignment = str(row.get(align_col, "unknown"))
            alignment_types_seen.add(alignment)
            h_dist = row.get(dist_col, 0)
            v_dist = row.get("vertical_dist", 0)

            bench_i = str(row.get("bench_i", ""))
            bench_k = str(row.get("bench_k", ""))
            same_bench = bench_i == bench_k

            features.append({
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": {
                    "well_i": wi,
                    "well_k": wk,
                    "alignment": alignment,
                    "h_dist": round(float(h_dist)) if pd.notna(h_dist) else 0,
                    "v_dist": round(float(v_dist)) if pd.notna(v_dist) else 0,
                    "same_bench": same_bench,
                    "bench_i": bench_i,
                    "bench_k": bench_k,
                },
            })

        # Build edge legend
        edge_colors = {
            "parallel_like": ("#2196F3", "Parallel"),
            "perpendicular": ("#4CAF50", "Perpendicular"),
            "oblique": ("#FF5722", "Oblique"),
        }
        legend_items = []
        for align_key, (color, label) in edge_colors.items():
            if align_key in alignment_types_seen:
                legend_items.append(
                    html.Div([
                        html.Span(style={
                            "display": "inline-block", "width": "20px", "height": "3px",
                            "backgroundColor": color, "marginRight": "6px",
                            "verticalAlign": "middle", "borderTop": f"2px dashed {color}",
                        }),
                        html.Span(f"{label} ({sum(1 for f in features if f['properties']['alignment'] == align_key)})",
                                  style={"fontSize": "0.72rem", "verticalAlign": "middle"}),
                    ], style={"lineHeight": "1.6"})
                )

        edge_legend = dbc.Card(
            dbc.CardBody(
                [html.H6("Edges", className="mb-1", style={"fontSize": "0.78rem"})] + legend_items,
                className="py-1 px-2",
            ),
            style={
                "backgroundColor": "rgba(255, 255, 255, 0.88)",
                "backdropFilter": "blur(4px)",
                "boxShadow": "0 1px 4px rgba(0,0,0,0.2)",
            },
        ) if legend_items else None

        return {"type": "FeatureCollection", "features": features}, edge_legend

    except Exception:
        import logging
        logging.getLogger("dashboard").exception("build_neighborhood_edges failed")
        return _EMPTY_GEOJSON, None


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
    return columns, _clean_dates_for_table(ik_display).to_dict("records"), options, visible


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
    return columns, _clean_dates_for_table(GB).to_dict("records"), options, visible


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
    return columns, _clean_dates_for_table(hdr_display).to_dict("records"), options, visible


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
    Output("geojson-edges", "data", allow_duplicate=True),
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
        _EMPTY_GEOJSON,  # clear edges
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
    logger.debug("apply_filters: search=%s benches=%s operators=%s rsv=%s statuses=%s",
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
    logger.debug("apply_filters: %d/%d wells pass filters", shown, total)

    # Rebuild GeoJSON with only filtered wells
    filtered_survey = survey_df[survey_df["uwi"].isin(filtered["uwi"])]
    gdf_traj = build_trajectory_geodataframe(filtered_survey, filtered)
    gdf_bh = build_bottomhole_geodataframe(gdf_traj)
    logger.debug("apply_filters: %d trajectories, %d bottomholes rebuilt", len(gdf_traj), len(gdf_bh))

    count_text = f"Showing {shown:,} of {total:,} wells"

    return (
        filtered_uwis,
        count_text,
        gdf_traj.__geo_interface__ if not gdf_traj.empty else _EMPTY_GEOJSON,
        gdf_bh.__geo_interface__ if not gdf_bh.empty else _EMPTY_GEOJSON,
    )


# ---------------------------------------------------------------------------
# Charts sub-tab toggle (Neighborhood / Statistics)
# ---------------------------------------------------------------------------

@callback(
    Output("subtab-neighborhood-content", "style"),
    Output("subtab-statistics-content", "style"),
    Input("charts-subtabs", "active_tab"),
)
def toggle_charts_subtab(active):
    show = {"display": "block"}
    hide = {"display": "none"}
    if active == "subtab-statistics":
        return hide, show
    return show, hide


# ---------------------------------------------------------------------------
# Fullscreen expand modals for Plotly charts
# ---------------------------------------------------------------------------

@callback(
    Output("modal-gb", "is_open"),
    Output("modal-gb-chart", "figure"),
    Input("btn-expand-gb", "n_clicks"),
    State("gun-barrel-chart", "figure"),
    prevent_initial_call=True,
)
def expand_gb(n_clicks, fig):
    if not fig:
        return False, empty_figure("No gun barrel data.")
    return True, fig


@callback(
    Output("modal-cum", "is_open"),
    Output("modal-cum-chart", "figure"),
    Input("btn-expand-cum", "n_clicks"),
    State("cum-oil-chart", "figure"),
    prevent_initial_call=True,
)
def expand_cum(n_clicks, fig):
    if not fig:
        return False, empty_figure("No production data.")
    return True, fig


@callback(
    Output("modal-daily", "is_open"),
    Output("modal-daily-chart", "figure"),
    Input("btn-expand-daily", "n_clicks"),
    State("daily-oil-chart", "figure"),
    prevent_initial_call=True,
)
def expand_daily(n_clicks, fig):
    if not fig:
        return False, empty_figure("No production data.")
    return True, fig


@callback(
    Output("modal-role-prod", "is_open"),
    Output("modal-role-prod-chart", "figure"),
    Input("btn-expand-role-prod", "n_clicks"),
    State("role-prod-chart", "figure"),
    prevent_initial_call=True,
)
def expand_role_prod(n_clicks, fig):
    if not fig:
        return False, empty_figure("No data.")
    return True, fig


# ---------------------------------------------------------------------------
# Production by Role — box plot
# ---------------------------------------------------------------------------

@callback(
    Output("role-prod-chart", "figure"),
    Input("pipeline-result-store", "data"),
    Input("role-prod-metric", "value"),
    Input("role-prod-facet-bench", "value"),
    prevent_initial_call=True,
)
def update_role_production(pipeline_result, metric, facet_bench):
    """Box plot of a production metric grouped by well role, optionally faceted by bench."""
    import logging
    _log = logging.getLogger("dashboard")
    try:
        if not pipeline_result:
            return empty_figure("Run pipeline first.")

        data = load_cached_pipeline(pipeline_result["cache_path"])
        header_df = data.get("header_df")

        if header_df is None or header_df.empty:
            return empty_figure("No header data.")

        if "role" not in header_df.columns:
            return empty_figure("No 'role' column — run pipeline with role assignment enabled.")

        metric = metric or "cum_oil_180d_per_ft"
        if metric not in header_df.columns:
            return empty_figure(f"Column '{metric}' not found in header data.")

        # Filter to wells that have both role and the metric
        plot_df = header_df[["role", "bench", metric]].dropna(subset=[metric]).copy()
        plot_df = plot_df[plot_df["role"] != "no_eligible_neighbor"]

        if plot_df.empty:
            return empty_figure("No wells with role and production data.")

        # Use role color map
        role_colors = {r: _ROLE_COLORS.get(r, "#607D8B") for r in plot_df["role"].unique()}

        # Desired role order: parent first, then child, infill, etc.
        role_order = ["parent", "child", "infill", "co-developed"]
        existing_roles = [r for r in role_order if r in plot_df["role"].unique()]
        extra_roles = sorted(set(plot_df["role"].unique()) - set(role_order))
        existing_roles.extend(extra_roles)

        # Pretty metric label
        label = metric.replace("_", " ").replace("per ft", "/ft").title()

        facet = "facet" in (facet_bench or [])

        if facet and "bench" in plot_df.columns and plot_df["bench"].nunique() > 1:
            fig = go.Figure()
            benches = sorted(plot_df["bench"].dropna().unique())
            for role in existing_roles:
                role_data = plot_df[plot_df["role"] == role]
                fig.add_trace(go.Box(
                    x=role_data["bench"],
                    y=role_data[metric],
                    name=role,
                    marker_color=role_colors.get(role, "#607D8B"),
                    boxmean="sd",
                ))
            fig.update_layout(
                boxmode="group",
                xaxis_title="Bench",
                yaxis_title=label,
            )
        else:
            fig = go.Figure()
            for role in existing_roles:
                role_data = plot_df[plot_df["role"] == role]
                fig.add_trace(go.Box(
                    y=role_data[metric],
                    name=role,
                    marker_color=role_colors.get(role, "#607D8B"),
                    boxmean="sd",
                ))
            fig.update_layout(
                xaxis_title="Role",
                yaxis_title=label,
            )

        fig.update_layout(
            template="plotly_white",
            margin=dict(t=30, b=50, l=60, r=20),
            legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="left", x=0, font=dict(size=10)),
            yaxis=dict(tickformat=","),
        )
        return fig

    except Exception as exc:
        _log.exception("Role production chart error: %s", exc)
        return empty_figure(f"Error: {exc}")
