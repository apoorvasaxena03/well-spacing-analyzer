"""
Step 2 — Column Mapping
User maps their source columns to canonical names required by src/.

- Pre-populated Enverus template (most common data source)
- Fuzzy auto-suggest for unknown column names (rapidfuzz)
- Separate mapping tables for header, directional, production
- Confirmed mapping → stored in column-map-store

DataTable components have STATIC IDs in the layout (data updated via callback).
This avoids Dash 4.x client-side validation errors for dynamically created component IDs.
"""

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dash_table, dcc, html
from rapidfuzz import process as fuzz_process

dash.register_page(__name__, path="/column-map", name="2 Map Columns", order=2)

# ---------------------------------------------------------------------------
# Canonical columns
# ---------------------------------------------------------------------------

CANONICAL_HEADER = [
    "uwi", "well_name", "operator", "bench", "spud_date",
    "first_prod_date", "hole_direction", "rsv_cat",
    "surface_lat", "surface_lon", "lateral_length_ft",
    "peak_oil_bopd", "peak_gas_mcfd",
]

CANONICAL_DIRECTIONAL = [
    "uwi", "uwi12", "md", "tvd", "latitude", "longitude",
    "azimuth", "inclination", "deviation_E/W", "deviation_N/S",
]

CANONICAL_PRODUCTION = [
    "uwi", "prod_date", "oil", "gas", "water",
    "daily_oil", "daily_gas", "daily_water",
    "cum_oil", "cum_gas", "cum_water",
]

# Enverus column templates
ENVERUS_HEADER = {
    "API_UWI_14_Unformatted": "uwi",
    "ENVInterval":             "bench",
    "FirstProdDate":           "first_prod_date",
    "Latitude":                "surface_lat",
    "Longitude":               "surface_lon",
    "WellName":                "well_name",
    "Operator":                "operator",
    "SpudDate":                "spud_date",
    "HoleDirection":           "hole_direction",
    "LateralLength_FT":        "lateral_length_ft",
    "PeakOil_BOPD":            "peak_oil_bopd",
    "PeakGas_MCFD":            "peak_gas_mcfd",
}

ENVERUS_DIRECTIONAL = {
    "API_UWI_12_Unformatted": "uwi12",
    "MeasuredDepth_FT":        "md",
    "TVD_FT":                  "tvd",
    "E_W":                     "deviation_E/W",
    "N_S":                     "deviation_N/S",
    "Azimuth":                 "azimuth",
    "Inclination":             "inclination",
    "Latitude":                "latitude",
    "Longitude":               "longitude",
}

ENVERUS_PRODUCTION = {
    "API_UWI_14_Unformatted": "uwi",
    "ProdDate":                "prod_date",
    "LiquidBBL":               "oil",
    "GasMCF":                  "gas",
    "WaterBBL":                "water",
}

TEMPLATES = {
    "Enverus": {"header": ENVERUS_HEADER, "directional": ENVERUS_DIRECTIONAL, "production": ENVERUS_PRODUCTION},
    "Custom":  {"header": {}, "directional": {}, "production": {}},
}

# Shared DataTable column definitions and style
_TABLE_COLUMNS = [
    {"name": "Your Column",          "id": "source_column",    "editable": False},
    {"name": "Maps To (canonical)",  "id": "canonical_column", "editable": True, "presentation": "dropdown"},
]
_TABLE_STYLE_CELL        = {"fontSize": "0.85rem", "padding": "4px 8px"}
_TABLE_STYLE_HEADER      = {"fontWeight": "bold", "backgroundColor": "#f8f9fa"}
_TABLE_STYLE_CONDITIONAL = [{"if": {"filter_query": '{canonical_column} = ""'}, "backgroundColor": "#fff3cd"}]


def _dropdown_options(canonical_cols: list[str]) -> dict:
    return {"canonical_column": {"options": [{"label": c, "value": c} for c in [""] + canonical_cols]}}


# ---------------------------------------------------------------------------
# Layout — DataTables have STATIC IDs; data is populated via callback
# ---------------------------------------------------------------------------

layout = dbc.Container(
    [
        html.H3("Step 2 — Map Columns", className="mb-1"),
        html.P(
            "Match your file's column names to the canonical names used by the spacing engine.",
            className="text-muted mb-3",
        ),

        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Source template:", className="fw-semibold"),
                        dbc.Select(
                            id="template-select",
                            options=[{"label": k, "value": k} for k in TEMPLATES],
                            value="Enverus",
                            className="mb-3",
                        ),
                    ],
                    md=3,
                ),
                dbc.Col(
                    dbc.Alert(
                        "Yellow rows are unmapped — they will be ignored by the spacing engine.",
                        color="warning",
                        className="mb-3 py-2",
                    ),
                    md=9,
                ),
            ]
        ),

        dbc.Tabs(
            [
                dbc.Tab(
                    [
                        html.Div(id="map-panel-header-msg", className="mt-2"),
                        dash_table.DataTable(
                            id="map-table-header",
                            columns=_TABLE_COLUMNS,
                            data=[],
                            editable=True,
                            dropdown=_dropdown_options(CANONICAL_HEADER),
                            style_cell=_TABLE_STYLE_CELL,
                            style_header=_TABLE_STYLE_HEADER,
                            style_data_conditional=_TABLE_STYLE_CONDITIONAL,
                            page_size=25,
                        ),
                    ],
                    label="Header",
                    tab_id="tab-header",
                    className="pt-3",
                ),
                dbc.Tab(
                    [
                        html.Div(id="map-panel-directional-msg", className="mt-2"),
                        dash_table.DataTable(
                            id="map-table-directional",
                            columns=_TABLE_COLUMNS,
                            data=[],
                            editable=True,
                            dropdown=_dropdown_options(CANONICAL_DIRECTIONAL),
                            style_cell=_TABLE_STYLE_CELL,
                            style_header=_TABLE_STYLE_HEADER,
                            style_data_conditional=_TABLE_STYLE_CONDITIONAL,
                            page_size=25,
                        ),
                    ],
                    label="Directional",
                    tab_id="tab-directional",
                    className="pt-3",
                ),
                dbc.Tab(
                    [
                        html.Div(id="map-panel-production-msg", className="mt-2"),
                        dash_table.DataTable(
                            id="map-table-production",
                            columns=_TABLE_COLUMNS,
                            data=[],
                            editable=True,
                            dropdown=_dropdown_options(CANONICAL_PRODUCTION),
                            style_cell=_TABLE_STYLE_CELL,
                            style_header=_TABLE_STYLE_HEADER,
                            style_data_conditional=_TABLE_STYLE_CONDITIONAL,
                            page_size=25,
                        ),
                    ],
                    label="Production",
                    tab_id="tab-production",
                    className="pt-3",
                ),
            ],
            active_tab="tab-header",
        ),

        dbc.Alert(id="column-map-error", color="danger", is_open=False, dismissable=True, className="mt-3"),

        html.Hr(),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Button("← Back", href="/", color="secondary", outline=True),
                    width="auto",
                ),
                dbc.Col(
                    dbc.Button(
                        "Confirm Mapping & Next →",
                        id="btn-confirm-mapping",
                        color="primary",
                        disabled=True,
                    ),
                    className="text-end",
                ),
            ],
            justify="between",
        ),
        dcc.Location(id="colmap-redirect", refresh=True),
    ],
    fluid=True,
    className="py-4",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fuzzy_match(source_cols: list[str], canonical_cols: list[str]) -> dict[str, str]:
    """
    Auto-suggest canonical matches. Score cutoff 75 — high enough to catch
    obvious matches (well_name→well_name, uwi→uwi) while avoiding false
    positives (well_status→well_name, permit_date→spud_date).
    """
    mapping = {}
    used_canonical: set[str] = set()
    for col in source_cols:
        match = fuzz_process.extractOne(col, canonical_cols, score_cutoff=75)
        if match and match[0] not in used_canonical:
            mapping[col] = match[0]
            used_canonical.add(match[0])
        else:
            mapping[col] = ""
    return mapping


def _build_rows(source_cols: list[str], canonical_cols: list[str], prefill: dict[str, str]) -> list[dict]:
    return [
        {"source_column": src, "canonical_column": prefill.get(src, "")}
        for src in source_cols
    ]


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@callback(
    Output("map-table-header",       "data"),
    Output("map-table-directional",  "data"),
    Output("map-table-production",   "data"),
    Output("map-panel-header-msg",      "children"),
    Output("map-panel-directional-msg", "children"),
    Output("map-panel-production-msg",  "children"),
    Input("upload-store",    "data"),
    Input("template-select", "value"),
    prevent_initial_call=False,   # must fire on page navigation with existing store
)
def build_mapping_tables(store, template_name):
    if not store:
        msg = dbc.Alert("No data loaded — go back to Step 1 and upload your files.", color="warning")
        return [], [], [], msg, msg, msg

    template = TEMPLATES.get(template_name, TEMPLATES["Custom"])

    def _rows(preview_key, canonical_cols, tmpl_map):
        source_cols = store.get(preview_key, [])
        if not source_cols:
            return [], dbc.Alert("No file uploaded for this dataset (optional).", color="light", className="mb-2")
        prefill = {**_fuzzy_match(source_cols, canonical_cols), **tmpl_map}
        return _build_rows(source_cols, canonical_cols, prefill), None

    h_rows,  h_msg  = _rows("header_preview_cols",      CANONICAL_HEADER,      template["header"])
    d_rows,  d_msg  = _rows("directional_preview_cols", CANONICAL_DIRECTIONAL, template["directional"])
    p_rows,  p_msg  = _rows("production_preview_cols",  CANONICAL_PRODUCTION,  template["production"])

    return h_rows, d_rows, p_rows, h_msg, d_msg, p_msg


@callback(
    Output("btn-confirm-mapping", "disabled"),
    Input("map-table-header",      "data"),
    Input("map-table-directional", "data"),
    prevent_initial_call=False,
)
def toggle_confirm_button(header_data, dir_data):
    def _has_uwi(rows):
        return bool(rows) and any(r.get("canonical_column") == "uwi" for r in rows)
    def _has_uwi_or_uwi12(rows):
        return bool(rows) and any(r.get("canonical_column") in ("uwi", "uwi12") for r in rows)
    return not (_has_uwi(header_data) and _has_uwi_or_uwi12(dir_data))


@callback(
    Output("column-map-store",      "data"),
    Output("colmap-redirect",       "href"),
    Output("column-map-error",      "children"),
    Output("column-map-error",      "is_open"),
    Input("btn-confirm-mapping",    "n_clicks"),
    State("map-table-header",       "data"),
    State("map-table-directional",  "data"),
    State("map-table-production",   "data"),
    prevent_initial_call=True,
)
def confirm_mapping(n_clicks, header_rows, dir_rows, prod_rows):
    def _to_map(rows):
        if not rows:
            return {}
        return {r["source_column"]: r["canonical_column"] for r in rows if r.get("canonical_column")}

    mapping = {
        "header":      _to_map(header_rows),
        "directional": _to_map(dir_rows),
        "production":  _to_map(prod_rows or []),
    }

    if "uwi" not in mapping["header"].values():
        return dash.no_update, dash.no_update, "Header mapping must include a 'uwi' column.", True
    dir_vals = set(mapping["directional"].values())
    if not dir_vals & {"uwi", "uwi12"}:
        return dash.no_update, dash.no_update, "Directional mapping must include a 'uwi' or 'uwi12' column.", True

    return mapping, "/configure", "", False
