"""
Step 2 — Column Mapping
User maps their source columns to canonical names required by src/.

Uses html.Datalist + dbc.Input per row (pattern-matching callbacks):
- Autocomplete suggestions from canonical column list (click or type to filter)
- Free-form text entry allowed (for custom passthrough column names)
- Backspace, clear, and full keyboard control
"""

import dash
import dash_bootstrap_components as dbc
from dash import ALL, Input, Output, State, callback, dcc, html
from rapidfuzz import process as fuzz_process

dash.register_page(__name__, path="/column-map", name="2 Map Columns", order=2)

# ---------------------------------------------------------------------------
# Canonical column lists
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

CANONICAL_BY_FILE = {
    "header":      CANONICAL_HEADER,
    "directional": CANONICAL_DIRECTIONAL,
    "production":  CANONICAL_PRODUCTION,
}

# Columns that MUST be mapped for the pipeline to run.
# Header uwi + directional uwi/uwi12 are enforced by toggle_confirm_button / confirm_mapping.
REQUIRED_CANONICAL = {
    "header":      {"uwi"},
    "directional": {"uwi", "uwi12"},   # one of these is required
    "production":  set(),              # production is optional overall
}

# Enverus templates
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


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

layout = dbc.Container(
    [
        html.H3("Step 2 — Map Columns", className="mb-1"),
        html.P(
            "Match your file's column names to the canonical names used by the spacing engine. "
            "Unmapped rows (no selection) are excluded from loading.",
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
                        [
                            html.Strong("Tip: "),
                            "Search or type in any dropdown. Leave blank to exclude a column. "
                            "Type a custom name to pass it through as-is.",
                        ],
                        color="info",
                        className="mb-3 py-2",
                    ),
                    md=9,
                ),
            ]
        ),

        dbc.Tabs(
            [
                dbc.Tab(html.Div(id="map-rows-header"),      label="Header",      tab_id="tab-header"),
                dbc.Tab(html.Div(id="map-rows-directional"), label="Directional", tab_id="tab-directional"),
                dbc.Tab(html.Div(id="map-rows-production"),  label="Production",  tab_id="tab-production"),
            ],
            active_tab="tab-header",
        ),

        dbc.Alert(id="column-map-error", color="danger", is_open=False, dismissable=True, className="mt-3"),

        html.Hr(),
        dbc.Row(
            [
                dbc.Col(dbc.Button("← Back", href="/", color="secondary", outline=True), width="auto"),
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
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for col in source_cols:
        match = fuzz_process.extractOne(col, canonical_cols, score_cutoff=75)
        if match and match[0] not in used:
            mapping[col] = match[0]
            used.add(match[0])
    return mapping


def _build_rows(file_key: str, source_cols: list[str], prefill: dict[str, str]) -> list:
    """Build one row per source column: datalist + text input for free-form canonical entry."""
    canonical = CANONICAL_BY_FILE[file_key]
    rows = []
    for idx, src in enumerate(source_cols):
        value = prefill.get(src) or ""
        mapped = bool(value)
        datalist_id = f"dl-{file_key}-{idx}"
        rows.append(
            dbc.Row(
                [
                    dbc.Col(
                        html.Code(src, style={"fontSize": "0.85rem"}),
                        width=5,
                        className="d-flex align-items-center",
                    ),
                    dbc.Col(
                        [
                            html.Datalist(
                                id=datalist_id,
                                children=[html.Option(value=c) for c in canonical],
                            ),
                            dbc.Input(
                                id={"type": "map-input", "file": file_key, "src": src},
                                type="text",
                                list=datalist_id,
                                value=value,
                                placeholder="— leave blank to exclude —",
                                debounce=False,
                                size="sm",
                                style={"fontSize": "0.85rem"},
                            ),
                        ],
                        width=7,
                    ),
                ],
                className="mb-1 align-items-center py-1",
                style={"backgroundColor": "#f0fff0" if mapped else "#fffbe6",
                       "borderRadius": "4px"},
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def _panel(file_key: str, store: dict, template_name: str) -> list:
    template = TEMPLATES.get(template_name, TEMPLATES["Custom"])
    source_cols = store.get(f"{file_key}_preview_cols", [])
    if not source_cols:
        return [dbc.Alert(f"No {file_key} file uploaded (optional).", color="light", className="mt-2")]
    prefill = {**_fuzzy_match(source_cols, CANONICAL_BY_FILE[file_key]), **template[file_key]}

    required = REQUIRED_CANONICAL[file_key]
    if required:
        required_note = dbc.Alert(
            [
                html.Strong("★ Required: "),
                html.Span(", ".join(sorted(required))),
                html.Span(" — the pipeline cannot run without these mappings.", className="text-muted"),
            ],
            color="warning",
            className="py-2 mb-2",
        )
    else:
        required_note = None

    col_header = dbc.Row(
        [
            dbc.Col(html.Strong("Your Column", style={"fontSize": "0.8rem"}), width=5),
            dbc.Col(html.Strong("Maps To (canonical)", style={"fontSize": "0.8rem"}), width=7),
        ],
        className="mb-2 text-muted",
    )
    children = [required_note, col_header] if required_note else [col_header]
    return children + _build_rows(file_key, source_cols, prefill)


@callback(
    Output("map-rows-header",      "children"),
    Output("map-rows-directional", "children"),
    Output("map-rows-production",  "children"),
    Input("upload-store",    "data"),
    Input("template-select", "value"),
    prevent_initial_call=False,
)
def build_mapping_rows(store, template_name):
    if not store:
        msg = [dbc.Alert("No data loaded — go back to Step 1.", color="warning")]
        return msg, msg, msg
    return (
        _panel("header",      store, template_name),
        _panel("directional", store, template_name),
        _panel("production",  store, template_name),
    )


@callback(
    Output("btn-confirm-mapping", "disabled"),
    Input({"type": "map-input", "file": "header",      "src": ALL}, "value"),
    Input({"type": "map-input", "file": "directional", "src": ALL}, "value"),
    prevent_initial_call=False,
)
def toggle_confirm_button(header_vals, dir_vals):
    has_uwi_header = "uwi" in (header_vals or [])
    has_uwi_dir    = bool({"uwi", "uwi12"} & set(v for v in (dir_vals or []) if v))
    return not (has_uwi_header and has_uwi_dir)


@callback(
    Output("column-map-store",   "data"),
    Output("colmap-redirect",    "href"),
    Output("column-map-error",   "children"),
    Output("column-map-error",   "is_open"),
    Input("btn-confirm-mapping", "n_clicks"),
    State({"type": "map-input", "file": ALL, "src": ALL}, "value"),
    State({"type": "map-input", "file": ALL, "src": ALL}, "id"),
    prevent_initial_call=True,
)
def confirm_mapping(n_clicks, values, ids):
    mapping: dict[str, dict] = {"header": {}, "directional": {}, "production": {}}
    for val, id_ in zip(values, ids):
        if val:
            mapping[id_["file"]][id_["src"]] = val

    if "uwi" not in mapping["header"].values():
        return dash.no_update, dash.no_update, "Header mapping must include a 'uwi' column.", True
    dir_vals = set(mapping["directional"].values())
    if not dir_vals & {"uwi", "uwi12"}:
        return dash.no_update, dash.no_update, "Directional mapping must include 'uwi' or 'uwi12'.", True

    return mapping, "/configure", "", False
