"""
Step 2 — Column Mapping
User maps their source columns to canonical names required by src/.

- Pre-populated Enverus template (most common data source)
- Fuzzy auto-suggest for unknown column names (rapidfuzz)
- Separate mapping tables for header, directional, production
- Confirmed mapping → stored in column-map-store
"""

import dash
import dash_bootstrap_components as dbc
import pandas as pd
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
    "uwi", "md", "tvd", "latitude", "longitude",
    "azimuth", "inclination", "deviation_E/W", "deviation_N/S",
]

CANONICAL_PRODUCTION = [
    "uwi", "prod_date", "oil", "gas", "water",
    "daily_oil", "daily_gas", "daily_water",
    "cum_oil", "cum_gas", "cum_water",
]

# Enverus column templates (from well_spacing_RingEnergy_v2.ipynb)
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
    "Enverus":  {"header": ENVERUS_HEADER, "directional": ENVERUS_DIRECTIONAL, "production": ENVERUS_PRODUCTION},
    "Custom":   {"header": {}, "directional": {}, "production": {}},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fuzzy_match(source_cols: list[str], canonical_cols: list[str]) -> dict[str, str]:
    """Auto-suggest canonical matches using rapidfuzz. Returns {source: canonical}."""
    mapping = {}
    for col in source_cols:
        match = fuzz_process.extractOne(col, canonical_cols, score_cutoff=60)
        mapping[col] = match[0] if match else ""
    return mapping


def _build_mapping_table(
    file_key: str,
    source_cols: list[str],
    canonical_cols: list[str],
    prefill: dict[str, str],
) -> dbc.Card:
    """Build an editable DataTable for column mapping."""
    rows = []
    for src in source_cols:
        rows.append({
            "source_column": src,
            "canonical_column": prefill.get(src, ""),
        })

    return dbc.Card(
        dbc.CardBody(
            [
                dash_table.DataTable(
                    id=f"map-table-{file_key}",
                    columns=[
                        {"name": "Your Column", "id": "source_column", "editable": False},
                        {
                            "name": "Maps To (canonical)",
                            "id": "canonical_column",
                            "editable": True,
                            "presentation": "dropdown",
                        },
                    ],
                    data=rows,
                    dropdown={
                        "canonical_column": {
                            "options": [{"label": c, "value": c} for c in [""] + canonical_cols]
                        }
                    },
                    style_cell={"fontSize": "0.85rem", "padding": "4px 8px"},
                    style_header={"fontWeight": "bold", "backgroundColor": "#f8f9fa"},
                    style_data_conditional=[
                        {
                            "if": {"filter_query": '{canonical_column} = ""'},
                            "backgroundColor": "#fff3cd",
                        }
                    ],
                    page_size=20,
                )
            ]
        )
    )


# ---------------------------------------------------------------------------
# Layout
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
                dbc.Tab(html.Div(id="map-panel-header"),      label="Header",      tab_id="tab-header"),
                dbc.Tab(html.Div(id="map-panel-directional"), label="Directional", tab_id="tab-directional"),
                dbc.Tab(html.Div(id="map-panel-production"),  label="Production",  tab_id="tab-production"),
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
# Callbacks
# ---------------------------------------------------------------------------

@callback(
    Output("map-panel-header", "children"),
    Output("map-panel-directional", "children"),
    Output("map-panel-production", "children"),
    Input("upload-store", "data"),
    Input("template-select", "value"),
    prevent_initial_call=True,
)
def build_mapping_tables(store, template_name):
    if not store:
        msg = dbc.Alert("No data loaded. Go back to Step 1.", color="warning")
        return msg, msg, msg

    template = TEMPLATES.get(template_name, TEMPLATES["Custom"])

    def _panel(file_key, preview_col_key, canonical_cols, tmpl_map):
        source_cols = store.get(preview_col_key, [])
        if not source_cols:
            return dbc.Alert(f"No {file_key} file uploaded (optional).", color="light")
        prefill = {**_fuzzy_match(source_cols, canonical_cols), **tmpl_map}
        return _build_mapping_table(file_key, source_cols, canonical_cols, prefill)

    return (
        _panel("header",      "header_preview_cols",      CANONICAL_HEADER,      template["header"]),
        _panel("directional", "directional_preview_cols", CANONICAL_DIRECTIONAL, template["directional"]),
        _panel("production",  "production_preview_cols",  CANONICAL_PRODUCTION,  template["production"]),
    )


@callback(
    Output("btn-confirm-mapping", "disabled"),
    Input("map-table-header", "data"),
    Input("map-table-directional", "data"),
    prevent_initial_call=True,
)
def toggle_confirm_button(header_data, dir_data):
    def _has_uwi(rows):
        if not rows:
            return False
        return any(r.get("canonical_column") == "uwi" for r in rows)

    return not (_has_uwi(header_data) and _has_uwi(dir_data))


@callback(
    Output("column-map-store", "data"),
    Output("colmap-redirect", "href"),
    Output("column-map-error", "children"),
    Output("column-map-error", "is_open"),
    Input("btn-confirm-mapping", "n_clicks"),
    State("map-table-header", "data"),
    State("map-table-directional", "data"),
    State("map-table-production", "data"),
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
    if "uwi" not in mapping["directional"].values() and "uwi12" not in mapping["directional"].values():
        return dash.no_update, dash.no_update, "Directional mapping must include a 'uwi' or 'uwi12' column.", True

    return mapping, "/configure", "", False
