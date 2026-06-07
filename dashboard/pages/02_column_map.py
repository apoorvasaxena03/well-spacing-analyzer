"""
Step 2 — Column Mapping
User maps their source columns to canonical names required by src/.

Five-column view per source column:
  1. Original column name (as-is from file)
  2. Standardized name (snake_case via convert_to_snake_case)
  3. Sample data (first 3 values from uploaded file)
  4. Canonical mapping (datalist + text input — autocomplete + free-form)
  5. Data type (auto-detected with override)

Unmapped columns can be excluded or passed through as-is (global toggle).
"""

import re
import time

import dash
import dash_bootstrap_components as dbc
from dash import ALL, Input, Output, State, callback, dcc, html
from rapidfuzz import process as fuzz_process

dash.register_page(__name__, path="/column-map", name="2 Map Columns", order=2)

# ---------------------------------------------------------------------------
# convert_to_snake_case — mirrors src/utils/utils.py
# ---------------------------------------------------------------------------

def _to_snake(name: str) -> str:
    """Convert a column name to snake_case (same logic as utils.standardize_column_names)."""
    name = name.strip()
    name = re.sub(r"^ENV", "Env", name)
    name = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", name)
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    name = name.replace(" ", "_")
    name = name.lower()
    name = re.sub(r"_+", "_", name)
    return name


# ---------------------------------------------------------------------------
# Dtype auto-detection from canonical name
# ---------------------------------------------------------------------------

_DTYPE_OPTIONS = [
    {"label": "String",   "value": "str"},
    {"label": "Float",    "value": "float64"},
    {"label": "Integer",  "value": "int64"},
    {"label": "Date",     "value": "datetime"},
    {"label": "Boolean",  "value": "bool"},
    {"label": "Auto",     "value": "auto"},
]

# Canonical name → expected dtype
_CANONICAL_DTYPES: dict[str, str] = {
    # String identifiers
    "uwi": "str", "uwi12": "str", "uwi10": "str",
    "well_name": "str", "lease_name": "str", "operator": "str",
    "bench": "str", "hole_direction": "str", "well_status": "str",
    "rsv_cat": "str",
    # Dates
    "spud_date": "datetime", "first_prod_date": "datetime",
    "last_prod_date": "datetime", "comp_date": "datetime",
    "permit_date": "datetime", "prod_date": "datetime",
    # Floats — coordinates
    "latitude": "float64", "longitude": "float64",
    "surface_lat": "float64", "surface_lon": "float64",
    "md": "float64", "tvd": "float64",
    "azimuth": "float64", "inclination": "float64",
    "deviation_E/W": "float64", "deviation_N/S": "float64",
    # Floats — metrics
    "lateral_length_ft": "float64",
    "oil": "float64", "gas": "float64", "water": "float64",
    "daily_oil": "float64", "daily_gas": "float64", "daily_water": "float64",
    "cum_oil": "float64", "cum_gas": "float64", "cum_water": "float64",
    "peak_oil_bopd": "float64", "peak_gas_mcfd": "float64",
    "fluid_intensity_bbl_per_ft": "float64",
    "proppant_intensity_lbs_per_ft": "float64",
}


def _infer_dtype(canonical_name: str, sample_values: list[str] | None = None) -> str:
    """Infer the best dtype for a column based on its canonical name or sample data."""
    if canonical_name in _CANONICAL_DTYPES:
        return _CANONICAL_DTYPES[canonical_name]
    # Heuristic fallbacks based on name patterns
    name_lower = canonical_name.lower()
    if "uwi" in name_lower or "api" in name_lower:
        return "str"
    if "date" in name_lower or "time" in name_lower:
        return "datetime"
    if any(kw in name_lower for kw in ("_ft", "_bbl", "_mcf", "_lbs", "lat", "lon", "length")):
        return "float64"
    return "auto"


_DTYPE_LABELS = {
    "str": "String", "float64": "Float", "int64": "Integer",
    "datetime": "Date", "bool": "Boolean", "auto": "Auto",
}

_DTYPE_COLORS = {
    "str": "#0d6efd", "float64": "#198754", "int64": "#6f42c1",
    "datetime": "#fd7e14", "bool": "#dc3545", "auto": "#6c757d",
}


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

# Minimum columns to enable the "Confirm" button (hard gate).
CONFIRM_REQUIRED = {
    "header":      {"uwi"},
    "directional": {"uwi", "uwi12"},   # at least one
    "production":  {"uwi", "prod_date"},
}

# Full required column sets by source — shown as guidance in each tab.
# These mirror WellDataLoader.HEADER_REQUIRED_COLS / DIRECTIONAL_REQUIRED_COLS.
PIPELINE_REQUIRED = {
    "header": {
        "uwi", "well_name", "operator", "bench",
        "first_prod_date", "hole_direction",
        "surface_lat", "surface_lon", "lateral_length_ft",
    },
    "directional_ihs": {
        "uwi", "md", "tvd", "inclination", "azimuth",
        "latitude", "longitude",
        "deviation_E/W", "deviation_N/S", "E/W", "N/S",
    },
    "directional_enverus": {
        "uwi12", "md", "tvd", "inclination", "azimuth",
        "latitude", "longitude",
        "deviation_E/W", "deviation_N/S",
    },
    "production": {
        "uwi", "prod_date", "oil", "gas", "water",
    },
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
# IHS templates (typical IHS Enerdeq / PIDM export column names)
IHS_HEADER = {
    "API 14":              "uwi",
    "Lease Name":          "lease_name",
    "Well Name":           "well_name",
    "Operator":            "operator",
    "Formation":           "bench",
    "First Prod Date":     "first_prod_date",
    "Spud Date":           "spud_date",
    "Hole Direction":      "hole_direction",
    "Final Status":        "well_status",
    "Surface Latitude":    "surface_lat",
    "Surface Longitude":   "surface_lon",
    "Lateral Length":      "lateral_length_ft",
}
IHS_DIRECTIONAL = {
    "UWI":              "uwi",
    "MD":               "md",
    "TVD":              "tvd",
    "Inclination":      "inclination",
    "Azimuth":          "azimuth",
    "Latitude":         "latitude",
    "Longitude":        "longitude",
    "Deviation E/W":    "deviation_E/W",
    "Deviation N/S":    "deviation_N/S",
}
IHS_PRODUCTION = {
    "UWI":              "uwi",
    "Production Date":  "prod_date",
    "Monthly Oil":      "oil",
    "Monthly Gas":      "gas",
    "Monthly Water":    "water",
}

# DrillingInfo templates (typical DI Desktop export column names)
DI_HEADER = {
    "API/UWI":                          "uwi",
    "Well Name":                        "well_name",
    "Operator Name":                    "operator",
    "Producing Formation":              "bench",
    "First Production Date":            "first_prod_date",
    "Spud Date":                        "spud_date",
    "Well Type":                        "hole_direction",
    "Surface Hole Latitude (WGS84)":    "surface_lat",
    "Surface Hole Longitude (WGS84)":   "surface_lon",
    "Lateral Length (ft)":              "lateral_length_ft",
    "Peak Oil (BOPD)":                  "peak_oil_bopd",
    "Peak Gas (MCFD)":                  "peak_gas_mcfd",
}
DI_DIRECTIONAL = {
    "API/UWI":          "uwi",
    "Measured Depth":   "md",
    "True Vertical Depth": "tvd",
    "Inclination":      "inclination",
    "Azimuth":          "azimuth",
    "Latitude":         "latitude",
    "Longitude":        "longitude",
}
DI_PRODUCTION = {
    "API/UWI":          "uwi",
    "Production Date":  "prod_date",
    "Oil (BBL)":        "oil",
    "Gas (MCF)":        "gas",
    "Water (BBL)":      "water",
}

TEMPLATES = {
    "Enverus":       {"header": ENVERUS_HEADER, "directional": ENVERUS_DIRECTIONAL, "production": ENVERUS_PRODUCTION},
    "IHS":           {"header": IHS_HEADER, "directional": IHS_DIRECTIONAL, "production": IHS_PRODUCTION},
    "DrillingInfo":  {"header": DI_HEADER, "directional": DI_DIRECTIONAL, "production": DI_PRODUCTION},
    "Custom":        {"header": {}, "directional": {}, "production": {}},
}


# ---------------------------------------------------------------------------
# Layout helpers (must be defined before layout)
# ---------------------------------------------------------------------------

_UNMAPPED_OPTIONS = [
    {"label": "Exclude", "value": "exclude"},
    {"label": "Keep original name", "value": "as-is"},
    {"label": "Use standardized (snake_case)", "value": "standardize"},
]


def _unmapped_radio(file_key: str) -> dbc.Row:
    """Static per-tab radio for unmapped column handling."""
    return dbc.Row(
        [
            dbc.Col(
                html.Small("Unmapped columns:", className="fw-semibold text-muted"),
                width="auto",
                className="d-flex align-items-center",
            ),
            dbc.Col(
                dbc.RadioItems(
                    id=f"unmapped-mode-{file_key}",
                    options=_UNMAPPED_OPTIONS,
                    value="exclude",
                    inline=True,
                    input_class_name="me-1",
                    label_class_name="me-3",
                    style={"fontSize": "0.82rem"},
                ),
            ),
        ],
        className="mt-2 mb-2 align-items-center",
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
                        [
                            html.Strong("Tip: "),
                            "Select a canonical name from the dropdown suggestions, or type a custom name "
                            "to pass it through as-is. Leave blank to exclude a column.",
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
                dbc.Tab(
                    [_unmapped_radio("header"), html.Div(id="map-rows-header")],
                    label="Header", tab_id="tab-header",
                ),
                dbc.Tab(
                    [_unmapped_radio("directional"), html.Div(id="map-rows-directional")],
                    label="Directional", tab_id="tab-directional",
                ),
                dbc.Tab(
                    [_unmapped_radio("production"), html.Div(id="map-rows-production")],
                    label="Production", tab_id="tab-production",
                ),
            ],
            active_tab="tab-header",
        ),

        dbc.Alert(id="column-map-error", color="danger", is_open=False, dismissable=True, className="mt-3"),
        dbc.Alert(id="column-map-success", color="success", is_open=False, dismissable=True, className="mt-3"),

        html.Hr(),
        dbc.Row(
            [
                dbc.Col(dbc.Button("← Back", href="/", color="secondary", outline=True), width="auto"),
                dbc.Col(
                    [
                        dbc.Button(
                            "Confirm Mapping",
                            id="btn-confirm-mapping",
                            color="primary",
                            disabled=True,
                            className="me-2",
                        ),
                        dbc.Button(
                            "Configure & Next →",
                            id="btn-go-configure",
                            href="/configure",
                            color="success",
                            disabled=True,
                        ),
                    ],
                    className="text-end",
                ),
            ],
            justify="between",
        ),
    ],
    fluid=True,
    className="py-4",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fuzzy_match(source_cols: list[str], canonical_cols: list[str]) -> dict[str, str]:
    """Fuzzy-match source columns to canonical names using standardized forms."""
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for col in source_cols:
        # Try matching on the snake_case form — much better for CamelCase source names
        snake = _to_snake(col)
        match = fuzz_process.extractOne(snake, canonical_cols, score_cutoff=75)
        if match and match[0] not in used:
            mapping[col] = match[0]
            used.add(match[0])
    return mapping


def _build_rows(
    file_key: str,
    source_cols: list[str],
    prefill: dict[str, str],
    sample_values: dict[str, list[str]],
    dtype_overrides: dict[str, str] | None = None,
) -> list:
    """Build one row per source column with 5 visual columns:
    Original | Standardized | Sample Values | Canonical Mapping | Dtype
    """
    canonical = CANONICAL_BY_FILE[file_key]
    dtype_overrides = dtype_overrides or {}
    rows = []

    for idx, src in enumerate(source_cols):
        value = prefill.get(src) or ""
        mapped = bool(value)
        snake = _to_snake(src)
        samples = sample_values.get(src, [])
        sample_text = ", ".join(samples) if samples else "—"
        datalist_id = f"dl-{file_key}-{idx}"

        # Auto-detect dtype from canonical mapping (or source name)
        detect_from = value if value else snake
        auto_dtype = _infer_dtype(detect_from, samples)
        current_dtype = dtype_overrides.get(src, auto_dtype)

        # Mark canonical options
        options = [html.Option(value=c) for c in canonical]

        rows.append(
            dbc.Row(
                [
                    # Col 1: Original name
                    dbc.Col(
                        html.Code(src, style={"fontSize": "0.82rem"}),
                        width=2,
                        className="d-flex align-items-center",
                    ),
                    # Col 2: Standardized (snake_case)
                    dbc.Col(
                        html.Span(
                            snake,
                            style={
                                "fontSize": "0.78rem",
                                "color": "#0d6efd" if snake in canonical else "#6c757d",
                                "fontFamily": "monospace",
                            },
                        ),
                        width=2,
                        className="d-flex align-items-center",
                    ),
                    # Col 3: Sample data
                    dbc.Col(
                        html.Small(
                            sample_text,
                            style={"fontSize": "0.72rem", "color": "#888"},
                            className="text-truncate d-block",
                            title=sample_text,
                        ),
                        width=3,
                        className="d-flex align-items-center",
                        style={"overflow": "hidden"},
                    ),
                    # Col 4: Canonical mapping (datalist + input)
                    dbc.Col(
                        [
                            html.Datalist(id=datalist_id, children=options),
                            dbc.Input(
                                id={"type": "map-input", "file": file_key, "src": src},
                                type="text",
                                list=datalist_id,
                                value=value,
                                placeholder="— leave blank to exclude —",
                                debounce=False,
                                size="sm",
                                style={"fontSize": "0.82rem"},
                            ),
                        ],
                        width=3,
                    ),
                    # Col 5: Dtype (auto-detected, overridable)
                    dbc.Col(
                        dbc.Select(
                            id={"type": "dtype-select", "file": file_key, "src": src},
                            options=_DTYPE_OPTIONS,
                            value=current_dtype,
                            size="sm",
                            style={
                                "fontSize": "0.75rem",
                                "color": _DTYPE_COLORS.get(current_dtype, "#6c757d"),
                                "padding": "2px 4px",
                            },
                        ),
                        width=2,
                    ),
                ],
                className="mb-1 align-items-center py-1 gx-2",
                style={
                    "backgroundColor": "#f0fff0" if mapped else "#fffbe6",
                    "borderRadius": "4px",
                },
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def _panel(file_key: str, store: dict, template_name: str, saved_mapping: dict | None = None) -> list:
    """Build the mapping rows for one file tab. Radio lives in the static layout."""
    template = TEMPLATES.get(template_name, TEMPLATES["Custom"])
    source_cols = store.get(f"{file_key}_preview_cols", [])
    sample_values = store.get(f"{file_key}_sample_values", {})

    if not source_cols:
        return [dbc.Alert(f"No {file_key} file uploaded — go back to Step 1.", color="warning", className="mt-2")]

    # If user already confirmed a mapping, restore it; otherwise use template + fuzzy
    if saved_mapping and saved_mapping.get(file_key):
        prefill = saved_mapping[file_key]
    else:
        prefill = {**_fuzzy_match(source_cols, CANONICAL_BY_FILE[file_key]), **template[file_key]}

    children = []

    # Show pipeline-required columns for this tab
    if file_key == "directional":
        # Show both IHS and Enverus requirements so the user knows what's needed
        ihs_req = sorted(PIPELINE_REQUIRED["directional_ihs"])
        env_req = sorted(PIPELINE_REQUIRED["directional_enverus"])
        children.append(
            dbc.Alert(
                [
                    html.Strong("★ Required columns depend on source:"),
                    html.Br(),
                    html.Small([
                        html.Strong("IHS: "),
                        ", ".join(ihs_req),
                    ]),
                    html.Br(),
                    html.Small([
                        html.Strong("Enverus: "),
                        ", ".join(env_req),
                    ]),
                ],
                color="warning",
                className="py-2 mb-2",
            )
        )
    elif file_key in PIPELINE_REQUIRED:
        req = sorted(PIPELINE_REQUIRED[file_key])
        children.append(
            dbc.Alert(
                [
                    html.Strong("★ Required: "),
                    html.Span(", ".join(req)),
                ],
                color="warning",
                className="py-2 mb-2",
            )
        )

    children.append(
        dbc.Row(
            [
                dbc.Col(html.Strong("Your Column", style={"fontSize": "0.78rem"}), width=2),
                dbc.Col(html.Strong("Standardized", style={"fontSize": "0.78rem"}), width=2),
                dbc.Col(html.Strong("Sample Data", style={"fontSize": "0.78rem"}), width=3),
                dbc.Col(html.Strong("Maps To (canonical)", style={"fontSize": "0.78rem"}), width=3),
                dbc.Col(html.Strong("Dtype", style={"fontSize": "0.78rem"}), width=2),
            ],
            className="mb-2 text-muted gx-2",
        )
    )
    return children + _build_rows(file_key, source_cols, prefill, sample_values)


@callback(
    Output("map-rows-header",      "children"),
    Output("map-rows-directional", "children"),
    Output("map-rows-production",  "children"),
    Input("upload-store",    "data"),
    Input("template-select", "value"),
    State("column-map-store", "data"),
    prevent_initial_call=False,
)
def build_mapping_rows(store, template_name, saved_mapping):
    if not store:
        msg = [dbc.Alert("No data loaded — go back to Step 1.", color="warning")]
        return msg, msg, msg
    return (
        _panel("header",      store, template_name, saved_mapping),
        _panel("directional", store, template_name, saved_mapping),
        _panel("production",  store, template_name, saved_mapping),
    )


@callback(
    Output("btn-confirm-mapping", "disabled"),
    Output("column-map-error", "children", allow_duplicate=True),
    Output("column-map-error", "is_open", allow_duplicate=True),
    Input({"type": "map-input", "file": "header",      "src": ALL}, "value"),
    Input({"type": "map-input", "file": "directional", "src": ALL}, "value"),
    Input({"type": "map-input", "file": "production",  "src": ALL}, "value"),
    State({"type": "map-input", "file": "header",      "src": ALL}, "id"),
    State({"type": "map-input", "file": "directional", "src": ALL}, "id"),
    State({"type": "map-input", "file": "production",  "src": ALL}, "id"),
    Input("unmapped-mode-header",      "value"),
    Input("unmapped-mode-directional", "value"),
    Input("unmapped-mode-production",  "value"),
    prevent_initial_call="initial_duplicate",
)
def toggle_confirm_button(header_vals, dir_vals, prod_vals,
                          header_ids, dir_ids, prod_ids,
                          mode_h, mode_d, mode_p):
    # Build effective canonical sets: explicit values + what unmapped mode will produce
    def _effective(vals, ids, mode):
        result = set()
        for val, id_ in zip(vals or [], ids or []):
            if val:
                result.add(val)
            elif mode == "as-is":
                result.add(id_["src"])
            elif mode == "standardize":
                result.add(_to_snake(id_["src"]))
        return result

    h = _effective(header_vals, header_ids, mode_h)
    d = _effective(dir_vals, dir_ids, mode_d)
    p = _effective(prod_vals, prod_ids, mode_p)

    has_uwi_header = "uwi" in h
    has_uwi_dir = bool({"uwi", "uwi12"} & d)

    # Enverus check: if directional uses uwi12, header MUST have both uwi and uwi12
    is_enverus = "uwi12" in d and "uwi" not in d
    if is_enverus and "uwi12" not in h:
        return True, (
            "Enverus directional uses uwi12 — header must also include 'uwi12' "
            "so the pipeline can link 14-digit UWIs to directional surveys."
        ), True

    # Production validation: uwi + prod_date + at least one volume column
    prod_missing = CONFIRM_REQUIRED["production"] - p
    if prod_missing:
        return True, (
            f"Production mapping must include: {', '.join(sorted(prod_missing))}."
        ), True

    has_volume = bool({"oil", "gas", "water"} & p)
    if p and not has_volume:
        return True, (
            "Production mapping must include at least one volume column (oil, gas, or water)."
        ), True

    # Header must include lateral_length_ft for per-ft normalization
    if "lateral_length_ft" not in h:
        return True, (
            "Header mapping must include 'lateral_length_ft' — required for per-foot production normalization."
        ), True

    disabled = not (has_uwi_header and has_uwi_dir)
    return disabled, "", False


@callback(
    Output("column-map-store",        "data"),
    Output("column-map-error",        "children"),
    Output("column-map-error",        "is_open"),
    Output("column-map-success",      "children"),
    Output("column-map-success",      "is_open"),
    Output("btn-go-configure",        "disabled"),
    Output("btn-confirm-mapping",     "disabled", allow_duplicate=True),
    Input("btn-confirm-mapping", "n_clicks"),
    State({"type": "map-input", "file": ALL, "src": ALL}, "value"),
    State({"type": "map-input", "file": ALL, "src": ALL}, "id"),
    State({"type": "dtype-select", "file": ALL, "src": ALL}, "value"),
    State({"type": "dtype-select", "file": ALL, "src": ALL}, "id"),
    State("unmapped-mode-header",      "value"),
    State("unmapped-mode-directional", "value"),
    State("unmapped-mode-production",  "value"),
    prevent_initial_call=True,
)
def confirm_mapping(n_clicks, values, ids, dtype_values, dtype_ids,
                    mode_header, mode_dir, mode_prod):
    modes = {"header": mode_header, "directional": mode_dir, "production": mode_prod}
    mapping: dict[str, dict] = {"header": {}, "directional": {}, "production": {}}

    for val, id_ in zip(values, ids):
        file_key = id_["file"]
        src_col = id_["src"]
        mode = modes.get(file_key, "exclude")
        if val:
            mapping[file_key][src_col] = val
        elif mode == "as-is":
            mapping[file_key][src_col] = src_col
        elif mode == "standardize":
            mapping[file_key][src_col] = _to_snake(src_col)

    # Collect dtype overrides (skip "auto" — only store explicit choices)
    dtype_map: dict[str, dict[str, str]] = {"header": {}, "directional": {}, "production": {}}
    for dt_val, dt_id in zip(dtype_values, dtype_ids):
        if dt_val and dt_val != "auto":
            dtype_map[dt_id["file"]][dt_id["src"]] = dt_val

    if "uwi" not in mapping["header"].values():
        return dash.no_update, "Header mapping must include a 'uwi' column.", True, "", False, True, False
    dir_vals = set(mapping["directional"].values())
    if not dir_vals & {"uwi", "uwi12"}:
        return dash.no_update, "Directional mapping must include 'uwi' or 'uwi12'.", True, "", False, True, False

    mapping["_version"] = int(time.time())
    mapping["_dtypes"] = dtype_map
    n_h = len(mapping["header"])
    n_d = len(mapping["directional"])
    n_p = len(mapping.get("production", {}))
    success_msg = (
        f"Mapping saved ({n_h} header, {n_d} directional, {n_p} production columns). "
        f"Click 'Configure & Next →' to set spacing parameters."
    )
    return mapping, "", False, success_msg, True, False, True


# ---------------------------------------------------------------------------
# Dynamic placeholder updates when unmapped-mode radio changes
# ---------------------------------------------------------------------------

def _make_placeholder(mode: str, src_col: str) -> str:
    """Return the placeholder text for an unmapped input based on radio mode."""
    if mode == "as-is":
        return f"\u2192 will keep: {src_col}"
    elif mode == "standardize":
        return f"\u2192 will use: {_to_snake(src_col)}"
    return "\u2014 leave blank to exclude \u2014"


# One callback per file tab — updates only placeholder, never wipes user values
for _fk in ("header", "directional", "production"):

    @callback(
        Output({"type": "map-input", "file": _fk, "src": ALL}, "placeholder"),
        Input(f"unmapped-mode-{_fk}", "value"),
        State({"type": "map-input", "file": _fk, "src": ALL}, "value"),
        State({"type": "map-input", "file": _fk, "src": ALL}, "id"),
        prevent_initial_call=True,
    )
    def _update_placeholders(mode, values, ids, _file_key=_fk):
        return [
            _make_placeholder(mode, id_["src"]) if not val else "\u2014 leave blank to exclude \u2014"
            for val, id_ in zip(values, ids)
        ]
