"""
Step 3 — Configure
Set spacing engine parameters before calculation.
UTM zone is auto-detected; all params stored in config-store.
RSV classification cutoffs are configurable with state-specific defaults.
"""

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dcc, html

dash.register_page(__name__, path="/configure", name="3 Configure", order=3)

# ---------------------------------------------------------------------------
# State-specific RSV cutoff reference data (from regulatory research)
# ---------------------------------------------------------------------------
_STATE_CUTOFFS = [
    {
        "state": "Texas (RRC)",
        "prod_cutoff": "12 months",
        "duc_age": "2 years (permit to spud)",
        "permit_window": "2 years",
        "notes": "Shut-in < 12 mo; inactive > 12 mo triggers Rule 15. "
                 "Permit W-1 valid 2 yr from approval.",
    },
    {
        "state": "New Mexico (OCD)",
        "prod_cutoff": "12 months",
        "duc_age": "2 years",
        "permit_window": "2 years",
        "notes": "1 yr continuous inactivity → must plug or get TA approval. "
                 "State lands: 15 mo → inactive well list.",
    },
    {
        "state": "Oklahoma (OCC)",
        "prod_cutoff": "12 months",
        "duc_age": "3 years",
        "permit_window": "2 years + 1 yr ext",
        "notes": "New shut-ins: max 7 yr inactive. "
                 "Pre-existing: 10 yr before plug requirement.",
    },
    {
        "state": "North Dakota (NDIC)",
        "prod_cutoff": "12 months",
        "duc_age": "2 years",
        "permit_window": "1 year",
        "notes": "1 yr no production → abandoned status. "
                 "Must return to production or plug within 6 mo.",
    },
    {
        "state": "Colorado (COGCC)",
        "prod_cutoff": "12 months",
        "duc_age": "2 years",
        "permit_window": "2 years",
        "notes": "Inactive > 12 mo triggers bonding/plugging review.",
    },
    {
        "state": "Wyoming (WOGCC)",
        "prod_cutoff": "12 months",
        "duc_age": "2 years",
        "permit_window": "2 years",
        "notes": "Standard 12-month inactive threshold. "
                 "TA status requires WOGCC approval.",
    },
    {
        "state": "Federal (BLM)",
        "prod_cutoff": "—",
        "duc_age": "—",
        "permit_window": "2 years + 2 yr ext",
        "notes": "APD valid 2 yr or until lease expires. "
                 "BLM may grant 2-yr extension.",
    },
    {
        "state": "Permian Basin (typical)",
        "prod_cutoff": "6 months",
        "duc_age": "3 years",
        "permit_window": "2 years",
        "notes": "Engineering default. Avg DUC completion: 4–6 mo (emerging), "
                 "1.7 mo (Eagle Ford), 2.6 mo (Bakken).",
    },
]


def _reference_table() -> dbc.Accordion:
    """Build a collapsible reference table with state-specific RSV cutoffs."""
    header = html.Thead(
        html.Tr([
            html.Th("State / Region", style={"fontSize": "0.8rem", "width": "15%"}),
            html.Th("Inactive Cutoff", style={"fontSize": "0.8rem", "width": "12%"}),
            html.Th("DUC Age", style={"fontSize": "0.8rem", "width": "12%"}),
            html.Th("Permit Window", style={"fontSize": "0.8rem", "width": "13%"}),
            html.Th("Regulatory Notes", style={"fontSize": "0.8rem"}),
        ])
    )
    rows = []
    for s in _STATE_CUTOFFS:
        rows.append(html.Tr([
            html.Td(html.Strong(s["state"]), style={"fontSize": "0.78rem"}),
            html.Td(s["prod_cutoff"], style={"fontSize": "0.78rem"}),
            html.Td(s["duc_age"], style={"fontSize": "0.78rem"}),
            html.Td(s["permit_window"], style={"fontSize": "0.78rem"}),
            html.Td(s["notes"], style={"fontSize": "0.75rem", "color": "#555"}),
        ]))

    return dbc.Accordion(
        dbc.AccordionItem(
            [
                html.P(
                    "Reference data from state regulatory agencies (TX RRC, NM OCD, OK OCC, "
                    "ND NDIC, CO COGCC, WY WOGCC, Federal BLM). Basin-level defaults reflect "
                    "industry engineering practice for unconventional horizontal wells.",
                    className="text-muted small mb-2",
                ),
                dbc.Table(
                    [header, html.Tbody(rows)],
                    bordered=True,
                    size="sm",
                    hover=True,
                    striped=True,
                    className="mb-0",
                ),
                html.P(
                    [
                        html.Small("Sources: "),
                        html.A("TX RRC Rule 15", href="https://www.rrc.texas.gov/oil-and-gas/applications-and-permits/", target="_blank", className="small me-2"),
                        html.A("NM OCD", href="https://www.emnrd.nm.gov/ocd/", target="_blank", className="small me-2"),
                        html.A("BLM APD", href="https://www.blm.gov/programs/energy-and-minerals/oil-and-gas/operations-and-production/permitting/applications-permits-drill", target="_blank", className="small me-2"),
                        html.A("EIA DPR", href="https://www.eia.gov/petroleum/drilling/", target="_blank", className="small me-2"),
                        html.A("IOGCC Idle Well Toolbox", href="https://iogcc.ok.gov/sites/g/files/gmc836/f/iogcc_idle_and_orphan_wells_2021_final_web.pdf", target="_blank", className="small"),
                    ],
                    className="mt-2 mb-0",
                ),
            ],
            title="State-by-State RSV Cutoff Reference",
        ),
        start_collapsed=True,
        className="mb-3",
    )


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

layout = dbc.Container(
    [
        html.H3("Step 3 — Configure", className="mb-1"),
        html.P("Adjust spacing engine parameters. Defaults work for most Midland Basin datasets.", className="text-muted mb-4"),

        # ---- UTM Zone Detection ----
        dbc.Card(
            dbc.CardBody([
                html.H6("UTM Zone", className="mb-3"),
                html.Div(id="cfg-utm-detection"),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label("UTM Zone to use"),
                                dbc.Input(
                                    id="cfg-utm-zone",
                                    placeholder="auto-detected (leave blank)",
                                    value="",
                                ),
                                dbc.FormText("Leave blank to use majority zone. Or type e.g. EPSG:32613 to override."),
                            ],
                            md=4,
                        ),
                    ],
                    className="mt-2",
                ),
            ]),
            className="mb-3",
        ),

        # ---- Card 1: Spacing Engine Parameters ----
        dbc.Card(
            dbc.CardBody(
                [
                    html.H6("Spacing Engine Parameters", className="mb-3"),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label("Max search radius (miles)"),
                                    dbc.Input(
                                        id="cfg-max-distance",
                                        type="number",
                                        value=4.0,
                                        min=0.1, max=20.0, step=0.5,
                                    ),
                                    dbc.FormText("Wells farther apart than this are excluded from pairing."),
                                ],
                                md=4,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Spacing cutoff (ft)"),
                                    dbc.Input(
                                        id="cfg-cutoff-ft",
                                        type="number",
                                        value=5280,
                                        min=100, max=50000, step=100,
                                    ),
                                    dbc.FormText("Pairs with horizontal spacing > cutoff are excluded."),
                                ],
                                md=4,
                            ),
                        ],
                        className="mb-3",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label("Batch size (pairs)"),
                                    dbc.Select(
                                        id="cfg-batch-size",
                                        options=[
                                            {"label": "50,000  (low memory)",    "value": 50_000},
                                            {"label": "200,000 (default)",        "value": 200_000},
                                            {"label": "500,000 (high memory)",    "value": 500_000},
                                        ],
                                        value=200_000,
                                    ),
                                    dbc.FormText("Reduce if you see memory errors on large datasets."),
                                ],
                                md=4,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Directional source"),
                                    dbc.Select(
                                        id="cfg-directional-source",
                                        options=[
                                            {"label": "Auto-detect from mapping", "value": "auto"},
                                            {"label": "IHS (uwi 14-digit)",       "value": "ihs"},
                                            {"label": "Enverus (uwi12 12-digit)", "value": "enverus"},
                                        ],
                                        value="auto",
                                    ),
                                    dbc.FormText("Controls required columns and UWI matching."),
                                ],
                                md=4,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Bench filter (optional)"),
                                    dbc.Input(
                                        id="cfg-bench-filter",
                                        placeholder="e.g. WOLFCAMP A, SPRABERRY",
                                        value="",
                                    ),
                                    dbc.FormText("Comma-separated. Leave blank to include all."),
                                ],
                                md=4,
                            ),
                        ],
                    ),
                ]
            ),
            className="mb-3",
        ),

        # ---- Card 2: RSV Classification ----
        dbc.Card(
            dbc.CardBody(
                [
                    html.H6("RSV Category Classification & Filter", className="mb-3"),
                    html.P(
                        "If your header data has 'well_status' and 'spud_date', the pipeline auto-computes "
                        "rsv_cat using these cutoffs. If rsv_cat already exists in your data, it is used as-is.",
                        className="text-muted small mb-3",
                    ),
                    dbc.Row(
                        [
                            # Cutoff inputs
                            dbc.Col(
                                [
                                    dbc.Label("Production cutoff (months)"),
                                    dbc.Input(
                                        id="cfg-prod-cutoff-months",
                                        type="number",
                                        value=6,
                                        min=1, max=60, step=1,
                                    ),
                                    dbc.FormText(
                                        "Months without production to classify as 02PDNP. "
                                        "TX/NM/ND regulatory: 12 mo. Engineering default: 6 mo."
                                    ),
                                ],
                                md=4,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("DUC age threshold (years)"),
                                    dbc.Input(
                                        id="cfg-duc-age-years",
                                        type="number",
                                        value=3,
                                        min=1, max=10, step=1,
                                    ),
                                    dbc.FormText(
                                        "Years after spud with no completion → 'Old DUC'. "
                                        "Permian avg completion: 4-6 mo; 3 yr is conservative."
                                    ),
                                ],
                                md=4,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Permit window (years)"),
                                    dbc.Input(
                                        id="cfg-permit-window-years",
                                        type="number",
                                        value=2,
                                        min=1, max=10, step=1,
                                    ),
                                    dbc.FormText(
                                        "Permits older than this → 'Expired Perm'. "
                                        "TX/NM/Federal: 2 yr. OK: 2+1 yr ext."
                                    ),
                                ],
                                md=4,
                            ),
                        ],
                        className="mb-3",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label("RSV categories to include"),
                                    dbc.Checklist(
                                        id="cfg-rsv-categories",
                                        options=[
                                            {"label": "01PDP — Proved Developed Producing",      "value": "01PDP"},
                                            {"label": "02PA — P&A Producer",                     "value": "02PA"},
                                            {"label": "02PDNP — Proved Developed Non-Producing", "value": "02PDNP"},
                                            {"label": "03PUD — Proved Undeveloped (DUC)",        "value": "03PUD"},
                                            {"label": "Old DUC — Spudded > DUC age, no completion", "value": "Old DUC"},
                                            {"label": "Expired Perm — Permit older than window", "value": "Expired Perm"},
                                        ],
                                        value=["01PDP", "02PA", "02PDNP", "03PUD"],
                                        inline=False,
                                        input_class_name="me-1",
                                        label_class_name="small",
                                        className="ms-1",
                                    ),
                                    dbc.FormText(
                                        "Only wells matching checked categories are included. "
                                        "Uncheck all to skip RSV filtering entirely."
                                    ),
                                ],
                                md=6,
                            ),
                        ],
                    ),
                ]
            ),
            className="mb-3",
        ),

        # ---- Collapsible reference table ----
        _reference_table(),

        dbc.Alert(id="cfg-save-success", color="success", is_open=False, dismissable=True, className="mt-3"),

        html.Hr(),
        dbc.Row(
            [
                dbc.Col(dbc.Button("← Back", href="/column-map", color="secondary", outline=True), width="auto"),
                dbc.Col(
                    [
                        dbc.Button("Save Config", id="btn-save-config", color="primary", className="me-2"),
                        dbc.Button(
                            "Calculate & Next →",
                            id="btn-go-calculate",
                            href="/calculate",
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
# UTM zone basin reference (from GeoSurveyProcessor docstring)
# ---------------------------------------------------------------------------
_BASIN_UTM_ZONES = [
    ("Permian — Delaware Basin",     13, "-105°", "Loving, Winkler, Ward, Reeves, Culberson, Pecos"),
    ("Permian — Midland Basin",      13, "-105°", "Midland, Martin, Howard, Glasscock, Reagan, Upton"),
    ("Permian — Central Basin Plat.", 13, "-105°", "Andrews, Ector, Crane"),
    ("Eagle Ford Shale",             14, "-99°",  "South TX (Karnes, DeWitt, Gonzales, La Salle)"),
    ("Haynesville Shale",            15, "-93°",  "East TX, NW Louisiana"),
    ("Anadarko (STACK/SCOOP)",       14, "-99°",  "Oklahoma, TX Panhandle"),
    ("Williston / Bakken",           13, "-105°", "North Dakota, Montana"),
    ("DJ Basin / Niobrara",          13, "-105°", "Colorado (Weld County), Wyoming"),
    ("Powder River Basin",           13, "-105°", "Wyoming, Montana"),
    ("Uinta Basin",                  12, "-111°", "Utah (Duchesne, Uintah)"),
    ("San Juan Basin",               12, "-111°", "New Mexico, Colorado"),
    ("Marcellus Shale",              17, "-81°",  "Pennsylvania, West Virginia, Ohio"),
    ("Utica Shale",                  17, "-81°",  "Ohio, Pennsylvania, West Virginia"),
    ("Fort Worth / Barnett",         14, "-99°",  "North-Central Texas"),
    ("Gulf Coast Basin",             15, "-93°",  "Texas, Louisiana Coast"),
    ("Austin Chalk",                 14, "-99°",  "South-Central Texas"),
    ("San Joaquin Basin",            10, "-123°", "California (Kern County)"),
    ("Cook Inlet / North Slope",      6, "-147°", "Alaska"),
]


def _utm_basin_table() -> dbc.Accordion:
    """Collapsible basin reference table for UTM zone selection."""
    header = html.Thead(html.Tr([
        html.Th("Basin / Play", style={"fontSize": "0.78rem"}),
        html.Th("Zone", style={"fontSize": "0.78rem", "width": "8%"}),
        html.Th("EPSG", style={"fontSize": "0.78rem", "width": "12%"}),
        html.Th("Central Meridian", style={"fontSize": "0.78rem", "width": "12%"}),
        html.Th("Region / Counties", style={"fontSize": "0.78rem"}),
    ]))
    rows = [
        html.Tr([
            html.Td(b[0], style={"fontSize": "0.76rem"}),
            html.Td(str(b[1]), style={"fontSize": "0.76rem", "fontWeight": "bold"}),
            html.Td(f"EPSG:326{b[1]:02d}", style={"fontSize": "0.76rem", "fontFamily": "monospace"}),
            html.Td(b[2], style={"fontSize": "0.76rem"}),
            html.Td(b[3], style={"fontSize": "0.74rem", "color": "#555"}),
        ])
        for b in _BASIN_UTM_ZONES
    ]
    return dbc.Accordion(
        dbc.AccordionItem(
            dbc.Table([header, html.Tbody(rows)], bordered=True, size="sm", hover=True, striped=True, className="mb-0"),
            title="UTM Zone by Basin Reference Table",
        ),
        start_collapsed=True,
        className="mt-2",
    )


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@callback(
    Output("cfg-utm-detection", "children"),
    Input("upload-store", "data"),
    prevent_initial_call=False,
)
def detect_utm_zones(upload_store):
    """Analyze well longitudes to detect UTM zones and show distribution."""
    if not upload_store:
        return dbc.Alert("Upload data first (Step 1) to auto-detect UTM zones.", color="light")

    # Get preview columns to find the lon column
    header_cols = upload_store.get("header_preview_cols", [])
    sample_vals = upload_store.get("header_sample_values", {})

    # Find longitude column from sample values
    lon_col = None
    for candidate in ["surface_lon", "longitude", "Longitude", "surface_longitude"]:
        if candidate in sample_vals:
            lon_col = candidate
            break

    if not lon_col:
        return dbc.Alert(
            "Cannot detect UTM zone — no longitude column found in header preview. "
            "Type the EPSG code manually (e.g. EPSG:32613 for Permian Basin).",
            color="info", className="py-2",
        )

    # Parse sample lons to estimate zone(s)
    try:
        lons = [float(v) for v in sample_vals[lon_col] if v and str(v).strip()]
    except (ValueError, TypeError):
        return dbc.Alert("Could not parse longitude samples for UTM detection.", color="info", className="py-2")

    if not lons:
        return dbc.Alert("No longitude values found in sample data.", color="info", className="py-2")

    # Compute zones from samples (limited preview — full analysis would need the actual file)
    zones = {}
    for lon in lons:
        z = int((lon + 180) / 6) + 1
        zones[z] = zones.get(z, 0) + 1

    total = sum(zones.values())
    majority_zone = max(zones, key=zones.get)
    majority_epsg = f"EPSG:326{majority_zone:02d}"

    if len(zones) == 1:
        # Single zone — all good
        return dbc.Alert(
            [
                html.Strong(f"UTM Zone {majority_zone} "),
                html.Span(f"({majority_epsg}) — all sample wells are in this zone."),
            ],
            color="success", className="py-2",
        )
    else:
        # Multiple zones — warn and show distribution
        dist_parts = []
        for z in sorted(zones, key=zones.get, reverse=True):
            pct = zones[z] / total * 100
            epsg = f"EPSG:326{z:02d}"
            dist_parts.append(html.Li(
                f"Zone {z} ({epsg}): {zones[z]} samples ({pct:.0f}%)",
                style={"fontSize": "0.82rem"},
            ))

        return html.Div([
            dbc.Alert(
                [
                    html.Strong("Multiple UTM zones detected in your data. "),
                    html.Span(
                        f"Majority zone: {majority_zone} ({majority_epsg}). "
                        "Leave the override field blank to use the majority zone, "
                        "or consult the basin table below and type an EPSG code to override."
                    ),
                ],
                color="warning", className="py-2 mb-2",
            ),
            html.Ul(dist_parts, className="mb-1"),
            _utm_basin_table(),
        ])


@callback(
    Output("config-store", "data"),
    Output("cfg-save-success", "children"),
    Output("cfg-save-success", "is_open"),
    Output("btn-go-calculate", "disabled"),
    Input("btn-save-config", "n_clicks"),
    State("cfg-utm-zone", "value"),
    State("cfg-max-distance", "value"),
    State("cfg-cutoff-ft", "value"),
    State("cfg-batch-size", "value"),
    State("cfg-directional-source", "value"),
    State("cfg-bench-filter", "value"),
    State("cfg-rsv-categories", "value"),
    State("cfg-prod-cutoff-months", "value"),
    State("cfg-duc-age-years", "value"),
    State("cfg-permit-window-years", "value"),
    State("column-map-store", "data"),
    prevent_initial_call=True,
)
def save_config(
    n_clicks, utm_zone, max_dist, cutoff_ft, batch_size,
    dir_source, bench_filter, rsv_cats, prod_cutoff, duc_age, permit_window,
    col_map_store,
):
    # utm_zone: blank = auto-detect, or user-typed EPSG override
    utm = utm_zone.strip() if utm_zone and utm_zone.strip() else None
    cfg = {
        "utm_zone": utm,
        "max_distance_miles": float(max_dist or 4.0),
        "cutoff_ft": float(cutoff_ft or 5280),
        "batch_size": int(batch_size or 200_000),
        "directional_source": dir_source if dir_source != "auto" else None,
        "bench_filter": [b.strip() for b in (bench_filter or "").split(",") if b.strip()],
        "rsv_categories": rsv_cats or [],
        "prod_cutoff_months": int(prod_cutoff or 6),
        "duc_age_years": int(duc_age or 3),
        "permit_window_years": int(permit_window or 2),
        "_mapping_version": (col_map_store or {}).get("_version"),
    }
    return cfg, "Configuration saved. Click 'Calculate & Next →' to proceed.", True, False
