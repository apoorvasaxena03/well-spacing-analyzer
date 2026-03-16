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

        # ---- Card 1: Spacing Engine Parameters ----
        dbc.Card(
            dbc.CardBody(
                [
                    html.H6("Spacing Engine Parameters", className="mb-3"),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label("UTM Zone (auto-detected)"),
                                    dbc.Input(
                                        id="cfg-utm-zone",
                                        placeholder="e.g. EPSG:32613",
                                        value="",
                                        disabled=True,
                                    ),
                                    dbc.FormText("Leave blank to auto-detect from data centroid."),
                                    dbc.Checklist(
                                        id="cfg-utm-override",
                                        options=[{"label": "Override UTM zone", "value": "override"}],
                                        value=[],
                                        className="mt-1",
                                    ),
                                ],
                                md=4,
                            ),
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

        html.Hr(),
        dbc.Row(
            [
                dbc.Col(dbc.Button("← Back", href="/column-map", color="secondary", outline=True), width="auto"),
                dbc.Col(
                    dbc.Button("Save & Next →", id="btn-save-config", color="primary"),
                    className="text-end",
                ),
            ],
            justify="between",
        ),
        dcc.Location(id="cfg-redirect", refresh=True),
    ],
    fluid=True,
    className="py-4",
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@callback(
    Output("cfg-utm-zone", "disabled"),
    Input("cfg-utm-override", "value"),
    prevent_initial_call=False,
)
def toggle_utm_input(override):
    return "override" not in (override or [])


@callback(
    Output("config-store", "data"),
    Output("cfg-redirect", "href"),
    Input("btn-save-config", "n_clicks"),
    State("cfg-utm-zone", "value"),
    State("cfg-utm-override", "value"),
    State("cfg-max-distance", "value"),
    State("cfg-cutoff-ft", "value"),
    State("cfg-batch-size", "value"),
    State("cfg-directional-source", "value"),
    State("cfg-bench-filter", "value"),
    State("cfg-rsv-categories", "value"),
    State("cfg-prod-cutoff-months", "value"),
    State("cfg-duc-age-years", "value"),
    State("cfg-permit-window-years", "value"),
    prevent_initial_call=True,
)
def save_config(
    n_clicks, utm_zone, utm_override, max_dist, cutoff_ft, batch_size,
    dir_source, bench_filter, rsv_cats, prod_cutoff, duc_age, permit_window,
):
    cfg = {
        "utm_zone": utm_zone if "override" in (utm_override or []) else None,
        "max_distance_miles": float(max_dist or 4.0),
        "cutoff_ft": float(cutoff_ft or 5280),
        "batch_size": int(batch_size or 200_000),
        "directional_source": dir_source if dir_source != "auto" else None,
        "bench_filter": [b.strip() for b in (bench_filter or "").split(",") if b.strip()],
        "rsv_categories": rsv_cats or [],
        "prod_cutoff_months": int(prod_cutoff or 6),
        "duc_age_years": int(duc_age or 3),
        "permit_window_years": int(permit_window or 2),
    }
    return cfg, "/calculate"
