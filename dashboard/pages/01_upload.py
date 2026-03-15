"""
Step 1 — Upload
Dual-mode data input:
  Tab A: Drag-and-drop CSV / Excel files (header, directional, production)
  Tab B: SQL query editor — write queries against a configured database

On completion → populates upload-store with file paths or query strings.
"""

import base64
import io
import tempfile
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, State, callback, dcc, html

dash.register_page(__name__, path="/", name="1 Upload", order=1)

# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def _file_upload_card(file_key: str, label: str, accept: str) -> dbc.Card:
    return dbc.Card(
        dbc.CardBody(
            [
                html.H6(label, className="card-title"),
                dcc.Upload(
                    id=f"upload-{file_key}",
                    children=html.Div(
                        [
                            html.I(className="bi bi-cloud-upload fs-3 text-muted"),
                            html.P("Drag & drop or click to select", className="text-muted mb-0"),
                            html.Small(accept, className="text-muted"),
                        ],
                        className="text-center py-3",
                    ),
                    accept=accept,
                    className="border border-dashed rounded",
                    style={"cursor": "pointer"},
                ),
                html.Div(id=f"upload-{file_key}-status", className="mt-2"),
            ]
        ),
        className="mb-3",
    )


def _sql_editor_card(file_key: str, label: str, placeholder: str) -> dbc.Card:
    return dbc.Card(
        dbc.CardBody(
            [
                html.H6(label, className="card-title"),
                dbc.Textarea(
                    id=f"sql-{file_key}",
                    placeholder=placeholder,
                    rows=5,
                    className="font-monospace",
                    style={"fontSize": "0.85rem"},
                ),
            ]
        ),
        className="mb-3",
    )


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

layout = dbc.Container(
    [
        html.H3("Step 1 — Load Your Data", className="mb-1"),
        html.P(
            "Upload CSV / Excel files or connect to a database to query data directly.",
            className="text-muted mb-4",
        ),

        dbc.Tabs(
            [
                # ── Tab A: File Upload ──────────────────────────────────────
                dbc.Tab(
                    dbc.Container(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(_file_upload_card("header", "Well Header File", ".csv,.xlsx"), md=4),
                                    dbc.Col(_file_upload_card("directional", "Directional Survey File", ".csv,.xlsx"), md=4),
                                    dbc.Col(_file_upload_card("production", "Production File (optional)", ".csv,.xlsx"), md=4),
                                ],
                                className="mt-3",
                            ),
                            dbc.Alert(
                                id="upload-file-error",
                                color="danger",
                                is_open=False,
                                dismissable=True,
                            ),
                        ],
                        fluid=True,
                    ),
                    label="File Upload",
                    tab_id="tab-file",
                ),

                # ── Tab B: Database Query ───────────────────────────────────
                dbc.Tab(
                    dbc.Container(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.H6("Database Connection", className="mt-3"),
                                            dbc.Select(
                                                id="db-backend-select",
                                                options=[
                                                    {"label": "Databricks",  "value": "databricks"},
                                                    {"label": "SQL Server",  "value": "mssql"},
                                                    {"label": "PostgreSQL",  "value": "postgresql"},
                                                    {"label": "Snowflake",   "value": "snowflake"},
                                                    {"label": "SQLite",      "value": "sqlite"},
                                                ],
                                                placeholder="Select database type...",
                                                className="mb-2",
                                            ),
                                            dbc.Input(
                                                id="db-connection-string",
                                                placeholder="Connection string / server URL",
                                                className="mb-2 font-monospace",
                                                style={"fontSize": "0.85rem"},
                                            ),
                                            dbc.Button(
                                                "Test Connection",
                                                id="btn-test-connection",
                                                color="secondary",
                                                size="sm",
                                                className="mb-2",
                                            ),
                                            html.Div(id="db-connection-status"),
                                        ],
                                        md=4,
                                    ),
                                    dbc.Col(
                                        [
                                            _sql_editor_card(
                                                "header-query",
                                                "Header Query",
                                                "SELECT uwi, well_name, bench, latitude, longitude, first_prod_date\nFROM well_header\nWHERE ...",
                                            ),
                                            _sql_editor_card(
                                                "directional-query",
                                                "Directional Survey Query",
                                                "SELECT uwi, md, tvd, latitude, longitude, azimuth\nFROM directional_survey\nWHERE ...",
                                            ),
                                            _sql_editor_card(
                                                "production-query",
                                                "Production Query (optional)",
                                                "SELECT uwi, prod_date, oil, gas, water\nFROM production\nWHERE ...",
                                            ),
                                        ],
                                        md=8,
                                    ),
                                ],
                                className="mt-3",
                            ),
                            dbc.Alert(
                                id="upload-db-error",
                                color="danger",
                                is_open=False,
                                dismissable=True,
                            ),
                        ],
                        fluid=True,
                    ),
                    label="Database Query",
                    tab_id="tab-db",
                ),
            ],
            id="upload-mode-tabs",
            active_tab="tab-file",
        ),

        # ── Footer: Next button ─────────────────────────────────────────────
        html.Hr(),
        dbc.Row(
            dbc.Col(
                dbc.Button(
                    "Next: Map Columns →",
                    id="btn-upload-next",
                    color="primary",
                    disabled=True,
                    href="/column-map",
                ),
                className="text-end",
            )
        ),
    ],
    fluid=True,
    className="py-4",
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def _parse_upload(contents: str, filename: str) -> tuple[pd.DataFrame | None, str | None]:
    """Decode a dcc.Upload content string and return a DataFrame + temp file path."""
    if contents is None:
        return None, None
    content_type, content_string = contents.split(",", 1)
    decoded = base64.b64decode(content_string)
    suffix = Path(filename).suffix.lower()
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(decoded)
            tmp_path = tmp.name
        if suffix == ".csv":
            df = pd.read_csv(tmp_path, nrows=5)
        else:
            df = pd.read_excel(tmp_path, nrows=5)
        return df, tmp_path
    except Exception as exc:
        return None, str(exc)


def _sample_values(df: pd.DataFrame, n: int = 3) -> dict[str, list[str]]:
    """Return first *n* non-null values per column as strings (for preview)."""
    samples: dict[str, list[str]] = {}
    for col in df.columns:
        vals = df[col].dropna().head(n).astype(str).tolist()
        samples[col] = vals
    return samples


@callback(
    Output("upload-header-status", "children"),
    Output("upload-store", "data", allow_duplicate=True),
    Input("upload-header", "contents"),
    State("upload-header", "filename"),
    State("upload-store", "data"),
    prevent_initial_call=True,
)
def on_header_upload(contents, filename, store):
    df, path = _parse_upload(contents, filename)
    if df is None:
        return dbc.Alert(f"Error: {path}", color="danger"), dash.no_update
    store = store or {}
    store["header_path"] = path
    store["header_filename"] = filename
    store["header_preview_cols"] = list(df.columns)
    store["header_sample_values"] = _sample_values(df)
    badge = dbc.Badge(f"{filename} ({len(df.columns)} cols)", color="success")
    return badge, store


@callback(
    Output("upload-directional-status", "children"),
    Output("upload-store", "data", allow_duplicate=True),
    Input("upload-directional", "contents"),
    State("upload-directional", "filename"),
    State("upload-store", "data"),
    prevent_initial_call=True,
)
def on_directional_upload(contents, filename, store):
    df, path = _parse_upload(contents, filename)
    if df is None:
        return dbc.Alert(f"Error: {path}", color="danger"), dash.no_update
    store = store or {}
    store["directional_path"] = path
    store["directional_filename"] = filename
    store["directional_preview_cols"] = list(df.columns)
    store["directional_sample_values"] = _sample_values(df)
    badge = dbc.Badge(f"{filename} ({len(df.columns)} cols)", color="success")
    return badge, store


@callback(
    Output("upload-production-status", "children"),
    Output("upload-store", "data", allow_duplicate=True),
    Input("upload-production", "contents"),
    State("upload-production", "filename"),
    State("upload-store", "data"),
    prevent_initial_call=True,
)
def on_production_upload(contents, filename, store):
    df, path = _parse_upload(contents, filename)
    if df is None:
        return dbc.Alert(f"Error: {path}", color="danger"), dash.no_update
    store = store or {}
    store["production_path"] = path
    store["production_filename"] = filename
    store["production_preview_cols"] = list(df.columns)
    store["production_sample_values"] = _sample_values(df)
    badge = dbc.Badge(f"{filename} ({len(df.columns)} cols)", color="success")
    return badge, store


@callback(
    Output("btn-upload-next", "disabled"),
    Input("upload-store", "data"),
    Input("upload-mode-tabs", "active_tab"),
    prevent_initial_call=True,
)
def toggle_next_button(store, active_tab):
    if active_tab == "tab-file":
        store = store or {}
        ready = "header_path" in store and "directional_path" in store
    else:
        # DB mode — enable once queries are entered (validated on next page)
        ready = True
    return not ready
