"""
Step 1 — Upload
Dual-mode data input:
  Tab A: Drag-and-drop CSV / Excel files (header, directional, production)
  Tab B: SQL query editor — write queries against a configured database

On completion → populates upload-store with file paths or query strings.
"""

import base64
import io
import json
import logging
import pickle
import tempfile
import zipfile
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, State, callback, dcc, html

logger = logging.getLogger("dashboard")

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

                # ── Tab C: Import Session ─────────────────────────────────
                dbc.Tab(
                    dbc.Container(
                        [
                            dbc.Row(
                                dbc.Col(
                                    [
                                        html.H5("Import a Previous Session", className="mt-3 mb-2"),
                                        html.P(
                                            "Upload a session package (.zip) exported from the Export page. "
                                            "This skips Steps 2–4 and takes you directly to Explore.",
                                            className="text-muted mb-3",
                                        ),
                                        dcc.Upload(
                                            id="upload-session-zip",
                                            children=dbc.Card(
                                                dbc.CardBody(
                                                    [
                                                        html.I(className="bi bi-box-arrow-in-down", style={"fontSize": "2rem"}),
                                                        html.P("Drag & drop or click to upload session package (.zip)", className="mb-0 mt-2"),
                                                    ],
                                                    className="text-center py-4",
                                                ),
                                                className="border-dashed",
                                                style={"borderStyle": "dashed", "borderColor": "#aaa", "cursor": "pointer"},
                                            ),
                                            accept=".zip",
                                        ),
                                        html.Div(id="session-import-status", className="mt-3"),
                                        dcc.Location(id="session-redirect", refresh=True),
                                    ],
                                    md=6,
                                ),
                            ),
                        ],
                        fluid=True,
                    ),
                    label="Import Session",
                    tab_id="tab-session",
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
        # Read first 5 rows for preview; force UWI-like columns to string
        # so 14-digit identifiers aren't read as float/int
        if suffix == ".csv":
            # Peek at header to find UWI columns
            header = pd.read_csv(tmp_path, nrows=0).columns
            uwi_dtypes = {c: str for c in header if "uwi" in c.lower() or "api" in c.lower()}
            df = pd.read_csv(tmp_path, nrows=5, dtype=uwi_dtypes or None)
        else:
            df = pd.read_excel(tmp_path, nrows=5, dtype=str)
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
    # Extract unique values for key categorical columns (used by Configure dropdowns)
    for cat_col in ("bench", "operator", "well_status", "rsv_cat", "hole_direction"):
        matches = [c for c in df.columns if c.lower().replace(" ", "_") == cat_col or c.lower() == cat_col]
        if matches:
            vals = sorted(df[matches[0]].dropna().astype(str).unique().tolist())
            store[f"header_unique_{cat_col}"] = vals
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


# ---------------------------------------------------------------------------
# Session package import
# ---------------------------------------------------------------------------

PIPELINE_CACHE_DIR = Path("./dashboard/.pipeline_cache")


@callback(
    Output("session-import-status", "children"),
    Output("pipeline-result-store", "data", allow_duplicate=True),
    Output("session-redirect", "href"),
    Input("upload-session-zip", "contents"),
    State("upload-session-zip", "filename"),
    prevent_initial_call=True,
)
def import_session_package(contents, filename):
    """Import a session package ZIP and redirect to Explore."""
    if not contents:
        return dash.no_update, dash.no_update, dash.no_update

    try:
        # Decode uploaded ZIP
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        zf = zipfile.ZipFile(io.BytesIO(decoded))

        # Read metadata
        metadata = {}
        if "metadata.json" in zf.namelist():
            metadata = json.loads(zf.read("metadata.json"))

        run_id = metadata.get("run_id", "imported")

        # Read parquet files into DataFrames
        cache_data = {}
        parquet_map = {
            "ik_pairs.parquet": "df_ik_pairs",
            "ik_pairs_raw.parquet": "df_ik_pairs_raw",
            "neighbor_summary.parquet": "df_spacing",
            "avg_spacing.parquet": "df_avg_spacing",
            "wps.parquet": "df_wps",
            "header.parquet": "header_df",
            "lateral.parquet": "lateral_df",
            "directional.parquet": "directional_df",
            "production.parquet": "production_df",
        }
        for zip_name, cache_key in parquet_map.items():
            if zip_name in zf.namelist():
                buf = io.BytesIO(zf.read(zip_name))
                cache_data[cache_key] = pd.read_parquet(buf)

        # Read stats
        if "stats.json" in zf.namelist():
            cache_data["stats"] = json.loads(zf.read("stats.json"))

        if "header_df" not in cache_data or "lateral_df" not in cache_data:
            return (
                dbc.Alert("Invalid session package: missing header or lateral data.", color="danger"),
                dash.no_update,
                dash.no_update,
            )

        # Save as pipeline cache pickle
        PIPELINE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = PIPELINE_CACHE_DIR / f"pipeline_{run_id}.pkl"
        with open(cache_path, "wb") as f:
            pickle.dump(cache_data, f)

        # Build summary
        datasets = metadata.get("datasets", [])
        summary_items = [f"{d['name']}: {d['rows']:,} rows" for d in datasets]

        # Read UI state if present and save into cache pickle
        # (Explore page's restore_ui_state callback reads it from cache on load)
        ui = {}
        if "ui_state.json" in zf.namelist():
            ui = json.loads(zf.read("ui_state.json"))
            # Persist into cache so restore_ui_state picks it up
            from dashboard.pipeline import save_ui_state
            save_ui_state(str(cache_path), ui)

        logger.info("Session imported: %s → %s (%d datasets, ui_state=%s)",
                     filename, cache_path.name, len(datasets), bool(ui))

        return (
            dbc.Alert(
                [
                    html.Strong(f"Session imported: {filename}"),
                    html.Br(),
                    html.Small(" | ".join(summary_items)),
                    html.Br(),
                    html.Small("Redirecting to Explore — settings restored." if ui else "Redirecting to Explore..."),
                ],
                color="success",
            ),
            {"cache_path": str(cache_path), "run_id": run_id, "crs_used": metadata.get("crs_used")},
            "/explore",
        )

    except Exception as exc:
        logger.exception("Session import failed: %s", exc)
        return (
            dbc.Alert(f"Import failed: {exc}", color="danger"),
            dash.no_update,
            dash.no_update,
        )
