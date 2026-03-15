"""
Well Spacing Analyzer — Dash Application
Entry point: python dashboard/app.py

6-step guided workflow:
  1. Upload       — drag-drop files or write SQL queries
  2. Column Map   — map source columns to canonical names
  3. Configure    — UTM zone, distance params, bench filters
  4. Calculate    — run spacing engine (background job)
  5. Explore      — map, gun barrel, production charts
  6. Export       — download results as CSV / Excel
"""

from diskcache import Cache as DiskCache
import dash
import dash_bootstrap_components as dbc
from flask_caching import Cache

# ---------------------------------------------------------------------------
# Background job cache — must be created BEFORE dash.Dash()
# ---------------------------------------------------------------------------
disk_cache = DiskCache("./dashboard/.cache")
background_callback_manager = dash.DiskcacheManager(disk_cache)

# ---------------------------------------------------------------------------
# App initialisation
# ---------------------------------------------------------------------------
app = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.FLATLY, dbc.icons.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="Well Spacing Analyzer",
    background_callback_manager=background_callback_manager,
)
server = app.server  # expose Flask server for caching

# Flask-caching for memoising expensive reads
cache = Cache(server, config={"CACHE_TYPE": "SimpleCache", "CACHE_DEFAULT_TIMEOUT": 3600})

# ---------------------------------------------------------------------------
# Shared dcc.Store components — passed between pages
# ---------------------------------------------------------------------------
stores = dbc.Container(
    [
        dash.dcc.Store(id="upload-store"),          # raw file bytes + filenames
        dash.dcc.Store(id="column-map-store"),      # {source: canonical} mappings per file
        dash.dcc.Store(id="config-store"),          # UTM zone, max_distance_miles, cutoff_ft
        dash.dcc.Store(id="pipeline-result-store"), # path to cached pipeline output on disk
        dash.dcc.Store(id="selected-wells-store"),  # clicked UWI + neighborhood UWIs from map
    ],
    fluid=True,
)

# ---------------------------------------------------------------------------
# Navigation bar
# ---------------------------------------------------------------------------
STEPS = [
    ("1 Upload",      "/"),
    ("2 Map Columns", "/column-map"),
    ("3 Configure",   "/configure"),
    ("4 Calculate",   "/calculate"),
    ("5 Explore",     "/explore"),
    ("6 Export",      "/export"),
]

navbar = dbc.Navbar(
    dbc.Container(
        [
            dbc.NavbarBrand("Well Spacing Analyzer", href="/", className="fw-bold"),
            dbc.Nav(
                [
                    dbc.NavItem(dbc.NavLink(label, href=href, active="exact"))
                    for label, href in STEPS
                ],
                navbar=True,
                className="ms-3",
            ),
        ],
        fluid=True,
    ),
    color="primary",
    dark=True,
    sticky="top",
    className="mb-3",
)

# ---------------------------------------------------------------------------
# Root layout
# ---------------------------------------------------------------------------
app.layout = dbc.Container(
    [
        stores,
        navbar,
        dash.page_container,
    ],
    fluid=True,
)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=8050)
