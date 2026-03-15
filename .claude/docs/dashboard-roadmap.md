# Dashboard Roadmap

## Vision

This is not just a dashboard — it is a **complete one-stop well spacing application** for layman users.
No Python knowledge required. No Jupyter notebooks. No manual config files.

A user opens the app in a browser, uploads their data, maps their columns,
hits **Calculate Spacing**, and within minutes has a fully interactive analysis
with maps, gun barrel diagrams, production charts, and neighbor identification.

**The app replaces the entire workflow:**

```text
Old: Excel → Python notebook → Spotfire → manual updates
New: Upload files → Map columns → Calculate → Explore results
```

Built with **Dash (Plotly)** — fully open-source, runs locally with `python dashboard/app.py`,
no Spotfire license, no cloud dependency.

**Core principles:**

- Any reservoir engineer can use it without touching code
- Works with any column naming convention (column mapper handles it)
- QGIS-like spatial capabilities — better than Spotfire
- One app, entire workflow: ingest → calculate → visualize → export

### Hybrid Design — Both Audiences Supported

```text
                    ┌─────────────────────────────┐
                    │       src/ library           │  ← unchanged, always works
                    │  (WellDataLoader, Calculator, │
                    │   GeoSurveyProcessor, etc.)   │
                    └──────────┬──────────────────-┘
                               │
               ┌───────────────┴───────────────┐
               │                               │
   ┌───────────▼──────────┐      ┌─────────────▼──────────┐
   │   Jupyter Notebooks  │      │    dashboard/app.py     │
   │   (power users /     │      │    (layman users /      │
   │    developers)       │      │     front-end)          │
   │                      │      │                         │
   │  Full control, custom│      │  Upload → Map → Run →   │
   │  column maps, custom │      │  Visualize → Export     │
   │  batch sizes, debug  │      │  No code required       │
   └──────────────────────┘      └─────────────────────────┘
```

The `src/` library is **never modified** for the dashboard. `dashboard/pipeline.py` simply
calls the same classes and functions that notebooks already use.
Power users keep full control via notebooks. Layman users get the guided app.
Both produce identical results from the same underlying code.

---

## End-to-End User Workflow (UX Design)

The app is organized as a **guided multi-step flow** with a persistent sidebar showing progress.
Each step unlocks the next. A layman user can complete the full analysis without reading any docs.

### Step 1 — Load Data

Each dataset (Header, Directional Survey, Production) has **two input modes** — the user
picks whichever fits their workflow. Both modes feed the same column mapper in Step 2.

```text
┌─ Well Header ───────────────────────────────────────────────────────────┐
│  [📂 File Upload]  [🗄 Database Query]  ← tabs                          │
│                                                                          │
│  ▌FILE UPLOAD tab:                                                       │
│  ┌──────────────────────────────────────────────┐                       │
│  │  Drag & drop CSV / Excel here                │                       │
│  │  — or —  [Browse Files]                      │                       │
│  └──────────────────────────────────────────────┘                       │
│  ✅ Loaded: well_header.csv  │  1,247 rows  │  18 columns  [Preview ▾]  │
└─────────────────────────────────────────────────────────────────────────┘

┌─ Directional Survey ────────────────────────────────────────────────────┐
│  [📂 File Upload]  [🗄 Database Query]  ← tabs                          │
│                                                                          │
│  ▌DATABASE QUERY tab:                                                    │
│                                                                          │
│  Backend:  [Databricks  ▼]                                               │
│                                                                          │
│  Server hostname:  [adb-3250236208859616.16.azuredatabricks.net     ]   │
│  HTTP path:        [/sql/1.0/warehouses/0cf5fe590cb7b313            ]   │
│  Access token:     [••••••••••••••••••••  (from env: DATABRICKS_TOKEN)] │
│  Catalog:          [bronze    ]   Schema: [enverus   ]                  │
│                                                    [Test Connection ▶]  │
│  ✅ Connection OK                                                        │
│                                                                          │
│  SQL Query:                                                              │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ SELECT                                                           │   │
│  │     API_UWI_12_Unformatted  AS uwi,                              │   │
│  │     MeasuredDepth_FT        AS md,                               │   │
│  │     TVD_FT                  AS tvd,                              │   │
│  │     Inclination_DEG         AS inclination,                      │   │
│  │     Azimuth_DEG             AS azimuth,                          │   │
│  │     Latitude                AS latitude,                         │   │
│  │     Longitude               AS longitude                         │   │
│  │ FROM bronze.enverus.fulldirectionalsurvey                        │   │
│  │ WHERE API_UWI_12_Unformatted IN :uwis                            │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  Bind Parameters (optional JSON):                                        │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ {"uwis": ["42227350890000", "42227350900000", ...]}              │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│  ℹ Large IN-lists auto-chunked (2,000/batch) — no action needed          │
│                                                                          │
│                                           [Run Query ▶]                 │
│  ✅ Loaded: 48,312 rows  │  7 columns  [Preview ▾]                      │
└─────────────────────────────────────────────────────────────────────────┘

┌─ Production Data ───────────────────────────────────────────────────────┐
│  [📂 File Upload]  [🗄 Database Query]        ⬜ Optional                │
└─────────────────────────────────────────────────────────────────────────┘
```

**Supported DB backends** (matches `database_manager.py` exactly):

| Backend    | Config Fields                                             |
| ---------- | --------------------------------------------------------- |
| Databricks | server_hostname, http_path, access_token, catalog, schema |
| SQL Server | host, database, username, password (or DSN)               |
| PostgreSQL | host, port, database, username, password                  |
| Snowflake  | account, database, schema, warehouse, username, password  |
| Oracle     | host, port, service_name, username, password              |
| MySQL      | host, port, database, username, password                  |
| SQLite     | file path                                                 |

**Credential security in the UI:**

- Password / token fields always masked (`type="password"`)
- Option to load from environment variable: user types `$DATABRICKS_TOKEN` → app reads `os.environ`
- Credentials stored only in server-side session, never in `dcc.Store` (client-visible)
- Connection is tested with `client.test_connection()` before any query runs

**SQL editor features:**

- Syntax highlighting via `dash-ace` or `dbc.Textarea` with monospace font
- Canonical column name hints shown as a collapsible reference panel beside the editor:

```text
  ℹ Required canonical names for Header:
     uwi, well_name, bench, latitude, longitude, first_prod_date
  ℹ Required canonical names for Directional:
     uwi (or uwi12), md, tvd, latitude, longitude, azimuth
```

- Alias hint: "Use `AS canonical_name` in your SELECT, or map columns in Step 2"
- Bind parameters: `:param_name` syntax, user provides JSON dict — large lists auto-chunked
  via `execute_query_chunked()` in `database_manager.py` (transparent, no UI change needed)

**How it maps to `WellDataLoader`:**

```python
# File upload path → source= argument
df = loader.get_header_data(source="uploaded_file.csv", column_map=user_map)

# Database query path → header_query= / directional_query= arguments
df = loader.get_header_data(header_query=user_sql)
df = loader.get_directional_data(
    directional_query=user_sql,
    directional_params=user_bind_params,   # auto-chunked if IN-list > 1000
)
```

Each dataset shows after load: row count, column count, **Preview** (first 5 rows in a `DataTable`).

### Step 2 — Map Columns

For each dataset, the user picks a **source template** (pre-fills the entire mapping instantly)
or manually maps columns. This is the key feature for layman users — if they're on Enverus,
they click one button and the mapping is done.

```text
┌─ Well Header — Column Mapping ──────────────────────────────────────────┐
│                                                                          │
│  Source template:  [Enverus (Databricks) ▼]   [Apply Template ✨]       │
│                     Enverus (Databricks)                                 │
│                     Enverus (CSV export)                                 │
│                     IHS / Enerdeq                                        │
│                     DrillingInfo / Enverus Legacy                        │
│                     Custom / Manual                                      │
│                                                                          │
│  Your Column                  →   Canonical Name          Status        │
│  ────────────────────────────────────────────────────────────────────   │
│  API_UWI_14_Unformatted       →   [uwi              ▼]    ✅ required   │
│  API_UWI_12_Unformatted       →   [uwi12            ▼]    ✅            │
│  LeaseName                    →   [lease_name       ▼]    ✅            │
│  WellName                     →   [well_name        ▼]    ✅ required   │
│  ENVOperator                  →   [operator         ▼]    ✅            │
│  ENVInterval                  →   [bench            ▼]    ✅ required   │
│  SpudDate                     →   [spud_date        ▼]    ✅            │
│  CompletionDate               →   [completion_date  ▼]    ✅            │
│  FirstProdDate                →   [first_prod_date  ▼]    ✅ required   │
│  LastProducingMonth           →   [last_prod_date   ▼]    ✅            │
│  Trajectory                   →   [hole_direction   ▼]    ✅            │
│  ENVWellStatus                →   [well_status      ▼]    ✅            │
│  LateralLength_FT             →   [lateral_length_ft▼]    ✅            │
│  Latitude                     →   [surface_lat      ▼]    ✅ required   │
│  Longitude                    →   [surface_lon      ▼]    ✅ required   │
│  FluidIntensity_BBLPerFT      →   [— skip —         ▼]    ⚪ optional   │
│  ProppantIntensity_LBSPerFT   →   [— skip —         ▼]    ⚪ optional   │
│  ────────────────────────────────────────────────────────────────────   │
│  Required: 6/6 ✅   Optional: 9 mapped                   [Confirm →]   │
└─────────────────────────────────────────────────────────────────────────┘
```

**Pre-populated templates** (stored in `dashboard/templates/column_maps.py`):

```python
# Exact column maps from well_spacing_RingEnergy_v2.ipynb — the reference notebook

ENVERUS_HEADER = {
    "API_UWI_14_Unformatted":       "uwi",
    "API_UWI_12_Unformatted":       "uwi12",
    "LeaseName":                    "lease_name",
    "WellName":                     "well_name",
    "ENVOperator":                  "operator",
    "ENVInterval":                  "bench",
    "PermitApprovedDate":           "permit_date",
    "SpudDate":                     "spud_date",
    "CompletionDate":               "completion_date",
    "FirstProdDate":                "first_prod_date",
    "LastProducingMonth":           "last_prod_date",
    "Trajectory":                   "hole_direction",
    "ENVWellStatus":                "well_status",
    "LateralLength_FT":             "lateral_length_ft",
    "Latitude":                     "surface_lat",
    "Longitude":                    "surface_lon",
    "FluidIntensity_BBLPerFT":      "fluid_intensity_bbl_per_ft",
    "ProppantIntensity_LBSPerFT":   "proppant_intensity_lbs_per_ft",
}

ENVERUS_DIRECTIONAL = {
    "API_UWI_12_Unformatted":       "uwi12",
    "MeasuredDepth_FT":             "md",
    "TVD_FT":                       "tvd",
    "Inclination_DEG":              "inclination",
    "Azimuth_DEG":                  "azimuth",
    "Latitude":                     "latitude",
    "Longitude":                    "longitude",
    "E_W":                          "deviation_E/W",
    "N_S":                          "deviation_N/S",
}

ENVERUS_PRODUCTION = {
    "API_UWI_14_Unformatted":       "uwi",
    "API_UWI_Unformatted":          "uwi_10",
    "ProducingMonth":               "prod_date",
    "Prod_OilBBL":                  "monthly_oil_bbl",
    "GasProd_MCF":                  "monthly_gas_mcf",
    "WaterProd_BBL":                "monthly_water_bbl",
}

# Add more templates as new data sources are onboarded:
# IHS_HEADER = { ... }
# DRILLINGINFO_HEADER = { ... }
```

**How it works for a layman user on Enverus:**

1. Load data (file or DB query)
2. Select template: **"Enverus (Databricks)"**
3. Click **Apply Template** — all rows fill in instantly
4. Unrecognized columns auto-set to "— skip —"
5. Review, correct if any are wrong, click **Confirm**
6. Done — no manual typing required

**How it works for a custom source:**

1. Load data
2. Select **"Custom / Manual"** — all rows start blank
3. Fuzzy auto-suggest fills in best guesses (e.g., `"API_14"` → suggests `uwi`)
4. User confirms/corrects each row
5. Optionally: **Save as new template** for reuse → stored in `dashboard/templates/column_maps.py`

Required canonical columns shown in red if unmapped. Cannot proceed until all required columns resolved.

### Step 3 — Configure Parameters

Collapsible panel with sensible defaults. Most users never need to change these.

```text
┌─ Calculation Settings ──────────────────────────────────┐
│  UTM Zone              [Auto-detect from data    ▼]     │
│  Max Search Radius     [3.0  miles  ────●──────] 0–10   │
│  Max Crossline Dist    [2500 ft     ──●────────] 0–5000  │
│  Batch Size            [200,000     ────────●──]         │
│                                                          │
│  ▼ Neighbor Settings                                     │
│  Horizontal Cutoff     [1800 ft     ──●────────] 0–5000  │
│  Vertical Cutoff TVD   [150 ft      ─●─────────] 0–500   │
│  Min Lateral Overlap   [30%         ──●────────] 0–100%  │
└─────────────────────────────────────────────────────────┘

            [← Back]        [🚀 Calculate Spacing]
```

**UTM auto-detect**: compute centroid of all lat/lon points → look up the correct UTM zone automatically.
No user needs to know what "EPSG:32613" means.

### Step 4 — Calculating (Progress View)

While the pipeline runs, show a live progress UI:

```text
┌─ Well Spacing Calculation ──────────────────────────────┐
│                                                          │
│  ✅  Loading & deduplicating wells          (0.3s)       │
│  ✅  Computing UTM coordinates              (1.2s)       │
│  ✅  Filtering lateral sections             (0.8s)       │
│  🔄  Computing pairwise spacing...                       │
│      Batch 3 / 12  ████████░░░░░░░░░░  38%  ~4 min left │
│  ⬜  Identifying neighbors                               │
│  ⬜  Building map layers                                  │
│                                                          │
│  Wells loaded: 1,247    Pairs in progress: 245,000+      │
└─────────────────────────────────────────────────────────┘
```

Progress driven by Dash `dcc.Interval` polling a background thread/process.
User can cancel and adjust parameters if results look wrong.

**Implementation**: Use `dash.long_callback` (built into Dash 2.x) with a
`DiskcacheManager` or `CeleryManager` for the background job. No extra infrastructure
needed for local use — `diskcache` is a pure-Python dependency.

### Step 5 — Explore Results

Once calculation is complete, the full visualization app unlocks. All panels are populated.
A persistent top bar shows the session summary:

```text
[Asset: Ring Energy]  [1,247 wells]  [312,450 pairs]  [Re-run ↺]  [Export ⬇]
```

Navigation tabs across the top:

```text
[🗺 Map]  [⬇ Gun Barrel]  [📈 Production]  [📊 Statistics]  [🔍 QC]  [⬇ Export]
```

Each tab is one of the visualization panels described below.

### Step 6 — Export

Download panel with format options:

```text
┌─ Export Results ────────────────────────────────────────┐
│  Spacing pairs (all)           [⬇ CSV]  [⬇ Excel]       │
│  Neighbor summary (one/well)   [⬇ CSV]  [⬇ Excel]       │
│  Well trajectories             [⬇ GeoJSON] [⬇ Shapefile]│
│  Gun barrel data               [⬇ CSV]                   │
│  Current map view              [⬇ PNG]                   │
└─────────────────────────────────────────────────────────┘
```

---

## App Architecture

```text
dashboard/
├── app.py                  # Entry point: python dashboard/app.py
├── layout.py               # Top-level layout: sidebar + tabs
├── pages/
│   ├── upload.py           # Step 1: file upload
│   ├── column_mapper.py    # Step 2: column mapping UI
│   ├── configure.py        # Step 3: parameter config
│   ├── progress.py         # Step 4: calculation progress
│   └── results.py          # Step 5: all result panels
├── callbacks/
│   ├── pipeline.py         # long_callback: runs full spacing pipeline
│   ├── map_callbacks.py    # map interactions → filter other panels
│   ├── gb_callbacks.py     # gun barrel: selected wells → GB data
│   └── production_callbacks.py
├── components/
│   ├── column_mapper.py    # reusable column mapping component
│   ├── map_panel.py        # dash-leaflet map + trajectory layers
│   ├── gun_barrel.py       # GB diagram (replicates Spotfire GB function)
│   ├── production_charts.py
│   └── progress_bar.py
├── pipeline.py             # wraps well_spacing_analyzer src/ pipeline
│                           # WellDataLoader → GeoSurveyProcessor →
│                           # WellSpacingCalculator → DirectionalBenchNeighbors
└── assets/
    ├── style.css
    └── map_extensions.js   # dash-leaflet JS callbacks (bench colors etc.)
```

**`pipeline.py`** is the bridge between the app and the `src/` library:

```python
def run_full_pipeline(
    header_path: str,
    survey_path: str,
    header_col_map: dict,
    survey_col_map: dict,
    utm_epsg: str,
    max_distance_miles: float,
    cutoff_ft: float,
    vertical_cutoff_ft: float,
    overlap_pct_min: float,
    progress_callback=None,   # called with (step_name, pct_complete)
) -> PipelineResult:
    ...
```

---

## Reference: Existing Spotfire Dashboard

**File locations**:

- Full dashboard: `C:\Users\ApoorvaSaxen_ct6z7vh\Downloads\A&D_GB_v2_To_Matt.dxp`
- Gun barrel mod: `C:\Users\ApoorvaSaxen_ct6z7vh\Downloads\Well spacing (gun barrel) diagram.mod`

**Spotfire Data Tables**:

- `header_standardized_spacing_o...` — well header
- `shapefiles_well_lateral` — well trajectory polylines
- `ik_pairs` — spacing pairs (well_i, well_k, horizontal_dist, elevation_i, drill_direction_i, mid_lat, mid_lon)
- `heel_toe_midpoints` — uwi, mid_lat, mid_lon
- `GB` — gun barrel data (computed by Python data function)
- `monthly_prod_standardized` — production time series
- `shp-Wells-BottomHole` — bottom-hole locations

**Filters**: `wps_corridor` (spatial corridor filter)

---

## Gun Barrel Python Logic (from Spotfire GB data function)

This is the exact logic to replicate in Dash:

```python
import pandas as pd
import numpy as np

def compute_gun_barrel(IK: pd.DataFrame, HeelToe: pd.DataFrame) -> pd.DataFrame:
    """
    Compute gun barrel positioning data.

    Args:
        IK: Spacing pairs df with columns:
            well_i, well_k, horizontal_dist, vertical_dist, 3D_dist,
            elevation_i, drill_direction_i
        HeelToe: Midpoint data with columns:
            uwi, mid_Lat, mid_Lon

    Returns:
        GB: Gun barrel df with cum_dist for X-axis positioning
    """
    if IK.shape[0] == 0:
        return pd.DataFrame(np.nan, index=[0], columns=[
            'well_i', 'elevation_i', 'drill_direction_i',
            'mid_Lat', 'mid_Lon', 'horizontal_dist', 'cum_dist', 'E_to_W_Rank'
        ])

    HeelToe = HeelToe.copy()
    HeelToe['mid_Lat'] = np.round(HeelToe['mid_Lat'], 9)
    HeelToe['mid_Lon'] = np.round(HeelToe['mid_Lon'], 9)

    # Get unique well_i entries from spacing pairs
    GB = IK[IK['well_i'].isin(IK['well_i'].unique())].drop_duplicates(
        subset=['well_i'], ignore_index=True)
    GB = GB[['well_i', 'elevation_i', 'drill_direction_i']].copy()

    # Join heel/toe midpoints
    GB = pd.merge(
        GB, HeelToe.rename(columns={"uwi": "well_i"}),
        how='left', on='well_i'
    ).reset_index(drop=True)

    # Sort wells for gun barrel positioning
    if GB['drill_direction_i'].mode().item() == 'NS':
        # NS wells → sort West to East by longitude
        GB['mid_Lon'] = np.round(GB['mid_Lon'], 9)
        GB = GB.sort_values(by=['mid_Lon']).reset_index(drop=True)
        GB['E_to_W_Rank'] = GB.index + 1
    elif GB['drill_direction_i'].mode().item() == 'EW':
        # EW wells → sort South to North by latitude
        GB['mid_Lat'] = np.round(GB['mid_Lat'], 9)
        GB = GB.sort_values(by=['mid_Lat']).reset_index(drop=True)
        GB['N_to_S_Rank'] = GB.index + 1

    if len(GB) == 1:
        GB['horizontal_dist'] = 0
        GB['cum_dist'] = 0
    else:
        # Get spacing between adjacent wells in sorted order
        GB['next_i_uwi'] = GB['well_i'].shift(-1)
        GB = GB.merge(
            IK[['well_i', 'well_k', 'horizontal_dist', 'vertical_dist', '3D_dist']],
            left_on=['well_i', 'next_i_uwi'],
            right_on=['well_i', 'well_k'],
            how='left'
        )
        # Cumulative distance (first well starts at 0)
        GB['cum_dist'] = GB['horizontal_dist'].shift(1, fill_value=0).cumsum()

    return GB
```

### Enhanced Gun Barrel Chart (Next-Gen)

The chart is a `go.Figure` built from multiple `go.Scatter` traces layered on top of each other.
It keeps full Spotfire parity (the `compute_gun_barrel()` function above is the data foundation)
and adds three upgrades: spacing zigzag lines, formation top horizons, and a centered x-axis option.

#### X-Axis Options

`cum_dist` is computed directly from the GB Python function:

```python
# Adjacent horizontal_dist values from IK, in sorted well order
GB['cum_dist'] = GB['horizontal_dist'].shift(1, fill_value=0).cumsum()
```

- First well (westernmost for NS, southernmost for EW) is always at **x = 0**
- Each subsequent well's x position = sum of `horizontal_dist` values between all preceding adjacent pairs
- `horizontal_dist` comes from the IK spacing DataFrame (already computed by `WellSpacingCalculator`)

| Mode          | Formula                          | Description                                        |
|---------------|----------------------------------|----------------------------------------------------|
| `cum_dist`    | from GB function above           | First well at 0, increases East (NS) or North (EW) |
| `sectionDist` | `cum_dist - cum_dist.max() / 2`  | Centered at 0; left-most well is most negative     |

Toggle via a `dcc.RadioItems` ("From west/south reference" / "Centered"). Store choice in `config-store`.

#### Layer 1 — Well Points (Spotfire baseline)

```python
for bench, group in GB.groupby("bench"):
    fig.add_trace(go.Scatter(
        x=group["sectionDist"],   # or cum_dist
        y=group["elevation_i"],
        mode="markers+text",
        name=bench,
        text=group["well_name"] + "<br>" + group["first_prod_date"].astype(str),
        textposition="top center",
        marker=dict(size=12, color=bench_colormap[bench]),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "TVD: %{y:,.0f} ft<br>"
            "Position: %{x:,.0f} ft<extra></extra>"
        ),
    ))
```

#### Layer 2 — Spacing Zigzag Lines

For each adjacent pair `(well_i, well_k)` in the sorted GB order, draw three line segments
connecting the two well points — forming a right-triangle "zigzag":

```
well_i point ──── horizontal line ──── corner point
                                           │
                                           │ vertical line
                                           │
                                       well_k point
```

The spacing values (`horizontal_dist`, `vertical_dist`, `dist3d`) come directly from the
IK spacing pairs DataFrame — they are already computed by `WellSpacingCalculator`.

```python
def add_spacing_zigzag_traces(fig, GB: pd.DataFrame, IK: pd.DataFrame, x_col: str = "sectionDist"):
    """
    Add connecting zigzag lines between adjacent well pairs in GB order.
    IK must contain: well_i, well_k, horizontal_dist, vertical_dist, dist3d
    """
    for idx in range(len(GB) - 1):
        wi = GB.iloc[idx]
        wk = GB.iloc[idx + 1]

        # Midpoint for annotation placement
        mid_x = (wi[x_col] + wk[x_col]) / 2
        corner_x = wk[x_col]
        corner_y = wi["elevation_i"]

        # Horizontal segment (wi → corner)
        fig.add_trace(go.Scatter(
            x=[wi[x_col], corner_x],
            y=[wi["elevation_i"], corner_y],
            mode="lines",
            line=dict(color="gray", dash="dot", width=1),
            showlegend=False,
            hoverinfo="skip",
        ))
        # Vertical segment (corner → wk)
        fig.add_trace(go.Scatter(
            x=[corner_x, wk[x_col]],
            y=[corner_y, wk["elevation_i"]],
            mode="lines",
            line=dict(color="gray", dash="dot", width=1),
            showlegend=False,
            hoverinfo="skip",
        ))
        # Hypotenuse (direct line wi → wk)
        fig.add_trace(go.Scatter(
            x=[wi[x_col], wk[x_col]],
            y=[wi["elevation_i"], wk["elevation_i"]],
            mode="lines",
            line=dict(color="lightgray", dash="dash", width=1),
            showlegend=False,
            hoverinfo="skip",
        ))

        # Look up spacing values from IK pairs
        pair = IK[
            (IK["well_i"] == wi["well_i"]) & (IK["well_k"] == wk["well_i"])
        ]
        if not pair.empty:
            h_dist = pair["horizontal_dist"].iloc[0]
            v_dist = pair["vertical_dist"].iloc[0]
            d3d   = pair["dist3d"].iloc[0]
            # Label at midpoint of hypotenuse
            fig.add_annotation(
                x=mid_x,
                y=(wi["elevation_i"] + wk["elevation_i"]) / 2,
                text=(
                    f"H: {h_dist:,.0f} ft<br>"
                    f"V: {v_dist:,.0f} ft<br>"
                    f"3D: {d3d:,.0f} ft"
                ),
                showarrow=False,
                font=dict(size=9, color="dimgray"),
                bgcolor="rgba(255,255,255,0.7)",
            )
```

#### Layer 3 — Formation Top Horizons (optional)

Formation tops are an optional input (the GB calculation does not require them).
When provided, draw one dashed horizontal-ish line per formation spanning the full x range.

**Input**: `df_formation_tops` — columns: `uwi`, `formation`, `top_tvd` (ft, negative convention).

```python
def add_formation_tops(fig, df_formation_tops: pd.DataFrame, GB: pd.DataFrame, x_col: str):
    """
    Draw formation top lines across the gun barrel x range.
    Interpolates TVD from per-well tops; falls back to mean when not per-well.
    """
    x_range = [GB[x_col].min(), GB[x_col].max()]
    formation_colors = {
        "Form A1": "#e41a1c",
        "Form B1": "#377eb8",
        "Form C1": "#4daf4a",
    }

    for formation, grp in df_formation_tops.groupby("formation"):
        # Merge tops TVD with well x-positions for a sloped line
        merged = grp.merge(GB[["well_i", x_col]], left_on="uwi", right_on="well_i", how="inner")
        merged = merged.sort_values(x_col)

        if merged.empty:
            # Fallback: flat line at mean TVD
            mean_tvd = grp["top_tvd"].mean()
            xs = x_range
            ys = [mean_tvd, mean_tvd]
        else:
            xs = list(merged[x_col]) + [x_range[0], x_range[-1]]
            ys = list(merged["top_tvd"]) + [merged["top_tvd"].iloc[0], merged["top_tvd"].iloc[-1]]
            # Sort by x for clean line
            pairs = sorted(zip(xs, ys))
            xs = [p[0] for p in pairs]
            ys = [p[1] for p in pairs]

        color = formation_colors.get(formation, "#888888")
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines",
            name=formation,
            line=dict(color=color, width=1.5, dash="longdash"),
            hovertemplate=f"{formation}: %{{y:,.0f}} ft TVD<extra></extra>",
        ))
```

#### Final Figure Layout

```python
fig.update_layout(
    title="Gun Barrel — Cross-Section View",
    xaxis=dict(
        title="Section Distance (ft)" if centered else "Cumulative Distance (ft)",
        zeroline=True, zerolinecolor="black", zerolinewidth=1,
    ),
    yaxis=dict(
        title="Depth TVD (ft)",
        autorange="reversed",          # deeper = lower on chart
    ),
    legend=dict(title="Bench / Formation"),
    hovermode="closest",
    template="plotly_white",
    height=550,
)
```

#### Data Flow Summary

```text
selected-wells-store  (map click → list of UWIs in neighborhood)
    → filter IK: keep rows where well_i OR well_k in selected UWIs
    → filter HeelToe: keep rows where uwi in selected UWIs

compute_gun_barrel(IK_filtered, HeelToe_filtered)
    → GB DataFrame (sectionDist, elevation_i, bench, well_name, first_prod_date)

IK_filtered spacing pairs
    → horizontal_dist, vertical_dist, dist3d for zigzag labels

df_formation_tops (optional upload in Step 1)
    → formation top lines per bench/formation

All three → layered go.Figure → dcc.Graph(id="gun-barrel-chart")
If selected-wells-store is empty or None → show empty figure with placeholder message
```

---

## Planned Dashboard Panels

### Panel 1: Interactive Map (QGIS-like)

**Library**: `dash-leaflet` with `dl.GeoJSON` — renders GeoPandas GeoJSON natively,
supports per-feature styling callbacks, click events, and spatial filtering.

#### Wellbore Sticks (trajectories)

Build from the directional survey data (post `filter_after_heel_point()`):

```python
from shapely.geometry import LineString, Point
import geopandas as gpd
import json

def build_trajectory_geodataframe(
    df_lateral: pd.DataFrame,
    df_header: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """
    Build a GeoDataFrame of wellbore sticks from directional survey.

    - Each well = one LineString (lat/lon per MD station, sorted by MD)
    - Last point (max MD) = bottom-hole location
    - Joined with header: well_name, bench, first_prod_date, operator
    """
    rows = []
    for uwi, group in df_lateral.groupby("uwi"):
        group = group.sort_values("md")
        # (longitude, latitude) order — GeoJSON / Leaflet convention
        coords = list(zip(group["longitude"], group["latitude"]))
        if len(coords) < 2:
            continue
        rows.append({
            "uwi": uwi,
            "geometry": LineString(coords),
            "bh_lon": coords[-1][0],   # bottom-hole longitude
            "bh_lat": coords[-1][1],   # bottom-hole latitude
        })

    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")

    # Join header attributes for styling and tooltips
    keep_cols = ["uwi", "well_name", "bench", "first_prod_date", "operator",
                 "hole_direction", "spud_date"]
    gdf = gdf.merge(
        df_header[[c for c in keep_cols if c in df_header.columns]],
        on="uwi", how="left"
    )
    return gdf


def build_bottomhole_geodataframe(gdf_trajectories: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Separate GeoDataFrame of bottom-hole Point markers."""
    gdf_bh = gdf_trajectories.copy()
    gdf_bh["geometry"] = gdf_bh.apply(
        lambda r: Point(r["bh_lon"], r["bh_lat"]), axis=1
    )
    return gdf_bh


# Convert to GeoJSON for dash-leaflet
trajectories_geojson = json.loads(gdf_trajectories.to_json())
bottomholes_geojson  = json.loads(gdf_bottomholes.to_json())
```

**Why GeoPandas + GeoJSON → dash-leaflet:**

- Already in `requirements.txt` — no new dependency
- `dl.GeoJSON` renders polylines natively with JS styling callbacks (color by bench)
- Enables spatial operations later: corridor filter = `gdf.within(polygon)`, distance queries
- Click on a feature → returns `feature["properties"]["uwi"]` → drives all other panels

#### dash-leaflet Map Layout

```python
dl.Map(
    center=[lat_center, lon_center],
    zoom=10,
    children=[
        dl.TileLayer(url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"),  # or satellite
        dl.LayersControl([
            dl.Overlay(
                dl.GeoJSON(
                    id="layer-trajectories",
                    data=trajectories_geojson,
                    options={"style": trajectory_style_fn},  # JS colorscale by bench
                ),
                name="Well Trajectories", checked=True,
            ),
            dl.Overlay(
                dl.GeoJSON(
                    id="layer-bottomholes",
                    data=bottomholes_geojson,
                    pointToLayer=bottomhole_marker_fn,  # custom marker icon
                ),
                name="Bottom-hole Markers", checked=True,
            ),
            dl.Overlay(
                dl.GeoJSON(id="layer-spacing-pairs"),
                name="Spacing Pairs", checked=False,
            ),
            dl.Overlay(
                dl.GeoJSON(id="layer-corridor"),
                name="Corridor Filter", checked=True,
            ),
        ]),
        dl.ScaleControl(position="bottomleft"),
    ],
    style={"height": "500px"},
)
```

#### Bench Color Mapping (JS callback injected into dl.GeoJSON)

```javascript
// Passed as options={"style": window.dashExtensions.trajectoryStyle}
window.dashExtensions = window.dashExtensions || {};
window.dashExtensions.trajectoryStyle = function(feature) {
    const benchColors = {
        "WOLFCAMP A":          "#1f77b4",
        "WOLFCAMP B UPPER":    "#ff7f0e",
        "WOLFCAMP B LOWER":    "#2ca02c",
        "3RD BONE SPRING":     "#d62728",
        "3RD BONE SPRING SAND":"#9467bd",
        "BARNETT":             "#8c564b",
        "DEAN":                "#e377c2",
    };
    const bench = feature.properties.bench || "Unknown";
    return {
        color: benchColors[bench] || "#7f7f7f",
        weight: 2,
        opacity: 0.85,
    };
};
```

#### Interaction: Click Well → Filter All Panels

```python
@app.callback(
    Output("selected-wells-store", "data"),
    Input("layer-trajectories", "clickData"),
    Input("layer-bottomholes", "clickData"),
    State("pipeline-result-store", "data"),
    prevent_initial_call=True,
)
def on_well_click(traj_click, bh_click, pipeline_result):
    click = traj_click or bh_click
    if not click:
        return dash.no_update

    clicked_uwi = click["properties"]["uwi"]

    # Load IK from cached pipeline output to find the full neighborhood
    IK = load_cached_ik(pipeline_result)  # thin helper; reads parquet/pickle from disk
    neighborhood = set(
        IK.loc[
            (IK["well_i"] == clicked_uwi) | (IK["well_k"] == clicked_uwi),
            ["well_i", "well_k"]
        ].values.flatten()
    )
    neighborhood.add(clicked_uwi)

    return {
        "clicked_uwi": clicked_uwi,
        "neighborhood_uwis": sorted(neighborhood),  # all wells in GB
    }
```

**Layers** (toggle-able via `dl.LayersControl`):

- ☑ Well trajectories — `LineString` per well, color by bench / year / operator
- ☑ Bottom-hole markers — `Point` at last survey station per well
- ☐ Spacing pair lines — `LineString` between well_i and well_k midpoints, color by `horizontal_dist`
- ☐ Parent-child connections — network edges, toggled separately
- ☑ Corridor filter polygon — `wps_corridor` geometry
- ☐ Township/range (PLSS) — static GeoJSON overlay
- ☐ Lease boundaries — user-uploaded shapefile/GeoJSON
- ☐ Frac hit risk heatmap — `dl.Colorbar` + grid

Basemap selector: OpenStreetMap / Esri Satellite / USGS Topo / CartoDB (dark)

**Tools**: zoom, pan, `dl.ScaleControl`, measure distance (future: `dl.MeasureControl`)

### Panel 2: Gun Barrel Diagram

**Input**: `selected-wells-store` → `neighborhood_uwis` list from map click.
**Never plots all wells** — only the clicked well and its spacing neighbors.

```python
@app.callback(
    Output("gun-barrel-chart", "figure"),
    Input("selected-wells-store", "data"),
    State("pipeline-result-store", "data"),
    prevent_initial_call=True,
)
def update_gun_barrel(selected, pipeline_result):
    if not selected or not selected.get("neighborhood_uwis"):
        return empty_figure("Click a well on the map to populate the gun barrel.")

    uwis = selected["neighborhood_uwis"]
    IK, HeelToe = load_cached_ik_heeltoe(pipeline_result)

    # Filter to only the selected neighborhood
    IK_filtered = IK[IK["well_i"].isin(uwis) & IK["well_k"].isin(uwis)]
    HeelToe_filtered = HeelToe[HeelToe["uwi"].isin(uwis)]

    if IK_filtered.empty:
        return empty_figure("No spacing pairs found for selected well.")

    GB = compute_gun_barrel(IK_filtered, HeelToe_filtered)
    fig = build_gun_barrel_figure(GB, IK_filtered, df_formation_tops)
    return fig
```

- X: `sectionDist` (centered) or `cum_dist` — toggled via radio button
- Y: `elevation_i` (TVD, `autorange="reversed"`)
- Color by: `bench`
- Three trace layers: well points, zigzag spacing lines, formation tops (optional)
- Default state: empty chart with "Click a well on the map" placeholder

### Panel 3: Cumulative Oil — Normalized Time

- X: `normalize_time_months`, Y: `cum_oil`
- Line per well, color by well
- Linked to map selection

### Panel 4: Daily Oil — Production Date

- X: `prod_date`, Y: daily_oil
- Time series lines per well

### Panel 5: PPF / GPF / Lateral Length by Well

- Grouped bar chart (dual Y-axis: production per ft + lateral length)
- X: well_name

### Panel 6: Box Plot

- `cum_oil_180d_per_ft` and `cum_oil_365d_per_ft` distributions
- Grouped by corridor/bench

---

## Column Mapping UI (Critical UX Feature)

Users have files with any column naming convention. The dashboard needs a mapping step:

### Flow

1. User uploads CSV or Excel (header or survey)
2. Dashboard reads headers from uploaded file
3. Shows two-column UI:

   ```text
   Your File Columns    →    Canonical Names
   ─────────────────────────────────────────
   "API 14"             →    [uwi          ▼]
   "Well Name"          →    [well_name    ▼]
   "Surface Lat"        →    [latitude     ▼]
   "Measured Depth FT"  →    [md           ▼]
   "True Vert Depth"    →    [tvd          ▼]
   ```

4. Auto-suggest with fuzzy matching (e.g., "API" → suggests "uwi")
5. User confirms/adjusts → stored in `dcc.Store`
6. All downstream calculations use the confirmed mapping

### Required Canonical Columns

- **Header**: `uwi`, `well_name`, `bench`, `latitude`, `longitude`, `first_prod_date`
- **Survey**: `uwi`, `md`, `tvd`, `latitude`, `longitude`, `azimuth`
- **Optional**: `operator`, `spud_date`, `hole_direction`, `rsv_cat`, `inclination`

---

## Tech Stack

```text
dash>=2.14.0
plotly>=5.18.0
dash-leaflet>=1.0.0           # QGIS-like interactive map
dash-bootstrap-components>=1.5.0
pandas>=2.0.0
geopandas>=0.14.0
pyproj>=3.6.0
thefuzz>=0.20.0               # fuzzy column name matching
openpyxl>=3.1.0               # Excel upload support
```

---

## Entry Point

`dashboard/app.py` (to be created at project root)

```text
well-spacing-analyzer/
├── src/                     (existing)
├── notebooks/               (existing)
├── dashboard/               (TO CREATE)
│   ├── app.py               # main Dash app
│   ├── layout.py            # page layout components
│   ├── callbacks/
│   │   ├── map_callbacks.py
│   │   ├── gb_callbacks.py
│   │   └── production_callbacks.py
│   ├── components/
│   │   ├── map_panel.py
│   │   ├── gun_barrel.py
│   │   ├── column_mapper.py    # column mapping UI component
│   │   └── production_charts.py
│   └── assets/              # CSS, icons
└── requirements-dashboard.txt
```

---

---

## Extended Vision: Parent-Child Diagnostics & Beyond

These go beyond the Spotfire reference — ideas to make this a genuinely better analytical tool.

### Parent-Child Interference Analysis

Inspired by `parent_child_clustering/` notebooks.

#### Panel: Parent-Child Relationship Explorer

- Visual: Network graph (NetworkX + Plotly) showing parent/child well connections
- Node size = lateral length, node color = first_prod_date vintage
- Edge weight/color = horizontal spacing (ft)
- Click a node → gun barrel and production charts update for that neighborhood
- Highlight "at-risk" child wells: those within 500 ft of a producing parent

#### Panel: Frac Hit Risk Heatmap

- Grid overlay on map: color by proximity-weighted density of active parents
- Child wells color-coded: green (safe spacing), yellow (moderate risk), red (close spacing)
- Configurable spacing threshold slider (e.g., 0–2000 ft)

#### Panel: Spacing vs. Production Scatter

- X: `horizontal_dist` to nearest parent (ft)
- Y: `cum_oil_365d_per_ft` (child well performance)
- Color by: bench, vintage, operator
- Trendline overlay — shows interference effect quantitatively
- Hypothesis: closer spacing → lower production → quantify the "sweet spot"

#### Panel: Depletion Timing Analysis

- X: time between parent first_prod_date and child spud_date (months)
- Y: child `cum_oil_365d_per_ft`
- Color by: horizontal spacing bin (<500 ft, 500–1000, 1000–1500, >1500 ft)
- Shows whether waiting longer reduces interference

---

### Well Clustering & Development Pattern Analysis

From `well_bundle_clustering` notebooks.

#### Panel: Well Bundle Map

- HDBSCAN clustering results overlaid on map
- Each cluster = a "development bundle" (wells drilled together)
- Color by cluster, shape by bench
- Click cluster → show all wells in bundle

#### Panel: Infill Opportunity Finder

- Overlay existing wells with a spacing grid
- Highlight grid cells with no wells and sufficient spacing from producers
- Color by recommended bench based on nearby well performance
- Exportable as a target list for drilling planning

---

### Geospatial / QGIS-like Advanced Features

#### Multi-layer Map (like QGIS)

Layer panel (checkbox list):

- ☑ Well trajectories (color by bench / year / operator)
- ☑ Bottom-hole markers
- ☑ Spacing pair lines (color by horizontal_dist gradient)
- ☑ Parent-child connections (network edges)
- ☑ Corridor polygons (wps_corridor filter)
- ☑ Township/range grid (PLSS)
- ☑ Lease boundaries (if uploaded as shapefile)
- ☑ Frac hit risk heatmap
- ☑ Infill opportunity grid

Basemap selector: OpenStreetMap / Satellite / USGS Topo / Blank

#### Spatial Query Tools

- Draw a rectangle/polygon on map → filter all panels to wells within polygon
- Measure tool: click two points → show distance in ft/miles
- Corridor tool: draw a line → show all wells within N ft of the line

#### Shapefile / GeoJSON Upload

- Upload your own lease boundaries, unit plats, or county lines as overlay layers

---

### Production Analytics (Beyond Spotfire)

#### Panel: Type Curve Builder

- Select wells by: bench, vintage, spacing bin, operator
- Auto-compute P10/P50/P90 type curves
- Overlay individual well curves (toggle on/off)
- Export type curve table

#### Panel: EUR Estimator

- Hyperbolic decline fitting (Arps) per well
- Show fitted EUR + uncertainty range
- Color map: EUR per lateral ft by location on the map

#### Panel: Vintage Analysis

- Box plots of `cum_oil_365d_per_ft` grouped by year of first production
- Shows technology improvement over time
- Faceted by bench

#### Panel: Operator Benchmarking

- Bar chart: median `cum_oil_365d_per_ft` by operator
- Scatter: lateral length vs. EUR colored by operator
- Table: statistics by operator (P50 prod, avg spacing, avg lateral length)

---

### Data Quality & QC Tools

#### Panel: Survey Quality Dashboard

- Flag wells with suspicious trajectories (large azimuth jumps, duplicate MD rows)
- Show wells with missing heel point detection
- Map view: color wells by data quality score

#### Panel: Spacing Result QC

- Histogram of horizontal_dist distribution (should be bell-curve-ish)
- Flag outlier pairs (spacing < 100 ft or > 5000 ft)
- Show pairs with `reject_reason` set

---

## Development Priority Order

1. **Column mapping UI** — blocks everything else, must work first
2. **Data loading** — read CSV/Excel via column mapper into canonical DataFrames
3. **Map panel** — well trajectories on interactive basemap with layer control
4. **Gun barrel** — replicate GB function, Plotly scatter
5. **Production charts** — normalized type curves + daily rate
6. **PPF/GPF bar chart + box plots**
7. **Linked filtering** — click map → update all panels
8. **Parent-child network graph** — high-value diagnostic
9. **Spacing vs. production scatter** — quantify interference
10. **Type curve builder** — P10/P50/P90 from selected well groups
11. **Frac hit risk heatmap**
12. **Infill opportunity finder**
13. **Advanced spatial tools** (measure, polygon query, shapefile upload)
14. **EUR estimator + decline fitting**
15. **Export** — download filtered CSV/Excel

---

## Notes

- Use the `dashboard-builder` agent (`.claude/agents/dashboard-builder.md`) when working on dashboard features
- The gun barrel Python function above is the authoritative reference — replicate it exactly
- Spotfire `.dxp` and `.mod` files are available at the locations above for reference
