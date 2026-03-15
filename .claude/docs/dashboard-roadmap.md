# Dashboard Roadmap

## Vision

Replace the Spotfire dashboard with a **fully open-source, locally launchable Dash (Plotly)** application that is:
- Better than Spotfire — QGIS-like spatial capabilities
- Self-contained — no license required, runs with `python dashboard/app.py`
- Interactive — click a well on the map → all panels update
- Geospatially rich — layer control, basemap switching, spatial queries
- Data-agnostic — column mapping UI lets any dataset work without code changes

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

**Gun barrel chart**:
- X-axis: `cum_dist` (ft from reference well)
- Y-axis: `elevation_i` (TVD, negative = deeper)
- Color by: `bench`
- Hover labels: `well_name + first_prod_date`
- Annotations: well name labels on points

---

## Planned Dashboard Panels

### Panel 1: Interactive Map (QGIS-like)
- **Library**: `dash-leaflet` or `plotly go.Scattermapbox`
- **Layers** (toggle-able):
  - Well trajectories (polylines, color by bench or vintage year)
  - Bottom-hole markers
  - Spacing pair lines (color by horizontal_dist)
  - Corridor filter polygon overlay
  - Basemap: OpenStreetMap / Satellite / USGS topo
- **Interactions**: click well → all other panels filter to that well's neighborhood
- **Tools**: zoom, pan, measure distance, layer visibility toggle

### Panel 2: Gun Barrel Diagram
- X: `cum_dist` (ft), Y: `elevation_i` (TVD)
- Scatter + line traces, color by bench
- Well name + date labels
- Dynamic: updates when wells are selected on map

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

### Flow:
1. User uploads CSV or Excel (header or survey)
2. Dashboard reads headers from uploaded file
3. Shows two-column UI:
   ```
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

### Required Canonical Columns:
- **Header**: `uwi`, `well_name`, `bench`, `latitude`, `longitude`, `first_prod_date`
- **Survey**: `uwi`, `md`, `tvd`, `latitude`, `longitude`, `azimuth`
- **Optional**: `operator`, `spud_date`, `hole_direction`, `rsv_cat`, `inclination`

---

## Tech Stack

```
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

```
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
