---
paths:
  - "dashboard/**/*.py"
---

# Rules when editing dashboard code

## Critical: Never Modify src/

The `src/` library is the authoritative backend. The dashboard wraps it — never forks it.

- `dashboard/pipeline.py` calls `WellDataLoader`, `GeoSurveyProcessor`, `WellSpacingCalculator`,
  `DirectionalBenchNeighbors`, `SpacingNeighborEnricher` exactly as notebooks do
- If a bug is found in `src/`, fix it in `src/` — not in `dashboard/pipeline.py`
- If a new feature is needed in the pipeline, add it to `src/` first, then expose it in `pipeline.py`
- The notebook workflows must continue to work unchanged after any dashboard change

## App Structure (6 steps)

The app is a guided flow: Upload → Column Mapping → Configure → Calculate → Explore → Export.
Each step is a separate page in `dashboard/pages/`. The `dcc.Store` components pass state between pages:

- `upload-store` → raw file bytes + filenames
- `column-map-store` → confirmed `{source: canonical}` mappings per file
- `config-store` → UTM zone, max_distance_miles, cutoff_ft, etc.
- `pipeline-result-store` → path to cached pipeline output (not the data itself — too large)
- `selected-wells-store` → currently selected UWI(s) from map click

## Framework: Dash (Plotly)
- Use `dash-bootstrap-components` for layout (not raw HTML divs)
- All callbacks must have `prevent_initial_call=True` unless the initial state is meaningful
- Use `dcc.Store` for sharing data between callbacks — never use global variables

## Column Mapping State
- The user's column mapping is stored in `dcc.Store(id='column-map-store')`
- ALL data callbacks must read from this store before processing
- If mapping is not set, show a warning and disable downstream panels

## Map ↔ Gun Barrel Link (Critical)

The gun barrel **must only show wells selected on the map**. It is never populated on page load.

- The `update_gun_barrel` callback takes `Input("selected-wells-store", "data")` — the store set by `on_well_click`
- `selected-wells-store` contains `{"clicked_uwi": str, "neighborhood_uwis": [str, ...]}`
- Filter IK pairs: keep rows where **both** `well_i` AND `well_k` are in `neighborhood_uwis`
  (this ensures only intra-neighborhood pairs are drawn — no dangling connections to outside wells)
- Filter HeelToe: keep rows where `uwi` in `neighborhood_uwis`
- If `selected-wells-store` is empty/None → return `empty_figure("Click a well on the map to populate the gun barrel.")`
- Never pass the full IK DataFrame to `compute_gun_barrel()` — always pre-filter

## Gun Barrel Diagram
- Use `compute_gun_barrel()` from `.claude/docs/dashboard-roadmap.md` as the data foundation
- Full enhanced design is in the "Enhanced Gun Barrel Chart" section of that file — follow it exactly
- Three mandatory trace layers: (1) well points, (2) spacing zigzag lines, (3) formation tops (optional)
- X-axis: togglable — "Cumulative Distance (ft)" (`cum_dist`) or "Section Distance (ft)" (`sectionDist`)
- `sectionDist` = `cum_dist - cum_dist.max() / 2` (centered at 0)
- Y-axis label: "Depth TVD (ft)" — use `autorange="reversed"` (deeper = lower on plot)
- Color by: `bench` column
- Spacing labels (horizontal_dist, vertical_dist, dist3d) come from the IK spacing DataFrame — never recompute them
- Formation tops are optional input; hide the trace layer gracefully when `df_formation_tops` is None

## Map Panel
- Use `dash-leaflet` for the interactive map (not `dcc.Graph` with Mapbox — requires token)
- Default zoom level: fit to data bounds
- Well trajectory polylines: use `dl.Polyline` with `color` from bench colormap
- Click on well → store `clicked_uwi` in `dcc.Store(id='selected-wells-store')`

## Performance
- Never load entire spacing DataFrame into a Dash callback — use filtered views
- Use `pd.DataFrame.to_dict('records')` for DataTable data (not `.to_json()`)
- Cache expensive computations with `@cache.memoize()` (use `flask-caching`)

## File Upload
- Use `dcc.Upload` for CSV/Excel intake
- Parse uploaded file immediately; store result as JSON in `dcc.Store`
- Show column mapping UI immediately after upload (before any analysis)

## Error Handling in Callbacks
- Always return a user-friendly error message (in an alert component) — never let callbacks crash silently
- Use `dash.no_update` to skip updates when input is invalid
