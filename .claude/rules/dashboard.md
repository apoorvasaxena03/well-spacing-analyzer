---
paths:
  - "dashboard/**/*.py"
---

# Rules when editing dashboard code

## Framework: Dash (Plotly)
- Use `dash-bootstrap-components` for layout (not raw HTML divs)
- All callbacks must have `prevent_initial_call=True` unless the initial state is meaningful
- Use `dcc.Store` for sharing data between callbacks — never use global variables

## Column Mapping State
- The user's column mapping is stored in `dcc.Store(id='column-map-store')`
- ALL data callbacks must read from this store before processing
- If mapping is not set, show a warning and disable downstream panels

## Gun Barrel Diagram
- Use `compute_gun_barrel()` from the GB function reference in `.claude/docs/dashboard-roadmap.md`
- X-axis label: "Cumulative Distance (ft)"
- Y-axis label: "Depth TVD (ft)" — note: values are negative
- Color by: `bench` column
- Hover template must include: well_name, first_prod_date, elevation_i, cum_dist

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
