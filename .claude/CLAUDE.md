# Well Spacing Analyzer — Claude Guide

## What This Project Does

A high-performance Python library for computing **parent/child well spacing** in unconventional (horizontal) oil & gas reservoirs. Given directional survey trajectories and well header data for ~20,000 wells, it computes pairwise spacing metrics (horizontal, vertical, 3D), overlap percentages, alignment classifications, and neighbor identification — in under 15 minutes.

**Author**: Apoorva Saxena (Reservoir Engineer)
**Domain**: Petroleum engineering — unconventional reservoir development (Permian Basin / Midland Basin focus)
**License**: View-only (© 2024–2025)

---

## Module Map

### Core library (`src/`)

| File | Purpose | Lines |
|------|---------|-------|
| `src/utils/custom_logger.py` | Unified logging with run-id correlation | 335 |
| `src/utils/database_manager.py` | Multi-DB abstraction (Postgres, SQL Server, Databricks, Snowflake, Oracle, SQLite) | 2,036 |
| `src/utils/utils.py` | Data wrangling, column standardization, deduplication, reservoir categorization, cumulative production | 1,308 |
| `src/well_data/well_data_manager.py` | Well data loading (CSV/Excel/DB), UTM projection, lateral section extraction | 2,482 |
| `src/well_data/well_spacing_stats.py` | Core spacing engine — all pairwise metrics, neighbor identification, clustering | 7,644 |
| `src/well_data/well_role_assignment.py` | `OverlappingNeighborhoodRoles` (V2) — parent/child role assignment from spacing pairs | 964 |

### Dashboard (`dashboard/`) — the Dash app that wraps the library

| File | Purpose | Lines |
|------|---------|-------|
| `dashboard/pipeline.py` | Bridge: drives the `src/` pipeline + role assignment + cumulative production from UI inputs | 1,113 |
| `dashboard/pages/01_upload.py` | Step 1 — file upload + DB query input (header, directional, production) | 485 |
| `dashboard/pages/02_column_map.py` | Step 2 — column mapping UI (templates + fuzzy matching) | 774 |
| `dashboard/pages/03_configure.py` | Step 3 — spacing, role assignment, and advanced engine params | 1,003 |
| `dashboard/pages/04_calculate.py` | Step 4 — run calculation | 343 |
| `dashboard/pages/05_explore.py` | Step 5 — Map / Gun Barrel / Charts (Neighborhood + Statistics) / Analysis tabs | 3,337 |
| `dashboard/pages/06_export.py` | Step 6 — export results + session package import/export | 336 |
| `dashboard/callbacks/explore_analysis.py` | On-demand DBN/Avg/WPS runs + matplotlib diagnostic plots | 698 |
| `dashboard/components/` | `gun_barrel.py`, `map_panel.py`, `matplotlib_render.py` | — |

**Entry point**: `python run_dashboard.py` (or `python dashboard/app.py`). Use `--debug` for verbose logging.

---

## Architecture & Data Flow

```
Raw Data (CSV / Excel / Database)
        │
        ▼
  WellDataLoader                    ← loads header_df + directional_df
        │
        ▼
  GeoSurveyProcessor                ← lat/lon → UTM(x,y), filter lateral section
        │
        ▼
  WellSpacingCalculator             ← batch pairwise spacing (200k pairs/batch)
        │
        ▼
  OverlappingNeighborhoodRoles      ← assign parent/child roles from spacing pairs (V2)
        │
        ▼
  DirectionalBenchNeighbors         ← identify same-bench / neighboring-bench wells (on-demand)
        │
        ▼
  SpacingNeighborEnricher           ← join header attributes onto spacing results
        │
        ▼
  Final DataFrame (CSV / DB / Dashboard)
```

---

## Key Concepts for Claude

### Column Mapping Convention
All data loaders use a single mapping format — always `{"Source Column": "canonical_name"}`. This is critical:
```python
column_map = {
    "API 14": "uwi",
    "Well Name": "well_name",
    "Surface Latitude": "latitude",
    "Surface Longitude": "longitude",
}
```

### Canonical Column Names
**Header**: `uwi`, `well_name`, `operator`, `bench`, `spud_date`, `first_prod_date`, `hole_direction`, `rsv_cat`
**Directional**: `uwi`, `md`, `tvd`, `latitude`, `longitude`, `azimuth`, `inclination`

### Coordinate System
- Default CRS: WGS84 (EPSG:4326) → UTM Zone 13N (EPSG:32613) for Midland Basin
- Configurable via `GeoSurveyProcessor(crs_from=..., crs_to=...)`
- Local i-frame: well_i defines the x-axis; crossline distance is Δy

### Alignment Classification (AlignmentType enum)
| Class | Angle | Algorithm |
|-------|-------|-----------|
| PARALLEL_LIKE | ≤ 25° | Crossline |Δy(x)| over overlap band |
| OBLIQUE | 25–65° | Nearest-projection from i-samples to k-polyline |
| PERPENDICULAR | ≥ 65° | Same as oblique |
| MISALIGNED | (optional) | Rejected pairs |

### Batch Processing & Checkpointing
```python
calculator._calculate_spacing_statistics(
    batch_size=200_000,          # memory limit per batch
    max_distance_miles=4.0,      # pre-filter by spatial bounds
    save_batches_dir="./batches" # checkpoint directory for resume
)
```

### Run ID Logging
Every session gets a unique run_id (format: `YYYYMMDD_HHMMSS_<8hex>`) injected into all log records. Use `set_run_id()` to correlate cross-module logs.

---

## Git Workflow

- **Main branch**: `main` — production-ready only
- **Dev branch**: `dev` — integration branch
- **Feature branches**: `feature/<description>` → PR → `dev` → merge to `main`
- **Commit style**: Conventional Commits (`feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `perf:`)
- **PRs**: Always include summary bullets + test plan checklist

---

## Quick Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the dashboard (primary entry point)
python run_dashboard.py            # add --debug for verbose logging

# Run a notebook
jupyter notebook notebooks/RingEnergy/well_spacing_RingEnergy_v2.ipynb

# Install package in editable mode
pip install -e .

# Run the test suite
pytest                             # 127 tests across tests/unit + tests/integration
```

---

## Slash Commands Reference

| Command | Purpose |
|---------|---------|
| `/audit` | Deep code audit across all modules (25+ yr senior engineer perspective) → saves report to `.claude/scratch/` |
| `/find-bugs [module]` | Targeted bug hunt on a specific module |
| `/walkthrough` | Full guided code walkthrough for deep understanding |
| `/run-analysis` | Step-by-step guide for running a spacing analysis |
| `/commit` | Stage changes and create a Conventional Commit |
| `/create-pr` | Create a GitHub PR from current branch |
| `/create-issue` | Create a GitHub issue from current bug/feature context |
| `/update-docs` | Sync CLAUDE.md and docs/ with current code state |

---

## Open-Source Dashboard (built — replaces Spotfire)

A **Dash (Plotly)** dashboard implements the full guided workflow end-to-end. It is the
primary way the tool is now used; the `src/` library is the unchanged foundation it calls.

**Entry point**: `python run_dashboard.py`
**Use `/dashboard-builder` agent** when working on dashboard features.

6-step guided flow (one page per step under `dashboard/pages/`):
**Upload → Map Columns → Configure → Calculate → Explore → Export**

Built and working today:

- **Map view**: well trajectories + bottom-holes on dash-leaflet basemap; color by bench/role/operator/year; neighborhood edge lines; measure tool; collapsible legend
- **Gun barrel diagram**: cross-sectional TVD vs horizontal position with spacing zigzag lines
- **Charts → Statistics**: production-by-role box plots (cum oil/gas/water 180d & 365d per ft)
- **Analysis tab**: on-demand DBN / Avg-spacing / Floating WPS runs + matplotlib diagnostic plots
- **Role assignment**: `OverlappingNeighborhoodRoles` (V2) wired through pipeline → header `role` column
- **Session persistence**: export/import a session package (results + UI state)

Still on the roadmap (not yet built): parent-child network graph, spacing-vs-production
scatter, type-curve builder, frac-hit risk heatmap, infill finder, QC panels.

See `.claude/docs/dashboard-roadmap.md` for the full vision and remaining panels.

---

## Important Notes for Claude

1. **Never assume column names** — always check the column mapping in the notebook/script being worked on
2. **Batch size matters** — 200k pairs/batch is the tested default; reducing it prevents OOM on large datasets
3. **UTM zone is configurable** — don't hardcode EPSG:32613; check the `GeoSurveyProcessor` initialization
4. **`filter_after_heel_point()`** — this is critical; spacing must only be computed on the lateral (horizontal) section, never the vertical/build section
5. **`drop_uwi_duplicates_keep_max_last_prod()`** — always deduplicate before loading into the calculator
6. **Tests live in `tests/`** — 127 tests across `tests/unit/` (alignment, gun barrel, map panel, spacing result, utils) and `tests/integration/` (geo survey, pipeline, spacing calculator). Run with `pytest`. Notebooks still serve as broader end-to-end checks.
