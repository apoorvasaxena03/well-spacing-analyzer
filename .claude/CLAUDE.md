# Well Spacing Analyzer — Claude Guide

## What This Project Does

A high-performance Python library for computing **parent/child well spacing** in unconventional (horizontal) oil & gas reservoirs. Given directional survey trajectories and well header data for ~20,000 wells, it computes pairwise spacing metrics (horizontal, vertical, 3D), overlap percentages, alignment classifications, and neighbor identification — in under 15 minutes.

**Author**: Apoorva Saxena (Reservoir Engineer)
**Domain**: Petroleum engineering — unconventional reservoir development (Permian Basin / Midland Basin focus)
**License**: View-only (© 2024–2025)

---

## Module Map

| File | Purpose | Lines |
|------|---------|-------|
| `src/utils/custom_logger.py` | Unified logging with run-id correlation | 335 |
| `src/utils/database_manager.py` | Multi-DB abstraction (Postgres, SQL Server, Databricks, Snowflake, Oracle, SQLite) | 2,036 |
| `src/utils/utils.py` | Data wrangling, column standardization, deduplication, reservoir categorization | 1,290 |
| `src/well_data/well_data_manager.py` | Well data loading (CSV/Excel/DB), UTM projection, lateral section extraction | 2,467 |
| `src/well_data/well_spacing_stats.py` | Core spacing engine — all pairwise metrics, neighbor identification, clustering | 7,644 |

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
  DirectionalBenchNeighbors         ← identify same-bench / neighboring-bench wells
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

# Run a notebook
jupyter notebook notebooks/RingEnergy/well_spacing_RingEnergy_v2.ipynb

# Install package in editable mode
pip install -e .
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

## Future: Open-Source Dashboard

A **Dash (Plotly)** dashboard is planned to replace the current Spotfire workflow.

**Planned entry point**: `dashboard/app.py`
**Use `/dashboard-builder` agent** when working on dashboard features.

Key visualizations planned:
- **Map view**: Well trajectories on satellite/topo basemap
- **Gun barrel diagram**: Cross-sectional TVD vs horizontal position
- **Spacing heatmap**: Pairwise spacing matrix by bench
- **Neighbor graph**: Network visualization of parent/child relationships
- **Production overlay**: Cumulative volumes on spacing plots

See `.claude/docs/dashboard-roadmap.md` for full vision.

---

## Important Notes for Claude

1. **Never assume column names** — always check the column mapping in the notebook/script being worked on
2. **Batch size matters** — 200k pairs/batch is the tested default; reducing it prevents OOM on large datasets
3. **UTM zone is configurable** — don't hardcode EPSG:32613; check the `GeoSurveyProcessor` initialization
4. **`filter_after_heel_point()`** — this is critical; spacing must only be computed on the lateral (horizontal) section, never the vertical/build section
5. **`drop_uwi_duplicates_keep_max_last_prod()`** — always deduplicate before loading into the calculator
6. **No unit tests exist yet** — notebooks serve as integration tests; when adding tests, use `pytest` and place in `tests/`
