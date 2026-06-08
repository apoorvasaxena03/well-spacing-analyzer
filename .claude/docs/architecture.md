# System Architecture

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT SOURCES                           │
│  CSV / Excel    │    Database (SQL Server, Snowflake, etc.)      │
└────────┬────────┴──────────────────┬────────────────────────────┘
         │                           │
         ▼                           ▼
┌────────────────────────────────────────────┐
│              WellDataLoader                │
│  src/well_data/well_data_manager.py        │
│  - Loads header_df + directional_df        │
│  - Applies column_map: {"Src":"canonical"} │
│  - Validates required canonical columns    │
└─────────────────┬──────────────────────────┘
                  │
                  ▼
┌────────────────────────────────────────────┐
│           GeoSurveyProcessor               │
│  src/well_data/well_data_manager.py        │
│  - lat/lon → UTM (x, y) via pyproj         │
│  - filter_after_heel_point() → laterals    │
│  - get_inclination_by_md() interpolation   │
└─────────────────┬──────────────────────────┘
                  │
                  ▼
┌────────────────────────────────────────────┐
│         WellSpacingCalculator              │
│  src/well_data/well_spacing_stats.py       │
│  - trajectories: Dict[uwi → DataFrame]     │
│  - _calculate_spacing_statistics()         │
│    ├── pre-filter by max_distance_miles    │
│    ├── batch processing (200k pairs/batch) │
│    ├── PARALLEL_LIKE: crossline |Δy(x)|    │
│    ├── OBLIQUE: nearest-projection dist    │
│    └── returns df_spacing (SpacingResult)  │
└─────────────────┬──────────────────────────┘
                  │
                  ▼
┌────────────────────────────────────────────┐
│       OverlappingNeighborhoodRoles         │
│  src/well_data/well_role_assignment.py     │
│  - assign_roles(pairs_df, header_df, ...)  │
│  - overlapping neighborhoods (no chaining) │
│  - parent / child / infill_candidate       │
│  - adds 'role' column → header_df          │
└─────────────────┬──────────────────────────┘
                  │
                  ▼
┌────────────────────────────────────────────┐
│        DirectionalBenchNeighbors           │
│  src/well_data/well_spacing_stats.py       │
│  - summarize(spacing_df, header_df, ...)   │
│  - same-bench neighbors (cutoff_ft=1800)   │
│  - near-bench neighbors (vertical_cutoff)  │
│  - overlap_pct_k_min filter                │
└─────────────────┬──────────────────────────┘
                  │
                  ▼
┌────────────────────────────────────────────┐
│         SpacingNeighborEnricher            │
│  src/well_data/well_spacing_stats.py       │
│  - Joins header attributes onto results    │
│  - Produces "one-line-per-well" output     │
│  - Columns: well_i, same_1, same_2, near_1 │
└─────────────────┬──────────────────────────┘
                  │
                  ▼
         Final DataFrame (CSV / Dashboard)
```

---

## Module Responsibilities

| Module | Responsibility | Key Class |
|--------|---------------|-----------|
| `custom_logger.py` | Logging + run-id tracking | `get_logger()`, `new_run_id()` |
| `database_manager.py` | Multi-DB SQL queries | `SQLAlchemyDBClient` |
| `utils.py` | Data wrangling, column standardization | standalone functions |
| `well_data_manager.py` | Data loading, UTM projection | `WellDataLoader`, `GeoSurveyProcessor` |
| `well_spacing_stats.py` | Spacing computation, neighbors, enrichment | `WellSpacingCalculator`, `DirectionalBenchNeighbors` |
| `well_role_assignment.py` | Parent/child/infill role assignment from spacing pairs (V2) | `OverlappingNeighborhoodRoles` |

---

## Key Design Patterns

### 1. Column Mapping Convention
Single direction: `{"Source Column in File": "canonical_name"}`
All loaders validate required canonical columns after load. Never reverse the mapping.

### 2. Canonical Column Names
- Header: `uwi`, `well_name`, `operator`, `bench`, `spud_date`, `first_prod_date`, `hole_direction`, `rsv_cat`
- Survey: `uwi`, `md`, `tvd`, `latitude`, `longitude`, `azimuth`, `inclination`
- Computed: `x`, `y` (UTM), `is_lateral` (after heel filter)

### 3. DBConfig Protocol
Any new database backend just needs to implement the `DBConfig` Protocol:
```python
class MyDBConfig:
    def get_connection_url(self) -> str: ...
    def get_engine_kwargs(self) -> dict: ...
```

### 4. Batch Processing + Checkpoint
```python
calc._calculate_spacing_statistics(
    save_batches_dir="./batches"  # each batch saved as parquet
)
# Resume after interruption:
calc._load_saved_batches("./batches")
```

### 5. Run-ID Logging
Every execution gets a unique run_id (format: `YYYYMMDD_HHMMSS_<8hex>`).
All modules share the same run_id → enables log correlation across modules.

---

## Dependency Graph

```
well_spacing_stats.py
    └── well_data_manager.py
            └── utils.py
                    └── custom_logger.py
    └── database_manager.py
            └── custom_logger.py
well_role_assignment.py
    └── consumes spacing pairs from well_spacing_stats.py
    └── custom_logger.py
```

---

## Tests & Notebooks

A `pytest` suite lives in `tests/` — **146 tests** across `tests/unit/` (alignment, gun
barrel, map panel, spacing result, utils) and `tests/integration/` (geo survey, pipeline,
spacing calculator). Run with `pytest`.

The notebooks additionally serve as broader end-to-end checks:

| Notebook | Asset | Purpose |
|----------|-------|---------|
| `RingEnergy/v1` | Ring Energy | Basic spacing |
| `RingEnergy/v2` | Ring Energy | Full pipeline + neighbors |
| `Ranch_74_EF/v1` | Ranch 74 | Asset-specific |
| `TPL/tpl_spacing` | TPL | Asset-specific |
| `parent_child_clustering/overlapping_neighborhoods_v1` | Ring Energy | Neighborhood analysis |
| `parent_child_clustering/well_bundle_clustering_v1` | Ring Energy | HDBSCAN clustering |
| `parent_child_clustering/well_bundle_clustering_v2` | Ring Energy | HDBSCAN v2 |

---

_Last updated: 2026-06-07_
