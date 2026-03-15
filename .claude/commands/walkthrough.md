# Full Code Walkthrough

You are an experienced reservoir engineering software architect. Give a comprehensive, guided walkthrough of the entire well-spacing-analyzer codebase — as if onboarding a senior Python developer who is new to petroleum engineering.

$ARGUMENTS

## Walkthrough Structure

### Part 1: Domain Context (explain first, before any code)
Explain to the reader:
- What "well spacing" means in unconventional reservoirs
- Why it matters: parent-child interference, frac hits, optimal development spacing
- What a directional survey is (MD, TVD, inclination, azimuth)
- What a lateral section is vs. build section vs. vertical section
- What "bench" means (reservoir zone / stratigraphic interval)
- What a gun barrel diagram shows
- Why UTM coordinates are needed (local Cartesian math, not spherical)

### Part 2: Architecture Overview
Read `src/__init__.py` and describe the overall package structure.

Draw the data flow:
```
Raw CSV/Excel/DB
    → WellDataLoader (canonical columns)
    → GeoSurveyProcessor (lat/lon → UTM, filter lateral)
    → WellSpacingCalculator (pairwise metrics)
    → DirectionalBenchNeighbors (neighbor identification)
    → SpacingNeighborEnricher (final enriched output)
```

### Part 3: Module Deep-Dives (read each file fully)

**3a. `src/utils/custom_logger.py`**
- Explain the run_id concept and why it matters for multi-module logging
- Show how to use it: `logger = get_logger(__name__)`

**3b. `src/utils/database_manager.py`**
- Explain the DBConfig Protocol pattern and why it's used
- Walk through supported backends: Postgres, SQL Server, Databricks, Snowflake, Oracle, SQLite
- Explain QueryResult dataclass and why it returns both data + metadata
- Show connection pool + retry pattern

**3c. `src/utils/utils.py`**
- Explain the column naming convention (`{"Source": "canonical"}`)
- Walk through `clean_column_names()` and `standardize_column_names()`
- Explain `drop_uwi_duplicates_keep_max_last_prod()` — why deduplication is critical
- Walk through `compute_rsv_cat()` — reservoir category mapping

**3d. `src/well_data/well_data_manager.py`**
- `WellDataLoader`: how it validates canonical columns after loading
- `GeoSurveyProcessor.compute_utm_coordinates()`: lat/lon → UTM using pyproj
- `GeoSurveyProcessor.filter_after_heel_point()`: how heel detection works, why laterals only
- Walk through a complete data loading sequence

**3e. `src/well_data/well_spacing_stats.py`** (most important)
- Explain `AlignmentType` enum and the three spacing regimes
- Walk through `WellSpacingCalculator` initialization
- Explain `_calculate_spacing_statistics()`: batch loop, pair filtering, checkpoint logic
- Deep dive into spacing algorithm for PARALLEL_LIKE:
  - Build local i-frame (well_i as x-axis)
  - Project well_k into i-frame
  - Find x-overlap band
  - Sample crossline distances |Δy(x)|
  - Compute mean/median/min/contact metrics
- Deep dive into OBLIQUE/PERPENDICULAR:
  - Sample points along well_i
  - Find nearest point on well_k polyline
  - Distance = crossline spacing at each sample
- Explain `SpacingResult` dataclass — all 30+ fields
- Explain `DirectionalBenchNeighbors` — cutoff logic, same-bench vs near-bench
- Explain `debug_pair_spacing()` — how to use it for troubleshooting

### Part 4: Running Your First Analysis
Walk through the `notebooks/RingEnergy/well_spacing_RingEnergy_v2.ipynb` notebook step by step, explaining every cell.

### Part 5: Common Gotchas
List the top 10 things that trip up new users:
1. Wrong UTM zone for a non-Permian Basin dataset
2. Forgetting to filter lateral section first
3. Duplicate UWIs causing pair explosion
4. Column map direction (Source → canonical, NOT canonical → Source)
5. Batch size too large causing OOM
6. ...and more from the code

### Part 6: Where to Add New Features
Explain where in the code to add:
- New spatial filter types (BoxSpec, CorridorSpec patterns)
- New database backends (DBConfig Protocol)
- New spacing metrics (SpacingResult fields)
- New neighbor identification strategies
