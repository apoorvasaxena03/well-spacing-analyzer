# Run a Spacing Analysis

Guide the user through a complete end-to-end well spacing analysis.

$ARGUMENTS

## Step-by-Step Analysis Guide

### Step 0: Prerequisites Check
```python
# Verify imports work
from src.well_data.well_data_manager import WellDataLoader, GeoSurveyProcessor
from src.well_data.well_spacing_stats import WellSpacingCalculator, DirectionalBenchNeighbors
from src.utils.utils import drop_uwi_duplicates_keep_max_last_prod
```
If imports fail, check `pip install -e .` was run.

---

### Step 1: Prepare Column Maps

**Header column map** — maps YOUR file's column names → canonical names:
```python
header_col_map = {
    "API 14":           "uwi",           # required
    "Well Name":        "well_name",     # required
    "Operator":         "operator",
    "Bench":            "bench",         # required for neighbor analysis
    "Spud Date":        "spud_date",
    "First Prod Date":  "first_prod_date",
    "Hole Direction":   "hole_direction",
    "Rsv Category":     "rsv_cat",
}
```

**Directional survey column map**:
```python
survey_col_map = {
    "API 14":           "uwi",           # required
    "Measured Depth":   "md",            # required
    "True Vert Depth":  "tvd",           # required
    "Latitude":         "latitude",      # required
    "Longitude":        "longitude",     # required
    "Azimuth":          "azimuth",
    "Inclination":      "inclination",
}
```

> The mapping direction is always `{"Source Column in Your File": "canonical_name"}`.
> Never reverse this.

---

### Step 2: Load Data

```python
loader = WellDataLoader()

# From CSV/Excel:
df_header = loader.get_header_data(
    source="path/to/header.csv",
    column_map=header_col_map
)

df_survey = loader.get_directional_data(
    source="path/to/surveys.csv",
    column_map=survey_col_map
)

# From database:
# df_header = loader.get_header_data(source=df_from_db, column_map=header_col_map)
```

**Validate**:
```python
print(f"Header: {len(df_header):,} wells")
print(f"Survey: {df_survey['uwi'].nunique():,} unique wells")
print(f"Header columns: {df_header.columns.tolist()}")
```

---

### Step 3: Deduplicate

```python
df_header = drop_uwi_duplicates_keep_max_last_prod(df_header, uwi_col='uwi')
df_survey = df_survey.drop_duplicates(subset=['uwi', 'md'])
print(f"After dedup: {len(df_header):,} wells")
```

---

### Step 4: Compute UTM Coordinates

```python
geo = GeoSurveyProcessor(
    crs_from="EPSG:4326",    # WGS84 (lat/lon)
    crs_to="EPSG:32613"      # UTM Zone 13N (Midland Basin)
    # For Delaware Basin: EPSG:32614 (UTM Zone 14N)
    # For Anadarko: EPSG:32614
    # Check: https://spatialreference.org/ref/epsg/
)

df_utm = geo.compute_utm_coordinates(df=df_survey)
print(f"UTM range X: {df_utm['x'].min():.0f} – {df_utm['x'].max():.0f} m")
print(f"UTM range Y: {df_utm['y'].min():.0f} – {df_utm['y'].max():.0f} m")
```

---

### Step 5: Filter to Lateral Section

```python
df_lateral = geo.filter_after_heel_point(df=df_utm)
print(f"Before heel filter: {len(df_utm):,} rows")
print(f"After heel filter: {len(df_lateral):,} rows")
print(f"Wells with laterals: {df_lateral['uwi'].nunique():,}")
```

> If `df_lateral` has far fewer wells than `df_utm`, check inclination threshold in `filter_after_heel_point()`.

---

### Step 6: Build Trajectories Dict

```python
trajectories = {
    uwi: group.reset_index(drop=True)
    for uwi, group in df_lateral.groupby('uwi')
}
print(f"Trajectories loaded: {len(trajectories):,} wells")
```

---

### Step 7: Run Spacing Calculation

```python
calc = WellSpacingCalculator(trajectories=trajectories)

df_spacing = calc._calculate_spacing_statistics(
    batch_size=200_000,           # reduce if running out of memory
    max_distance_miles=3.0,       # pre-filter: only consider pairs within this distance
    max_crossline_ft=2500,        # reject parallel pairs wider than this
    save_batches_dir="./batches"  # enable checkpoint/resume
)

print(f"Spacing pairs computed: {len(df_spacing):,}")
print(df_spacing[['well_i', 'well_k', 'horizontal_dist', 'pair_alignment']].head(10))
```

---

### Step 8: Identify Neighbors

```python
neighbors_calc = DirectionalBenchNeighbors()

neighbors = neighbors_calc.summarize(
    spacing_df=df_spacing,
    header_df=df_header,
    cutoff_ft=1800,             # max horizontal distance to call a well a "neighbor"
    vertical_cutoff_ft=150,     # max TVD difference between benches
    overlap_pct_k_min=0.30,     # min 30% lateral overlap to qualify
)
```

---

### Step 9: Enrich and Export

```python
from src.well_data.well_spacing_stats import SpacingNeighborEnricher

enricher = SpacingNeighborEnricher(
    header=df_header,
    spacing=df_spacing,
    neighbors=neighbors
)

df_final = enricher.build(column_order=[
    'uwi', 'well_name', 'bench', 'operator', 'first_prod_date',
    'same_bench_1', 'same_bench_2',
    'near_bench_1', 'near_bench_2',
])

df_final.to_csv("spacing_results.csv", index=False)
print(f"Final output: {len(df_final):,} rows × {len(df_final.columns)} columns")
```

---

### Common Issues & Fixes

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| 0 spacing pairs | max_distance_miles too small | Increase to 4.0 or 5.0 |
| Memory error | batch_size too large | Reduce to 50,000 |
| Wrong coordinates | Wrong UTM zone | Check EPSG for your basin |
| No heel detected | Inclination data missing | Check inclination column exists |
| Duplicate pairs | Duplicate UWIs in survey | Run dedup on survey df |
