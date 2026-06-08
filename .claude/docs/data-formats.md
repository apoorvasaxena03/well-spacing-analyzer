# Data Formats & Column Conventions

## Column Mapping Convention

**Always**: `{"Source Column in Your File": "canonical_name"}`

```python
# ✅ Correct
column_map = {"API 14": "uwi", "Well Name": "well_name"}

# ❌ Wrong — reversed!
column_map = {"uwi": "API 14", "well_name": "Well Name"}
```

The loader uses your map to rename FROM your file's columns TO canonical names.

---

## Header DataFrame (`header_df`)

| Canonical Name | Type | Required | Description |
|---------------|------|----------|-------------|
| `uwi` | str | ✅ | Unique Well Identifier (API-14 format) |
| `well_name` | str | ✅ | Well name/label |
| `operator` | str | | Operating company |
| `bench` | str | ✅* | Reservoir zone (e.g., "WOLFCAMP A", "3RD BONE SPRING") |
| `spud_date` | datetime | | Drilling start date |
| `first_prod_date` | datetime | | First production date |
| `hole_direction` | str | | "HORIZONTAL" or "VERTICAL" |
| `rsv_cat` | str | | Reserves category (e.g., "01PDP", "02PA") |

*Required for neighbor identification by bench.

---

## Directional Survey DataFrame (`directional_df`)

| Canonical Name | Type | Required | Description |
|---------------|------|----------|-------------|
| `uwi` | str | ✅ | Unique Well Identifier |
| `md` | float | ✅ | Measured depth (ft), monotonically increasing |
| `tvd` | float | ✅ | True vertical depth (ft, positive = deeper) |
| `latitude` | float | ✅ | Surface or survey latitude (decimal degrees) |
| `longitude` | float | ✅ | Surface or survey longitude (decimal degrees) |
| `azimuth` | float | ✅ | Wellbore azimuth (degrees, 0=N, 90=E, 180=S, 270=W) |
| `inclination` | float | | Inclination from vertical (degrees, 0=vertical, 90=horizontal) |

After `compute_utm_coordinates()`, adds:
- `x` (float): UTM easting (meters)
- `y` (float): UTM northing (meters)

After `filter_after_heel_point()`, adds:
- `is_lateral` (bool): True for rows in the lateral section

---

## Spacing Results DataFrame (`df_spacing`)

Primary output of `_calculate_spacing_statistics()`.

| Column | Type | Description |
|--------|------|-------------|
| `well_i` | str | Reference well UWI |
| `well_k` | str | Comparison well UWI |
| `horizontal_dist` | float | Mean crossline spacing (ft) |
| `vertical_dist` | float | Mean TVD difference (ft) |
| `3D_dist` | float | 3D distance (ft) |
| `angle_deg` | float | Angle between azimuths (0–90°) |
| `pair_alignment` | str | PARALLEL_LIKE / OBLIQUE / PERPENDICULAR |
| `overlap_len_common_ft` | float | Lateral overlap length (ft), parallel-like only |
| `overlap_pct_i` | float | overlap / LL_i, parallel-like only |
| `overlap_pct_k` | float | overlap / LL_k, parallel-like only |
| `LL_i`, `LL_k` | float | Lateral lengths (ft) |
| `drill_direction_i` | str | "NS" or "EW" |
| `drill_direction_k` | str | "NS" or "EW" |
| `direction_to_k_from_i_axis` | str | Cardinal direction N/S/E/W |
| `elevation_i` | float | TVD of well_i midpoint (ft, negative = depth) |
| `mid_lat`, `mid_lon` | float | Midpoint lat/lon of well_i |
| `n_samples` | int | Sample points used in calculation |
| `reject_reason` | str | Why pair was rejected (empty if accepted) |

---

## Neighbor Summary Output

Output of `DirectionalBenchNeighbors.summarize()`.

One row per well (`well_i`), with neighbor columns:

| Column Pattern | Description |
|---------------|-------------|
| `same_bench_1_uwi` | Closest same-bench neighbor UWI |
| `same_bench_1_dist` | Horizontal distance to same_bench_1 |
| `same_bench_1_overlap` | Overlap % with same_bench_1 |
| `same_bench_2_*` | Second closest same-bench neighbor |
| `near_bench_1_*` | Closest different-bench neighbor |
| `near_bench_2_*` | Second closest different-bench neighbor |

---

## Gun Barrel Data (`GB DataFrame`)

Input to gun barrel diagram. Derived from spacing pairs + heel/toe midpoints.

| Column | Description |
|--------|-------------|
| `well_i` | Well UWI |
| `elevation_i` | TVD depth (negative, ft) |
| `drill_direction_i` | NS or EW |
| `mid_lat` | Midpoint latitude |
| `mid_lon` | Midpoint longitude |
| `horizontal_dist` | Spacing to adjacent well in sorted order (ft) |
| `cum_dist` | Cumulative distance from leftmost/southernmost well (ft) |
| `E_to_W_Rank` | Rank W→E (NS wells, sorted by longitude) |
| `N_to_S_Rank` | Rank S→N (EW wells, sorted by latitude) |

**Gun barrel plot**: X = `cum_dist`, Y = `elevation_i`, Color = `bench`

---

## Production DataFrame

Monthly production input (one row per well per month):

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `uwi` | str | ✅ | Well UWI |
| `prod_date` | datetime | ✅ | Production month |
| `oil` | float | ✅* | Monthly oil production (bbl) |
| `gas` | float | ✅* | Monthly gas production (mcf) |
| `water` | float | ✅* | Monthly water production (bbl) |

*At least one of oil/gas/water is required. Per-foot normalization also needs
`lateral_length_ft` on the **header** (joined onto production before the cum calc).

### Cumulative metrics (`calculate_cumulative_volumes_by_period`)

One row per UWI. Windows are based on actual calendar days per production month
(`cum_days ≤ 180` / `≤ 365`). Computed in `src/utils/utils.py` and merged onto
`header_df` by the dashboard pipeline:

| Column | Description |
|--------|-------------|
| `cum_oil_180d`, `cum_gas_180d`, `cum_water_180d` | 180-day cumulative volumes |
| `cum_oil_365d`, `cum_gas_365d`, `cum_water_365d` | 365-day cumulative volumes |
| `cum_oil_180d_per_ft` … `cum_water_365d_per_ft` | Above, normalized by `lateral_length_ft` |

The `*_per_ft` columns drive the dashboard's **production-by-role** box plots.

---

## Role Assignment Output (`OverlappingNeighborhoodRoles`)

One row per well. Produced by `src/well_data/well_role_assignment.py` and merged
onto `header_df` (column `role`). See `algorithms.md` for the assignment logic.

| Column | Type | Description |
|--------|------|-------------|
| `uwi` | str | Well UWI |
| `role` | str | `parent` / `child` / `infill_candidate` / `no_eligible_neighbor` |
| `child_gen` | str | `gen1_child` / `late_child` (child wells only) |
| `isolated` | bool | True when the well has no eligible neighbors |
| `parent_uwi` | str | Nearest older eligible neighbor (the parent) |
| `parent_dist_ft` | float | Distance to parent (ft) |
| `parent_vertical_ft` | float | Vertical (TVD) distance to parent (ft) |
| `parent_pair_type` | str | Relationship type to parent (e.g. `parallel_same_bench`) |
| `parent_confidence` | str | `high` / `medium` / `low` |
| `parent_comp_date` | datetime | Parent completion date |
| `days_since_parent` | float | Days between parent and this well's completion |
| `child_above_parent` | bool | True when this well's TVD is shallower than the parent |
| `n_eligible_neighbors` | int | Count of eligible neighbors |
| `n_older_eligible` | int | Count of older eligible neighbors |
| `dominant_pair_type` | str | Most common contributing relationship type |

---

_Last updated: 2026-06-07_
