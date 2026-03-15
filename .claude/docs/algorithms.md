# Spacing Algorithms

## Overview

The `WellSpacingCalculator` uses three distinct algorithms depending on the geometric relationship between two well laterals. The algorithm is selected based on the angle between the two wells' azimuths.

---

## Step 1: Alignment Classification

```
angle = absolute angle between well_i and well_k azimuths (0–90°)

if angle ≤ 25°    → PARALLEL_LIKE
if 25° < angle < 65° → OBLIQUE
if angle ≥ 65°    → PERPENDICULAR
```

The angle is computed from the mean azimuth of each well's lateral section.

---

## Step 2a: PARALLEL_LIKE Algorithm

**When**: Wells are roughly going the same direction (e.g., both EW or both NS).

**Physical meaning**: Classic "in-bench" or "cross-bench" parallel spacing — the number that matters most for development planning.

### Steps:
1. **Build local i-frame**: Place well_i along the x-axis. Rotate coordinate system so well_i's mean azimuth = 0.
2. **Project well_k** into this local frame → get `(x_k, y_k)` in i-frame coordinates
3. **Find x-overlap band**: `[max(min_xi, min_xk), min(max_xi, max_xk)]`
4. **Sample crossline distances** at regular intervals along the overlap band
5. **|Δy(x)|** = crossline distance at each sample point
6. **Compute metrics**:
   - `horizontal_dist` = mean |Δy| over overlap band (primary metric)
   - `dy_p5` = 5th percentile (closest approach)
   - `overlap_len_common_ft` = length of overlap band in feet
   - `overlap_pct_i` = overlap_len / LL_i (what % of well_i is "parallel" to well_k)
   - `overlap_pct_k` = overlap_len / LL_k

### Key parameters:
- `max_crossline_ft`: reject pairs where mean |Δy| > this value (default 3,000 ft)
- `n_samples`: number of x-sample points in overlap band

---

## Step 2b: OBLIQUE Algorithm

**When**: Wells cross at an intermediate angle (25–65°).

**Physical meaning**: Common in transition zones, or when wells in different benches have slightly different azimuth. Less common but needs proper handling.

### Steps:
1. **Sample points along well_i** at regular MD intervals
2. **For each sample point on well_i**, find the nearest point on well_k's polyline
3. **Distance** = straight-line distance from sample point to nearest point on well_k
4. **Compute metrics**:
   - `horizontal_dist` = mean distance
   - `contact_len_i_ft` = length of well_i within `contact_threshold_ft` of well_k
   - `contact_pct_i` = contact_len_i_ft / LL_i
   - `proj_coverage_i_pct` = % of well_i that has a valid projection onto well_k

---

## Step 2c: PERPENDICULAR Algorithm

**When**: Wells cross at near-right angles (≥ 65°).

**Physical meaning**: Perpendicular wells — rare in practice but occurs in crossing development programs or when mixing NS and EW wells.

**Algorithm**: Same as OBLIQUE (nearest-projection), but the interpretation differs. The "spacing" here is the closest approach distance, not a parallel spacing.

---

## Coordinate System Details

### UTM Transformation
```
Input: latitude, longitude (WGS84, EPSG:4326)
Output: x, y (UTM meters, default EPSG:32613 = UTM Zone 13N)

UTM Zone by Basin:
- Midland Basin: Zone 13N (EPSG:32613)  ← project default
- Delaware Basin: Zone 14N (EPSG:32614)
- Anadarko: Zone 14N (EPSG:32614)
- Bakken: Zone 13N or 14N depending on longitude
```

### Local i-Frame
The local i-frame is constructed per-pair:
- **Origin**: heel point of well_i
- **x-axis**: direction of well_i's mean azimuth (along-well direction)
- **y-axis**: perpendicular to x-axis (crossline direction)
- This means |Δy| = true crossline spacing, independent of absolute coordinates

---

## SpacingResult Field Reference

| Field | Units | Applies to |
|-------|-------|------------|
| `horizontal_dist` | ft | All pairs |
| `vertical_dist` | ft | All pairs |
| `dist3d` | ft | All pairs |
| `angle_deg` | degrees | All pairs |
| `pair_alignment` | enum | All pairs |
| `overlap_len_common_ft` | ft | PARALLEL_LIKE |
| `overlap_pct_i` | 0–1 | PARALLEL_LIKE |
| `overlap_pct_k` | 0–1 | PARALLEL_LIKE |
| `LL_i`, `LL_k` | ft | PARALLEL_LIKE |
| `dy_p5` | ft | PARALLEL_LIKE |
| `contact_threshold_ft` | ft | OBLIQUE/PERP |
| `contact_len_i_ft` | ft | OBLIQUE/PERP |
| `contact_pct_i` | 0–1 | OBLIQUE/PERP |
| `proj_coverage_i_pct` | 0–1 | OBLIQUE/PERP |
| `n_samples` | count | All pairs |
| `reject_reason` | string | Rejected pairs |
| `direction_to_k_from_i_axis` | N/S/E/W | All pairs |
| `drill_direction_i/k` | NS/EW | All pairs |

---

## Neighbor Identification (DirectionalBenchNeighbors)

After computing all pairwise spacing, neighbors are identified by:

1. **Filter to PARALLEL_LIKE pairs** (neighbors must be roughly parallel)
2. **Apply spatial cutoffs**:
   - `horizontal_dist ≤ cutoff_ft` (default: 1,800 ft)
   - `vertical_dist ≤ vertical_cutoff_ft` (default: 150 ft TVD)
   - `overlap_pct_k ≥ overlap_pct_k_min` (default: 0.30 = 30% overlap)
3. **Classify as same-bench or near-bench** based on `bench` column in header
4. **Rank by distance** → same_1 (closest), same_2 (second closest), near_1, near_2
