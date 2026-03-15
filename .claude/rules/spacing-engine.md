---
paths:
  - "src/well_data/well_spacing_stats.py"
---

# Rules when editing well_spacing_stats.py

This is the core spacing engine (~7,644 lines). Extra care required:

## Numeric Safety
- Never use `==` to compare floats — use `abs(a - b) < epsilon` or `np.isclose()`
- Always guard `arccos` / `arcsin` inputs: clamp to `[-1.0, 1.0]` before calling
- Check for `np.inf`, `np.nan` in all distance/angle results before returning

## AlignmentType Boundaries
- `PARALLEL_LIKE`: angle ≤ 25.0° (inclusive)
- `OBLIQUE`: 25.0° < angle < 65.0°
- `PERPENDICULAR`: angle ≥ 65.0° (inclusive)
- Boundary wells (exactly 25° or 65°) go to PARALLEL_LIKE and PERPENDICULAR respectively

## Coordinate Convention
- `x` = along-well direction in local i-frame
- `y` = crossline (perpendicular) direction in local i-frame
- `horizontal_dist` = mean |Δy| over the x-overlap band — NEVER the Euclidean distance
- All distances in **feet** (UTM is meters — convert: `meters * 3.28084`)

## Batch Processing
- Never load all pairs into memory at once for datasets > 10k wells
- Always respect `batch_size` parameter
- If adding new output columns to `SpacingResult`, add to dataclass AND update batch assembly code

## SpacingResult
- Fields are frozen dataclass — adding fields requires updating ALL callers
- `reject_reason` should be a non-empty string when pair is rejected, empty string otherwise
- `n_samples` must always be ≥ 1 for non-rejected pairs
