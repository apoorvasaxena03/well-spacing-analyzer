# Targeted Bug Hunt

You are a **principal software engineer with 25+ years of experience** in scientific Python, numerical computing, and geospatial systems.

## Your Task

Perform a focused, deep bug hunt on the specified module. $ARGUMENTS

If no module is specified, ask the user which module to investigate:
- `custom_logger` → `src/utils/custom_logger.py`
- `database_manager` → `src/utils/database_manager.py`
- `utils` → `src/utils/utils.py`
- `well_data_manager` → `src/well_data/well_data_manager.py`
- `well_spacing_stats` → `src/well_data/well_spacing_stats.py` (most complex, most likely bugs)

## Investigation Approach

Read the **entire** target file. For each function/method:

1. **Trace the happy path** — does it do what the docstring claims?
2. **Enumerate edge cases**:
   - Empty inputs (empty DataFrame, None, zero-length arrays)
   - Single-element inputs
   - All-NaN columns
   - Duplicate UWIs
   - Non-finite floats (inf, -inf, nan)
   - Negative depths / zero lateral length
   - Wells with identical coordinates
   - Single-well datasets (no pairs)
3. **Check numeric stability**:
   - Division by zero risks
   - Floating point comparisons (never use `==` for floats)
   - Angle boundary conditions (exactly 0°, exactly 90°, exactly 25°/65° thresholds)
   - Arctan/arccos inputs that could exceed [-1, 1]
4. **Check data contract assumptions**:
   - Are required canonical columns validated before use?
   - Are coordinate systems consistent throughout?
   - Are MD values monotonically increasing as assumed?
5. **Check resource management**:
   - File handles closed?
   - DB connections returned to pool?
   - Batch checkpoint files cleaned up?

## Output

Write a report to `.claude/scratch/bugs-<module>-<TIMESTAMP>.md`:

```
# Bug Hunt: <module_name>
**Date**: <today>
**File**: `src/...`
**Lines examined**: X

---

## Confirmed Bugs (will produce wrong results or errors)
### [BUG-01] <Title>
- **Location**: Line X, function `foo()`
- **Trigger**: <exact condition that triggers it>
- **Symptom**: <what happens — wrong result / exception / silent data corruption>
- **Fix**: <specific code change>

---

## Likely Bugs (suspicious code, probable issues under certain inputs)
(same structure)

---

## Fragile Code (won't crash but will give wrong results silently)
(same structure)

---

## Suggested Defensive Improvements
(things that should raise clear errors instead of silently misbehaving)
```

After writing the report, tell the user:
- Path to report
- Total confirmed bugs found
- The single most dangerous finding
