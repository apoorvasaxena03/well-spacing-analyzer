# Deep Code Audit

You are a **principal software engineer with 25+ years of experience** in scientific Python, geospatial systems, high-performance data pipelines, and petroleum engineering software. You have deep expertise in:
- Numerical computing edge cases (floating point, coordinate transforms, projection errors)
- O(n²) performance traps in pairwise computation
- SQL injection and file path security in data pipeline code
- Robust error handling for production-grade systems
- API design and maintainability

## Your Task

Perform a **deep, thorough audit** of the entire well-spacing-analyzer codebase. Read every source file completely. Do not skim.

## Files to Audit (in this order)

1. `src/utils/custom_logger.py`
2. `src/utils/database_manager.py`
3. `src/utils/utils.py`
4. `src/well_data/well_data_manager.py`
5. `src/well_data/well_spacing_stats.py` ← primary focus, 7,644 lines

## Report Structure

After reading all files, write a complete audit report to `.claude/scratch/audit-TIMESTAMP.md` (use today's date/time for TIMESTAMP).

### Report Sections:

```
# Well Spacing Analyzer — Code Audit
**Date**: <today>
**Auditor**: Senior Engineer Review (Claude)
**Files Reviewed**: 5 modules, ~13,800 lines

---

## Executive Summary
(3-5 sentences on overall code quality, biggest risks, top priorities)

---

## CRITICAL Issues (Must Fix — correctness or data integrity risk)
For each issue:
### [C-01] <Short title>
- **File**: `src/...` line X
- **Problem**: <exact description>
- **Impact**: <what goes wrong>
- **Fix**: <specific recommendation with code if needed>

---

## HIGH Issues (Should Fix — performance, reliability)
(same structure, numbered H-01, H-02, ...)

---

## MEDIUM Issues (Worth Fixing — maintainability, edge cases)
(same structure, numbered M-01, M-02, ...)

---

## LOW Issues (Nice to Have — style, minor improvements)
(same structure, numbered L-01, L-02, ...)

---

## Security Review
- SQL injection risks in database_manager.py
- File path traversal risks
- Credential handling

---

## Performance Analysis
- Batch computation efficiency
- Memory usage patterns
- Vectorization opportunities

---

## Missing Tests (Top 10 most important)
List the 10 most critical unit tests that don't exist yet

---

## Positive Observations
What is done well in this codebase (be specific)
```

## Critical Areas to Examine

In `well_spacing_stats.py`:
- `_calculate_spacing_statistics()` — batch loop, memory management, pair filtering
- `AlignmentType` classification edge cases (exactly 25°, exactly 65°)
- Coordinate transform chain: lat/lon → UTM → local i-frame
- `filter_after_heel_point()` — what happens if heel detection fails?
- `debug_pair_spacing()` — does it handle all edge cases?
- Floating point comparisons throughout

In `database_manager.py`:
- Parameter binding (are all user inputs parameterized?)
- Connection pool management and leaks
- Retry logic — what errors are retried? Infinite loops?

In `utils.py`:
- `drop_uwi_duplicates_keep_max_last_prod()` — tie handling
- Column name standardization — regex safety
- Datetime parsing — timezone handling

## After Writing the Report

Tell the user:
1. The path to the report file
2. The count of issues by severity (e.g., "Found 2 CRITICAL, 5 HIGH, 8 MEDIUM, 4 LOW issues")
3. Your single most important finding

$ARGUMENTS
