# Update Documentation

Keep CLAUDE.md and `.claude/docs/` in sync with the current state of the code.

$ARGUMENTS

## Phase 1: Read Ground Truth (current code)

Read these files completely.

**Core library (`src/`):**

1. `src/utils/custom_logger.py` — note any new functions/classes
2. `src/utils/database_manager.py` — note supported DB backends, new config classes
3. `src/utils/utils.py` — note new utility functions, changed canonical column names, production helpers (e.g. `calculate_cumulative_volumes_by_period`)
4. `src/well_data/well_data_manager.py` — note new methods, changed CRS defaults
5. `src/well_data/well_spacing_stats.py` — note new classes, new SpacingResult fields, changed thresholds
6. `src/well_data/well_role_assignment.py` — `OverlappingNeighborhoodRoles` (V2): relationship types, role taxonomy, output columns, thresholds

**Dashboard (`dashboard/`):**

7. `dashboard/pipeline.py` — the bridge to `src/`: what runs in batch vs. on-demand, role assignment, cumulative production merge
8. `dashboard/pages/*.py` — the 6-step flow (upload → column_map → configure → calculate → explore → export); note new tabs/panels
9. `dashboard/callbacks/explore_analysis.py` — on-demand DBN/Avg/WPS + diagnostic plots

**Tests:**

10. `tests/` — count test files and `def test_` functions (unit + integration); note what is covered

Also check `requirements.txt` (and `pyproject.toml` / `setup.py`) for new dependencies, and `run_dashboard.py` for the launch entry point.

## Phase 2: Diff Against Docs

Read these docs and identify gaps vs. current code:
1. `.claude/CLAUDE.md`
2. `.claude/docs/architecture.md`
3. `.claude/docs/algorithms.md`
4. `.claude/docs/data-formats.md`
5. `.claude/docs/dashboard-roadmap.md` — mark built vs. still-planned panels accurately
6. `README.md`

**Check each doc for** (core engine):
- [ ] Module map table — line counts correct? All modules listed (incl. `well_role_assignment.py`)?
- [ ] Canonical column names — still accurate?
- [ ] `AlignmentType` thresholds — still 25°/65°?
- [ ] Default UTM zone — still EPSG:32613?
- [ ] Batch size default — still 200,000?
- [ ] `SpacingResult` field list — any new fields added?
- [ ] Database backends list — any new ones?
- [ ] Slash commands table in CLAUDE.md — all commands still exist?
- [ ] Git workflow section — still accurate?

**Check each doc for** (newer surface area — easy to miss):
- [ ] Role assignment (`OverlappingNeighborhoodRoles`) — documented in algorithms.md + architecture.md? Role taxonomy + output columns in data-formats.md?
- [ ] Dashboard — described as built (not "future/planned")? Entry point `run_dashboard.py` correct? 6-step flow + actual `dashboard/` tree match disk?
- [ ] Production metrics — cumulative per-ft columns (oil/gas/water, 180d/365d) and `lateral_length_ft` requirement documented in data-formats.md?
- [ ] Tests — does any doc still claim "no tests exist"? Update with the real count.
- [ ] README.md — does it reference non-existent files (e.g. `src/spacing_calculator.py`) or frame Spotfire as the target rather than what the dashboard replaces?

## Phase 3: Update

For each gap found:
1. Update the relevant doc with accurate information
2. Keep edits minimal — only fix what's wrong, don't rewrite everything
3. Add `_Last updated: <date>_` note at bottom of each file you change

## Phase 4: Report

Tell the user:
- Which files were updated
- What changed (bullet list)
- Any code TODOs found that should become GitHub issues

**Do NOT**:
- Rewrite docs from scratch (preserve structure)
- Add content not in the code
- Remove content that's still accurate
