# Update Documentation

Keep CLAUDE.md and `.claude/docs/` in sync with the current state of the code.

$ARGUMENTS

## Phase 1: Read Ground Truth (current code)

Read these files completely:
1. `src/utils/custom_logger.py` — note any new functions/classes
2. `src/utils/database_manager.py` — note supported DB backends, new config classes
3. `src/utils/utils.py` — note new utility functions, changed canonical column names
4. `src/well_data/well_data_manager.py` — note new methods, changed CRS defaults
5. `src/well_data/well_spacing_stats.py` — note new classes, new SpacingResult fields, changed thresholds

Also check `requirements.txt` for new dependencies.

## Phase 2: Diff Against Docs

Read these docs and identify gaps vs. current code:
1. `.claude/CLAUDE.md`
2. `.claude/docs/architecture.md`
3. `.claude/docs/algorithms.md`
4. `.claude/docs/data-formats.md`
5. `README.md`

**Check each doc for**:
- [ ] Module map table — line counts correct? All modules listed?
- [ ] Canonical column names — still accurate?
- [ ] `AlignmentType` thresholds — still 25°/65°?
- [ ] Default UTM zone — still EPSG:32613?
- [ ] Batch size default — still 200,000?
- [ ] `SpacingResult` field list — any new fields added?
- [ ] Database backends list — any new ones?
- [ ] Slash commands table in CLAUDE.md — all commands still exist?
- [ ] Git workflow section — still accurate?

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
