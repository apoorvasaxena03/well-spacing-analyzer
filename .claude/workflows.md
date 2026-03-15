# Well Spacing Analyzer — Common Workflows

These are the multi-step patterns you'll use most often. Each workflow references the relevant slash commands.

---

## Workflow 1: New Asset / Field Analysis (End-to-End)

**When**: You have header + directional survey data for a new field and want spacing results.

**Steps**:
1. Create a new notebook directory: `notebooks/<AssetName>/`
2. Copy the template from `notebooks/RingEnergy/well_spacing_RingEnergy_v2.ipynb`
3. Set up column maps for your source data (header + directional)
4. Configure `GeoSurveyProcessor` with the correct UTM zone for your region
5. Run `WellDataLoader` → verify row counts and canonical column presence
6. Run `GeoSurveyProcessor.compute_utm_coordinates()` → check for lat/lon outliers
7. Run `GeoSurveyProcessor.filter_after_heel_point()` → verify heel detection
8. Run `WellSpacingCalculator._calculate_spacing_statistics()` with conservative `max_distance_miles` first
9. Inspect a sample of pairs using `debug_pair_spacing()` to validate results
10. Run `DirectionalBenchNeighbors.summarize()` → tune cutoff_ft and overlap_pct_k_min
11. Run `SpacingNeighborEnricher.build()` → export final CSV

**Use `/run-analysis`** for guided step-by-step instructions.

---

## Workflow 2: Debugging a Spacing Result

**When**: A specific well pair has suspicious spacing output (too large, too small, wrong alignment).

**Steps**:
1. Identify the pair: `well_i` and `well_k` UWIs from the spacing DataFrame
2. Call `debug_pair_spacing(well_i, well_k)` to get `PairArtifacts`
3. Inspect: `pair_alignment`, `angle_deg`, `n_samples`, `reject_reason`
4. Plot the `Xi_utm`, `Xk_utm` trajectory arrays to visualize relative positions
5. Check `Xi_if`, `Xk_if` (local i-frame) for crossline distances
6. If PARALLEL_LIKE: check the overlap band and `dy_p5` for outlier samples
7. If OBLIQUE/PERP: check nearest-projection distances

**Use `/find-bugs well_spacing_stats`** to look for systematic issues.

---

## Workflow 3: Adding a New Database Source

**When**: Data is in SQL Server, Databricks, Snowflake, etc. instead of CSV/Excel.

**Steps**:
1. Choose the right config class: `SqlServerConfig`, `DatabricksConfig`, `SnowflakeConfig`, etc.
2. Initialize `SQLAlchemyDBClient(config=...)`
3. Test connection: `client.test_connection()`
4. Query header data: `client.read_sql("SELECT ... FROM wells WHERE ...")`
5. Pass the resulting DataFrame directly to `WellDataLoader.get_header_data(source=df, ...)`
6. Do the same for directional surveys
7. Never hardcode credentials — use environment variables

---

## Workflow 4: Performance Tuning for Large Datasets

**When**: Processing is taking too long or running out of memory.

**Knobs to tune** (in `_calculate_spacing_statistics()`):

| Parameter | Default | Effect |
|-----------|---------|--------|
| `batch_size` | 200,000 | Reduce if OOM; increase if fast machine |
| `max_distance_miles` | 4.0 | Reduce to pre-filter more aggressively |
| `max_crossline_ft` | 3,000 | Reduce to skip wide parallel pairs |
| `save_batches_dir` | None | Set a path to enable checkpoint/resume |

**Steps**:
1. Start with `max_distance_miles=2.0` for initial exploration
2. Enable `save_batches_dir` before running on full dataset (checkpoint safety)
3. If interrupted, reload with `_load_saved_batches(save_batches_dir)`
4. Monitor memory with `htop` or Task Manager during batch runs

---

## Workflow 5: Feature Branch → PR

**When**: Adding a new feature or fixing a bug.

**Steps**:
1. `git checkout dev && git pull` — start from latest dev
2. `git checkout -b feature/<short-description>`
3. Make changes to `src/` or `notebooks/`
4. Use **`/audit`** or **`/find-bugs`** to review your changes before committing
5. Use **`/commit`** to create a Conventional Commit with proper message
6. Push: `git push -u origin feature/<short-description>`
7. Use **`/create-pr`** to open a PR targeting `dev`
8. After review and merge to `dev`, open a second PR: `dev` → `main`

---

## Workflow 6: Keeping Documentation In Sync

**When**: Code has changed and docs/CLAUDE.md are stale.

**Steps**:
1. Use **`/update-docs`** — it audits code vs. docs in 3 phases
2. Check that canonical column names in CLAUDE.md match current code
3. Check that the module map table still has correct line counts
4. Update algorithm descriptions in `.claude/docs/algorithms.md` if spacing logic changed
5. Commit with `docs: update CLAUDE.md and algorithm docs`

---

## Workflow 7: Starting Dashboard Development

**When**: Ready to build a panel of the Dash dashboard.

**Steps**:
1. Reference `.claude/docs/dashboard-roadmap.md` for architecture plan
2. Use **`dashboard-builder` agent** for Dash-specific guidance
3. Entry point: `dashboard/app.py` (to be created)
4. Data inputs: spacing output CSV + header CSV + production CSV
5. Start with the gun barrel diagram (most unique/valuable visualization)
6. See GB function reference in `.claude/docs/dashboard-roadmap.md`

---

## Workflow 8: Investigating a Bug or Anomaly

**When**: You notice unexpected results or get an error.

**Steps**:
1. Check the run log in `logs/` — find the run_id from the session
2. Use **`/find-bugs <module>`** to investigate the relevant module
3. If it's a data issue: check column maps and `filter_after_heel_point()` output
4. If it's a geometry issue: use `debug_pair_spacing()` to visualize the pair
5. If it's a DB issue: run `client.test_connection()` and check credentials
6. Use **`/create-issue`** to document the bug in GitHub before fixing it
