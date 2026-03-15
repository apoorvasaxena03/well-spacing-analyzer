---
paths:
  - "notebooks/**/*.ipynb"
---

# Rules when editing Jupyter notebooks

## Notebook Structure
- First cell: imports only (no logic)
- Second cell: configuration / column maps
- Third cell onward: data loading, processing, analysis
- Final cells: results export / visualization

## Column Maps in Notebooks
Always define column maps as standalone dict cells with a comment header:
```python
# Header column map — maps source columns to canonical names
header_col_map = {
    "Source Column": "canonical_name",
}
```
Never scatter column renames throughout the notebook.

## Data Validation Cells
After every load step, include a validation cell:
```python
# Validate
assert 'uwi' in df.columns, "Missing canonical 'uwi' column"
print(f"Loaded {len(df):,} rows, {df['uwi'].nunique():,} unique wells")
```

## No Hardcoded Paths
- File paths should be at the top of the notebook in a config cell
- Never embed absolute paths like `C:\Users\...` in the middle of notebook cells

## Reproducibility
- Set random seeds where applicable (`np.random.seed(42)`)
- Clear outputs before committing: `jupyter nbconvert --ClearOutputPreprocessor.enabled=True`

## UTM Zone
- Always document which UTM zone is being used and why
- Comment: `# UTM Zone 13N (EPSG:32613) — Midland Basin`
