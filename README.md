# Well Spacing Analyzer

A high-performance Python tool for computing **parent/child well spacing** in unconventional reservoirs, plus an interactive **Dash dashboard** that runs the whole workflow without code. It processes directional survey and header data from Excel, CSV, or database sources, delivering results for large datasets (e.g., ~20,000 wells in the Midland Basin) in under 15 minutes.

## Features

- **Flexible data input**: directional survey, header, and production data from CSV, Excel, or database queries (Postgres, MySQL, SQL Server, Databricks, Snowflake, Oracle, SQLite).
- **Efficient processing**: batch pairwise spacing (200k pairs/batch) with checkpoint/resume for large datasets.
- **Spacing metrics**: horizontal / vertical / 3D distances, overlap percentages, alignment classification (parallel / oblique / perpendicular), and neighbor identification.
- **Parent/child role assignment**: `OverlappingNeighborhoodRoles` (V2) labels each well parent / child / infill_candidate from spacing pairs and completion dates.
- **Cumulative production**: 180/365-day cumulative oil/gas/water, normalized per lateral foot.
- **Interactive dashboard**: a guided, open-source [Dash](https://dash.plotly.com/) app (map, gun barrel, production-by-role charts, on-demand diagnostics) that **replaces the legacy Spotfire workflow** — no notebooks or config files required.

## Repository Structure

```text
├── src/                         # Core Python library
│   ├── utils/                   # logging, multi-DB manager, data wrangling
│   └── well_data/               # data loading + UTM, spacing engine, role assignment
├── dashboard/                   # Dash app (upload → map → configure → calculate → explore → export)
├── notebooks/                   # Jupyter notebooks (power-user / integration checks)
├── tests/                       # pytest suite (unit + integration)
├── run_dashboard.py             # dashboard launcher
├── requirements.txt             # Python dependencies
└── README.md
```

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/apoorvasaxena03/well-spacing-analyzer.git
   cd well-spacing-analyzer
   ```

2. **Create a virtual environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate     # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Option A — Dashboard (no code)

```bash
python run_dashboard.py          # add --debug for verbose logging
```

Then open the app in your browser and follow the guided flow: **Upload → Map Columns →
Configure → Calculate → Explore → Export**. Header, directional survey, and production
files are mapped to canonical column names in the app — any naming convention works.

### Option B — Library / notebooks (power users)

Use the `src/` classes directly (`WellDataLoader` → `GeoSurveyProcessor` →
`WellSpacingCalculator` → `OverlappingNeighborhoodRoles`). See
`notebooks/RingEnergy/well_spacing_RingEnergy_v2.ipynb` for a complete reference run, and
`.claude/docs/` for architecture, algorithms, and data-format references.

## Tests

```bash
pytest                           # unit + integration suite under tests/
```

## License

© 2025 Apoorva Saxena. This repository is shared for **viewing purposes only**. Redistribution, modification, or commercial use is prohibited without written permission.

## Author

**Apoorva Saxena**  
Reservoir Engineer  
[LinkedIn](https://www.linkedin.com/in/apoorvasaxena)  
[GitHub](https://github.com/apoorvasaxena03)
