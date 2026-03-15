"""
Launch the Well Spacing Analyzer dashboard.

Run from the project root:
    python run_dashboard.py

The project root must be on sys.path so both `dashboard` and `src` are importable.
This file lives at the project root, which is exactly what ensures that.
"""

from dashboard.app import app

if __name__ == "__main__":
    app.run(debug=True, port=8050)
