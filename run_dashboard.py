"""
Launch the Well Spacing Analyzer dashboard.

Run from the project root:
    python run_dashboard.py

The project root must be on sys.path so both `dashboard` and `src` are importable.
This file lives at the project root, which is exactly what ensures that.
"""

import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

CACHE_DIR = Path("./dashboard/.cache")


def _clear_cache():
    """Delete the DiskcacheManager cache directory.

    Removes stale background job state so the next run starts clean.
    """
    if CACHE_DIR.exists():
        try:
            shutil.rmtree(CACHE_DIR)
            print(f"  Cleared cache: {CACHE_DIR}")
        except Exception as exc:
            print(f"  Warning: could not clear cache: {exc}")


def _kill_process_tree():
    """Kill the entire process tree (this process + all children).

    On Windows, Ctrl+C only stops the main Flask process — background
    callback workers spawned by DiskcacheManager keep running and hold
    the cache DB open.  This kills everything.
    """
    pid = os.getpid()
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        os.killpg(pid, signal.SIGTERM)


def _sigint_handler(signum, frame):
    """Handle Ctrl+C: clear cache, kill child processes, exit."""
    print("\nShutting down dashboard...")
    _clear_cache()
    print("  Killing child processes...")
    _kill_process_tree()
    sys.exit(0)


if __name__ == "__main__":
    # Clear any stale cache from previous runs BEFORE starting
    _clear_cache()

    signal.signal(signal.SIGINT, _sigint_handler)
    signal.signal(signal.SIGTERM, _sigint_handler)

    from dashboard.app import app

    # use_reloader=False prevents Werkzeug from spawning a second watcher process,
    # which would cause duplicate log entries in dashboard.log.
    app.run(debug=True, port=8050, use_reloader=False)
