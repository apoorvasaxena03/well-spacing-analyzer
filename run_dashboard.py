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
    """Remove stale cache subdirectories from previous runs.

    Each app run creates a unique subdirectory under CACHE_DIR (via uuid).
    On startup, we remove ALL old subdirectories.  Locked files from
    orphaned processes are skipped silently — they become harmless since
    the new run uses a fresh subdirectory and will never replay them.
    """
    if not CACHE_DIR.exists():
        return
    cleared = 0
    for child in CACHE_DIR.iterdir():
        if child.is_dir():
            try:
                shutil.rmtree(child)
                cleared += 1
            except Exception:
                pass  # locked by orphaned process — harmless
        elif child.is_file():
            try:
                child.unlink()
                cleared += 1
            except Exception:
                pass
    if cleared:
        print(f"  Cleared {cleared} stale cache entries")


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
    # --debug flag sets log level to DEBUG (otherwise defaults to INFO)
    if "--debug" in sys.argv:
        os.environ["DASHBOARD_LOG_LEVEL"] = "DEBUG"
        sys.argv.remove("--debug")
        print("  Log level: DEBUG")

    # Clear any stale cache from previous runs BEFORE starting
    _clear_cache()

    signal.signal(signal.SIGINT, _sigint_handler)
    signal.signal(signal.SIGTERM, _sigint_handler)

    from dashboard.app import app

    # use_reloader=False prevents Werkzeug from spawning a second watcher process,
    # which would cause duplicate log entries in dashboard.log.
    app.run(debug=True, port=8050, use_reloader=False)
