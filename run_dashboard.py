"""
Launch the Well Spacing Analyzer dashboard.

Run from the project root:
    python run_dashboard.py

The project root must be on sys.path so both `dashboard` and `src` are importable.
This file lives at the project root, which is exactly what ensures that.
"""

import os
import signal
import subprocess
import sys


def _kill_process_tree():
    """Kill the entire process tree (this process + all children).

    On Windows, Ctrl+C only stops the main Flask process — background
    callback workers spawned by DiskcacheManager keep running and hold
    the cache DB open.  This kills everything.
    """
    pid = os.getpid()
    if sys.platform == "win32":
        # /T = kill process tree, /F = force
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        os.killpg(pid, signal.SIGTERM)


def _sigint_handler(signum, frame):
    """Handle Ctrl+C: print message, then kill entire process tree."""
    print("\nShutting down dashboard (killing all child processes)...")
    _kill_process_tree()
    # If taskkill didn't terminate us, force exit
    sys.exit(0)


if __name__ == "__main__":
    # Register Ctrl+C handler BEFORE importing/starting the app
    signal.signal(signal.SIGINT, _sigint_handler)
    signal.signal(signal.SIGTERM, _sigint_handler)

    from dashboard.app import app

    # use_reloader=False prevents Werkzeug from spawning a second watcher process,
    # which would cause duplicate log entries in dashboard.log.
    app.run(debug=True, port=8050, use_reloader=False)
