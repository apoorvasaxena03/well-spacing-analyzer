"""
Launch the Well Spacing Analyzer dashboard.

Run from the project root:
    python run_dashboard.py

The project root must be on sys.path so both `dashboard` and `src` are importable.
This file lives at the project root, which is exactly what ensures that.
"""

import atexit
import os
import signal
import sys

from dashboard.app import app


def _cleanup_children():
    """Kill child processes (DiskcacheManager workers) on exit.

    On Windows, Ctrl+C only stops the main process — background callback
    workers spawned by DiskcacheManager keep running and hold the cache
    DB open.  This handler terminates the entire process tree.
    """
    if sys.platform == "win32":
        # taskkill /T = kill process tree, /F = force, /PID = this process
        os.system(f"taskkill /F /T /PID {os.getpid()} >nul 2>&1")
    else:
        # On Unix, kill the process group
        os.killpg(os.getpid(), signal.SIGTERM)


if __name__ == "__main__":
    atexit.register(_cleanup_children)

    # use_reloader=False prevents Werkzeug from spawning a second watcher process,
    # which would cause duplicate log entries in dashboard.log.
    app.run(debug=True, port=8050, use_reloader=False)
