#!/usr/bin/env python3
"""
verify-before-stop.py

Runs ruff on any Python files modified by Claude during the session.
Called by the Stop hook in settings.json.

Exit codes:
  0 = all clear (or nothing to check)
  2 = ruff found errors (blocks session stop, prompts Claude to fix)
"""

import json
import os
import subprocess
import sys


def main():
    # Read the hook payload from stdin
    try:
        payload = json.loads(sys.stdin.read())
    except (json.JSONDecodeError, ValueError):
        # Can't parse payload — don't block
        sys.exit(0)

    # Prevent infinite loop — check both JSON payload field AND env var
    # The payload's stop_hook_active field is the canonical check per Claude Code docs
    if payload.get("stop_hook_active") or os.environ.get("STOP_HOOK_ACTIVE"):
        sys.exit(0)
    os.environ["STOP_HOOK_ACTIVE"] = "1"

    # Extract modified Python files from Claude's tool calls in the transcript
    modified_files = set()
    transcript = payload.get("transcript", [])

    for message in transcript:
        if not isinstance(message, dict):
            continue
        for content_block in message.get("content", []):
            if not isinstance(content_block, dict):
                continue
            tool_use = content_block.get("tool_use", {})
            if not isinstance(tool_use, dict):
                tool_use = content_block  # handle flat structure

            tool_name = tool_use.get("name", "")
            if tool_name not in ("Write", "Edit"):
                continue

            file_path = (
                tool_use.get("input", {}).get("file_path")
                or tool_use.get("input", {}).get("path")
            )
            if not file_path:
                continue

            # Only check Python source files in src/
            if file_path.endswith(".py") and "src/" in file_path:
                # Skip if file is in excluded dirs
                skip_dirs = [".claude/", "__pycache__/", "logs/", ".ipynb_checkpoints/"]
                if not any(skip in file_path for skip in skip_dirs):
                    if os.path.isfile(file_path):
                        modified_files.add(file_path)

    if not modified_files:
        # No Python source files modified — nothing to check
        sys.exit(0)

    print(f"\n[verify-before-stop] Checking {len(modified_files)} modified file(s)...")

    # Run ruff on each modified file
    errors_found = False
    for file_path in sorted(modified_files):
        result = subprocess.run(
            ["ruff", "check", file_path],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            errors_found = True
            print(f"\n❌ ruff errors in {file_path}:")
            print(result.stdout)
            if result.stderr:
                print(result.stderr)
        else:
            print(f"  ✅ {os.path.basename(file_path)} — clean")

    if errors_found:
        print("\n[verify-before-stop] Ruff errors found. Please fix before finishing.")
        sys.exit(2)  # Exit code 2 = block stop, prompt Claude to fix
    else:
        print("\n[verify-before-stop] All checks passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
