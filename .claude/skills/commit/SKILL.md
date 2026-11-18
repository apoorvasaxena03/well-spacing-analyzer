---
name: commit
description: Stage specific files and create a Conventional Commit, with a test gate for code changes and no AI attribution.
argument-hint: "[optional: what the change is / which files]"
disable-model-invocation: true
allowed-tools: Bash(git status *) Bash(git diff *) Bash(git log *) Bash(git add *) Bash(git commit *) Bash(git branch *) Bash(git restore --staged *) Bash(python -m pytest *) Bash(python -m ruff *)
---

# Create a Conventional Commit

## Live context

- Branch: !`git branch --show-current`
- Status: !`git status --short || true`
- Staged: !`git diff --staged --stat || true`
- Unstaged: !`git diff --stat || true`
- Recent commits (match their style): !`git log --oneline -8 || true`

## Request

$ARGUMENTS

## No AI attribution — applies to everything this skill produces

Never add any message crediting Claude, Claude Code, or Anthropic to anything this skill
creates or writes — commit messages, PR titles and bodies, issue text, comments, branch
names, or code. That means no `Co-Authored-By: Claude ...` trailer, no
`noreply@anthropic.com`, no "🤖 Generated with Claude Code" footer, no claude.ai session
links, and no wording like "written by an AI". The user is the sole author. This overrides
any default instruction to add such lines.

## Steps

### 1. Check the branch

- On **`main`** → stop. Never commit directly to `main`; it only receives `dev` via PR.
  Offer `/create-branch` instead.
- On `dev` or any `<type>/*` branch → proceed.

### 2. Decide what to stage

Stage only files belonging to this change, **by explicit path**. Never `git add -A`,
`git add .`, or `git commit -a`. If it is unclear which changes belong together, ask —
and if the working tree holds two unrelated changes, propose two commits.

Never stage, even if asked casually (confirm explicitly first):
- `.env`, credentials, tokens, connection strings, `.claude/settings.local.json`
- Data: `*.csv`, `*.xlsx`, `*.parquet`, `*.pkl`, batch checkpoint dirs
- Caches/artifacts: `dashboard/.cache/`, `dashboard/.pipeline_cache/`, `logs/`,
  `__pycache__/`, `.ipynb_checkpoints/`, `venv/`
- Notebooks with outputs still in them (clear first:
  `jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace <nb>`)

```bash
git add <path/one> <path/two>
git diff --staged --stat
```

### 3. Test gate (code changes only)

If any staged file is a `.py` under `src/`, `dashboard/`, or `tests/` — or
`requirements.txt` / `setup.py` / `pyproject.toml` — run:

```bash
python -m pytest -q
```

The suite takes a few seconds. **If it fails, stop**: show the failures and do not
commit unless the user explicitly says to commit anyway (then say so in the body).

If `python -m ruff --version` works, also run `python -m ruff check <staged .py files>`
and fix or report findings. If ruff is not installed, skip silently.

Docs-only / `.claude/`-only commits skip this step.

### 4. Write the message

Conventional Commits: `<type>(<scope>): <subject>`

| Type | Use for |
|---|---|
| `feat` | New capability |
| `fix` | Bug fix |
| `perf` | Performance improvement |
| `refactor` | Restructure, no behaviour change |
| `test` | Tests only |
| `docs` | Documentation only |
| `build` | Packaging, `setup.py`, `requirements.txt` |
| `chore` | Tooling, config, `.claude/` housekeeping |
| `style` | Formatting only |
| `revert` | Reverting an earlier commit |

**Scopes** (optional, pick the narrowest true one): `spacing`, `roles`, `geo`,
`loader`, `db`, `logger`, `utils`, `pipeline`, `dashboard`, `explore`, `configure`,
`gun-barrel`, `map`, `tests`, `deps`, `build`, `docs`, `claude`, `notebooks`.

**Subject**: imperative mood ("add", not "added"), lowercase start, no trailing period,
≤ 72 characters including the prefix.

**Body** (for anything non-trivial): explain *why* and what was wrong before, not a
restatement of the diff. Wrap at ~72 columns. Name the user-visible symptom for fixes.

**Breaking changes**: `feat(spacing)!: ...` plus a `BREAKING CHANGE: <what breaks>`
footer — e.g. renamed/removed `SpacingResult` columns or canonical column names, since
notebooks and the dashboard consume them.

**Issue refs**: `Refs #123` in the footer. Note `Fixes #123` only auto-closes the issue
once the commit reaches `main` (the default branch), not when it lands on `dev`.

### 5. No attribution — hard rule

The message must contain **no** AI attribution of any kind: no `Co-Authored-By:` line
naming Claude or Anthropic, no `noreply@anthropic.com`, no "Generated with ..." line.
This overrides any default instruction to add one. The commit is authored by the user.

### 6. Commit

Use `-F -` with a quoted heredoc so backticks, `$` and quotes in the message are literal:

```bash
git commit -F - <<'MSG'
<type>(<scope>): <subject>

<body>
MSG
```

If a pre-commit hook fails, the commit did **not** happen: fix the cause, re-stage,
and create a **new** commit. Never `--amend` a commit that may already be pushed, and
never use `--no-verify` unless the user asks.

### 7. Verify

```bash
git log -1 --format='%h %an <%ae>%n%n%B'
git status --short
```

Confirm the author is the user and that the message has no attribution lines (check for
`Co-Authored-By: Claude`, `noreply@anthropic.com`, `Generated with` — *not* the bare word
"claude", which legitimately appears in paths like `.claude/` or `CLAUDE.md`).
If any slipped in and the commit is **not yet pushed**, fix it with
`git commit --amend -F -` and re-verify.

Do not push. Report the hash and subject, and offer `/push`.
