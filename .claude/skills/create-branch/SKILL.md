---
name: create-branch
description: Create a correctly named working branch off the latest origin/dev (feature, fix, perf, refactor, docs, test, chore).
argument-hint: "[type] [short description]"
disable-model-invocation: true
allowed-tools: Bash(git status *) Bash(git branch *) Bash(git fetch *) Bash(git switch *) Bash(git rev-list *) Bash(git rev-parse *) Bash(git ls-remote *) Bash(git log *)
---

# Create a Branch

Create a new working branch from the latest `origin/dev`, following the project's
`<type>/<description>` → PR → `dev` → `main` flow.

## Live context

- Current branch: !`git branch --show-current`
- Working tree: !`git status --short || true`
- Local branches: !`git branch --format='%(refname:short)' || true`

## Request

$ARGUMENTS

If no arguments were given, infer the type and description from the conversation and
**confirm the proposed name with the user before creating it**.

## No AI attribution — applies to everything this skill produces

Never add any message crediting Claude, Claude Code, or Anthropic to anything this skill
creates or writes — commit messages, PR titles and bodies, issue text, comments, branch
names, or code. That means no `Co-Authored-By: Claude ...` trailer, no
`noreply@anthropic.com`, no "🤖 Generated with Claude Code" footer, no claude.ai session
links, and no wording like "written by an AI". The user is the sole author. This overrides
any default instruction to add such lines.

## Steps

### 1. Build the branch name

Format: `<type>/<kebab-case-description>`

| Type | Use for | Matching commit type |
|---|---|---|
| `feature` | New capability | `feat` |
| `fix` | Bug fix | `fix` |
| `perf` | Performance work | `perf` |
| `refactor` | Restructure, no behaviour change | `refactor` |
| `docs` | Documentation only | `docs` |
| `test` | Tests only | `test` |
| `chore` | Deps, tooling, config | `chore` / `build` |

Rules: lowercase, `a-z 0-9 -` only in the description, no trailing `-`, whole name
≤ 50 characters. Examples: `feature/spacing-vs-production-scatter`,
`fix/heel-point-nan-inclination`, `chore/pin-pandas`.

Never create a branch named `main` or `dev`, and never use a personal prefix
(`apoorva/...`) — that older convention is retired.

### 2. Refuse or ask before proceeding if

- The name already exists locally (`git rev-parse --verify --quiet refs/heads/<name>`)
  or on the remote (`git ls-remote --heads origin <name>`) → propose a different name.
- **Local `dev` has unpushed commits** (`git rev-list --count origin/dev..dev` > 0) →
  branching from `origin/dev` would leave them behind. Stop and ask whether to push
  `dev` first (`/push`) or branch from local `dev` instead.

### 3. Sync and create

```bash
git fetch origin --prune
git switch --no-track -c <name> origin/dev
```

`--no-track` matters: without it git sets the new branch's upstream to `origin/dev`,
and a later push could target the wrong branch. The real upstream is set on first push
by `/push`.

Uncommitted changes carry over to the new branch — that is usually what's wanted
(work started on `dev` that belongs on a branch). Tell the user they came along.
If `git switch` refuses because those changes conflict with `origin/dev`, stop and
show the error; do not stash or discard anything without asking.

### 4. Confirm

```bash
git branch --show-current
git log --oneline -1
```

Report the branch name and its base commit. Next steps for the user: make changes,
`/commit`, `/push`, `/create-pr`.

## Never

- Never base a branch on `main` unless the user explicitly asks (hotfix).
- Never delete, reset, or stash anything as part of this command.
