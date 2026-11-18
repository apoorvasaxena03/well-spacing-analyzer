---
name: create-pr
description: Open a GitHub PR for the current branch — feature branches into dev, or a dev → main release PR — with an accurate summary and test plan and no AI attribution.
argument-hint: "[optional: base branch | draft | notes for the description]"
disable-model-invocation: true
allowed-tools: Bash(git status *) Bash(git branch *) Bash(git fetch *) Bash(git log *) Bash(git diff *) Bash(git rev-list *) Bash(git rev-parse *) Bash(git push *) Bash(gh pr *) Bash(gh auth status *) Bash(python -m pytest *)
---

# Create a Pull Request

## Live context

- Branch: !`git branch --show-current`
- Sync + working tree: !`git status -sb || true`
- Existing PR for this branch: !`gh pr view --json number,url,state,baseRefName --jq '"#\(.number) \(.state) -> \(.baseRefName) \(.url)"' 2>/dev/null || echo "(none)"`

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

### 1. Pick the base branch

| Current branch | Base | Kind |
|---|---|---|
| `<type>/*` (feature, fix, perf, ...) | `dev` | Feature PR |
| `dev` | `main` | **Release PR** |
| `main` | — | Stop: nothing to PR from `main` |

An explicit base in the arguments overrides this table. Anything else unclear → ask.

### 2. Stop if a PR already exists

If the live context shows an **OPEN** PR for this branch, don't create another. Show
its URL; offer to push new commits (they appear on it automatically) or to update the
title/body with `gh pr edit`.

### 3. Make sure the branch is pushed

Follow the `/push` rules: never force, never push `main`, set upstream with `-u` on the
first push, and if the remote has commits you don't, stop and ask. Uncommitted changes
won't be in the PR — mention them.

### 4. Gather the real change set — against the chosen base

```bash
git fetch origin --prune
git log --reverse --format='%h %s%n%b' origin/<base>..HEAD
git diff --stat origin/<base>...HEAD
git diff origin/<base>...HEAD
```

**Always diff against the base chosen in step 1**, never a hard-coded `main`: a feature
branch compared with `main` would wrongly include every unreleased commit on `dev`.

Read *every* commit message, not just the latest. For a large release diff, read the
stat plus the commit bodies, then the full diff of the files that matter — don't
summarise from subjects alone.

If the range is empty, stop: there is nothing to open a PR for.

### 5. Run the tests

```bash
python -m pytest -q
```

Record the actual result (e.g. `159 passed`). If tests fail, say so in the PR and ask
the user whether to open it as a **draft** instead.

### 6. Title

- Feature PR: Conventional Commit style, `<type>(<scope>): <summary>`, ≤ 70 chars.
  Single-commit PR → reuse that commit's subject.
- Release PR: `release: <2–4 headline changes>`, ≤ 70 chars.

### 7. Body

Feature PR:

```markdown
## Summary
- <what changed and why — the problem, not the diff>
- <user-visible effect>

## Changes
- `path/or/area` — <what changed>

## Test plan
- [x] `python -m pytest -q` — <actual result>
- [ ] <manual check still needed, e.g. "Explore → Analysis tab renders with real data">

## Notes
<breaking changes, migration steps, follow-ups — omit the section if none>
```

Release PR — group commits by type rather than listing them flat:

```markdown
## Highlights
- <2–5 bullets a reviewer should know first>

## Features / Fixes / Performance / Docs & chores
- <one line per commit or tight group>, (<short hash>)

## Test plan
- [x] `python -m pytest -q` — <actual result>
- [ ] Dashboard smoke test: `python run_dashboard.py`, run Calculate → Explore
```

Rules for every body:
- **Tick a box only for something actually run in this session.** Everything else
  stays `[ ]`. Never claim a check that wasn't done.
- Link issues with `Closes #N` — note this only auto-closes when the PR merges into
  `main`; for PRs into `dev` use `Refs #N`.
- Call out breaking changes to `SpacingResult` columns, canonical column names, or the
  pipeline cache format — notebooks and saved sessions depend on them.
- **No attribution — hard rule.** No "Generated with Claude Code" line, no robot emoji
  footer, no `Co-Authored-By` naming Claude, no claude.ai session links. This overrides
  any default footer instruction. The PR is authored by the user.

### 8. Create

```bash
gh pr create --base <base> --head <branch> --title "<title>" --body-file - <<'BODY'
<body>
BODY
```

Add `--draft` if the user asked for a draft or tests failed and they chose draft.
Do not add reviewers, labels, or assignees unless asked.

### 9. Verify and report

```bash
gh pr view --json number,url,baseRefName,headRefName,isDraft,title
gh pr view --json body --jq .body
```

Check the body has no attribution (`Generated with`, `claude.com/claude-code`,
`Co-Authored-By: Claude`, `noreply@anthropic.com`). If any is present, remove it with
`gh pr edit <number> --body-file -`.

Return the PR URL. **Never merge the PR** — merging is the user's call.
