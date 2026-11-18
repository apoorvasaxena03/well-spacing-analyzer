---
name: create-issue
description: File a well-structured GitHub issue (bug, feature, enhancement, docs, performance, question) using only labels that exist in the repo.
argument-hint: "[optional: what the issue is about]"
disable-model-invocation: true
allowed-tools: Bash(gh issue *) Bash(gh label list *) Bash(git log *) Bash(git branch *) Bash(git rev-parse *) Bash(python --version) Bash(python -m pip show *)
---

# Create a GitHub Issue

## Live context

- Repo labels: !`gh label list --limit 100 --json name --jq '[.[].name] | join(", ")' 2>/dev/null || echo "(could not list labels)"`
- Branch / commit: !`git branch --show-current` @ !`git rev-parse --short HEAD`

## Request

$ARGUMENTS

If no arguments were given, infer the issue from the conversation (a bug just found, a
feature discussed). Show the drafted title and body and **get confirmation before
creating** — an issue is public on this repo.

## No AI attribution — applies to everything this skill produces

Never add any message crediting Claude, Claude Code, or Anthropic to anything this skill
creates or writes — commit messages, PR titles and bodies, issue text, comments, branch
names, or code. That means no `Co-Authored-By: Claude ...` trailer, no
`noreply@anthropic.com`, no "🤖 Generated with Claude Code" footer, no claude.ai session
links, and no wording like "written by an AI". The user is the sole author. This overrides
any default instruction to add such lines.

## Steps

### 1. Classify and pick a label

| Kind | Title prefix | Label |
|---|---|---|
| Bug — wrong result or crash | `[Bug]` | `bug` |
| Feature — new capability | `[Feature]` | `enhancement` |
| Enhancement — improve existing | `[Enhancement]` | `enhancement` |
| Performance — too slow / too much memory | `[Perf]` | `enhancement` |
| Documentation | `[Docs]` | `documentation` |
| Question | `[Question]` | `question` |

**Only use labels present in the live context list** — `gh issue create` fails on an
unknown label. Never create a new label unless the user asks.

Title: `<prefix> <short description>`, ≤ 80 chars.

### 2. Check for duplicates

```bash
gh issue list --state all --search "<2–4 key terms>" --limit 10
```

If a likely duplicate exists, show it and ask whether to comment there instead.

### 3. Collect real environment facts (bugs and performance)

Fill these from the machine — never leave placeholders:

```bash
python --version
python -m pip show pandas numpy geopandas pyproj shapely dash 2>/dev/null | grep -E '^(Name|Version):'
```

Include dataset size (wells / pairs) and the `run_id` from `logs/` if known. **Strip**
anything sensitive before it goes in a public issue: file paths with usernames or
company names, connection strings, API-14 UWIs or operator names from client data
(use synthetic or truncated examples).

### 4. Body

**Bug**
```markdown
## Problem
## Steps to reproduce
1. ...
## Expected
## Actual
<error text or wrong value — trimmed traceback in a code block>
## Environment
- Python / key packages: ...
- Commit: <short hash> on <branch>
- Dataset size: ...
## Suspected cause / proposed fix
<optional — file:line if known>
```

**Feature / enhancement / performance**
```markdown
## Motivation
## Proposed solution
## Acceptance criteria
- [ ] <specific, testable>
## Alternatives considered
```

For performance, include the measured baseline (time, memory, wells, pairs) so the fix
can be verified against it.

**No attribution — hard rule.** No "Generated with Claude Code" footer and no AI
attribution of any kind.

### 5. Create

```bash
gh issue create --title "<title>" --label "<label>" --body-file - <<'BODY'
<body>
BODY
```

Return the issue URL. Suggest `/create-branch fix <description>` (or `feature ...`) if
the user wants to start on it, and `Refs #N` in the eventual commits.
