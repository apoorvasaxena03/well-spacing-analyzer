---
name: push
description: Push the current branch to origin safely — sets upstream on first push, never force-pushes, never pushes main.
argument-hint: "[optional: branch, defaults to current]"
disable-model-invocation: true
allowed-tools: Bash(git status *) Bash(git branch *) Bash(git fetch *) Bash(git push *) Bash(git rev-list *) Bash(git rev-parse *) Bash(git log *) Bash(git pull --rebase *) Bash(gh auth status *)
---

# Push the Current Branch

## Live context

- Branch: !`git branch --show-current`
- Upstream: !`git rev-parse --abbrev-ref --symbolic-full-name @{u} 2>/dev/null || echo "(none — first push)"`
- Sync + working tree: !`git status -sb || true`

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

### 1. Hard stops

- Branch is **`main`** → stop. `main` only changes through a `dev` → `main` PR. (The
  project's `deny` rules also block `git push ... main`.)
- Detached HEAD → stop and explain.
- Nothing to push (no upstream needed and 0 commits ahead) → say "already up to date"
  and stop.

Uncommitted changes are fine but are **not** included — tell the user, and offer
`/commit` first if they look like part of this work.

### 2. Check divergence

```bash
git fetch origin --prune
git rev-list --left-right --count origin/<branch>...HEAD   # skip if no upstream yet
```

Left number = commits on the remote you don't have; right = commits you'd push.

- **Behind = 0** → push (step 3).
- **Behind > 0** → do **not** push and do **not** force. Show the incoming commits
  (`git log --oneline HEAD..origin/<branch>`), then propose
  `git pull --rebase origin <branch>` to replay local commits on top. Ask before
  running it. If the rebase hits conflicts, stop and show them — never resolve by
  discarding either side without the user's say-so.

### 3. Push

First push of the branch (no upstream):
```bash
git push -u origin <branch>
```

Subsequent pushes:
```bash
git push origin <branch>
```

Always name `origin` and the branch explicitly; never rely on `push.default`.

### 4. Verify

```bash
git status -sb | head -1
git rev-parse HEAD @{u}
```

The two hashes must match and status must show no `[ahead N]`. Report the branch,
the pushed range, and the head commit.

Then suggest the next step:
- `<type>/*` branch → `/create-pr` (targets `dev`)
- `dev` → `/create-pr` for a release PR into `main`, when the user is ready to release

## If authentication fails

`Invalid username or token` / `Authentication failed`:
1. `gh auth status` — if logged out: `gh auth login` (GitHub.com → HTTPS → Yes → browser).
2. `gh auth setup-git` — makes gh the git credential source for github.com, so a stale
   Git Credential Manager entry can't be used.
3. Retry the push. Do not ask the user to paste a token into the chat.

## Never

- Never `--force`, `-f`, or `--force-with-lease` (also blocked by `deny` rules). If
  history genuinely must be rewritten on a shared branch, explain why and let the user
  do it themselves.
- Never `--no-verify`.
- Never push tags, other branches, or `--all` / `--mirror` as a side effect.
