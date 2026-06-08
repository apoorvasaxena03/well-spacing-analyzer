# Create Pull Request

Create a GitHub Pull Request from the current branch.

## Steps

### 1. Gather context
Run these commands to understand what's changing:

```bash
git status
git log main...HEAD --oneline
git diff main...HEAD --stat
```

### 2. Determine target branch
- If current branch is a `feature/*` branch → target is `dev`
- If current branch is `dev` → target is `main`
- If unclear, ask the user

### 3. Check for unpushed commits
```bash
git status -sb
```
If behind remote, push first: `git push -u origin <branch-name>`

### 4. Read ALL commits in the branch
```bash
git log main...HEAD --format="%H %s%n%b"
```
Read the full diff for context:
```bash
git diff main...HEAD
```

### 5. Draft the PR

**Title format**: `<type>: <concise description>` (max 70 chars)
Types: `feat`, `fix`, `refactor`, `perf`, `docs`, `test`

**Body structure**:
```markdown
## Summary
- <bullet 1: what changed and why>
- <bullet 2>
- <bullet 3 if needed>

## Changes
- `src/...` — <what changed in this file>
- `notebooks/...` — <what changed>

## Test Plan
- [ ] Ran notebook end-to-end on sample data
- [ ] Verified spacing results match expected for known well pairs
- [ ] No regressions in existing notebooks
- [ ] <any specific test cases relevant to this change>

## Notes
<Any breaking changes, migration steps, or reviewer callouts>

🤖 Generated with [Claude Code](https://claude.ai/claude-code)
```

### 6. Create the PR
```bash
gh pr create \
  --title "<title>" \
  --body "$(cat <<'EOF'
<body content>
EOF
)" \
  --base <target_branch>
```

### 7. Output
Return the PR URL to the user.

$ARGUMENTS
