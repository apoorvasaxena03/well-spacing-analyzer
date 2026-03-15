# Create a Conventional Commit

Stage and commit changes with a properly formatted Conventional Commit message.

## Steps

### 1. Assess current state
```bash
git status
git diff --staged
git diff
git log --oneline -5
```

### 2. Identify what to stage
Only stage files directly related to the current change. **Never use `git add -A` or `git add .`** — always add specific paths.

Ask the user to confirm which files to stage if it's not clear from context.

Do NOT stage:
- `.env` files or any file with credentials
- Large binary files not tracked before
- Unrelated changes in other modules
- Log files (`logs/`)
- Notebook checkpoint files (`.ipynb_checkpoints/`)
- `__pycache__/` directories

### 3. Stage the files
```bash
git add <specific/path/to/file.py>
git add <another/file.py>
```

### 4. Draft the commit message

**Format**: `<type>(<scope>): <short description>`

| Type | Use for |
|------|---------|
| `feat` | New function/class/capability |
| `fix` | Bug fix |
| `perf` | Performance improvement |
| `refactor` | Code restructure, no behavior change |
| `docs` | Documentation only |
| `test` | Adding or fixing tests |
| `chore` | Dependencies, configs, tooling |

**Scope** (optional): `spacing`, `logger`, `db`, `utils`, `geo`, `neighbors`, `dashboard`, `notebooks`

**Examples**:
- `feat(spacing): add floating-window spacing calculator`
- `fix(geo): handle missing heel point in filter_after_heel_point`
- `perf(spacing): reduce memory usage in batch pairwise computation`
- `docs(claude): update CLAUDE.md with new canonical column names`

Short description: imperative mood, lowercase, no period, max 72 chars.

Body (optional): explain WHY, not what. Reference issue numbers with `Fixes #123` or `Closes #456`.

### 5. Create the commit
```bash
git commit -m "$(cat <<'EOF'
<type>(<scope>): <short description>

<optional body explaining why>

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

### 6. Confirm success
```bash
git log --oneline -3
```

$ARGUMENTS
