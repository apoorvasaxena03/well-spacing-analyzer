# Create GitHub Issue

Analyze the current context and create a well-structured GitHub issue.

## Instructions

$ARGUMENTS

If no arguments are given, infer the issue from the current conversation context (e.g., a bug just found, a feature requested, a question raised).

### Determine Issue Type

- **Bug**: Something is producing wrong results or crashing
- **Feature**: New capability needed
- **Enhancement**: Improvement to existing functionality
- **Documentation**: Docs are missing or incorrect
- **Performance**: Something is too slow or uses too much memory
- **Question**: Uncertainty about behavior

### Draft the Issue

**Title**: `[Type] <short description>` (max 80 chars)

**Body** (adapt to issue type):

**For bugs**:
```markdown
## Problem
<Clear description of what's wrong>

## Steps to Reproduce
1. Load data with...
2. Call `WellSpacingCalculator._calculate_spacing_statistics()`
3. Observe...

## Expected Behavior
<What should happen>

## Actual Behavior
<What actually happens — include error message or wrong output>

## Environment
- Python version:
- Key package versions (pandas, geopandas, pyproj):
- Dataset size (~wells, ~pairs):

## Possible Cause
<If known or suspected>

## Proposed Fix
<If known>
```

**For features**:
```markdown
## Motivation
<Why is this needed? What problem does it solve?>

## Proposed Solution
<How it should work>

## Acceptance Criteria
- [ ] <specific testable requirement>
- [ ] <another requirement>

## Alternatives Considered
<Other approaches and why they were rejected>
```

### Create the Issue
```bash
gh issue create \
  --title "<title>" \
  --body "$(cat <<'EOF'
<body>
EOF
)" \
  --label "<bug|enhancement|documentation|performance>"
```

Return the issue URL to the user.
