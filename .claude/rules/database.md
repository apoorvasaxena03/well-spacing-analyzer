---
paths:
  - "src/utils/database_manager.py"
---

# Rules when editing database_manager.py

## Security — Non-Negotiable
- NEVER use f-strings or string concatenation to build SQL queries
- ALL user-provided values must go through SQLAlchemy parameter binding: `text("SELECT ... WHERE col = :val")` with `{"val": value}`
- NEVER log connection URLs with passwords — use the masked URL method already in the module

## Connection Management
- Always use the existing `NullPool` pattern for stateless workloads
- New database backends must implement the `DBConfig` Protocol — not inherit from a base class
- `test_connection()` must be called before any production query sequence

## Retry Logic
- Only retry on transient errors (connection timeout, temp unavailable)
- NEVER retry on authentication failures, syntax errors, or constraint violations
- Maximum 3 retries with exponential backoff already implemented — don't change these defaults

## QueryResult Dataclass
- Always return `QueryResult` from `read_sql()` — never raw DataFrames
- `elapsed_ms` must be measured and populated
- `rows` must match `len(df)` exactly

## New Database Backends
To add a new backend:
1. Create a `MyConfig` dataclass implementing `DBConfig` Protocol
2. Add a `get_connection_url()` method returning a masked-safe SQLAlchemy URL
3. Add the config class to `__all__` in `database_manager.py`
4. Document the required environment variables in the class docstring
