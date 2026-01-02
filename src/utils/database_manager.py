from __future__ import annotations

import logging
import urllib.parse
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Mapping, Optional, Protocol, Union
import time

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, URL
from sqlalchemy.pool import NullPool
from sqlalchemy.exc import OperationalError, DBAPIError
#%%

# =============================================================================
# Public return type
# =============================================================================
@dataclass(frozen=True)
class QueryResult:
    """
    Rich metadata returned from a database operation.

    This dataclass is useful when you want *more than just a DataFrame*:
    timing, row counts, and the SQL statement for logging or debugging.

    Attributes
    ----------
    sql:
        The SQL statement executed (as provided by the caller).
    elapsed_ms:
        Execution time in milliseconds (measured around the query execution).
    rows:
        Row count when available.
        - For SELECT queries, this is typically `len(df)` (if df is captured).
        - For INSERT/UPDATE/DELETE, this is the DBAPI/SQLAlchemy `rowcount`.
        - Some backends return -1 for certain statements (notably some DDL).
    df:
        DataFrame result for SELECT queries. For non-SELECT statements this is None.
    """
    sql: str
    elapsed_ms: float
    rows: Optional[int] = None
    df: Optional[pd.DataFrame] = None

# =============================================================================
# Config Protocol (plug-in interface)
# =============================================================================
class DBConfig(Protocol):
    """
    Protocol for database config objects.

    Any config class can work with SQLAlchemyDBClient as long as it implements:

    - build_url() -> URL or str:
        Returns a SQLAlchemy URL (or string URL) for create_engine().

    - ping_sql() -> str:
        Returns a minimal SQL statement used by test_connection().

    - display_name() -> str:
        Human-readable name for logging (should not contain secrets).
    """

    def build_url(self) -> Union[str, URL]: ...
    def ping_sql(self) -> str: ...
    def display_name(self) -> str: ...


# =============================================================================
# URL / ODBC helpers
# =============================================================================
def _safe_url_str(url: Union[str, URL]) -> str:
    """
    Render a URL safely for logs without leaking passwords/tokens.

    Notes
    -----
    - For SQLAlchemy URL objects, render_as_string(hide_password=True) is used.
    - For raw string URLs, we avoid echoing them because they may contain secrets.
    """
    if isinstance(url, URL):
        try:
            return url.render_as_string(hide_password=True)
        except Exception:
            return "<sqlalchemy-url>"
    # For string URLs, do best-effort redaction.
    return "<string-url>"


def _build_odbc_connect_value(parts: Mapping[str, str]) -> str:
    """
    Build and URL-encode an ODBC connection string for SQLAlchemy's `odbc_connect=`.

    This avoids common quoting issues with spaces/special characters.

    Example (before encoding)
    -------------------------
    DRIVER={ODBC Driver 18 for SQL Server};
    SERVER=myhost,1433;
    DATABASE=MyDB;
    UID=user;
    PWD=pass;
    Encrypt=yes;
    TrustServerCertificate=yes;

    Returns
    -------
    str
        URL-encoded string suitable for:
            mssql+pyodbc:///?odbc_connect=<encoded>
    """
    odbc_str = ";".join(f"{k}={v}" for k, v in parts.items() if v is not None) + ";"
    return urllib.parse.quote_plus(odbc_str)

@dataclass(frozen=True)
class OdbcDsnConfig:
    """
    Generic ODBC DSN config (works for Databricks DSN, SQL Server DSN, etc.)
    using the pyodbc SQLAlchemy dialect.

    Requirements
    ------------
    pip install pyodbc sqlalchemy pandas

    Notes
    -----
    - Requires a DSN configured on the machine (ODBC Data Sources / unixODBC).
    - If your DSN already contains authentication, you may not need username/password.
    """
    dsn: str
    username: Optional[str] = None
    password: Optional[str] = None
    autocommit: bool = True

    def build_url(self) -> str:
        # Build an ODBC connection string and URL-encode it
        parts = [f"DSN={self.dsn}"]
        if self.username is not None:
            parts.append(f"UID={self.username}")
        if self.password is not None:
            parts.append(f"PWD={self.password}")
        if self.autocommit:
            parts.append("AUTOCOMMIT=1")

        odbc_str = ";".join(parts) + ";"
        encoded = urllib.parse.quote_plus(odbc_str)

        # Generic ODBC dialect for SQLAlchemy
        return f"pyodbc:///?odbc_connect={encoded}"

    def ping_sql(self) -> str:
        return "SELECT 1"

    def display_name(self) -> str:
        return f"ODBC(DSN={self.dsn})"
    
# =============================================================================
# Database config dataclasses
# =============================================================================
@dataclass(frozen=True)
class PostgresConfig:
    """
    PostgreSQL config.

    Requirements
    ------------
    - SQLAlchemy + pandas
    - A PostgreSQL DBAPI driver:
        pip install psycopg[binary]
      (or pip install psycopg2-binary)

    Notes
    -----
    - driver="psycopg" uses psycopg v3 (recommended)
    - driver="psycopg2" uses psycopg2
    """
    host: str
    database: str
    username: str
    password: str
    port: int = 5432
    driver: str = "psycopg"  # "psycopg" or "psycopg2"

    def build_url(self) -> URL:
        return URL.create(
            drivername=f"postgresql+{self.driver}",
            username=self.username,
            password=self.password,
            host=self.host,
            port=self.port,
            database=self.database,
        )

    def ping_sql(self) -> str:
        return "SELECT 1"

    def display_name(self) -> str:
        return f"Postgres({self.host}:{self.port}/{self.database})"


@dataclass(frozen=True)
class MySQLConfig:
    """
    MySQL config.

    Requirements
    ------------
    - SQLAlchemy + pandas
    - A MySQL DBAPI driver:
        pip install pymysql
      (or mysql-connector-python)

    Notes
    -----
    - driver="pymysql" is common and stable
    - driver="mysqlconnector" uses mysql-connector-python
    """
    host: str
    database: str
    username: str
    password: str
    port: int = 3306
    driver: str = "pymysql"  # "pymysql" or "mysqlconnector"

    def build_url(self) -> URL:
        return URL.create(
            drivername=f"mysql+{self.driver}",
            username=self.username,
            password=self.password,
            host=self.host,
            port=self.port,
            database=self.database,
        )

    def ping_sql(self) -> str:
        return "SELECT 1"

    def display_name(self) -> str:
        return f"MySQL({self.host}:{self.port}/{self.database})"


@dataclass(frozen=True)
class SQLiteConfig:
    """
    SQLite config (file-based or in-memory).

    Requirements
    ------------
    pip install sqlalchemy pandas

    Notes
    -----
    SQLite is great for testing and local prototyping.
    """
    path: str = ":memory:"  # ":memory:" or "mydb.sqlite"

    def build_url(self) -> URL:
        if self.path == ":memory:":
            return URL.create("sqlite+pysqlite", database=":memory:")
        return URL.create("sqlite+pysqlite", database=self.path)

    def ping_sql(self) -> str:
        return "SELECT 1"

    def display_name(self) -> str:
        return f"SQLite({self.path})"


@dataclass(frozen=True)
class SqlServerConfig:
    """
    SQL Server config via pyodbc, supporting BOTH:
      1) DSN-based connection
      2) Hostname/driver-based connection

    This config uses SQLAlchemy's `odbc_connect=` mechanism to avoid quoting issues.

    Requirements
    ------------
    - pip install sqlalchemy pandas pyodbc
    - Install the ODBC driver on the machine:
      e.g. "ODBC Driver 18 for SQL Server"

    DSN Mode
    --------
    Provide:
      dsn="YourDSN"
    Optionally:
      database, username, password, extra_odbc

    Host/Driver Mode
    ----------------
    Provide:
      host="server" (or "server\\instance")
      odbc_driver="ODBC Driver 18 for SQL Server"  (default)
    Optionally:
      port (default 1433), database, username, password, extra_odbc

    TLS / Security Notes
    --------------------
    ODBC Driver 18 enables Encrypt by default in many environments.
    This class defaults to:
      encrypt="yes"
      trust_server_certificate="yes"
    Adjust to your organization's policy.
    """

    # Choose one:
    dsn: Optional[str] = None
    host: Optional[str] = None

    database: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None

    port: int = 1433
    odbc_driver: str = "ODBC Driver 18 for SQL Server"

    # TLS/security options (common defaults you can override)
    encrypt: str = "yes"
    trust_server_certificate: str = "yes"

    # Add any extra ODBC key/value pairs:
    extra_odbc: Optional[Mapping[str, str]] = None

    def build_url(self) -> str:
        if not self.dsn and not self.host:
            raise ValueError("SqlServerConfig requires either dsn=... or host=...")

        odbc_parts: Dict[str, str] = {}

        if self.dsn:
            odbc_parts["DSN"] = self.dsn
        else:
            # SERVER can include port as "host,1433" for SQL Server ODBC
            server = f"{self.host},{self.port}" if self.port else (self.host or "")
            odbc_parts["DRIVER"] = f"{{{self.odbc_driver}}}"
            odbc_parts["SERVER"] = server

        if self.database:
            odbc_parts["DATABASE"] = self.database

        if self.username is not None:
            odbc_parts["UID"] = self.username
        if self.password is not None:
            odbc_parts["PWD"] = self.password

        # Security toggles (especially relevant for ODBC Driver 18)
        if self.encrypt:
            odbc_parts["Encrypt"] = self.encrypt
        if self.trust_server_certificate:
            odbc_parts["TrustServerCertificate"] = self.trust_server_certificate

        if self.extra_odbc:
            odbc_parts.update(dict(self.extra_odbc))

        encoded = _build_odbc_connect_value(odbc_parts)
        # mssql+pyodbc + odbc_connect pattern
        return f"mssql+pyodbc:///?odbc_connect={encoded}"

    def ping_sql(self) -> str:
        return "SELECT 1"

    def display_name(self) -> str:
        if self.dsn:
            return f"SQLServer(DSN={self.dsn}, DB={self.database})"
        return f"SQLServer({self.host}:{self.port}, DB={self.database})"


@dataclass(frozen=True)
class OracleConfig:
    """
    Oracle config via python-oracledb.

    Requirements
    ------------
    pip install sqlalchemy pandas oracledb
    """
    host: str
    service_name: str
    username: str
    password: str
    port: int = 1521
    driver: str = "oracledb"  # python-oracledb

    def build_url(self) -> URL:
        return URL.create(
            drivername=f"oracle+{self.driver}",
            username=self.username,
            password=self.password,
            host=self.host,
            port=self.port,
            query={"service_name": self.service_name},
        )

    def ping_sql(self) -> str:
        return "SELECT 1 FROM DUAL"

    def display_name(self) -> str:
        return f"Oracle({self.host}:{self.port}/{self.service_name})"


@dataclass(frozen=True)
class SnowflakeConfig:
    """
    Snowflake config via snowflake-sqlalchemy.

    Requirements
    ------------
    pip install sqlalchemy pandas snowflake-sqlalchemy

    Notes
    -----
    This builds a URL string in the common form:
      snowflake://user:pass@account/database/schema?warehouse=...&role=...
    """
    account: str
    username: str
    password: str
    database: str
    schema: str
    warehouse: Optional[str] = None
    role: Optional[str] = None

    def build_url(self) -> str:
        # snowflake://user:pass@account/database/schema?warehouse=...&role=...
        base = f"snowflake://{urllib.parse.quote_plus(self.username)}:{urllib.parse.quote_plus(self.password)}@{self.account}/{self.database}/{self.schema}"
        q: Dict[str, str] = {}
        if self.warehouse:
            q["warehouse"] = self.warehouse
        if self.role:
            q["role"] = self.role
        if q:
            return base + "?" + urllib.parse.urlencode(q)
        return base

    def ping_sql(self) -> str:
        return "SELECT 1"

    def display_name(self) -> str:
        return f"Snowflake({self.account}/{self.database}.{self.schema})"


@dataclass(frozen=True)
class DatabricksConfig:
    """
    Databricks config via databricks-sqlalchemy dialect.

    Requirements
    ------------
    pip install sqlalchemy pandas databricks-sqlalchemy

    Parameters
    ----------
    server_hostname:
        The workspace hostname (e.g. adb-xxx.azuredatabricks.net)
    http_path:
        SQL Warehouse or endpoint HTTP path (e.g. /sql/1.0/warehouses/xxxx)
    token:
        Databricks personal access token
    catalog, schema:
        Optional defaults (useful in multi-catalog environments)
    """
    server_hostname: str              # e.g. adb-xxxx.azuredatabricks.net
    http_path: str                    # e.g. /sql/1.0/warehouses/xxxx
    token: str                        # Databricks personal access token
    catalog: Optional[str] = None
    schema: Optional[str] = None

    def build_url(self) -> URL:
        q: Dict[str, str] = {"http_path": self.http_path}
        if self.catalog:
            q["catalog"] = self.catalog
        if self.schema:
            q["schema"] = self.schema

        return URL.create(
            drivername="databricks",
            username="token",
            password=self.token,
            host=self.server_hostname,
            query=q,
        )

    def ping_sql(self) -> str:
        return "SELECT 1"

    def display_name(self) -> str:
        return f"Databricks({self.server_hostname}, http_path={self.http_path})"


AnyDBConfig = Union[
    PostgresConfig,
    MySQLConfig,
    SQLiteConfig,
    SqlServerConfig,
    OracleConfig,
    SnowflakeConfig,
    DatabricksConfig,
]


# =============================================================================
# Main client
# =============================================================================
class SQLAlchemyDBClient:
    """
    Unified database client built on SQLAlchemy.

    Design goals
    ------------
    1) One consistent API across many databases
       You swap *config objects*, not code paths.

    2) Safe, reusable connection handling
       The client creates a SQLAlchemy Engine (lazy) and uses context managers
       to open/close connections per operation.

    3) Production-friendly ergonomics
       - test_connection() for quick validation
       - chunked reads via iter_query()
       - retry helpers for transient network/DB issues
       - structured metadata returns via QueryResult

    Parameters
    ----------
    config:
        A database config object implementing DBConfig (one of the dataclasses above).
    logger:
        Optional logger. If omitted, a logger named after the class is used.
    echo:
        If True, SQLAlchemy will log SQL statements (noisy; useful for debugging).
    pool_pre_ping:
        If True, SQLAlchemy checks connections before using them (reduces stale-conn errors).
    use_null_pool:
        If True, disables pooling (useful for serverless / short-lived scripts).
    engine_kwargs:
        Additional keyword arguments passed into sqlalchemy.create_engine().

        Examples:
        - {"connect_args": {...}}
        - {"execution_options": {"stream_results": True}}
        - {"isolation_level": "AUTOCOMMIT"}

    Notes
    -----
    - Do NOT f-string user input into SQL. Use bind params instead:
        "SELECT * FROM t WHERE id=:id", params={"id": 123}
    - Some backends require extra dialect packages:
        Databricks -> databricks-sqlalchemy
        Snowflake -> snowflake-sqlalchemy
        Oracle    -> oracledb
    """

    def __init__(
        self,
        config: AnyDBConfig,
        *,
        logger: Optional[logging.Logger] = None,
        echo: bool = False,
        pool_pre_ping: bool = True,
        use_null_pool: bool = False,
        engine_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.echo = echo
        self.pool_pre_ping = pool_pre_ping
        self.use_null_pool = use_null_pool
        self.engine_kwargs: Dict[str, Any] = dict(engine_kwargs or {})
        self._engine: Optional[Engine] = None

    @property
    def engine(self) -> Engine:
        """
        Return the active SQLAlchemy Engine.

        Raises
        ------
        RuntimeError
            If connect() has not been called yet.
        """
        if self._engine is None:
            raise RuntimeError("Engine not initialized. Call connect() first.")
        return self._engine

    def connect(self) -> Engine:
        """
        Create and return the SQLAlchemy Engine (idempotent).

        Returns
        -------
        Engine
            SQLAlchemy engine configured for the target database.
        """
        if self._engine is not None:
            return self._engine

        url = self.config.build_url()
        self.logger.info("Creating engine for %s | %s", self.config.display_name(), _safe_url_str(url))

        kwargs: Dict[str, Any] = {
            "echo": self.echo,
            "pool_pre_ping": self.pool_pre_ping,
            **self.engine_kwargs,
        }
        if self.use_null_pool:
            kwargs["poolclass"] = NullPool

        self._engine = create_engine(url, **kwargs)
        return self._engine
    
    def close(self) -> None:
        """
        Dispose the engine and free pooled connections.

        Safe to call multiple times.
        """
        if self._engine is not None:
            self.logger.info("Disposing engine for %s", self.config.display_name())
            self._engine.dispose()
            self._engine = None

    def __enter__(self) -> "SQLAlchemyDBClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


    def test_connection(self) -> None:
        """
        Validate connectivity using a lightweight ping query.

        Raises
        ------
        SQLAlchemyError / DBAPIError
            If the connection fails.
        """
        self.connect()
        sql = self.config.ping_sql()
        self.logger.info("Testing connection: %s", self.config.display_name())
        with self.engine.connect() as conn:
            conn.execute(text(sql))
        self.logger.info("Connection OK")

    # ---------------- Retry helpers ----------------
    @staticmethod
    def _sleep_backoff(attempt: int, base: float = 0.5, cap: float = 8.0) -> None:
        """
        Exponential backoff sleep: base * 2^attempt, capped at `cap` seconds.
        """
        delay = min(cap, base * (2 ** attempt))
        time.sleep(delay)

    # ---------------- Query methods (simple returns) ----------------
    def execute_query(
        self,
        sql_query: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Execute a SELECT query and return a DataFrame.

        Parameters
        ----------
        sql_query:
            SQL SELECT statement using :param style bind parameters.
        params:
            Bind parameters mapping.

        Returns
        -------
        pd.DataFrame
        """
        self.connect()
        self.logger.debug("execute_query | sql=%s | params=%s", sql_query, params)
        with self.engine.connect() as conn:
            return pd.read_sql_query(sql=text(sql_query), con=conn, params=params)
        
    def iter_query(
        self,
        sql_query: str,
        *,
        chunksize: int = 100_000,
        params: Optional[Mapping[str, Any]] = None,
    ) -> Iterator[pd.DataFrame]:
        """
        Stream a large SELECT query as DataFrame chunks.

        Parameters
        ----------
        sql_query:
            SQL SELECT statement.
        chunksize:
            Number of rows per chunk.
        params:
            Bind parameters mapping.

        Yields
        ------
        pd.DataFrame
            Chunk of results.
        """
        self.connect()
        self.logger.debug("iter_query | chunksize=%s | sql=%s", chunksize, sql_query)
        with self.engine.connect() as conn:
            yield from pd.read_sql_query(
                sql=text(sql_query),
                con=conn,
                params=params,
                chunksize=chunksize,
            )

    def execute_non_query(
        self,
        sql_stmt: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
    ) -> int:
        """
        Execute INSERT/UPDATE/DELETE/DDL and return rowcount when available.

        This method uses `engine.begin()` which opens a transaction and commits
        automatically if no exception is raised.

        Parameters
        ----------
        sql_stmt:
            SQL statement (INSERT/UPDATE/DELETE/DDL).
        params:
            Bind parameters mapping.

        Returns
        -------
        int
            rowcount when available (may be -1 for some DDL/backends).
        """
        self.connect()
        self.logger.debug("execute_non_query | sql=%s | params=%s", sql_stmt, params)
        with self.engine.begin() as conn:
            result = conn.execute(text(sql_stmt), params or {})
            return int(getattr(result, "rowcount", -1))

    def write_dataframe(
        self,
        df: pd.DataFrame,
        *,
        table_name: str,
        schema: Optional[str] = None,
        if_exists: str = "append",
        index: bool = False,
        chunksize: int = 10_000,
        method: Optional[Union[str, Any]] = "multi",
    ) -> None:
        """
        Write a DataFrame to a SQL table using pandas.DataFrame.to_sql().

        Parameters
        ----------
        df:
            The DataFrame to write.
        table_name:
            Target table name.
        schema:
            Optional schema name.
        if_exists:
            "fail" | "replace" | "append"
        index:
            If True, write the DataFrame index as a column.
        chunksize:
            Batch size for inserts.
        method:
            Insert method; "multi" is usually faster. Some backends may require None.

        Notes
        -----
        For very large loads, you may prefer backend-specific bulk loaders, but
        this is an excellent general-purpose option.
        """
        self.connect()
        self.logger.info(
            "write_dataframe | %s rows -> %s%s (if_exists=%s)",
            len(df),
            f"{schema}." if schema else "",
            table_name,
            if_exists,
        )
        df.to_sql(
            name=table_name,
            con=self.engine,
            schema=schema,
            if_exists=if_exists,
            index=index,
            chunksize=chunksize,
            method=method,
        )

    # ---------------- Query methods (rich returns) ----------------
    def execute_query_result(
        self,
        sql_query: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
    ) -> QueryResult:
        """
        Execute INSERT/UPDATE/DELETE/DDL and return rowcount when available.

        This method uses `engine.begin()` which opens a transaction and commits
        automatically if no exception is raised.

        Parameters
        ----------
        sql_stmt:
            SQL statement (INSERT/UPDATE/DELETE/DDL).
        params:
            Bind parameters mapping.

        Returns
        -------
        int
            rowcount when available (may be -1 for some DDL/backends).
        """
        self.connect()
        t0 = time.perf_counter()
        with self.engine.connect() as conn:
            df = pd.read_sql_query(sql=text(sql_query), con=conn, params=params)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return QueryResult(sql=sql_query, elapsed_ms=elapsed_ms, rows=len(df), df=df)

    def execute_non_query_result(
        self,
        sql_stmt: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
    ) -> QueryResult:
        """
        Execute non-SELECT SQL and return QueryResult with timing + row count.

        Returns
        -------
        QueryResult
            Includes elapsed_ms and rows=rowcount (df is None).
        """
        self.connect()
        t0 = time.perf_counter()
        with self.engine.begin() as conn:
            result = conn.execute(text(sql_stmt), params or {})
            rows = int(getattr(result, "rowcount", -1))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return QueryResult(sql=sql_stmt, elapsed_ms=elapsed_ms, rows=rows, df=None)

    # ---------------- Retry wrappers ----------------
    def execute_query_with_retry(
        self,
        sql_query: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        max_retries: int = 3,
    ) -> pd.DataFrame:
        """
        Execute a SELECT query with retries for transient DB/network errors.

        Retries on:
        - sqlalchemy.exc.OperationalError
        - sqlalchemy.exc.DBAPIError

        Parameters
        ----------
        max_retries:
            Number of retry attempts (in addition to the first attempt).
        backoff_base_s:
            Base backoff in seconds.
        backoff_cap_s:
            Max backoff in seconds.

        Returns
        -------
        pd.DataFrame
        """
        self.connect()
        last_err: Optional[BaseException] = None

        for attempt in range(max_retries + 1):
            try:
                return self.execute_query(sql_query, params=params)
            except (OperationalError, DBAPIError) as e:
                last_err = e
                if attempt >= max_retries:
                    raise
                self.logger.warning(
                    "DB error (attempt %s/%s). Retrying... %s",
                    attempt + 1,
                    max_retries + 1,
                    e,
                )
                self._sleep_backoff(attempt)

        raise last_err or RuntimeError("Retry failed")

    def execute_non_query_with_retry(
        self,
        sql_stmt: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        max_retries: int = 3,
        backoff_base_s: float = 0.5,
        backoff_cap_s: float = 8.0,
    ) -> int:
        """
        Execute non-SELECT SQL with retries for transient DB/network errors.

        Returns
        -------
        int
            rowcount when available.
        """
        self.connect()
        last_err: Optional[BaseException] = None

        for attempt in range(max_retries + 1):
            try:
                return self.execute_non_query(sql_stmt, params=params)
            except (OperationalError, DBAPIError) as e:
                last_err = e
                if attempt >= max_retries:
                    raise
                self.logger.warning(
                    "DB error (attempt %s/%s). Retrying... %s",
                    attempt + 1,
                    max_retries + 1,
                    e,
                )
                self._sleep_backoff(attempt, base=backoff_base_s, cap=backoff_cap_s)

        raise last_err or RuntimeError("Retry failed")
    
#%%

"""
EXAMPLES / REFERENCE USAGE FOR `database_manager.py`
===================================================

Copy-paste this whole file anywhere (e.g., `notebooks/db_examples.py` or
`examples/database_manager_examples.py`) and edit imports + credentials.

Assumptions
-----------
- Your module lives at: src/utils/database_manager.py
- So imports look like: from utils.database_manager import ...

If your imports differ, adjust the import lines accordingly.

These examples cover:
- connect / close / context manager usage
- test_connection()
- execute_query()
- execute_query_with_retry()
- iter_query() (chunking)
- execute_non_query() (DDL/DML)
- execute_non_query_with_retry()
- write_dataframe()
- QueryResult returns: execute_query_result() / execute_non_query_result()
- client_from_env() environment-driven configuration
- DB configs: Postgres, MySQL, SQLite, SQL Server (DSN + host/driver), Databricks

Run strategy
------------
Uncomment ONE section at a time and run it.

⚠️ Security
-----------
Do NOT hardcode real passwords/tokens in committed code.
Prefer environment variables or secret managers.
"""

# ---------------------------------------------------------------------
# Adjust this import path to match your project
# ---------------------------------------------------------------------
# from utils.database_manager import (
#     SQLAlchemyDBClient,
#     QueryResult,
#     client_from_env,
#     PostgresConfig,
#     MySQLConfig,
#     SQLiteConfig,
#     SqlServerConfig,
#     DatabricksConfig,
#     SnowflakeConfig,
#     OracleConfig,
# )

# ---------------------------------------------------------------------
# Logging setup (recommended)
# ---------------------------------------------------------------------
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("db_examples")


# =====================================================================
# 1) BASIC PATTERN (WORKS FOR ALL DBS)
# =====================================================================
# def example_basic_postgres() -> None:
#     """
#     Demonstrates:
#     - context manager usage
#     - test_connection()
#     - execute_query()
#     """
#     cfg = PostgresConfig(
#         host="localhost",
#         port=5432,
#         database="analytics",
#         username="postgres",
#         password="secret",
#     )

#     with SQLAlchemyDBClient(cfg) as db:
#         db.test_connection()
#         df = db.execute_query("SELECT 1 AS ok")
#         print(df)


# =====================================================================
# 2) DATABRICKS
# =====================================================================
# def example_databricks() -> None:
#     """
#     Demonstrates Databricks via databricks-sqlalchemy dialect.
#     """
#     cfg = DatabricksConfig(
#         server_hostname="adb-1234567890123456.7.azuredatabricks.net",
#         http_path="/sql/1.0/warehouses/abcd1234efgh5678",
#         token="dapi_xxxxxxxxxxxxxxxxx",
#         catalog="main",
#         schema="default",
#     )

#     # For scripts/jobs, NullPool can be a nice default.
#     with SQLAlchemyDBClient(cfg, use_null_pool=True) as db:
#         db.test_connection()
#         df = db.execute_query("SELECT current_user() AS who")
#         print(df)


# =====================================================================
# 3) SQL SERVER VIA DSN (closest to your current pyodbc DSN usage)
# =====================================================================
# def example_sqlserver_dsn() -> None:
#     """
#     Demonstrates SQL Server DSN-based connection.
#     """
#     cfg = SqlServerConfig(
#         dsn="MyCompanySqlServerDSN",
#         database="MyDB",
#         username="myuser",
#         password="mypassword",
#     )

#     with SQLAlchemyDBClient(cfg) as db:
#         db.test_connection()
#         df = db.execute_query("SELECT TOP (10) * FROM dbo.SomeTable")
#         print(df.head())


# =====================================================================
# 4) SQL SERVER VIA HOSTNAME + DRIVER (no DSN)
# =====================================================================
# def example_sqlserver_host_driver() -> None:
#     """
#     Demonstrates SQL Server hostname/driver mode.
#     """
#     cfg = SqlServerConfig(
#         host="myserver.company.com",
#         port=1433,
#         database="MyDB",
#         username="myuser",
#         password="mypassword",
#         odbc_driver="ODBC Driver 18 for SQL Server",
#         encrypt="yes",
#         trust_server_certificate="yes",
#         extra_odbc={"ApplicationIntent": "ReadOnly"},  # optional
#     )

#     with SQLAlchemyDBClient(cfg) as db:
#         db.test_connection()
#         df = db.execute_query(
#             "SELECT TOP (10) * FROM dbo.SomeTable WHERE id = :id",
#             params={"id": 123},
#         )
#         print(df)


# =====================================================================
# 5) CHUNKED READS FOR BIG TABLES: iter_query()
# =====================================================================
# def example_iter_query_chunking() -> None:
#     """
#     Demonstrates iter_query() to avoid loading an entire large table in memory.
#     """
#     cfg = PostgresConfig(
#         host="localhost",
#         database="analytics",
#         username="postgres",
#         password="secret",
#     )

#     with SQLAlchemyDBClient(cfg) as db:
#         for chunk_df in db.iter_query("SELECT * FROM big_table", chunksize=50_000):
#             print("chunk rows:", len(chunk_df))
            # process chunk_df here


# =====================================================================
# 6) DDL/DML: execute_non_query()
# =====================================================================
# def example_execute_non_query_sqlite() -> None:
#     """
#     Demonstrates:
#     - CREATE TABLE (DDL)
#     - INSERT/UPDATE using params (DML)
#     - SELECT back out
#     """
#     cfg = SQLiteConfig(":memory:")

#     with SQLAlchemyDBClient(cfg) as db:
#         db.execute_non_query("CREATE TABLE t (id INTEGER, name TEXT)")
#         db.execute_non_query(
#             "INSERT INTO t (id, name) VALUES (:id, :name)",
#             params={"id": 1, "name": "Apoorva"},
#         )
#         rows = db.execute_non_query(
#             "UPDATE t SET name=:name WHERE id=:id",
#             params={"id": 1, "name": "Updated"},
#         )
#         print("updated rows:", rows)

#         df = db.execute_query("SELECT * FROM t")
#         print(df)


# =====================================================================
# 7) WRITE A DATAFRAME: write_dataframe()
# =====================================================================
# def example_write_dataframe_postgres() -> None:
#     """
#     Demonstrates DataFrame -> SQL table.
#     """
#     cfg = PostgresConfig(
#         host="localhost",
#         database="analytics",
#         username="postgres",
#         password="secret",
#     )

#     df_in = pd.DataFrame({"id": [1, 2], "name": ["a", "b"]})

#     with SQLAlchemyDBClient(cfg) as db:
#         db.write_dataframe(
#             df_in,
#             table_name="example_table",
#             schema="public",
#             if_exists="replace",
#         )
#         df_out = db.execute_query("SELECT * FROM public.example_table ORDER BY id")
#         print(df_out)


# =====================================================================
# 8) RICH METADATA RETURNS: execute_query_result() / execute_non_query_result()
# =====================================================================
# def example_queryresult_returns() -> None:
#     """
#     Demonstrates QueryResult usage (timing + rows + df).
#     """
#     cfg = SQLiteConfig(":memory:")

#     with SQLAlchemyDBClient(cfg) as db:
#         r1: QueryResult = db.execute_non_query_result("CREATE TABLE t (id INTEGER)")
#         print("DDL:", r1.elapsed_ms, r1.rows)

#         r2: QueryResult = db.execute_non_query_result("INSERT INTO t (id) VALUES (1)")
#         print("INSERT:", r2.elapsed_ms, r2.rows)

#         r3: QueryResult = db.execute_query_result("SELECT * FROM t")
#         print("SELECT:", r3.elapsed_ms, r3.rows)
#         print(r3.df)


# =====================================================================
# 9) RETRIES: execute_query_with_retry() / execute_non_query_with_retry()
# =====================================================================
# def example_retries_sqlserver() -> None:
#     """
#     Demonstrates retry wrappers for transient errors.
#     """
#     cfg = SqlServerConfig(
#         dsn="MyCompanySqlServerDSN",
#         database="MyDB",
#         username="myuser",
#         password="mypassword",
#     )

#     with SQLAlchemyDBClient(cfg) as db:
#         df = db.execute_query_with_retry(
#             "SELECT TOP (10) * FROM dbo.SomeTable",
#             max_retries=3,
#             backoff_base_s=0.5,
#             backoff_cap_s=5.0,
#         )
#         print(df.head())

#         rows = db.execute_non_query_with_retry(
#             "UPDATE dbo.SomeTable SET flag=1 WHERE id=:id",
#             params={"id": 123},
#             max_retries=3,
#         )
#         print("rows updated:", rows)


# =====================================================================
# 10) ENVIRONMENT-DRIVEN CONFIG: client_from_env()
# =====================================================================
# def example_client_from_env_postgres() -> None:
#     """
#     Demonstrates client_from_env() for Postgres.

#     Equivalent environment variables (shell):
#       export DB_KIND=postgres
#       export POSTGRES_HOST=localhost
#       export POSTGRES_DB=analytics
#       export POSTGRES_USER=postgres
#       export POSTGRES_PASSWORD=secret
#     """
#     os.environ["DB_KIND"] = "postgres"
#     os.environ["POSTGRES_HOST"] = "localhost"
#     os.environ["POSTGRES_DB"] = "analytics"
#     os.environ["POSTGRES_USER"] = "postgres"
#     os.environ["POSTGRES_PASSWORD"] = "secret"

#     db = client_from_env()
#     try:
#         db.test_connection()
#         df = db.execute_query("SELECT 1 AS ok")
#         print(df)
#     finally:
#         db.close()


# def example_client_from_env_sqlserver_dsn() -> None:
#     """
#     Demonstrates client_from_env() for SQL Server DSN.

#     Equivalent environment variables (shell):
#       export DB_KIND=sqlserver
#       export MSSQL_DSN=MyCompanySqlServerDSN
#       export MSSQL_DB=MyDB
#       export MSSQL_USER=myuser
#       export MSSQL_PASSWORD=mypassword
#     """
#     os.environ["DB_KIND"] = "sqlserver"
#     os.environ["MSSQL_DSN"] = "MyCompanySqlServerDSN"
#     os.environ["MSSQL_DB"] = "MyDB"
#     os.environ["MSSQL_USER"] = "myuser"
#     os.environ["MSSQL_PASSWORD"] = "mypassword"

#     db = client_from_env()
#     try:
#         db.test_connection()
#         df = db.execute_query("SELECT 1 AS ok")
#         print(df)
#     finally:
#         db.close()

# =====================================================================
# Usage with a Databricks DSN
# =====================================================================

# from utils.database_manager import SQLAlchemyDBClient, OdbcDsnConfig

# cfg = OdbcDsnConfig(dsn="Databricks", autocommit=True)
# with SQLAlchemyDBClient(cfg, use_null_pool=True) as db:
#     df = db.execute_query("SELECT 1")
#     print(df)


# =====================================================================
# MAIN: choose ONE example to run
# =====================================================================
# if __name__ == "__main__":
    # Uncomment ONE at a time:
    # example_basic_postgres()
    # example_databricks()
    # example_sqlserver_dsn()
    # example_sqlserver_host_driver()
    # example_iter_query_chunking()
    # example_execute_non_query_sqlite()
    # example_write_dataframe_postgres()
    # example_queryresult_returns()
    # example_retries_sqlserver()
    # example_client_from_env_postgres()
    # example_client_from_env_sqlserver_dsn()
    # pass

#