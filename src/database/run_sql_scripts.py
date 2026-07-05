"""Run ordered SQL files against the retail in-stock SQLite database."""

import sqlite3
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DATABASE_PATH = REPO_ROOT / "data" / "database" / "retail_instock.db"
SQL_DIR = REPO_ROOT / "sql"


def find_sql_scripts() -> list[Path]:
    """Validate inputs and return SQL files in ascending filename order."""
    if not DATABASE_PATH.is_file():
        raise FileNotFoundError(
            f"SQLite database not found: {DATABASE_PATH}\n"
            "Run src/database/build_sqlite_db.py first."
        )

    sql_scripts = sorted(
        (path for path in SQL_DIR.glob("*.sql") if path.is_file()),
        key=lambda path: path.name,
    )
    if not sql_scripts:
        raise FileNotFoundError(
            f"No .sql files found in: {SQL_DIR}\n"
            "Add at least one SQL script before running this file."
        )

    return sql_scripts


def run_sql_script(connection: sqlite3.Connection, script_path: Path) -> None:
    """Execute one non-empty SQL file and identify it clearly on failure."""
    print(f"[START] {script_path.relative_to(REPO_ROOT)}", flush=True)
    sql_text = script_path.read_text(encoding="utf-8-sig")

    if not sql_text.strip():
        print(
            f"[WARNING] Empty SQL file skipped: {script_path.relative_to(REPO_ROOT)}",
            flush=True,
        )
        print(f"[DONE]  {script_path.name} (skipped)", flush=True)
        return

    try:
        connection.executescript(sql_text)
        connection.commit()
    except sqlite3.Error as error:
        connection.rollback()
        raise RuntimeError(
            f"SQL execution failed in {script_path.relative_to(REPO_ROOT)}: {error}"
        ) from error

    print(f"[DONE]  {script_path.name}", flush=True)


def quote_identifier(identifier: str) -> str:
    """Safely quote a table or view name read from sqlite_master."""
    return '"' + identifier.replace('"', '""') + '"'


def print_database_objects(connection: sqlite3.Connection) -> None:
    """Print user-created tables/views and their row counts when available."""
    objects = connection.execute(
        """
        SELECT type, name
        FROM sqlite_master
        WHERE type IN ('table', 'view')
          AND name NOT LIKE 'sqlite_%'
        ORDER BY type, name
        """
    ).fetchall()

    print("\n[DATABASE OBJECTS]", flush=True)
    if not objects:
        print("No user-created tables or views found.", flush=True)
        return

    for object_type, object_name in objects:
        try:
            row_count = connection.execute(
                f"SELECT COUNT(*) FROM {quote_identifier(object_name)}"
            ).fetchone()[0]
            print(f"[{object_type.upper()}] {object_name}: {row_count} rows")
        except sqlite3.Error as error:
            # A complex or temporarily invalid view may not support COUNT(*).
            print(
                f"[{object_type.upper()}] {object_name}: "
                f"row count unavailable ({error})"
            )


def main() -> None:
    """Run all SQL scripts and report the resulting database objects."""
    sql_scripts = find_sql_scripts()
    print(f"SQLite database: {DATABASE_PATH.relative_to(REPO_ROOT)}", flush=True)
    print(f"SQL scripts found: {len(sql_scripts)}", flush=True)

    with sqlite3.connect(DATABASE_PATH) as connection:
        for script_path in sql_scripts:
            run_sql_script(connection, script_path)
        print_database_objects(connection)

    print("\nAll SQL scripts completed successfully.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, OSError, RuntimeError, sqlite3.Error) as error:
        print(f"[ERROR] {error}", file=sys.stderr, flush=True)
        raise SystemExit(1) from error
