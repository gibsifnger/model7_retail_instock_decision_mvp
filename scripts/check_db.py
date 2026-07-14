from pathlib import Path
import sqlite3

db_path = Path("data/database/retail_instock.db")

if not db_path.exists():
    raise FileNotFoundError(f"DB file not found: {db_path.resolve()}")

with sqlite3.connect(db_path) as conn:
    cur = conn.cursor()

    tables = [
        row[0]
        for row in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
    ]

    print(f"DB path: {db_path.resolve()}")
    print("\nTables and row counts:")

    for table in tables:
        row_count = cur.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        print(f"- {table}: {row_count:,} rows")
