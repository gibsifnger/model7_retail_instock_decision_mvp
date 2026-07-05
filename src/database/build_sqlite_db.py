"""Load the nine raw CSV files into a fresh SQLite staging database."""

import sqlite3
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = REPO_ROOT / "data" / "raw"
DATABASE_DIR = REPO_ROOT / "data" / "database"
DATABASE_PATH = DATABASE_DIR / "retail_instock.db"

RAW_CSV_FILES = [
    "sku_master.csv",
    "vendor_master.csv",
    "sku_code_mapping.csv",
    "oms_sales_orders.csv",
    "wms_inventory_snapshot.csv",
    "wms_goods_receipts.csv",
    "erp_purchase_orders.csv",
    "md_promotion_calendar.csv",
    "manual_overrides.csv",
]


def validate_input_files() -> list[Path]:
    """Return all raw CSV paths, or fail before replacing an existing DB."""
    csv_paths = [RAW_DATA_DIR / filename for filename in RAW_CSV_FILES]
    missing_files = [path for path in csv_paths if not path.is_file()]

    if missing_files:
        missing_list = "\n".join(f"- {path}" for path in missing_files)
        raise FileNotFoundError(
            "Cannot build the SQLite database because raw CSV files are missing:\n"
            f"{missing_list}\n"
            "Run src/data_generation/03_generate_all_raw_data.py first."
        )

    return csv_paths


def prepare_database_path() -> None:
    """Create the database directory and remove a previous generated DB."""
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    if DATABASE_PATH.exists():
        DATABASE_PATH.unlink()


def load_csv_to_table(
    connection: sqlite3.Connection,
    csv_path: Path,
) -> int:
    """Load one CSV as a same-named staging table and return its DB row count."""
    table_name = csv_path.stem

    # Do not parse or transform date columns in the staging layer. Empty source
    # values also remain empty strings rather than being inferred as NaN.
    dataframe = pd.read_csv(csv_path, keep_default_na=False)
    dataframe.to_sql(table_name, connection, if_exists="replace", index=False)

    row_count = connection.execute(
        f'SELECT COUNT(*) FROM "{table_name}"'
    ).fetchone()[0]
    if row_count != len(dataframe):
        raise RuntimeError(
            f"Row-count mismatch for {table_name}: "
            f"CSV={len(dataframe)}, SQLite={row_count}"
        )

    return int(row_count)


def build_database() -> None:
    """Create a fresh SQLite DB and load all raw files as staging tables."""
    csv_paths = validate_input_files()
    prepare_database_path()

    with sqlite3.connect(DATABASE_PATH) as connection:
        for csv_path in csv_paths:
            row_count = load_csv_to_table(connection, csv_path)
            print(f"Loaded {csv_path.stem}: {row_count} rows", flush=True)

    relative_database_path = DATABASE_PATH.relative_to(REPO_ROOT)
    print(f"SQLite database created at {relative_database_path}", flush=True)


if __name__ == "__main__":
    try:
        build_database()
    except (FileNotFoundError, OSError, RuntimeError, sqlite3.Error) as error:
        print(f"[ERROR] {error}", file=sys.stderr, flush=True)
        raise SystemExit(1) from error
