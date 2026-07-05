"""Run all raw-data generators and verify their nine CSV outputs."""

import csv
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_GENERATION_DIR = REPO_ROOT / "src" / "data_generation"
RAW_DATA_DIR = REPO_ROOT / "data" / "raw"

GENERATION_STEPS = [
    ("Master data generation", DATA_GENERATION_DIR / "01_generate_master_data.py"),
    (
        "Transaction data generation",
        DATA_GENERATION_DIR / "02_generate_transaction_data.py",
    ),
]

EXPECTED_OUTPUTS = [
    RAW_DATA_DIR / "sku_master.csv",
    RAW_DATA_DIR / "vendor_master.csv",
    RAW_DATA_DIR / "sku_code_mapping.csv",
    RAW_DATA_DIR / "oms_sales_orders.csv",
    RAW_DATA_DIR / "wms_inventory_snapshot.csv",
    RAW_DATA_DIR / "wms_goods_receipts.csv",
    RAW_DATA_DIR / "erp_purchase_orders.csv",
    RAW_DATA_DIR / "md_promotion_calendar.csv",
    RAW_DATA_DIR / "manual_overrides.csv",
]


def run_generation_step(step_name: str, script_path: Path) -> None:
    """Run one generator with the current Python interpreter."""
    if not script_path.is_file():
        raise FileNotFoundError(f"Generator script not found: {script_path}")

    print(f"\n[START] {step_name}", flush=True)
    try:
        subprocess.run(
            [sys.executable, str(script_path)],
            cwd=REPO_ROOT,
            check=True,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"{step_name} failed with exit code {error.returncode}: {script_path}"
        ) from error
    print(f"[DONE]  {step_name}", flush=True)


def count_csv_rows(csv_path: Path) -> int:
    """Count data rows in a CSV, excluding its header."""
    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.reader(csv_file)
        try:
            next(reader)
        except StopIteration as error:
            raise ValueError(f"CSV is empty and has no header: {csv_path}") from error
        return sum(1 for _ in reader)


def verify_outputs() -> None:
    """Fail if an expected CSV is missing; otherwise print all row counts."""
    missing_files = [path for path in EXPECTED_OUTPUTS if not path.is_file()]
    if missing_files:
        missing_list = "\n".join(f"- {path}" for path in missing_files)
        raise FileNotFoundError(
            "Raw-data generation completed, but expected CSV files are missing:\n"
            f"{missing_list}"
        )

    print("\n[VERIFY] Generated raw-data files", flush=True)
    for output_path in EXPECTED_OUTPUTS:
        row_count = count_csv_rows(output_path)
        print(f"[OK] {output_path} ({row_count} rows)")


def main() -> None:
    """Run master and transaction generators, then verify all outputs."""
    for step_name, script_path in GENERATION_STEPS:
        run_generation_step(step_name, script_path)

    verify_outputs()
    print("\nAll raw-data generation steps completed successfully.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"\n[ERROR] {error}", file=sys.stderr, flush=True)
        raise SystemExit(1) from error
