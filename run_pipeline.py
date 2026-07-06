"""Run the full Retail InStock Decision MVP pipeline.

This file is an orchestration wrapper only. It does not implement data
generation, SQL mart, feature engineering, modeling, rule engine, or reporting
logic directly; each step is delegated to the existing stage script.
"""

import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent

PIPELINE_STEPS = [
    ("Generate all raw synthetic data", "src/data_generation/03_generate_all_raw_data.py"),
    ("Build SQLite database", "src/database/build_sqlite_db.py"),
    ("Run SQL scripts", "src/database/run_sql_scripts.py"),
    ("Build final feature table", "src/features/build_final_feature_table.py"),
    ("Train demand forecast model", "src/models/train_demand_forecast_model.py"),
    ("Train stockout risk classifier", "src/models/train_stockout_classifier.py"),
    ("Run reorder rule engine", "src/decision_engine/reorder_rule_engine.py"),
    ("Make KPI summary", "src/reporting/make_kpi_summary.py"),
    ("Export Power BI datasets", "src/reporting/export_powerbi_dataset.py"),
]

FINAL_OUTPUTS = [
    "data/raw/sku_master.csv",
    "data/database/retail_instock.db",
    "data/mart/final_modeling_table.csv",
    "outputs/predictions/demand_forecast_result.csv",
    "outputs/predictions/stockout_risk_result.csv",
    "outputs/decisions/reorder_action_result.csv",
    "outputs/metrics/kpi_summary.csv",
    "data/powerbi/powerbi_overview_kpi.csv",
    "data/powerbi/powerbi_reorder_action.csv",
]


def format_elapsed(seconds: float) -> str:
    """Format elapsed runtime in a compact human-readable form."""
    minutes, remaining_seconds = divmod(seconds, 60)
    return f"{int(minutes)}m {remaining_seconds:.1f}s"


def run_step(step_number: int, total_steps: int, step_name: str, script_path: str) -> None:
    """Run one pipeline step and stop immediately on failure."""
    absolute_script_path = REPO_ROOT / script_path
    if not absolute_script_path.is_file():
        raise FileNotFoundError(f"Pipeline step script not found: {script_path}")

    print(
        f"\n[Step {step_number:02d}/{total_steps}] START - {step_name}",
        flush=True,
    )
    print(f"Running: {sys.executable} {script_path}", flush=True)

    step_started_at = time.perf_counter()
    try:
        subprocess.run(
            [sys.executable, str(absolute_script_path)],
            cwd=REPO_ROOT,
            check=True,
        )
    except subprocess.CalledProcessError as error:
        elapsed = format_elapsed(time.perf_counter() - step_started_at)
        print(
            f"[Step {step_number:02d}/{total_steps}] FAILED - {step_name} "
            f"after {elapsed}",
            file=sys.stderr,
            flush=True,
        )
        raise RuntimeError(
            f"Pipeline stopped because step {step_number} failed: {step_name}"
        ) from error

    elapsed = format_elapsed(time.perf_counter() - step_started_at)
    print(
        f"[Step {step_number:02d}/{total_steps}] DONE - {step_name} "
        f"({elapsed})",
        flush=True,
    )


def verify_final_outputs() -> None:
    """Print existence and file size for key final outputs."""
    print("\n[Final Output Check]", flush=True)
    missing_outputs = []
    for relative_path in FINAL_OUTPUTS:
        output_path = REPO_ROOT / relative_path
        if output_path.is_file():
            print(
                f"OK      {relative_path} "
                f"({output_path.stat().st_size:,} bytes)",
                flush=True,
            )
        else:
            print(f"MISSING {relative_path}", flush=True)
            missing_outputs.append(relative_path)

    if missing_outputs:
        raise FileNotFoundError(
            "Pipeline completed but required outputs are missing: "
            + ", ".join(missing_outputs)
        )


def main() -> None:
    """Execute the full MVP pipeline in dependency order."""
    pipeline_started_at = time.perf_counter()
    total_steps = len(PIPELINE_STEPS)

    print("Retail InStock Decision MVP pipeline started.", flush=True)
    print(f"Repository root: {REPO_ROOT}", flush=True)

    for step_number, (step_name, script_path) in enumerate(PIPELINE_STEPS, start=1):
        run_step(step_number, total_steps, step_name, script_path)

    verify_final_outputs()

    total_elapsed = format_elapsed(time.perf_counter() - pipeline_started_at)
    print(f"\nPipeline completed successfully in {total_elapsed}.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError) as error:
        print(f"\n[PIPELINE ERROR] {error}", file=sys.stderr, flush=True)
        raise SystemExit(1) from error
