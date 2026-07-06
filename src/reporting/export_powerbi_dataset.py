"""Export existing MVP outputs as Power BI-friendly CSV datasets.

This script does not train models and does not rerun the rule engine. It only
reshapes already generated tables into dashboard-ready CSV files.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]

FINAL_MODELING_TABLE_PATH = REPO_ROOT / "data" / "mart" / "final_modeling_table.csv"
DEMAND_FORECAST_PATH = (
    REPO_ROOT / "outputs" / "predictions" / "demand_forecast_result.csv"
)
STOCKOUT_RISK_PATH = (
    REPO_ROOT / "outputs" / "predictions" / "stockout_risk_result.csv"
)
REORDER_ACTION_PATH = (
    REPO_ROOT / "outputs" / "decisions" / "reorder_action_result.csv"
)
KPI_SUMMARY_PATH = REPO_ROOT / "outputs" / "metrics" / "kpi_summary.csv"

POWERBI_DIR = REPO_ROOT / "data" / "powerbi"


OVERVIEW_KPI_COLUMNS = [
    "kpi_group",
    "kpi_name",
    "kpi_value",
    "description",
]

DEMAND_FORECAST_COLUMNS = [
    "sku_id",
    "channel_id",
    "center_id",
    "week_start_date",
    "target_demand_next_1w",
    "baseline_pred",
    "demand_pred",
    "abs_error",
    "ape",
    "error",
    "promo_flag",
    "stockout_flag",
    "inventory_cover_weeks",
]

STOCKOUT_RISK_COLUMNS = [
    "sku_id",
    "channel_id",
    "center_id",
    "week_start_date",
    "target_stockout_risk_next_2w",
    "stockout_risk_pred_proba",
    "stockout_risk_pred_label",
    "baseline_risk_pred",
    "inventory_cover_weeks",
    "stockout_days_last_4w",
    "available_qty",
    "inbound_qty_next_2w",
    "promo_flag",
    "overdue_po_qty",
]

REORDER_ACTION_COLUMNS = [
    "sku_id",
    "channel_id",
    "center_id",
    "week_start_date",
    "category_l1",
    "category_l2",
    "brand",
    "demand_pred",
    "stockout_risk_pred_proba",
    "stockout_risk_pred_label",
    "available_qty",
    "inbound_qty_next_1w",
    "inbound_qty_next_2w",
    "inbound_qty_next_4w",
    "inventory_position_qty",
    "inventory_cover_weeks",
    "safety_stock_qty",
    "lead_time_weeks",
    "review_period_weeks",
    "target_cover_weeks",
    "target_inventory_qty",
    "raw_order_need",
    "recommended_order_qty",
    "recommended_action",
    "action_reason",
    "moq_qty",
    "order_multiple",
    "min_order_amount",
    "unit_cost",
    "open_po_qty",
    "overdue_po_qty",
    "promo_flag",
    "manual_override_flag",
    "override_type",
    "override_reason",
]

SKU_DETAIL_COLUMNS = [
    "sku_id",
    "category_l1",
    "category_l2",
    "brand",
    "default_vendor_id",
    "unit_cost",
    "list_price",
    "gross_margin_rate",
    "shelf_life_days",
    "moq_qty",
    "order_multiple",
    "min_order_amount",
    "active_flag",
]


def read_required_csv(path: Path, input_name: str) -> pd.DataFrame:
    """Read a required input file or fail with a clear message."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing {input_name}: {path}")
    return pd.read_csv(path)


def require_columns(data: pd.DataFrame, columns: list[str], input_name: str) -> None:
    """Validate required columns before export."""
    missing_columns = [column for column in columns if column not in data.columns]
    if missing_columns:
        raise ValueError(
            f"{input_name} is missing required columns: "
            + ", ".join(missing_columns)
        )


def make_risk_bucket(probability: pd.Series) -> np.ndarray:
    """Create a three-level risk bucket from stockout probability."""
    probability = pd.to_numeric(probability, errors="coerce").fillna(0)
    return np.select(
        [probability >= 0.7, probability >= 0.4],
        ["HIGH", "MEDIUM"],
        default="LOW",
    )


def export_csv(data: pd.DataFrame, filename: str) -> None:
    """Save one Power BI dataset and print its validation summary."""
    output_path = POWERBI_DIR / filename
    data.to_csv(output_path, index=False, encoding="utf-8")
    print(
        f"{filename}: rows={len(data)}, columns={len(data.columns)}, "
        f"path={output_path}"
    )


def build_overview_kpi() -> pd.DataFrame:
    """Build Power BI overview KPI dataset."""
    data = read_required_csv(KPI_SUMMARY_PATH, "KPI summary")
    require_columns(data, OVERVIEW_KPI_COLUMNS, "kpi_summary.csv")
    return data[OVERVIEW_KPI_COLUMNS].copy()


def build_demand_forecast() -> pd.DataFrame:
    """Build Power BI demand forecast dataset with error flags."""
    data = read_required_csv(DEMAND_FORECAST_PATH, "demand forecast result")
    require_columns(data, DEMAND_FORECAST_COLUMNS, "demand_forecast_result.csv")

    output = data[DEMAND_FORECAST_COLUMNS].copy()
    error = pd.to_numeric(output["error"], errors="coerce").fillna(0)
    ape = pd.to_numeric(output["ape"], errors="coerce")

    output["forecast_error_direction"] = np.select(
        [error > 0, error < 0],
        ["OVER_FORECAST", "UNDER_FORECAST"],
        default="EXACT",
    )
    output["high_ape_flag"] = (ape >= 0.5).astype(int)
    return output


def build_stockout_risk() -> pd.DataFrame:
    """Build Power BI stockout risk dataset with buckets and result labels."""
    data = read_required_csv(STOCKOUT_RISK_PATH, "stockout risk result")
    require_columns(data, STOCKOUT_RISK_COLUMNS, "stockout_risk_result.csv")

    output = data[STOCKOUT_RISK_COLUMNS].copy()
    target = pd.to_numeric(
        output["target_stockout_risk_next_2w"], errors="coerce"
    )
    prediction = pd.to_numeric(
        output["stockout_risk_pred_label"], errors="coerce"
    )

    output["risk_bucket"] = make_risk_bucket(output["stockout_risk_pred_proba"])
    output["prediction_result"] = np.select(
        [
            target.eq(1) & prediction.eq(1),
            target.eq(0) & prediction.eq(1),
            target.eq(1) & prediction.eq(0),
            target.eq(0) & prediction.eq(0),
        ],
        [
            "TRUE_POSITIVE",
            "FALSE_POSITIVE",
            "FALSE_NEGATIVE",
            "TRUE_NEGATIVE",
        ],
        default="UNKNOWN",
    )
    return output


def build_reorder_action() -> pd.DataFrame:
    """Build Power BI reorder action dataset with priority and amount fields."""
    data = read_required_csv(REORDER_ACTION_PATH, "reorder action result")
    require_columns(data, REORDER_ACTION_COLUMNS, "reorder_action_result.csv")

    output = data[REORDER_ACTION_COLUMNS].copy()
    priority_map = {"EXPEDITE": 1, "BUY": 2, "REDUCE": 3, "HOLD": 4}
    output["action_priority"] = output["recommended_action"].map(priority_map)
    output["estimated_order_amount"] = (
        pd.to_numeric(output["recommended_order_qty"], errors="coerce").fillna(0)
        * pd.to_numeric(output["unit_cost"], errors="coerce").fillna(0)
    )
    output["risk_bucket"] = make_risk_bucket(output["stockout_risk_pred_proba"])
    return output


def build_sku_detail() -> pd.DataFrame:
    """Build SKU-level dimension table from the final modeling table."""
    data = read_required_csv(FINAL_MODELING_TABLE_PATH, "final modeling table")
    require_columns(data, SKU_DETAIL_COLUMNS, "final_modeling_table.csv")

    output = (
        data[SKU_DETAIL_COLUMNS]
        .sort_values("sku_id", kind="stable")
        .drop_duplicates("sku_id", keep="first")
        .reset_index(drop=True)
    )
    return output


def main() -> None:
    """Create all Power BI CSV exports."""
    POWERBI_DIR.mkdir(parents=True, exist_ok=True)

    exports = {
        "powerbi_overview_kpi.csv": build_overview_kpi(),
        "powerbi_demand_forecast.csv": build_demand_forecast(),
        "powerbi_stockout_risk.csv": build_stockout_risk(),
        "powerbi_reorder_action.csv": build_reorder_action(),
        "powerbi_sku_detail.csv": build_sku_detail(),
    }

    for filename, data in exports.items():
        export_csv(data, filename)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"[ERROR] {error}", file=sys.stderr, flush=True)
        raise SystemExit(1) from error
