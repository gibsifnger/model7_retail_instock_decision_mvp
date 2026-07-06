"""Create a portfolio/dashboard KPI summary from existing MVP outputs.

This script does not train models and does not rerun the rule engine. It only
reads generated metrics and decision outputs, then summarizes them into a
single long-format KPI table.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEMAND_METRICS_PATH = REPO_ROOT / "outputs" / "metrics" / "demand_model_metrics.csv"
STOCKOUT_METRICS_PATH = (
    REPO_ROOT / "outputs" / "metrics" / "stockout_model_metrics.csv"
)
REORDER_DECISIONS_PATH = (
    REPO_ROOT / "outputs" / "decisions" / "reorder_action_result.csv"
)
OUTPUT_PATH = REPO_ROOT / "outputs" / "metrics" / "kpi_summary.csv"


def read_required_csv(path: Path, input_name: str) -> pd.DataFrame:
    """Read a required CSV or fail with a clear prerequisite message."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing {input_name}: {path}")
    return pd.read_csv(path)


def require_columns(data: pd.DataFrame, columns: set[str], input_name: str) -> None:
    """Validate that an input contains the columns used by this summary."""
    missing_columns = sorted(columns.difference(data.columns))
    if missing_columns:
        raise ValueError(
            f"{input_name} is missing required columns: "
            + ", ".join(missing_columns)
        )


def select_demand_rows(metrics: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return the demand model row and rolling-mean baseline row."""
    require_columns(
        metrics,
        {"model_name", "wape", "bias", "forecast_accuracy"},
        "demand_model_metrics.csv",
    )

    baseline_rows = metrics[
        metrics["model_name"].str.contains("baseline", case=False, na=False)
    ]
    model_rows = metrics[
        ~metrics["model_name"].str.contains("baseline", case=False, na=False)
    ]
    if model_rows.empty:
        raise ValueError("No demand model row found in demand_model_metrics.csv")
    if baseline_rows.empty:
        raise ValueError("No demand baseline row found in demand_model_metrics.csv")

    return model_rows.iloc[0], baseline_rows.iloc[0]


def select_stockout_rows(metrics: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return the threshold 0.4 stockout model row and rule baseline row."""
    require_columns(
        metrics,
        {
            "model_name",
            "threshold",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
            "false_negative",
        },
        "stockout_model_metrics.csv",
    )

    model_rows = metrics[
        (~metrics["model_name"].str.contains("baseline", case=False, na=False))
        & np.isclose(pd.to_numeric(metrics["threshold"], errors="coerce"), 0.4)
    ]
    baseline_rows = metrics[
        metrics["model_name"].eq("inventory_cover_stockout_rule_baseline")
    ]
    if model_rows.empty:
        raise ValueError(
            "No stockout classifier row found for threshold 0.4 in "
            "stockout_model_metrics.csv"
        )
    if baseline_rows.empty:
        raise ValueError(
            "No inventory_cover_stockout_rule_baseline row found in "
            "stockout_model_metrics.csv"
        )

    return model_rows.iloc[0], baseline_rows.iloc[0]


def add_kpi(
    rows: list[dict[str, object]],
    kpi_group: str,
    kpi_name: str,
    kpi_value: float | int,
    description: str,
) -> None:
    """Append one long-format KPI row."""
    rows.append(
        {
            "kpi_group": kpi_group,
            "kpi_name": kpi_name,
            "kpi_value": kpi_value,
            "description": description,
        }
    )


def build_demand_kpis(
    rows: list[dict[str, object]],
    demand_model: pd.Series,
    demand_baseline: pd.Series,
) -> None:
    """Add demand forecast KPIs."""
    model_wape = float(demand_model["wape"])
    baseline_wape = float(demand_baseline["wape"])

    add_kpi(
        rows,
        "Demand Forecast",
        "model_wape",
        model_wape,
        "WAPE of demand forecast model",
    )
    add_kpi(
        rows,
        "Demand Forecast",
        "baseline_wape",
        baseline_wape,
        "WAPE of sales rolling mean 4-week baseline",
    )
    add_kpi(
        rows,
        "Demand Forecast",
        "wape_improvement_pp",
        (baseline_wape - model_wape) * 100,
        "WAPE improvement of model versus baseline in percentage points",
    )
    add_kpi(
        rows,
        "Demand Forecast",
        "model_bias",
        float(demand_model["bias"]),
        "Bias of demand forecast model",
    )
    add_kpi(
        rows,
        "Demand Forecast",
        "baseline_bias",
        float(demand_baseline["bias"]),
        "Bias of sales rolling mean 4-week baseline",
    )
    add_kpi(
        rows,
        "Demand Forecast",
        "model_forecast_accuracy",
        float(demand_model["forecast_accuracy"]),
        "Forecast Accuracy of demand forecast model, defined as 1 - WAPE",
    )
    add_kpi(
        rows,
        "Demand Forecast",
        "baseline_forecast_accuracy",
        float(demand_baseline["forecast_accuracy"]),
        "Forecast Accuracy of baseline, defined as 1 - WAPE",
    )


def build_stockout_kpis(
    rows: list[dict[str, object]],
    stockout_model: pd.Series,
    stockout_baseline: pd.Series,
) -> None:
    """Add stockout risk classifier KPIs."""
    add_kpi(
        rows,
        "Stockout Risk",
        "model_precision",
        float(stockout_model["precision"]),
        "Precision of stockout risk classifier at threshold 0.4",
    )
    add_kpi(
        rows,
        "Stockout Risk",
        "model_recall",
        float(stockout_model["recall"]),
        "Recall of stockout risk classifier at threshold 0.4",
    )
    add_kpi(
        rows,
        "Stockout Risk",
        "model_f1",
        float(stockout_model["f1"]),
        "F1 score of stockout risk classifier at threshold 0.4",
    )
    add_kpi(
        rows,
        "Stockout Risk",
        "model_pr_auc",
        float(stockout_model["pr_auc"]),
        "PR-AUC of stockout risk classifier",
    )
    add_kpi(
        rows,
        "Stockout Risk",
        "model_roc_auc",
        float(stockout_model["roc_auc"]),
        "ROC-AUC of stockout risk classifier",
    )
    add_kpi(
        rows,
        "Stockout Risk",
        "baseline_precision",
        float(stockout_baseline["precision"]),
        "Precision of inventory cover stockout rule baseline",
    )
    add_kpi(
        rows,
        "Stockout Risk",
        "baseline_recall",
        float(stockout_baseline["recall"]),
        "Recall of inventory cover stockout rule baseline",
    )
    add_kpi(
        rows,
        "Stockout Risk",
        "baseline_f1",
        float(stockout_baseline["f1"]),
        "F1 score of inventory cover stockout rule baseline",
    )
    add_kpi(
        rows,
        "Stockout Risk",
        "baseline_pr_auc",
        float(stockout_baseline["pr_auc"]),
        "PR-AUC of inventory cover stockout rule baseline",
    )
    add_kpi(
        rows,
        "Stockout Risk",
        "false_negative_model",
        int(stockout_model["false_negative"]),
        "False negatives of stockout risk classifier at threshold 0.4",
    )
    add_kpi(
        rows,
        "Stockout Risk",
        "false_negative_baseline",
        int(stockout_baseline["false_negative"]),
        "False negatives of inventory cover stockout rule baseline",
    )


def build_reorder_kpis(rows: list[dict[str, object]], decisions: pd.DataFrame) -> None:
    """Add reorder decision KPIs."""
    require_columns(
        decisions,
        {"recommended_action", "recommended_order_qty"},
        "reorder_action_result.csv",
    )

    action_counts = decisions["recommended_action"].value_counts()
    total_rows = int(len(decisions))
    total_order_qty = int(pd.to_numeric(decisions["recommended_order_qty"]).sum())

    buy_count = int(action_counts.get("BUY", 0))
    hold_count = int(action_counts.get("HOLD", 0))
    expedite_count = int(action_counts.get("EXPEDITE", 0))
    reduce_count = int(action_counts.get("REDUCE", 0))

    safe_total = total_rows if total_rows else np.nan
    add_kpi(
        rows,
        "Reorder Decision",
        "total_decision_rows",
        total_rows,
        "Total number of SKU-channel-center-week decision rows",
    )
    add_kpi(
        rows,
        "Reorder Decision",
        "total_recommended_order_qty",
        total_order_qty,
        "Total recommended order quantity across all decisions",
    )
    add_kpi(
        rows,
        "Reorder Decision",
        "buy_count",
        buy_count,
        "Number of BUY recommendations",
    )
    add_kpi(
        rows,
        "Reorder Decision",
        "hold_count",
        hold_count,
        "Number of HOLD recommendations",
    )
    add_kpi(
        rows,
        "Reorder Decision",
        "expedite_count",
        expedite_count,
        "Number of EXPEDITE recommendations",
    )
    add_kpi(
        rows,
        "Reorder Decision",
        "reduce_count",
        reduce_count,
        "Number of REDUCE recommendations",
    )
    add_kpi(
        rows,
        "Reorder Decision",
        "buy_ratio",
        buy_count / safe_total,
        "Share of BUY recommendations",
    )
    add_kpi(
        rows,
        "Reorder Decision",
        "hold_ratio",
        hold_count / safe_total,
        "Share of HOLD recommendations",
    )
    add_kpi(
        rows,
        "Reorder Decision",
        "expedite_ratio",
        expedite_count / safe_total,
        "Share of EXPEDITE recommendations",
    )
    add_kpi(
        rows,
        "Reorder Decision",
        "reduce_ratio",
        reduce_count / safe_total,
        "Share of REDUCE recommendations",
    )
    add_kpi(
        rows,
        "Reorder Decision",
        "avg_recommended_order_qty",
        float(pd.to_numeric(decisions["recommended_order_qty"]).mean()),
        "Average recommended order quantity across all decisions",
    )
    add_kpi(
        rows,
        "Reorder Decision",
        "max_recommended_order_qty",
        int(pd.to_numeric(decisions["recommended_order_qty"]).max()),
        "Maximum recommended order quantity across all decisions",
    )


def main() -> None:
    """Build and save the long-format KPI summary table."""
    demand_metrics = read_required_csv(
        DEMAND_METRICS_PATH, "demand model metrics"
    )
    stockout_metrics = read_required_csv(
        STOCKOUT_METRICS_PATH, "stockout model metrics"
    )
    reorder_decisions = read_required_csv(
        REORDER_DECISIONS_PATH, "reorder action result"
    )

    demand_model, demand_baseline = select_demand_rows(demand_metrics)
    stockout_model, stockout_baseline = select_stockout_rows(stockout_metrics)

    kpi_rows: list[dict[str, object]] = []
    build_demand_kpis(kpi_rows, demand_model, demand_baseline)
    build_stockout_kpis(kpi_rows, stockout_model, stockout_baseline)
    build_reorder_kpis(kpi_rows, reorder_decisions)

    kpi_summary = pd.DataFrame(
        kpi_rows,
        columns=["kpi_group", "kpi_name", "kpi_value", "description"],
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    kpi_summary.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"KPI summary rows: {len(kpi_summary)}")
    print(f"Saved KPI summary to: {OUTPUT_PATH}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"[ERROR] {error}", file=sys.stderr, flush=True)
        raise SystemExit(1) from error
