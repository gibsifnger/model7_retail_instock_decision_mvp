"""Create constrained reorder actions from existing model predictions.

This module is a deterministic MVP rule engine. It does not train models, and
its outputs are decision recommendations rather than ML targets.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURE_TABLE_PATH = REPO_ROOT / "data" / "mart" / "final_modeling_table.csv"
DEMAND_PREDICTION_PATH = (
    REPO_ROOT / "outputs" / "predictions" / "demand_forecast_result.csv"
)
STOCKOUT_PREDICTION_PATH = (
    REPO_ROOT / "outputs" / "predictions" / "stockout_risk_result.csv"
)
OUTPUT_PATH = REPO_ROOT / "outputs" / "decisions" / "reorder_action_result.csv"

KEY_COLUMNS = ["sku_id", "channel_id", "center_id", "week_start_date"]
ACTION_LABELS = {"BUY", "HOLD", "EXPEDITE", "REDUCE"}

REQUIRED_FEATURE_COLUMNS = {
    *KEY_COLUMNS,
    "category_l1",
    "category_l2",
    "brand",
    "available_qty",
    "inbound_qty_next_1w",
    "inbound_qty_next_2w",
    "inbound_qty_next_4w",
    "inventory_position_qty",
    "inventory_cover_weeks",
    "safety_stock_qty",
    "standard_lead_time_days",
    "order_cycle_days",
    "open_po_qty",
    "overdue_po_qty",
    "moq_qty",
    "order_multiple",
    "min_order_amount",
    "unit_cost",
    "active_flag",
    "order_block_flag",
    "manual_override_flag",
    "override_type",
    "override_reason",
    "promo_flag",
}

OUTPUT_COLUMNS = [
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


def read_csv_input(path: Path, input_name: str) -> pd.DataFrame:
    """Read one required input or fail with a clear prerequisite message."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing {input_name}: {path}")
    return pd.read_csv(path)


def validate_unique_keys(data: pd.DataFrame, input_name: str) -> None:
    """Require one row per decision key in every input before joining."""
    missing_keys = sorted(set(KEY_COLUMNS).difference(data.columns))
    if missing_keys:
        raise ValueError(
            f"{input_name} is missing join keys: {', '.join(missing_keys)}"
        )
    duplicate_count = int(data.duplicated(KEY_COLUMNS).sum())
    if duplicate_count:
        raise ValueError(
            f"{input_name} contains {duplicate_count} duplicate decision keys."
        )


def load_and_join_inputs() -> pd.DataFrame:
    """Left-join both prediction outputs onto the final modeling table."""
    feature_table = read_csv_input(FEATURE_TABLE_PATH, "final modeling table")
    demand_predictions = read_csv_input(
        DEMAND_PREDICTION_PATH, "demand forecast predictions"
    )
    stockout_predictions = read_csv_input(
        STOCKOUT_PREDICTION_PATH, "stockout risk predictions"
    )

    missing_features = sorted(
        REQUIRED_FEATURE_COLUMNS.difference(feature_table.columns)
    )
    if missing_features:
        raise ValueError(
            "Final modeling table is missing required columns: "
            + ", ".join(missing_features)
        )
    for frame, name in [
        (feature_table, "final modeling table"),
        (demand_predictions, "demand forecast predictions"),
        (stockout_predictions, "stockout risk predictions"),
    ]:
        validate_unique_keys(frame, name)
        frame["week_start_date"] = pd.to_datetime(
            frame["week_start_date"], errors="raise"
        )

    if "demand_pred" not in demand_predictions.columns:
        raise ValueError("Demand prediction input is missing: demand_pred")
    missing_risk_columns = {
        "stockout_risk_pred_proba",
        "stockout_risk_pred_label",
    }.difference(stockout_predictions.columns)
    if missing_risk_columns:
        raise ValueError(
            "Stockout prediction input is missing: "
            + ", ".join(sorted(missing_risk_columns))
        )

    joined = feature_table.merge(
        demand_predictions[KEY_COLUMNS + ["demand_pred"]],
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    joined = joined.merge(
        stockout_predictions[
            KEY_COLUMNS
            + ["stockout_risk_pred_proba", "stockout_risk_pred_label"]
        ],
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )

    # The v1 engine acts only where both model outputs are available. The two
    # model holdouts differ by one week, so this naturally keeps their overlap.
    joined = joined.loc[
        joined["demand_pred"].notna()
        & joined["stockout_risk_pred_proba"].notna()
    ].copy()
    if joined.empty:
        raise ValueError("No rows have both demand and stockout model predictions.")

    return joined.sort_values(KEY_COLUMNS, kind="stable").reset_index(drop=True)


def normalize_numeric_inputs(data: pd.DataFrame) -> pd.DataFrame:
    """Apply safe v1 defaults needed for quantity and action calculations."""
    data = data.copy()
    numeric_columns = [
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
        "standard_lead_time_days",
        "order_cycle_days",
        "open_po_qty",
        "overdue_po_qty",
        "moq_qty",
        "order_multiple",
        "min_order_amount",
        "unit_cost",
        "active_flag",
        "order_block_flag",
        "manual_override_flag",
        "promo_flag",
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    zero_default_columns = [
        "available_qty",
        "inbound_qty_next_1w",
        "inbound_qty_next_2w",
        "inbound_qty_next_4w",
        "safety_stock_qty",
        "standard_lead_time_days",
        "order_cycle_days",
        "open_po_qty",
        "overdue_po_qty",
        "moq_qty",
        "min_order_amount",
        "unit_cost",
        "active_flag",
        "order_block_flag",
        "manual_override_flag",
        "promo_flag",
    ]
    data[zero_default_columns] = data[zero_default_columns].fillna(0)
    data["order_multiple"] = data["order_multiple"].fillna(1)
    data.loc[data["order_multiple"].le(0), "order_multiple"] = 1
    data["demand_pred"] = data["demand_pred"].clip(lower=0)
    data["stockout_risk_pred_proba"] = data[
        "stockout_risk_pred_proba"
    ].clip(lower=0, upper=1)

    return data


def apply_order_constraints(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate replenishment need and round it to executable order quantities."""
    data = normalize_numeric_inputs(data)
    data["lead_time_weeks"] = np.ceil(
        data["standard_lead_time_days"].clip(lower=0) / 7.0
    ).astype(int)
    data["review_period_weeks"] = np.ceil(
        data["order_cycle_days"].clip(lower=0) / 7.0
    ).astype(int)
    data["target_cover_weeks"] = (
        data["lead_time_weeks"] + data["review_period_weeks"]
    )
    data["target_inventory_qty"] = (
        data["demand_pred"] * data["target_cover_weeks"]
        + data["safety_stock_qty"]
    )
    data["eligible_confirmed_inbound_qty"] = data["inbound_qty_next_4w"]
    data["current_inventory_position"] = (
        data["available_qty"] + data["eligible_confirmed_inbound_qty"]
    )
    # Keep the output inventory position aligned with the rule calculation.
    data["inventory_position_qty"] = data["current_inventory_position"]
    data["raw_order_need"] = (
        data["target_inventory_qty"] - data["current_inventory_position"]
    ).clip(lower=0)

    data["recommended_order_qty"] = data.apply(
        calculate_constrained_order_quantity,
        axis=1,
    )
    return data


def calculate_constrained_order_quantity(row: pd.Series) -> int:
    """Apply MOQ, order multiple, and minimum amount to positive raw need."""
    raw_order_need = max(float(row["raw_order_need"]), 0.0)
    if raw_order_need <= 0:
        return 0

    order_multiple = max(int(np.ceil(float(row["order_multiple"]))), 1)
    moq_qty = max(float(row["moq_qty"]), 0.0)
    min_order_amount = max(float(row["min_order_amount"]), 0.0)
    unit_cost = max(float(row["unit_cost"]), 0.0)

    minimum_amount_qty = (
        np.ceil(min_order_amount / unit_cost)
        if min_order_amount > 0 and unit_cost > 0
        else 0.0
    )
    constrained_need = max(raw_order_need, moq_qty, minimum_amount_qty)
    rounded_quantity = np.ceil(constrained_need / order_multiple) * order_multiple
    return max(int(rounded_quantity), 0)


def override_reason_text(row: pd.Series) -> str:
    """Return a readable manual reason even when source text is missing."""
    reason = row.get("override_reason")
    if pd.isna(reason) or not str(reason).strip():
        return "No override reason provided"
    return str(reason).strip()


def decide_action(row: pd.Series) -> tuple[str, int, str]:
    """Apply the required A-to-F action priority to one decision row."""
    calculated_qty = max(int(row["recommended_order_qty"]), 0)
    risk_probability = float(row["stockout_risk_pred_proba"])
    inventory_cover = float(row["inventory_cover_weeks"])
    lead_time_weeks = float(row["lead_time_weeks"])
    target_cover_weeks = float(row["target_cover_weeks"])

    # A. Inactive or blocked products always stop ordering.
    if int(row["active_flag"]) == 0 or int(row["order_block_flag"]) == 1:
        return "HOLD", 0, "HOLD: Inactive or blocked SKU"

    # B. Approved manual overrides take precedence over model-driven actions.
    override_type = str(row.get("override_type", "")).upper()
    if int(row["manual_override_flag"]) == 1:
        reason = override_reason_text(row)
        if override_type in {"FORCE_HOLD", "BLOCK_BUY"}:
            return "HOLD", 0, f"HOLD: manual override - {reason}"
        if override_type == "EXPEDITE":
            return "EXPEDITE", calculated_qty, f"EXPEDITE: manual override - {reason}"

    # C. High risk inside normal lead time with an existing PO needs escalation.
    has_supply_issue = float(row["overdue_po_qty"]) > 0 or float(
        row["open_po_qty"]
    ) > 0
    if (
        risk_probability >= 0.4
        and inventory_cover < lead_time_weeks
        and has_supply_issue
    ):
        return (
            "EXPEDITE",
            calculated_qty,
            "EXPEDITE: high stockout risk with overdue/open PO before normal "
            "replenishment; expedite existing supply",
        )

    # D. Buy only when executable quantity exists and risk or cover justifies it.
    if calculated_qty > 0 and (
        risk_probability >= 0.25 or inventory_cover < target_cover_weeks
    ):
        return (
            "BUY",
            calculated_qty,
            "BUY: demand forecast exceeds inventory position after lead time "
            "and order constraints",
        )

    # E. REDUCE is reserved for severe over-supply situations where an open PO
    # and excessive near-term inbound supply make reduction/delay/cancel review
    # operationally meaningful. High cover alone falls through to HOLD.
    reduce_cover_threshold = max(26.0, target_cover_weeks * 4.0)
    if (
        inventory_cover >= reduce_cover_threshold
        and risk_probability < 0.10
        and float(row["raw_order_need"]) == 0
        and float(row["open_po_qty"]) > 0
        and float(row["inbound_qty_next_4w"]) >= float(row["demand_pred"]) * 4.0
        and int(row["promo_flag"]) == 0
    ):
        return (
            "REDUCE",
            0,
            "REDUCE: severe excess cover with open PO and excessive near-term "
            "inbound supply to review",
        )

    # F. All other situations hold the current position.
    return "HOLD", 0, "HOLD: inventory position sufficient"


def apply_action_rules(data: pd.DataFrame) -> pd.DataFrame:
    """Apply prioritized actions and overwrite quantity where the rule requires."""
    decisions = data.apply(decide_action, axis=1, result_type="expand")
    decisions.columns = [
        "recommended_action",
        "final_recommended_order_qty",
        "action_reason",
    ]
    data = data.copy()
    data["recommended_action"] = decisions["recommended_action"]
    data["recommended_order_qty"] = decisions[
        "final_recommended_order_qty"
    ].astype(int)
    data["action_reason"] = decisions["action_reason"]
    return data


def validate_decisions(data: pd.DataFrame) -> dict[str, int | float]:
    """Validate action labels, quantities, and decision-grain uniqueness."""
    duplicate_keys = int(data.duplicated(KEY_COLUMNS).sum())
    invalid_actions = sorted(set(data["recommended_action"]) - ACTION_LABELS)
    buy_zero = int(
        (
            data["recommended_action"].eq("BUY")
            & data["recommended_order_qty"].le(0)
        ).sum()
    )
    hold_reduce_positive = int(
        (
            data["recommended_action"].isin(["HOLD", "REDUCE"])
            & data["recommended_order_qty"].gt(0)
        ).sum()
    )
    negative_quantity = int(data["recommended_order_qty"].lt(0).sum())

    if duplicate_keys:
        raise ValueError(f"Decision output has {duplicate_keys} duplicate keys.")
    if invalid_actions:
        raise ValueError(f"Invalid action labels: {invalid_actions}")
    if buy_zero or hold_reduce_positive or negative_quantity:
        raise ValueError(
            "Decision quantity validation failed: "
            f"BUY with zero={buy_zero}, HOLD/REDUCE with positive="
            f"{hold_reduce_positive}, negative={negative_quantity}"
        )

    action_counts = data["recommended_action"].value_counts()
    return {
        "output_rows": len(data),
        "recommended_order_qty_sum": int(data["recommended_order_qty"].sum()),
        "BUY": int(action_counts.get("BUY", 0)),
        "EXPEDITE": int(action_counts.get("EXPEDITE", 0)),
        "REDUCE": int(action_counts.get("REDUCE", 0)),
        "HOLD": int(action_counts.get("HOLD", 0)),
        "buy_zero": buy_zero,
        "hold_reduce_positive": hold_reduce_positive,
        "negative_quantity": negative_quantity,
    }


def save_and_report(data: pd.DataFrame) -> None:
    """Save the final action file and print all requested validation metrics."""
    metrics = validate_decisions(data)
    output = data[OUTPUT_COLUMNS].copy()
    output["week_start_date"] = output["week_start_date"].dt.strftime("%Y-%m-%d")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    action_counts = output["recommended_action"].value_counts().to_dict()
    print(f"output row count: {metrics['output_rows']}")
    print(f"recommended_action counts: {action_counts}")
    print(
        "recommended_order_qty sum: "
        f"{metrics['recommended_order_qty_sum']}"
    )
    print(f"BUY row count: {metrics['BUY']}")
    print(f"EXPEDITE row count: {metrics['EXPEDITE']}")
    print(f"REDUCE row count: {metrics['REDUCE']}")
    print(f"HOLD row count: {metrics['HOLD']}")
    print(f"BUY with recommended_order_qty = 0: {metrics['buy_zero']}")
    print(
        "HOLD/REDUCE with recommended_order_qty > 0: "
        f"{metrics['hold_reduce_positive']}"
    )
    print(f"negative recommended_order_qty rows: {metrics['negative_quantity']}")
    print(f"Saved reorder decisions to: {OUTPUT_PATH}")


def main() -> None:
    """Join model predictions, calculate constraints, and create rule actions."""
    joined = load_and_join_inputs()
    with_quantities = apply_order_constraints(joined)
    decisions = apply_action_rules(with_quantities)
    save_and_report(decisions)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"[ERROR] {error}", file=sys.stderr, flush=True)
        raise SystemExit(1) from error
