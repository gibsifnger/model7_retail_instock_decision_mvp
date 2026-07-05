"""Build the final MVP modeling table from the weekly SQLite source mart.

This step creates features and labels only. It does not train ML models or run
reorder decision rules.
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DATABASE_PATH = REPO_ROOT / "data" / "database" / "retail_instock.db"
OUTPUT_PATH = REPO_ROOT / "data" / "mart" / "final_modeling_table.csv"
MART_TABLE = "mart_retail_instock_weekly"

KEY_COLUMNS = ["sku_id", "channel_id", "center_id", "week_start_date"]
REQUIRED_MART_COLUMNS = {
    *KEY_COLUMNS,
    "default_vendor_id",
    "launch_date",
    "discontinue_date",
    "category_l1",
    "category_l2",
    "brand",
    "unit_cost",
    "list_price",
    "moq_qty",
    "order_multiple",
    "min_order_amount",
    "active_flag",
    "standard_lead_time_days",
    "reliability_tier",
    "lead_time_profile",
    "fill_rate_profile",
    "ordered_qty_1w",
    "fulfilled_qty_1w",
    "partial_fulfillment_flag",
    "stockout_flag",
    "promo_flag",
    "promo_type",
    "promo_depth",
    "available_qty",
    "reserved_qty",
    "inbound_qty_next_1w",
    "inbound_qty_next_2w",
    "inbound_qty_next_4w",
    "open_po_qty",
    "overdue_po_qty",
    "manual_override_flag",
}


def load_source_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the weekly mart and PO/receipt history from SQLite."""
    if not DATABASE_PATH.is_file():
        raise FileNotFoundError(
            f"SQLite database not found: {DATABASE_PATH}\n"
            "Build the database and SQL mart before running this script."
        )

    with sqlite3.connect(DATABASE_PATH) as connection:
        table_exists = connection.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE type = 'table' AND name = ?
            """,
            (MART_TABLE,),
        ).fetchone()
        if table_exists is None:
            raise ValueError(
                f"Required SQLite table not found: {MART_TABLE}. "
                "Run src/database/run_sql_scripts.py first."
            )

        mart = pd.read_sql_query(f'SELECT * FROM "{MART_TABLE}"', connection)
        purchase_orders = pd.read_sql_query(
            "SELECT * FROM erp_purchase_orders", connection
        )
        receipts = pd.read_sql_query(
            "SELECT * FROM wms_goods_receipts", connection
        )

    missing_columns = sorted(REQUIRED_MART_COLUMNS.difference(mart.columns))
    if missing_columns:
        raise ValueError(
            "The weekly mart is missing required columns: "
            + ", ".join(missing_columns)
        )
    if mart.empty:
        raise ValueError(f"SQLite table is empty: {MART_TABLE}")

    return mart, purchase_orders, receipts


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide numeric Series while returning NaN for invalid denominators."""
    numerator_array = pd.to_numeric(numerator, errors="coerce").to_numpy(dtype=float)
    denominator_array = pd.to_numeric(denominator, errors="coerce").to_numpy(
        dtype=float
    )
    result = np.full(len(numerator_array), np.nan, dtype=float)
    valid = np.isfinite(denominator_array) & (denominator_array > 0)
    np.divide(numerator_array, denominator_array, out=result, where=valid)
    return pd.Series(result, index=numerator.index)


def add_sales_and_distortion_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create point-in-time-safe demand, lag, rolling, and distortion features."""
    data = data.copy()
    group = data.groupby(KEY_COLUMNS[:3], sort=False, group_keys=False)

    data["observed_sales_qty"] = data["fulfilled_qty_1w"]
    data["demand_gap_qty"] = (
        data["ordered_qty_1w"] - data["fulfilled_qty_1w"]
    ).clip(lower=0)
    data["sales_censored_flag"] = (
        data["partial_fulfillment_flag"].eq(1) | data["stockout_flag"].eq(1)
    ).astype(int)

    # MVP approximation for censored demand: unfulfilled ordered demand is
    # added back to observed sales. This intentionally remains a simple v1 rule.
    data["stockout_adjusted_sales"] = (
        data["observed_sales_qty"] + data["demand_gap_qty"]
    )

    data["sales_lag_1w"] = group["observed_sales_qty"].shift(1)
    data["sales_lag_4w"] = group["observed_sales_qty"].shift(4)
    data["sales_rolling_mean_4w"] = group["observed_sales_qty"].transform(
        lambda values: values.shift(1).rolling(4, min_periods=1).mean()
    )
    data["sales_rolling_std_4w"] = group["observed_sales_qty"].transform(
        lambda values: values.shift(1).rolling(4, min_periods=1).std(ddof=0)
    )
    data["demand_volatility_index"] = safe_divide(
        data["sales_rolling_std_4w"], data["sales_rolling_mean_4w"]
    )
    data["stockout_adjusted_sales_rolling_mean_4w"] = group[
        "stockout_adjusted_sales"
    ].transform(lambda values: values.shift(1).rolling(4, min_periods=1).mean())

    data["stockout_days_last_4w"] = group["stockout_flag"].transform(
        lambda values: values.shift(1).rolling(4, min_periods=1).sum()
    ).fillna(0)
    data["in_stock_rate_4w"] = (
        1.0 - data["stockout_days_last_4w"] / 4.0
    ).clip(lower=0.0, upper=1.0)
    data["sales_history_weeks"] = group.cumcount()

    return data


def add_historical_promo_uplift(data: pd.DataFrame) -> pd.DataFrame:
    """Estimate SKU promo uplift using completed prior weeks only."""
    promo_history = data[
        ["sku_id", "week_start_date", "promo_flag", "stockout_adjusted_sales"]
    ].copy()
    promo_history["promo_sales"] = promo_history[
        "stockout_adjusted_sales"
    ].where(promo_history["promo_flag"].eq(1), 0.0)
    promo_history["regular_sales"] = promo_history[
        "stockout_adjusted_sales"
    ].where(promo_history["promo_flag"].eq(0), 0.0)
    promo_history["promo_observation"] = promo_history["promo_flag"].eq(1).astype(int)
    promo_history["regular_observation"] = promo_history["promo_flag"].eq(0).astype(int)

    sku_week = (
        promo_history.groupby(["sku_id", "week_start_date"], as_index=False)
        .agg(
            promo_sales=("promo_sales", "sum"),
            regular_sales=("regular_sales", "sum"),
            promo_observation=("promo_observation", "sum"),
            regular_observation=("regular_observation", "sum"),
        )
        .sort_values(["sku_id", "week_start_date"])
    )
    sku_group = sku_week.groupby("sku_id", sort=False)

    # Shift after each cumulative sum so the current week's realized outcome
    # never contributes to its own historical uplift feature.
    for column in [
        "promo_sales",
        "regular_sales",
        "promo_observation",
        "regular_observation",
    ]:
        sku_week[f"past_{column}"] = sku_group[column].transform(
            lambda values: values.cumsum().shift(1)
        )

    past_promo_mean = safe_divide(
        sku_week["past_promo_sales"], sku_week["past_promo_observation"]
    )
    past_regular_mean = safe_divide(
        sku_week["past_regular_sales"], sku_week["past_regular_observation"]
    )
    sku_week["historical_promo_uplift"] = (
        safe_divide(past_promo_mean, past_regular_mean)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1.0)
        .clip(lower=0.25, upper=4.0)
    )

    return data.merge(
        sku_week[["sku_id", "week_start_date", "historical_promo_uplift"]],
        on=["sku_id", "week_start_date"],
        how="left",
        validate="many_to_one",
    )


def add_promotion_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add the simplified promotion fields available in the v1 mart."""
    data = add_historical_promo_uplift(data)
    # Total promotion duration is not carried by the SQL mart in v1.
    data["promo_duration_days"] = 0
    data["promo_days_in_week"] = np.where(data["promo_flag"].eq(1), 7, 0)
    data["promo_day_index"] = 0
    return data


def build_vendor_performance_history(
    data: pd.DataFrame,
    purchase_orders: pd.DataFrame,
    receipts: pd.DataFrame,
) -> pd.DataFrame:
    """Build weekly vendor metrics from PO lines completed before each cutoff.

    This implements option B from the design: metrics use actual linked PO and
    receipt history. Static vendor profile proxies are used only as cold-start
    fallbacks when no completed history is available before a weekly cutoff.
    """
    purchase_orders = purchase_orders.copy()
    receipts = receipts.copy()
    purchase_orders["po_created_date"] = pd.to_datetime(
        purchase_orders["po_created_date"], errors="coerce"
    )
    purchase_orders["promised_delivery_date"] = pd.to_datetime(
        purchase_orders["promised_delivery_date"], errors="coerce"
    )
    receipts["receipt_datetime"] = pd.to_datetime(
        receipts["receipt_datetime"], errors="coerce"
    )

    receipt_summary = receipts.groupby(
        ["po_id", "po_line_id"], as_index=False
    ).agg(
        accepted_qty=("accepted_qty", "sum"),
        completion_datetime=("receipt_datetime", "max"),
    )
    po_performance = purchase_orders.merge(
        receipt_summary,
        on=["po_id", "po_line_id"],
        how="left",
        validate="one_to_one",
    )
    completed = po_performance.loc[
        po_performance["confirmed_qty"].gt(0)
        & po_performance["accepted_qty"].ge(po_performance["confirmed_qty"])
        & po_performance["completion_datetime"].notna()
    ].copy()
    completed["actual_lead_time_days"] = (
        completed["completion_datetime"].dt.normalize()
        - completed["po_created_date"].dt.normalize()
    ).dt.days
    completed["po_fill_rate"] = (
        completed["accepted_qty"] / completed["ordered_qty"].replace(0, np.nan)
    ).clip(lower=0.0, upper=1.0)
    completed["on_time_delivery"] = (
        completed["completion_datetime"].dt.normalize()
        <= completed["promised_delivery_date"].dt.normalize()
    ).astype(int)

    vendor_attributes = data[
        [
            "default_vendor_id",
            "standard_lead_time_days",
            "reliability_tier",
            "lead_time_profile",
            "fill_rate_profile",
        ]
    ].drop_duplicates("default_vendor_id")
    week_starts = sorted(data["week_start_date"].drop_duplicates())
    lead_time_std_proxy = {
        "STABLE": 2.0,
        "MODERATE": 5.0,
        "VARIABLE": 9.0,
        "HIGH_VARIANCE": 15.0,
    }
    fill_rate_proxy = {"HIGH": 0.97, "MEDIUM": 0.88, "LOW": 0.75}
    on_time_proxy = {"HIGH": 0.94, "MEDIUM": 0.82, "LOW": 0.65}
    rows: list[dict[str, object]] = []

    for vendor in vendor_attributes.itertuples(index=False):
        vendor_history = completed.loc[
            completed["vendor_id"].eq(vendor.default_vendor_id)
        ].sort_values("completion_datetime")
        for week_start in week_starts:
            cutoff = week_start + pd.Timedelta(hours=8)
            history = vendor_history.loc[
                vendor_history["completion_datetime"].lt(cutoff)
            ]

            if history.empty:
                average_lead_time = float(vendor.standard_lead_time_days)
                lead_time_std = lead_time_std_proxy[str(vendor.lead_time_profile)]
                fill_rate = fill_rate_proxy[str(vendor.fill_rate_profile)]
                on_time_rate = on_time_proxy[str(vendor.reliability_tier)]
            else:
                average_lead_time = float(history["actual_lead_time_days"].mean())
                lead_time_std = (
                    float(history["actual_lead_time_days"].std(ddof=0))
                    if len(history) >= 2
                    else lead_time_std_proxy[str(vendor.lead_time_profile)]
                )
                fill_rate = float(history["po_fill_rate"].mean())
                on_time_rate = float(history["on_time_delivery"].mean())

            rows.append(
                {
                    "default_vendor_id": vendor.default_vendor_id,
                    "week_start_date": week_start,
                    "vendor_avg_lead_time": average_lead_time,
                    "vendor_lead_time_std": lead_time_std,
                    "po_fill_rate": fill_rate,
                    "on_time_delivery_rate": on_time_rate,
                }
            )

    return pd.DataFrame(rows)


def add_inventory_supply_and_product_features(
    data: pd.DataFrame,
    purchase_orders: pd.DataFrame,
    receipts: pd.DataFrame,
) -> pd.DataFrame:
    """Add inventory, vendor performance, constraints, and product lifecycle."""
    vendor_performance = build_vendor_performance_history(
        data, purchase_orders, receipts
    )
    data = data.merge(
        vendor_performance,
        on=["default_vendor_id", "week_start_date"],
        how="left",
        validate="many_to_one",
    )

    data["inventory_position_qty"] = (
        data["available_qty"] + data["inbound_qty_next_4w"]
    )
    data["inventory_cover_weeks"] = safe_divide(
        data["inventory_position_qty"], data["sales_rolling_mean_4w"]
    )
    lead_time_weeks = np.maximum(
        pd.to_numeric(data["standard_lead_time_days"], errors="coerce"), 1.0
    ) / 7.0
    data["safety_stock_qty"] = data["sales_rolling_std_4w"] * np.sqrt(
        lead_time_weeks
    )

    discontinue_date = pd.to_datetime(data["discontinue_date"], errors="coerce")
    data["order_block_flag"] = (
        data["active_flag"].eq(0)
        | (discontinue_date.notna() & discontinue_date.le(data["week_start_date"]))
    ).astype(int)

    launch_date = pd.to_datetime(data["launch_date"], errors="coerce")
    data["product_age_weeks"] = (
        (data["week_start_date"] - launch_date).dt.days // 7
    ).clip(lower=0)
    data["new_product_flag"] = data["product_age_weeks"].le(12).astype(int)

    return data


def add_targets(data: pd.DataFrame) -> pd.DataFrame:
    """Create future labels without using them in any feature calculation."""
    group = data.groupby(KEY_COLUMNS[:3], sort=False, group_keys=False)
    data["target_demand_next_1w"] = group["stockout_adjusted_sales"].shift(-1)

    stockout_next_1w = group["stockout_flag"].shift(-1)
    stockout_next_2w = group["stockout_flag"].shift(-2)
    data["target_stockout_risk_next_2w"] = np.maximum(
        stockout_next_1w, stockout_next_2w
    )
    return data


def validate_and_save(data: pd.DataFrame) -> None:
    """Validate decision-grain uniqueness, save CSV, and print quality metrics."""
    unique_key_count = data[KEY_COLUMNS].drop_duplicates().shape[0]
    duplicate_key_count = len(data) - unique_key_count
    if duplicate_key_count != 0:
        raise ValueError(
            f"Final modeling table has {duplicate_key_count} duplicate decision keys."
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output = data.copy()
    output["week_start_date"] = output["week_start_date"].dt.strftime("%Y-%m-%d")
    output.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"final_modeling_table row count: {len(output)}")
    print(f"unique key count: {unique_key_count}")
    print(f"duplicate key count: {duplicate_key_count}")
    print(
        "target_demand_next_1w non-null count: "
        f"{output['target_demand_next_1w'].notna().sum()}"
    )
    print(
        "target_stockout_risk_next_2w non-null count: "
        f"{output['target_stockout_risk_next_2w'].notna().sum()}"
    )
    print(
        "stockout_adjusted_sales sum: "
        f"{output['stockout_adjusted_sales'].sum():.2f}"
    )
    print(f"sales_lag_1w null count: {output['sales_lag_1w'].isna().sum()}")
    print(
        "inventory_cover_weeks null count: "
        f"{output['inventory_cover_weeks'].isna().sum()}"
    )
    print(f"Saved final modeling table to: {OUTPUT_PATH}")


def main() -> None:
    """Build all MVP v1 features and targets from the weekly source mart."""
    mart, purchase_orders, receipts = load_source_data()
    mart["week_start_date"] = pd.to_datetime(
        mart["week_start_date"], errors="raise"
    )
    mart = mart.sort_values(KEY_COLUMNS, kind="stable").reset_index(drop=True)

    final_table = add_sales_and_distortion_features(mart)
    final_table = add_promotion_features(final_table)
    final_table = add_inventory_supply_and_product_features(
        final_table, purchase_orders, receipts
    )
    final_table = add_targets(final_table)
    validate_and_save(final_table)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, OSError, ValueError, sqlite3.Error) as error:
        print(f"[ERROR] {error}", file=sys.stderr, flush=True)
        raise SystemExit(1) from error
