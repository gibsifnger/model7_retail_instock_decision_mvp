import sqlite3
from pathlib import Path

import pandas as pd


DB_PATH = Path("data/database/retail_instock.db")
MART_PATH = Path("data/mart/final_modeling_table.csv")
OUTPUT_DIR = Path("outputs/validation")


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(DB_PATH)
    if not MART_PATH.exists():
        raise FileNotFoundError(MART_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        oms = pd.read_sql_query(
            """
            SELECT
                order_datetime,
                ordered_qty,
                fulfilled_qty,
                cancelled_qty
            FROM oms_sales_orders
            """,
            conn,
        )

        inventory = pd.read_sql_query(
            """
            SELECT snapshot_datetime
            FROM wms_inventory_snapshot
            """,
            conn,
        )

        overrides = pd.read_sql_query(
            """
            SELECT
                override_id,
                sku_id,
                channel_id,
                center_id,
                effective_from,
                effective_to,
                approval_status,
                created_at
            FROM manual_overrides
            """,
            conn,
        )

    mart = pd.read_csv(
        MART_PATH,
        parse_dates=["week_start_date"],
    )

    oms["order_datetime"] = pd.to_datetime(
        oms["order_datetime"],
        errors="raise",
    )
    inventory["snapshot_datetime"] = pd.to_datetime(
        inventory["snapshot_datetime"],
        errors="raise",
    )

    oms["week_start_date"] = (
        oms["order_datetime"]
        - pd.to_timedelta(
            oms["order_datetime"].dt.weekday,
            unit="D",
        )
    ).dt.normalize()

    oms["decision_cutoff"] = (
        oms["week_start_date"]
        + pd.Timedelta(hours=8)
    )

    oms["order_after_week_cutoff"] = (
        oms["order_datetime"]
        >= oms["decision_cutoff"]
    )

    inventory["week_start_date"] = (
        inventory["snapshot_datetime"]
        - pd.to_timedelta(
            inventory["snapshot_datetime"].dt.weekday,
            unit="D",
        )
    ).dt.normalize()

    inventory["expected_cutoff"] = (
        inventory["week_start_date"]
        + pd.Timedelta(hours=8)
    )

    inventory["snapshot_at_cutoff"] = (
        inventory["snapshot_datetime"]
        == inventory["expected_cutoff"]
    )

    current_realized_features = [
        "observed_sales_qty",
        "demand_gap_qty",
        "sales_censored_flag",
        "stockout_adjusted_sales",
    ]

    lagged_sales_features = [
        "sales_lag_1w",
        "sales_lag_4w",
        "sales_rolling_mean_4w",
        "sales_rolling_std_4w",
        "demand_volatility_index",
        "stockout_adjusted_sales_rolling_mean_4w",
        "stockout_days_last_4w",
        "in_stock_rate_4w",
        "sales_history_weeks",
    ]

    current_realized_nonzero_rows = {
        feature: int(
            pd.to_numeric(
                mart[feature],
                errors="coerce",
            )
            .fillna(0)
            .ne(0)
            .sum()
        )
        for feature in current_realized_features
    }

    partial_only_rows = int(
        (
            mart["partial_fulfillment_flag"].eq(1)
            & mart["stockout_flag"].eq(0)
        ).sum()
    )

    stockout_only_rows = int(
        (
            mart["partial_fulfillment_flag"].eq(0)
            & mart["stockout_flag"].eq(1)
        ).sum()
    )

    both_rows = int(
        (
            mart["partial_fulfillment_flag"].eq(1)
            & mart["stockout_flag"].eq(1)
        ).sum()
    )

    # Manual Override의 소급 가능성 점검
    overrides["created_at"] = pd.to_datetime(
        overrides["created_at"],
        errors="coerce",
    )
    overrides["effective_from"] = pd.to_datetime(
        overrides["effective_from"],
        errors="coerce",
    )

    overrides["effective_cutoff"] = (
        overrides["effective_from"]
        + pd.Timedelta(hours=8)
    )

    overrides["created_after_effective_cutoff"] = (
        overrides["created_at"]
        >= overrides["effective_cutoff"]
    )

    summary_rows = [
        ("oms_order_rows", len(oms)),
        (
            "oms_orders_after_monday_08_cutoff",
            int(oms["order_after_week_cutoff"].sum()),
        ),
        (
            "oms_orders_before_or_at_cutoff",
            int((~oms["order_after_week_cutoff"]).sum()),
        ),
        (
            "oms_after_cutoff_ratio",
            float(oms["order_after_week_cutoff"].mean()),
        ),
        ("inventory_snapshot_rows", len(inventory)),
        (
            "inventory_snapshots_exactly_monday_08",
            int(inventory["snapshot_at_cutoff"].sum()),
        ),
        (
            "inventory_snapshots_not_at_monday_08",
            int((~inventory["snapshot_at_cutoff"]).sum()),
        ),
        (
            "partial_fulfillment_only_rows",
            partial_only_rows,
        ),
        (
            "stockout_only_rows",
            stockout_only_rows,
        ),
        (
            "partial_and_stockout_rows",
            both_rows,
        ),
        (
            "approved_override_rows",
            int(
                overrides["approval_status"]
                .eq("APPROVED")
                .sum()
            ),
        ),
        (
            "override_created_after_effective_cutoff",
            int(
                (
                    overrides["approval_status"].eq("APPROVED")
                    & overrides[
                        "created_after_effective_cutoff"
                    ]
                ).sum()
            ),
        ),
    ]

    for feature, count in current_realized_nonzero_rows.items():
        summary_rows.append(
            (
                f"nonzero_rows__{feature}",
                count,
            )
        )

    summary = pd.DataFrame(
        summary_rows,
        columns=["metric", "value"],
    )

    feature_contract = pd.DataFrame(
        [
            (
                "observed_sales_qty",
                "현재 주 실현 판매",
                "월요일 08시 이후 발생",
                "LEAKAGE",
            ),
            (
                "demand_gap_qty",
                "현재 주 주문-출고 차이",
                "월요일 08시 이후 확정",
                "LEAKAGE",
            ),
            (
                "sales_censored_flag",
                "현재 주 부분출고 또는 결품",
                "부분출고는 월요일 08시 이후 확정",
                "LEAKAGE",
            ),
            (
                "stockout_adjusted_sales",
                "현재 주 주문수요",
                "월요일 08시 이후 발생",
                "LEAKAGE",
            ),
            *[
                (
                    feature,
                    "이전 완료 주 판매이력",
                    "shift(1) 이상 적용",
                    "SAFE",
                )
                for feature in lagged_sales_features
            ],
            (
                "available_qty",
                "월요일 08시 가용재고",
                "판단시점 Snapshot",
                "SAFE",
            ),
            (
                "inbound_qty_next_1w/2w/4w",
                "판단시점 기준 미래 입고예정",
                "월요일 08시 As-of",
                "SAFE",
            ),
            (
                "promo_flag/promo_depth",
                "현재 주 계획 프로모션",
                "등록·승인시각 없음",
                "CONDITIONAL",
            ),
            (
                "manual_override_flag",
                "승인된 수동 Override",
                "created_at Cutoff 필터 없음",
                "WARNING",
            ),
            (
                "historical_promo_uplift",
                "과거 완료 주 프로모션 성과",
                "shift 적용",
                "SAFE_WITH_DEFINITION_FIX",
            ),
            (
                "vendor_performance",
                "판단시점 이전 완료 PO 성과",
                "월요일 08시 Cutoff 적용",
                "SAFE",
            ),
        ],
        columns=[
            "feature",
            "business_meaning",
            "availability",
            "asof_status",
        ],
    )

    summary.to_csv(
        OUTPUT_DIR / "09_asof_leakage_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    feature_contract.to_csv(
        OUTPUT_DIR / "09_feature_asof_contract.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n[AS-OF / LEAKAGE SUMMARY]")
    print(summary.to_string(index=False))

    print("\n[FEATURE AS-OF CONTRACT]")
    print(feature_contract.to_string(index=False))


if __name__ == "__main__":
    main()
