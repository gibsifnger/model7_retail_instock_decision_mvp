from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    REPO_ROOT
    / "data"
    / "mart"
    / "final_modeling_table.csv"
)

OUTPUT_DIR = (
    REPO_ROOT
    / "outputs"
    / "validation"
)

TARGET = "target_stockout_risk_next_2w"

ROW_KEYS = [
    "sku_id",
    "channel_id",
    "center_id",
    "week_start_date",
]

EVENT_KEYS = [
    "sku_id",
    "center_id",
    "week_start_date",
]

CURRENT_WEEK_REALIZED = [
    "observed_sales_qty",
    "demand_gap_qty",
    "sales_censored_flag",
    "stockout_adjusted_sales",
]

CENTER_REPEATED_NUMERIC = [
    "available_qty",
    "reserved_qty",
    "inbound_qty_next_1w",
    "inbound_qty_next_2w",
    "inbound_qty_next_4w",
    "inventory_position_qty",
    "open_po_qty",
    "overdue_po_qty",
    "standard_lead_time_days",
    "vendor_avg_lead_time",
    "vendor_lead_time_std",
    "po_fill_rate",
    "on_time_delivery_rate",
    "moq_qty",
    "order_multiple",
    "min_order_amount",
    "unit_cost",
    "list_price",
    "product_age_weeks",
    "new_product_flag",
    "active_flag",
    "order_block_flag",
]

STATIC_CATEGORICAL = [
    "category_l1",
    "category_l2",
    "brand",
    "default_vendor_id",
    "vendor_country",
    "import_flag",
    "reliability_tier",
    "lead_time_profile",
    "fill_rate_profile",
]


def build_disagreement_table(
    data: pd.DataFrame,
    columns: list[str],
    feature_type: str,
) -> pd.DataFrame:
    rows = []

    for column in columns:
        if column not in data.columns:
            continue

        variation = (
            data.groupby(EVENT_KEYS)[column]
            .nunique(dropna=False)
        )

        rows.append(
            {
                "feature": column,
                "feature_type": feature_type,
                "events_with_disagreement": int(
                    variation.gt(1).sum()
                ),
                "event_count": len(variation),
                "disagreement_rate": float(
                    variation.gt(1).mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = pd.read_csv(
        INPUT_PATH,
        parse_dates=["week_start_date"],
    )

    duplicate_row_keys = int(
        data.duplicated(ROW_KEYS).sum()
    )

    data[TARGET] = pd.to_numeric(
        data[TARGET],
        errors="coerce",
    )

    numeric_columns = [
        column
        for column in CENTER_REPEATED_NUMERIC
        if column in data.columns
    ]

    categorical_columns = [
        column
        for column in STATIC_CATEGORICAL
        if column in data.columns
    ]

    numeric_disagreement = (
        build_disagreement_table(
            data,
            numeric_columns,
            "center_repeated_numeric",
        )
    )

    categorical_disagreement = (
        build_disagreement_table(
            data,
            categorical_columns,
            "static_categorical",
        )
    )

    disagreement = pd.concat(
        [
            numeric_disagreement,
            categorical_disagreement,
        ],
        ignore_index=True,
    )

    base_aggregation = {
        TARGET: "max",
        "channel_id": "nunique",
    }

    for column in CURRENT_WEEK_REALIZED:
        if column in data.columns:
            base_aggregation[column] = "sum"

    center = (
        data.groupby(
            EVENT_KEYS,
            as_index=False,
        )
        .agg(base_aggregation)
        .rename(
            columns={
                "channel_id": "channel_count",
                "observed_sales_qty":
                    "center_observed_sales_qty",
                "demand_gap_qty":
                    "center_demand_gap_qty",
                "sales_censored_flag":
                    "censored_channel_count",
                "stockout_adjusted_sales":
                    "center_adjusted_sales_qty",
            }
        )
    )

    if numeric_columns:
        numeric_center = (
            data.groupby(
                EVENT_KEYS,
                as_index=False,
            )[numeric_columns]
            .first()
        )

        center = center.merge(
            numeric_center,
            on=EVENT_KEYS,
            how="left",
            validate="one_to_one",
        )

    if categorical_columns:
        categorical_center = (
            data.groupby(
                EVENT_KEYS,
                as_index=False,
            )[categorical_columns]
            .first()
        )

        center = center.merge(
            categorical_center,
            on=EVENT_KEYS,
            how="left",
            validate="one_to_one",
        )

    promo_aggregation = {}

    if "promo_flag" in data.columns:
        promo_aggregation[
            "promo_any_flag"
        ] = (
            "promo_flag",
            "max",
        )

        promo_aggregation[
            "promo_channel_count"
        ] = (
            "promo_flag",
            "sum",
        )

    if "promo_depth" in data.columns:
        promo_aggregation[
            "promo_depth_max"
        ] = (
            "promo_depth",
            "max",
        )

    if "promo_days_in_week" in data.columns:
        promo_aggregation[
            "promo_days_in_week_max"
        ] = (
            "promo_days_in_week",
            "max",
        )

    if promo_aggregation:
        promo_center = (
            data.groupby(
                EVENT_KEYS,
                as_index=False,
            )
            .agg(**promo_aggregation)
        )

        center = center.merge(
            promo_center,
            on=EVENT_KEYS,
            how="left",
            validate="one_to_one",
        )

    center = center.sort_values(
        [
            "sku_id",
            "center_id",
            "week_start_date",
        ],
        kind="stable",
    ).reset_index(drop=True)

    group = center.groupby(
        [
            "sku_id",
            "center_id",
        ],
        sort=False,
    )

    center["center_sales_lag_1w"] = (
        group[
            "center_observed_sales_qty"
        ].shift(1)
    )

    center["center_sales_lag_4w"] = (
        group[
            "center_observed_sales_qty"
        ].shift(4)
    )

    center[
        "center_sales_rolling_mean_4w"
    ] = group[
        "center_observed_sales_qty"
    ].transform(
        lambda series:
            series.shift(1)
            .rolling(
                window=4,
                min_periods=1,
            )
            .mean()
    )

    center[
        "center_sales_rolling_std_4w"
    ] = group[
        "center_observed_sales_qty"
    ].transform(
        lambda series:
            series.shift(1)
            .rolling(
                window=4,
                min_periods=1,
            )
            .std(ddof=0)
    )

    center["sales_history_weeks"] = (
        group.cumcount()
    )

    rolling_mean = center[
        "center_sales_rolling_mean_4w"
    ]

    rolling_std = center[
        "center_sales_rolling_std_4w"
    ]

    center[
        "center_demand_volatility_index"
    ] = np.where(
        rolling_mean.gt(0),
        rolling_std / rolling_mean,
        0.0,
    )

    if "available_qty" in center.columns:
        center["current_stockout_flag"] = (
            center["available_qty"]
            .fillna(0)
            .le(0)
            .astype(int)
        )

        center[
            "stockout_snapshots_last_4w"
        ] = group[
            "current_stockout_flag"
        ].transform(
            lambda series:
                series.shift(1)
                .rolling(
                    window=4,
                    min_periods=1,
                )
                .sum()
        )

        observed_window = np.minimum(
            center["sales_history_weeks"],
            4,
        ).replace(0, np.nan)

        center[
            "in_stock_snapshot_rate_4w"
        ] = (
            1
            - (
                center[
                    "stockout_snapshots_last_4w"
                ]
                / observed_window
            )
        )

    if {
        "available_qty",
        "inbound_qty_next_2w",
    }.issubset(center.columns):
        center[
            "inventory_position_next_2w"
        ] = (
            center["available_qty"]
            + center[
                "inbound_qty_next_2w"
            ]
        )

        center[
            "inventory_cover_next_2w_basis"
        ] = np.where(
            rolling_mean.gt(0),
            center[
                "inventory_position_next_2w"
            ]
            / rolling_mean,
            np.nan,
        )

    duplicate_center_keys = int(
        center.duplicated(
            EVENT_KEYS
        ).sum()
    )

    channel_count_mismatch = int(
        center[
            "channel_count"
        ].ne(3).sum()
    )

    target_disagreement = (
        data.loc[
            data[TARGET].notna()
        ]
        .groupby(EVENT_KEYS)[TARGET]
        .nunique()
        .gt(1)
        .sum()
    )

    labeled_center = center.loc[
        center[TARGET].notna()
    ].copy()

    summary = pd.DataFrame(
        [
            {
                "metric":
                    "source_channel_rows_all",
                "value":
                    len(data),
            },
            {
                "metric":
                    "source_duplicate_row_keys",
                "value":
                    duplicate_row_keys,
            },
            {
                "metric":
                    "center_event_rows_all",
                "value":
                    len(center),
            },
            {
                "metric":
                    "row_to_center_ratio",
                "value":
                    len(data) / len(center),
            },
            {
                "metric":
                    "duplicate_center_event_keys",
                "value":
                    duplicate_center_keys,
            },
            {
                "metric":
                    "events_not_three_channels",
                "value":
                    channel_count_mismatch,
            },
            {
                "metric":
                    "labeled_center_event_rows",
                "value":
                    len(labeled_center),
            },
            {
                "metric":
                    "target_disagreement_events",
                "value":
                    int(target_disagreement),
            },
            {
                "metric":
                    "positive_center_targets",
                "value":
                    int(
                        labeled_center[
                            TARGET
                        ].sum()
                    ),
            },
            {
                "metric":
                    "center_sales_lag_1w_null_rows",
                "value":
                    int(
                        center[
                            "center_sales_lag_1w"
                        ].isna().sum()
                    ),
            },
            {
                "metric":
                    "center_sales_lag_4w_null_rows",
                "value":
                    int(
                        center[
                            "center_sales_lag_4w"
                        ].isna().sum()
                    ),
            },
            {
                "metric":
                    "center_rolling_mean_null_rows",
                "value":
                    int(
                        center[
                            "center_sales_rolling_mean_4w"
                        ].isna().sum()
                    ),
            },
            {
                "metric":
                    "current_week_realized_features_excluded",
                "value":
                    len(
                        CURRENT_WEEK_REALIZED
                    ),
            },
        ]
    )

    contract_rows = [
        {
            "source_feature":
                "observed_sales_qty",
            "center_feature":
                "center_sales_lag_and_rolling",
            "aggregation":
                "채널 합산 후 센터 시계열에서 Lag·Rolling 재계산",
            "model_usage":
                "LAGGED_ONLY",
            "reason":
                "현재 주 실적은 제외하고 과거값만 사용",
        },
        {
            "source_feature":
                "demand_gap_qty",
            "center_feature":
                "center_demand_gap_qty",
            "aggregation":
                "채널 합산",
            "model_usage":
                "EXCLUDED_CURRENT_FORM",
            "reason":
                "현재 주 주문·출고 결과이며 판단시점 이후 정보",
        },
        {
            "source_feature":
                "stockout_adjusted_sales",
            "center_feature":
                "center_adjusted_sales_qty",
            "aggregation":
                "채널 합산",
            "model_usage":
                "EXCLUDED_CURRENT_FORM",
            "reason":
                "현재 주 정보이며 수요정의도 수정 필요",
        },
        {
            "source_feature":
                "available/inbound/open_po",
            "center_feature":
                "동일 이름 센터 Feature",
            "aggregation":
                "채널 간 동일성 검증 후 1회 사용",
            "model_usage":
                "CENTER_LEVEL",
            "reason":
                "센터 공용 재고·공급정보",
        },
        {
            "source_feature":
                "promo_flag/promo_depth",
            "center_feature":
                "promo_any/count/max",
            "aggregation":
                "채널별 행사를 센터 수준으로 요약",
            "model_usage":
                "PROVISIONAL",
            "reason":
                "예측대상 주차와 확정시각 재설계 필요",
        },
        {
            "source_feature":
                "channel_id",
            "center_feature":
                "없음",
            "aggregation":
                "제거",
            "model_usage":
                "EXCLUDED",
            "reason":
                "결품판단 단위가 SKU×센터×주차",
        },
    ]

    aggregation_contract = pd.DataFrame(
        contract_rows
    )

    center.to_csv(
        OUTPUT_DIR
        / "23_stockout_center_grain_candidate.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        OUTPUT_DIR
        / "23_stockout_center_grain_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    disagreement.to_csv(
        OUTPUT_DIR
        / "23_stockout_center_field_disagreement.csv",
        index=False,
        encoding="utf-8-sig",
    )

    aggregation_contract.to_csv(
        OUTPUT_DIR
        / "23_stockout_center_aggregation_contract.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n[STOCKOUT CENTER GRAIN RECONSTRUCTION SUMMARY]"
    )
    print(
        summary.to_string(index=False)
    )

    print(
        "\n[CENTER-REPEATED FIELD DISAGREEMENT]"
    )
    print(
        disagreement.to_string(index=False)
    )

    print(
        "\n[CENTER AGGREGATION CONTRACT]"
    )
    print(
        aggregation_contract.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
