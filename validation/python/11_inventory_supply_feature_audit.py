from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path("data/mart/final_modeling_table.csv")
OUTPUT_DIR = Path("outputs/validation")

CENTER_KEYS = [
    "sku_id",
    "center_id",
    "week_start_date",
]

ROW_KEYS = [
    "sku_id",
    "channel_id",
    "center_id",
    "week_start_date",
]

REPEATED_CENTER_COLUMNS = [
    "available_qty",
    "reserved_qty",
    "inbound_qty_next_1w",
    "inbound_qty_next_2w",
    "inbound_qty_next_4w",
    "open_po_qty",
    "overdue_po_qty",
    "stockout_flag",
]


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    numerator = pd.to_numeric(
        numerator,
        errors="coerce",
    ).to_numpy(dtype=float)

    denominator = pd.to_numeric(
        denominator,
        errors="coerce",
    ).to_numpy(dtype=float)

    result = np.full(len(numerator), np.nan)

    valid = (
        np.isfinite(denominator)
        & (denominator > 0)
    )

    np.divide(
        numerator,
        denominator,
        out=result,
        where=valid,
    )

    return pd.Series(result)


def mismatch_count(
    actual: pd.Series,
    expected: pd.Series,
) -> int:
    return int(
        (
            ~np.isclose(
                pd.to_numeric(actual, errors="coerce"),
                pd.to_numeric(expected, errors="coerce"),
                equal_nan=True,
            )
        ).sum()
    )


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(
        INPUT_PATH,
        parse_dates=["week_start_date"],
    ).sort_values(
        ROW_KEYS,
        kind="stable",
    ).reset_index(drop=True)

    numeric_columns = [
        *REPEATED_CENTER_COLUMNS,
        "inventory_position_qty",
        "inventory_cover_weeks",
        "safety_stock_qty",
        "sales_rolling_mean_4w",
        "sales_rolling_std_4w",
        "observed_sales_qty",
        "standard_lead_time_days",
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    # ---------------------------------------------------------
    # 1. 현재 Feature 계산식 독립 재계산
    # ---------------------------------------------------------
    expected_inventory_position = (
        data["available_qty"]
        + data["inbound_qty_next_4w"]
    )

    expected_inventory_cover = safe_divide(
        expected_inventory_position,
        data["sales_rolling_mean_4w"],
    )

    lead_time_weeks = (
        data["standard_lead_time_days"]
        .clip(lower=1)
        / 7
    )

    expected_safety_stock = (
        data["sales_rolling_std_4w"]
        * np.sqrt(lead_time_weeks)
    )

    # ---------------------------------------------------------
    # 2. 입고예정·PO 수량 불변조건
    # ---------------------------------------------------------
    negative_supply_rows = int(
        (
            data[
                [
                    "available_qty",
                    "inbound_qty_next_1w",
                    "inbound_qty_next_2w",
                    "inbound_qty_next_4w",
                    "open_po_qty",
                    "overdue_po_qty",
                ]
            ]
            .lt(0)
            .any(axis=1)
        ).sum()
    )

    inbound_1_gt_2_rows = int(
        data["inbound_qty_next_1w"]
        .gt(data["inbound_qty_next_2w"])
        .sum()
    )

    inbound_2_gt_4_rows = int(
        data["inbound_qty_next_2w"]
        .gt(data["inbound_qty_next_4w"])
        .sum()
    )

    inbound_4_gt_open_po_rows = int(
        data["inbound_qty_next_4w"]
        .gt(data["open_po_qty"])
        .sum()
    )

    overdue_gt_open_po_rows = int(
        data["overdue_po_qty"]
        .gt(data["open_po_qty"])
        .sum()
    )

    # ---------------------------------------------------------
    # 3. 센터 값의 채널 반복 일관성
    # ---------------------------------------------------------
    center_repetition = (
        data.groupby(
            CENTER_KEYS,
            as_index=False,
        )
        .agg(
            channel_row_count=("channel_id", "count"),
            **{
                f"{column}_nunique": (
                    column,
                    lambda values:
                        values.nunique(dropna=False),
                )
                for column in REPEATED_CENTER_COLUMNS
            },
        )
    )

    unexpected_channel_row_count = int(
        center_repetition["channel_row_count"]
        .ne(3)
        .sum()
    )

    repeated_value_mismatch_keys = int(
        center_repetition[
            [
                f"{column}_nunique"
                for column in REPEATED_CENTER_COLUMNS
            ]
        ]
        .gt(1)
        .any(axis=1)
        .sum()
    )

    # ---------------------------------------------------------
    # 4. 채널 반복으로 합계가 몇 배가 되는지
    # ---------------------------------------------------------
    center_unique = (
        data.drop_duplicates(CENTER_KEYS)
    )

    channel_available_sum = data["available_qty"].sum()
    center_available_sum = center_unique["available_qty"].sum()

    channel_inventory_position_sum = (
        data["inventory_position_qty"].sum()
    )
    center_inventory_position_sum = (
        center_unique["inventory_position_qty"].sum()
    )

    available_repetition_ratio = (
        channel_available_sum / center_available_sum
        if center_available_sum > 0
        else np.nan
    )

    inventory_position_repetition_ratio = (
        channel_inventory_position_sum
        / center_inventory_position_sum
        if center_inventory_position_sum > 0
        else np.nan
    )

    # ---------------------------------------------------------
    # 5. 센터 총수요 기준 Cover 재계산
    # ---------------------------------------------------------
    center_week = (
        data.groupby(
            CENTER_KEYS,
            as_index=False,
        )
        .agg(
            available_qty=("available_qty", "first"),
            inbound_qty_next_4w=(
                "inbound_qty_next_4w",
                "first",
            ),
            current_inventory_position=(
                "inventory_position_qty",
                "first",
            ),
            center_sales_rolling_mean_4w=(
                "sales_rolling_mean_4w",
                lambda values: values.sum(min_count=1),
            ),
            channel_cover_min=(
                "inventory_cover_weeks",
                "min",
            ),
            channel_cover_max=(
                "inventory_cover_weeks",
                "max",
            ),
            channel_safety_stock_nunique=(
                "safety_stock_qty",
                lambda values:
                    values.nunique(dropna=True),
            ),
        )
    )

    center_week["expected_center_inventory_position"] = (
        center_week["available_qty"]
        + center_week["inbound_qty_next_4w"]
    )

    center_week["expected_center_cover_weeks"] = safe_divide(
        center_week["expected_center_inventory_position"],
        center_week["center_sales_rolling_mean_4w"],
    ).to_numpy()

    center_week["channel_cover_spread"] = (
        center_week["channel_cover_max"]
        - center_week["channel_cover_min"]
    )

    channel_cover_differs_within_center_keys = int(
        center_week["channel_cover_spread"]
        .fillna(0)
        .gt(1e-9)
        .sum()
    )

    channel_safety_stock_differs_within_center_keys = int(
        center_week["channel_safety_stock_nunique"]
        .gt(1)
        .sum()
    )

    valid_center_cover = (
        center_week["expected_center_cover_weeks"]
        .notna()
    )

    # ---------------------------------------------------------
    # 6. 센터 총판매 기준 안전재고 Proxy
    # ---------------------------------------------------------
    center_sales = (
        data.groupby(
            CENTER_KEYS,
            as_index=False,
        )
        .agg(
            center_observed_sales=(
                "observed_sales_qty",
                "sum",
            ),
            standard_lead_time_days=(
                "standard_lead_time_days",
                "first",
            ),
        )
        .sort_values(CENTER_KEYS)
    )

    center_group = center_sales.groupby(
        ["sku_id", "center_id"],
        sort=False,
        group_keys=False,
    )

    center_sales["center_sales_rolling_std_4w"] = (
        center_group["center_observed_sales"]
        .transform(
            lambda values:
                values.shift(1)
                .rolling(4, min_periods=1)
                .std(ddof=0)
        )
    )

    center_sales["center_safety_stock_proxy"] = (
        center_sales["center_sales_rolling_std_4w"]
        * np.sqrt(
            center_sales["standard_lead_time_days"]
            .clip(lower=1)
            / 7
        )
    )

    # ---------------------------------------------------------
    # 결과
    # ---------------------------------------------------------
    summary = pd.DataFrame(
        [
            ("row_count", len(data)),
            (
                "center_week_key_count",
                len(center_week),
            ),

            (
                "inventory_position_mismatch_rows",
                mismatch_count(
                    data["inventory_position_qty"],
                    expected_inventory_position,
                ),
            ),
            (
                "inventory_cover_mismatch_rows",
                mismatch_count(
                    data["inventory_cover_weeks"],
                    expected_inventory_cover,
                ),
            ),
            (
                "safety_stock_mismatch_rows",
                mismatch_count(
                    data["safety_stock_qty"],
                    expected_safety_stock,
                ),
            ),

            (
                "negative_supply_rows",
                negative_supply_rows,
            ),
            (
                "inbound_1_gt_2_rows",
                inbound_1_gt_2_rows,
            ),
            (
                "inbound_2_gt_4_rows",
                inbound_2_gt_4_rows,
            ),
            (
                "inbound_4_gt_open_po_rows",
                inbound_4_gt_open_po_rows,
            ),
            (
                "overdue_gt_open_po_rows",
                overdue_gt_open_po_rows,
            ),

            (
                "unexpected_channel_row_count",
                unexpected_channel_row_count,
            ),
            (
                "repeated_center_value_mismatch_keys",
                repeated_value_mismatch_keys,
            ),

            (
                "available_qty_channel_sum",
                channel_available_sum,
            ),
            (
                "available_qty_center_sum",
                center_available_sum,
            ),
            (
                "available_qty_repetition_ratio",
                available_repetition_ratio,
            ),

            (
                "inventory_position_channel_sum",
                channel_inventory_position_sum,
            ),
            (
                "inventory_position_center_sum",
                center_inventory_position_sum,
            ),
            (
                "inventory_position_repetition_ratio",
                inventory_position_repetition_ratio,
            ),

            (
                "channel_cover_differs_within_center_keys",
                channel_cover_differs_within_center_keys,
            ),
            (
                "channel_safety_stock_differs_within_center_keys",
                channel_safety_stock_differs_within_center_keys,
            ),
            (
                "median_center_cover_weeks",
                center_week.loc[
                    valid_center_cover,
                    "expected_center_cover_weeks",
                ].median(),
            ),
            (
                "median_channel_cover_min",
                center_week.loc[
                    valid_center_cover,
                    "channel_cover_min",
                ].median(),
            ),
            (
                "median_channel_cover_max",
                center_week.loc[
                    valid_center_cover,
                    "channel_cover_max",
                ].median(),
            ),
        ],
        columns=["metric", "value"],
    )

    sample = center_week.loc[
        center_week["channel_cover_spread"]
        .fillna(0)
        .gt(0),
        [
            "sku_id",
            "center_id",
            "week_start_date",
            "current_inventory_position",
            "center_sales_rolling_mean_4w",
            "expected_center_cover_weeks",
            "channel_cover_min",
            "channel_cover_max",
            "channel_cover_spread",
            "channel_safety_stock_nunique",
        ],
    ].sort_values(
        "channel_cover_spread",
        ascending=False,
    ).head(20)

    summary.to_csv(
        OUTPUT_DIR
        / "11_inventory_supply_feature_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    sample.to_csv(
        OUTPUT_DIR
        / "11_inventory_supply_feature_samples.csv",
        index=False,
        encoding="utf-8-sig",
    )

    center_sales.to_csv(
        OUTPUT_DIR
        / "11_center_safety_stock_proxy.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n[INVENTORY / SUPPLY FEATURE SUMMARY]")
    print(summary.to_string(index=False))

    print("\n[CENTER COVER SAMPLE]")
    print(sample.to_string(index=False))


if __name__ == "__main__":
    main()
