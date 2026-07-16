from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path("data/mart/final_modeling_table.csv")
OUTPUT_DIR = Path("outputs/validation")

GROUP_KEYS = [
    "sku_id",
    "channel_id",
    "center_id",
]

ROW_KEYS = GROUP_KEYS + ["week_start_date"]


def mismatch_count(
    actual: pd.Series,
    expected: pd.Series,
) -> int:
    actual_numeric = pd.to_numeric(actual, errors="coerce")
    expected_numeric = pd.to_numeric(expected, errors="coerce")

    matches = np.isclose(
        actual_numeric,
        expected_numeric,
        equal_nan=True,
    )
    return int((~matches).sum())


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(
        INPUT_PATH,
        parse_dates=["week_start_date"],
    )

    data = data.sort_values(
        ROW_KEYS,
        kind="stable",
    ).reset_index(drop=True)

    numeric_columns = [
        "ordered_qty_1w",
        "cancelled_qty_1w",
        "stockout_adjusted_sales",
        "stockout_flag",
        "target_demand_next_1w",
        "target_stockout_risk_next_2w",
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    # ---------------------------------------------------------
    # 1. 기본 Grain 및 시간 연속성 확인
    # ---------------------------------------------------------
    duplicate_key_count = (
        len(data)
        - data[ROW_KEYS].drop_duplicates().shape[0]
    )

    group_sizes = data.groupby(
        GROUP_KEYS,
        sort=False,
    ).size()

    week_difference_days = (
        data.groupby(
            GROUP_KEYS,
            sort=False,
        )["week_start_date"]
        .diff()
        .dt.days
    )

    non_seven_day_gap_rows = int(
        (
            week_difference_days.notna()
            & week_difference_days.ne(7)
        ).sum()
    )

    # ---------------------------------------------------------
    # 2. Self Join으로 실제 +1주 값을 독립적으로 생성
    # ---------------------------------------------------------
    future_1 = data[
        ROW_KEYS
        + [
            "ordered_qty_1w",
            "cancelled_qty_1w",
            "stockout_adjusted_sales",
            "stockout_flag",
        ]
    ].copy()

    future_1["_future_1_exists"] = 1
    future_1["future_1_week_start_date"] = (
        future_1["week_start_date"]
    )

    # 미래 행의 날짜를 1주 앞당겨 현재 행과 연결
    future_1["week_start_date"] = (
        future_1["week_start_date"]
        - pd.Timedelta(weeks=1)
    )

    future_1 = future_1.rename(
        columns={
            "ordered_qty_1w": "future_1_ordered_qty",
            "cancelled_qty_1w": "future_1_cancelled_qty",
            "stockout_adjusted_sales":
                "future_1_stockout_adjusted_sales",
            "stockout_flag": "future_1_stockout_flag",
        }
    )

    # ---------------------------------------------------------
    # 3. Self Join으로 실제 +2주 값을 독립적으로 생성
    # ---------------------------------------------------------
    future_2 = data[
        ROW_KEYS + ["stockout_flag"]
    ].copy()

    future_2["_future_2_exists"] = 1
    future_2["future_2_week_start_date"] = (
        future_2["week_start_date"]
    )

    future_2["week_start_date"] = (
        future_2["week_start_date"]
        - pd.Timedelta(weeks=2)
    )

    future_2 = future_2.rename(
        columns={
            "stockout_flag": "future_2_stockout_flag",
        }
    )

    audit = data.merge(
        future_1,
        on=ROW_KEYS,
        how="left",
        validate="one_to_one",
    )

    audit = audit.merge(
        future_2,
        on=ROW_KEYS,
        how="left",
        validate="one_to_one",
    )

    audit["_future_1_exists"] = (
        audit["_future_1_exists"]
        .fillna(0)
        .astype(int)
    )

    audit["_future_2_exists"] = (
        audit["_future_2_exists"]
        .fillna(0)
        .astype(int)
    )

    # ---------------------------------------------------------
    # 4. 현재 As-Is Demand Target 재계산
    # ---------------------------------------------------------
    audit["expected_current_demand_target"] = (
        audit["future_1_stockout_adjusted_sales"]
    )

    # 향후 수정 후보인 Net Demand Target
    audit["expected_net_demand_target"] = (
        audit["future_1_ordered_qty"]
        - audit["future_1_cancelled_qty"]
    ).clip(lower=0)

    # ---------------------------------------------------------
    # 5. Stockout Target 재계산
    # 현재 구현은 +1주와 +2주가 모두 있어야 Target 생성
    # ---------------------------------------------------------
    both_future_weeks_exist = (
        audit["_future_1_exists"].eq(1)
        & audit["_future_2_exists"].eq(1)
    )

    audit["expected_stockout_target"] = np.nan

    audit.loc[
        both_future_weeks_exist,
        "expected_stockout_target",
    ] = np.maximum(
        audit.loc[
            both_future_weeks_exist,
            "future_1_stockout_flag",
        ],
        audit.loc[
            both_future_weeks_exist,
            "future_2_stockout_flag",
        ],
    )

    # ---------------------------------------------------------
    # 6. Boundary 검사
    # ---------------------------------------------------------
    demand_missing_despite_future = int(
        (
            audit["_future_1_exists"].eq(1)
            & audit["target_demand_next_1w"].isna()
        ).sum()
    )

    demand_present_without_future = int(
        (
            audit["_future_1_exists"].eq(0)
            & audit["target_demand_next_1w"].notna()
        ).sum()
    )

    stockout_missing_despite_two_future_weeks = int(
        (
            both_future_weeks_exist
            & audit["target_stockout_risk_next_2w"].isna()
        ).sum()
    )

    stockout_present_without_two_future_weeks = int(
        (
            ~both_future_weeks_exist
            & audit["target_stockout_risk_next_2w"].notna()
        ).sum()
    )

    # ---------------------------------------------------------
    # 7. 센터 결품 Target의 채널별 반복 확인
    # ---------------------------------------------------------
    center_week_target = (
        data.groupby(
            [
                "sku_id",
                "center_id",
                "week_start_date",
            ],
            as_index=False,
        )
        .agg(
            channel_row_count=("channel_id", "count"),
            nonnull_target_count=(
                "target_stockout_risk_next_2w",
                "count",
            ),
            target_nunique=(
                "target_stockout_risk_next_2w",
                lambda values: values.nunique(dropna=True),
            ),
            min_target=(
                "target_stockout_risk_next_2w",
                "min",
            ),
            max_target=(
                "target_stockout_risk_next_2w",
                "max",
            ),
        )
    )

    stockout_channel_value_mismatch_keys = int(
        center_week_target["target_nunique"]
        .gt(1)
        .sum()
    )

    unexpected_channel_row_count = int(
        center_week_target["channel_row_count"]
        .ne(3)
        .sum()
    )

    positive_stockout_rows_at_channel_grain = int(
        data["target_stockout_risk_next_2w"]
        .eq(1)
        .sum()
    )

    positive_stockout_keys_at_center_grain = int(
        center_week_target["max_target"]
        .eq(1)
        .sum()
    )

    repetition_ratio = (
        positive_stockout_rows_at_channel_grain
        / positive_stockout_keys_at_center_grain
        if positive_stockout_keys_at_center_grain > 0
        else np.nan
    )

    # ---------------------------------------------------------
    # 8. 결과 요약
    # ---------------------------------------------------------
    valid_net_comparison = (
        audit["target_demand_next_1w"].notna()
        & audit["expected_net_demand_target"].notna()
    )

    current_vs_net_difference_rows = int(
        (
            ~np.isclose(
                audit.loc[
                    valid_net_comparison,
                    "target_demand_next_1w",
                ],
                audit.loc[
                    valid_net_comparison,
                    "expected_net_demand_target",
                ],
                equal_nan=True,
            )
        ).sum()
    )

    summary = pd.DataFrame(
        [
            ("row_count", len(data)),
            ("duplicate_key_count", duplicate_key_count),
            ("group_count", len(group_sizes)),
            ("min_weeks_per_group", group_sizes.min()),
            ("max_weeks_per_group", group_sizes.max()),
            ("non_seven_day_gap_rows", non_seven_day_gap_rows),

            (
                "expected_future_1_rows",
                int(audit["_future_1_exists"].sum()),
            ),
            (
                "actual_demand_target_nonnull_rows",
                int(
                    audit["target_demand_next_1w"]
                    .notna()
                    .sum()
                ),
            ),
            (
                "current_demand_target_mismatch_rows",
                mismatch_count(
                    audit["target_demand_next_1w"],
                    audit["expected_current_demand_target"],
                ),
            ),
            (
                "demand_missing_despite_future_rows",
                demand_missing_despite_future,
            ),
            (
                "demand_present_without_future_rows",
                demand_present_without_future,
            ),

            (
                "expected_two_future_week_rows",
                int(both_future_weeks_exist.sum()),
            ),
            (
                "actual_stockout_target_nonnull_rows",
                int(
                    audit["target_stockout_risk_next_2w"]
                    .notna()
                    .sum()
                ),
            ),
            (
                "stockout_target_mismatch_rows",
                mismatch_count(
                    audit["target_stockout_risk_next_2w"],
                    audit["expected_stockout_target"],
                ),
            ),
            (
                "stockout_missing_despite_two_future_weeks",
                stockout_missing_despite_two_future_weeks,
            ),
            (
                "stockout_present_without_two_future_weeks",
                stockout_present_without_two_future_weeks,
            ),

            (
                "current_vs_net_target_different_rows",
                current_vs_net_difference_rows,
            ),

            (
                "unexpected_channel_row_count",
                unexpected_channel_row_count,
            ),
            (
                "stockout_channel_value_mismatch_keys",
                stockout_channel_value_mismatch_keys,
            ),
            (
                "positive_stockout_rows_channel_grain",
                positive_stockout_rows_at_channel_grain,
            ),
            (
                "positive_stockout_keys_center_grain",
                positive_stockout_keys_at_center_grain,
            ),
            (
                "stockout_target_repetition_ratio",
                repetition_ratio,
            ),
        ],
        columns=["metric", "value"],
    )

    # 대표 시계열 샘플
    first_group = data[GROUP_KEYS].drop_duplicates().iloc[0]

    sample_mask = (
        audit["sku_id"].eq(first_group["sku_id"])
        & audit["channel_id"].eq(first_group["channel_id"])
        & audit["center_id"].eq(first_group["center_id"])
    )

    sample = audit.loc[
        sample_mask,
        [
            "sku_id",
            "channel_id",
            "center_id",
            "week_start_date",
            "stockout_adjusted_sales",
            "target_demand_next_1w",
            "future_1_week_start_date",
            "expected_current_demand_target",
            "stockout_flag",
            "future_1_stockout_flag",
            "future_2_stockout_flag",
            "target_stockout_risk_next_2w",
            "expected_stockout_target",
        ],
    ].tail(8)

    summary.to_csv(
        OUTPUT_DIR / "08_target_alignment_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    sample.to_csv(
        OUTPUT_DIR / "08_target_alignment_samples.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n[TARGET ALIGNMENT SUMMARY]")
    print(summary.to_string(index=False))

    print("\n[TARGET ALIGNMENT SAMPLE - LAST 8 WEEKS]")
    print(sample.to_string(index=False))


if __name__ == "__main__":
    main()
