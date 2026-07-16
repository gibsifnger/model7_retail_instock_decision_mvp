from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path("data/mart/final_modeling_table.csv")
OUTPUT_DIR = Path("outputs/validation")

KEY_COLUMNS = [
    "sku_id",
    "channel_id",
    "center_id",
]


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(
        INPUT_PATH,
        parse_dates=["week_start_date"],
    ).sort_values(KEY_COLUMNS + ["week_start_date"])

    numeric_columns = [
        "ordered_qty_1w",
        "fulfilled_qty_1w",
        "cancelled_qty_1w",
        "observed_sales_qty",
        "demand_gap_qty",
        "stockout_adjusted_sales",
        "target_demand_next_1w",
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    # 현재 구현을 별도로 다시 계산
    data["current_formula_recalc"] = (
        data["fulfilled_qty_1w"]
        + (
            data["ordered_qty_1w"]
            - data["fulfilled_qty_1w"]
        ).clip(lower=0)
    )

    # 수요 정의 후보
    data["gross_order_demand"] = data["ordered_qty_1w"]

    data["net_order_demand"] = (
        data["ordered_qty_1w"]
        - data["cancelled_qty_1w"]
    ).clip(lower=0)

    data["unfulfilled_net_demand"] = (
        data["net_order_demand"]
        - data["fulfilled_qty_1w"]
    ).clip(lower=0)

    # 현재 Target과 Net 주문수요 Target 비교
    group = data.groupby(
        KEY_COLUMNS,
        sort=False,
        group_keys=False,
    )

    data["candidate_target_net_next_1w"] = (
        group["net_order_demand"].shift(-1)
    )

    current_formula_match = np.isclose(
        data["stockout_adjusted_sales"],
        data["current_formula_recalc"],
        equal_nan=True,
    )

    adjusted_equals_ordered = np.isclose(
        data["stockout_adjusted_sales"],
        data["ordered_qty_1w"],
        equal_nan=True,
    )

    target_mask = (
        data["target_demand_next_1w"].notna()
        & data["candidate_target_net_next_1w"].notna()
    )

    target_difference = (
        data.loc[target_mask, "target_demand_next_1w"]
        - data.loc[target_mask, "candidate_target_net_next_1w"]
    )

    net_demand_sum = data["net_order_demand"].sum()
    gross_net_difference = (
        data["gross_order_demand"].sum()
        - net_demand_sum
    )

    overstatement_rate = (
        gross_net_difference / net_demand_sum
        if net_demand_sum > 0
        else np.nan
    )

    summary = pd.DataFrame(
        [
            ("row_count", len(data)),
            ("ordered_qty_sum", data["ordered_qty_1w"].sum()),
            ("fulfilled_qty_sum", data["fulfilled_qty_1w"].sum()),
            ("cancelled_qty_sum", data["cancelled_qty_1w"].sum()),
            ("gross_order_demand_sum", data["gross_order_demand"].sum()),
            ("net_order_demand_sum", net_demand_sum),
            (
                "stockout_adjusted_sales_sum",
                data["stockout_adjusted_sales"].sum(),
            ),
            (
                "current_formula_mismatch_rows",
                int((~current_formula_match).sum()),
            ),
            (
                "adjusted_vs_ordered_mismatch_rows",
                int((~adjusted_equals_ordered).sum()),
            ),
            (
                "rows_with_cancelled_qty",
                int(data["cancelled_qty_1w"].gt(0).sum()),
            ),
            (
                "ordered_less_than_fulfilled_rows",
                int(
                    data["ordered_qty_1w"]
                    .lt(data["fulfilled_qty_1w"])
                    .sum()
                ),
            ),
            (
                "cancelled_greater_than_ordered_rows",
                int(
                    data["cancelled_qty_1w"]
                    .gt(data["ordered_qty_1w"])
                    .sum()
                ),
            ),
            (
                "fulfilled_plus_cancelled_exceeds_ordered_rows",
                int(
                    (
                        data["fulfilled_qty_1w"]
                        + data["cancelled_qty_1w"]
                    )
                    .gt(data["ordered_qty_1w"])
                    .sum()
                ),
            ),
            (
                "gross_minus_net_demand_sum",
                gross_net_difference,
            ),
            (
                "gross_vs_net_overstatement_rate",
                overstatement_rate,
            ),
            (
                "current_target_sum",
                data.loc[
                    target_mask,
                    "target_demand_next_1w",
                ].sum(),
            ),
            (
                "candidate_net_target_sum",
                data.loc[
                    target_mask,
                    "candidate_target_net_next_1w",
                ].sum(),
            ),
            (
                "current_minus_net_target_sum",
                target_difference.sum(),
            ),
            (
                "current_vs_net_target_different_rows",
                int((~np.isclose(target_difference, 0)).sum()),
            ),
        ],
        columns=["metric", "value"],
    )

    cancelled_samples = (
        data.loc[
            data["cancelled_qty_1w"].gt(0),
            [
                "sku_id",
                "channel_id",
                "center_id",
                "week_start_date",
                "ordered_qty_1w",
                "fulfilled_qty_1w",
                "cancelled_qty_1w",
                "stockout_adjusted_sales",
                "net_order_demand",
                "unfulfilled_net_demand",
            ],
        ]
        .sort_values(
            "cancelled_qty_1w",
            ascending=False,
        )
        .head(20)
    )

    summary.to_csv(
        OUTPUT_DIR / "07_demand_definition_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    cancelled_samples.to_csv(
        OUTPUT_DIR / "07_demand_definition_samples.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n[DEMAND DEFINITION SUMMARY]")
    print(summary.to_string(index=False))

    print("\n[CANCELLED ORDER SAMPLE]")
    print(cancelled_samples.to_string(index=False))


if __name__ == "__main__":
    main()
