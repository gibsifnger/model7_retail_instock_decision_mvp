from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path("data/mart/final_modeling_table.csv")
OUTPUT_DIR = Path("outputs/validation")
TEST_WEEKS = 8


def main() -> None:
    data = pd.read_csv(
        INPUT_PATH,
        parse_dates=["week_start_date"],
    )

    data = data.loc[
        data["target_demand_next_1w"].notna()
    ].copy()

    weeks = sorted(data["week_start_date"].unique())
    test_weeks = weeks[-TEST_WEEKS:]
    test = data.loc[
        data["week_start_date"].isin(test_weeks)
    ].copy()

    train = data.loc[
        data["week_start_date"] < test_weeks[0]
    ].copy()

    fallback = float(
        train["target_demand_next_1w"].median()
    )

    test["baseline_source_missing"] = (
        test["sales_rolling_mean_4w"].isna()
    )

    test["baseline_prediction"] = (
        pd.to_numeric(
            test["sales_rolling_mean_4w"],
            errors="coerce",
        )
        .fillna(fallback)
        .clip(lower=0)
    )

    test["baseline_error"] = (
        test["baseline_prediction"]
        - test["target_demand_next_1w"]
    )

    test["baseline_abs_error"] = (
        test["baseline_error"].abs()
    )

    summary = pd.DataFrame(
        [
            ("test_rows", len(test)),
            (
                "baseline_fallback_rows",
                int(
                    test["baseline_source_missing"]
                    .sum()
                ),
            ),
            (
                "baseline_fallback_rate",
                float(
                    test["baseline_source_missing"]
                    .mean()
                ),
            ),
            (
                "baseline_prediction_sum",
                test["baseline_prediction"].sum(),
            ),
            (
                "target_sum",
                test["target_demand_next_1w"].sum(),
            ),
            (
                "baseline_bias_sum",
                test["baseline_error"].sum(),
            ),
            (
                "baseline_underforecast_rows",
                int(
                    test["baseline_error"].lt(0).sum()
                ),
            ),
            (
                "baseline_overforecast_rows",
                int(
                    test["baseline_error"].gt(0).sum()
                ),
            ),
        ],
        columns=["metric", "value"],
    )

    segment = (
        test.groupby(
            ["promo_flag", "stockout_flag"],
            as_index=False,
        )
        .agg(
            rows=("sku_id", "size"),
            actual_sum=(
                "target_demand_next_1w",
                "sum",
            ),
            prediction_sum=(
                "baseline_prediction",
                "sum",
            ),
            abs_error_sum=(
                "baseline_abs_error",
                "sum",
            ),
            bias_sum=(
                "baseline_error",
                "sum",
            ),
        )
    )

    segment["wape"] = (
        segment["abs_error_sum"]
        / segment["actual_sum"].replace(0, np.nan)
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        OUTPUT_DIR
        / "14_demand_baseline_audit_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    segment.to_csv(
        OUTPUT_DIR
        / "14_demand_baseline_segment.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n[DEMAND BASELINE SUMMARY]")
    print(summary.to_string(index=False))

    print("\n[DEMAND BASELINE SEGMENT]")
    print(segment.to_string(index=False))


if __name__ == "__main__":
    main()
