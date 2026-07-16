from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path("data/mart/final_modeling_table.csv")
OUTPUT_DIR = Path("outputs/validation")

GROUP_KEYS = ["sku_id", "channel_id", "center_id"]
ROW_KEYS = GROUP_KEYS + ["week_start_date"]


def count_mismatch(actual: pd.Series, expected: pd.Series) -> int:
    actual = pd.to_numeric(actual, errors="coerce")
    expected = pd.to_numeric(expected, errors="coerce")

    return int(
        (~np.isclose(actual, expected, equal_nan=True)).sum()
    )


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

    group = data.groupby(
        GROUP_KEYS,
        sort=False,
        group_keys=False,
    )

    expected = pd.DataFrame(index=data.index)

    expected["sales_lag_1w"] = (
        group["observed_sales_qty"].shift(1)
    )

    expected["sales_lag_4w"] = (
        group["observed_sales_qty"].shift(4)
    )

    expected["sales_rolling_mean_4w"] = (
        group["observed_sales_qty"]
        .transform(
            lambda values:
                values.shift(1)
                .rolling(4, min_periods=1)
                .mean()
        )
    )

    expected["sales_rolling_std_4w"] = (
        group["observed_sales_qty"]
        .transform(
            lambda values:
                values.shift(1)
                .rolling(4, min_periods=1)
                .std(ddof=0)
        )
    )

    expected["demand_volatility_index"] = safe_divide(
        expected["sales_rolling_std_4w"],
        expected["sales_rolling_mean_4w"],
    ).to_numpy()

    expected[
        "stockout_adjusted_sales_rolling_mean_4w"
    ] = (
        group["stockout_adjusted_sales"]
        .transform(
            lambda values:
                values.shift(1)
                .rolling(4, min_periods=1)
                .mean()
        )
    )

    expected["stockout_days_last_4w"] = (
        group["stockout_flag"]
        .transform(
            lambda values:
                values.shift(1)
                .rolling(4, min_periods=1)
                .sum()
        )
        .fillna(0)
    )

    expected["in_stock_rate_4w"] = (
        1
        - expected["stockout_days_last_4w"] / 4
    ).clip(lower=0, upper=1)

    expected["sales_history_weeks"] = (
        group.cumcount()
    )

    prior_history_count = (
        expected["sales_history_weeks"]
        .clip(upper=4)
    )

    expected["dynamic_in_stock_rate_4w"] = np.where(
        prior_history_count.gt(0),
        (
            1
            - expected["stockout_days_last_4w"]
            / prior_history_count
        ),
        np.nan,
    )

    feature_columns = [
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

    summary_rows = [
        ("row_count", len(data)),
        (
            "group_count",
            data[GROUP_KEYS]
            .drop_duplicates()
            .shape[0],
        ),
    ]

    for feature in feature_columns:
        summary_rows.append(
            (
                f"{feature}_mismatch_rows",
                count_mismatch(
                    data[feature],
                    expected[feature],
                ),
            )
        )

    early_history_mask = (
        prior_history_count.gt(0)
        & prior_history_count.lt(4)
    )

    fixed_vs_dynamic_difference = (
        ~np.isclose(
            data.loc[
                early_history_mask,
                "in_stock_rate_4w",
            ],
            expected.loc[
                early_history_mask,
                "dynamic_in_stock_rate_4w",
            ],
            equal_nan=True,
        )
    )

    summary_rows.extend(
        [
            (
                "first_week_rows",
                int(
                    expected["sales_history_weeks"]
                    .eq(0)
                    .sum()
                ),
            ),
            (
                "lag_1w_null_rows",
                int(data["sales_lag_1w"].isna().sum()),
            ),
            (
                "lag_4w_null_rows",
                int(data["sales_lag_4w"].isna().sum()),
            ),
            (
                "early_history_rows_1_to_3_weeks",
                int(early_history_mask.sum()),
            ),
            (
                "fixed_vs_dynamic_in_stock_rate_different_rows",
                int(fixed_vs_dynamic_difference.sum()),
            ),
            (
                "max_stockout_days_last_4w",
                float(
                    data["stockout_days_last_4w"].max()
                ),
            ),
        ]
    )

    summary = pd.DataFrame(
        summary_rows,
        columns=["metric", "value"],
    )

    sample_columns = (
        ROW_KEYS
        + [
            "observed_sales_qty",
            "stockout_flag",
        ]
        + feature_columns
    )

    samples = data.loc[
        data["sales_history_weeks"].le(5),
        sample_columns,
    ].head(30)

    summary.to_csv(
        OUTPUT_DIR
        / "10_lag_rolling_feature_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    samples.to_csv(
        OUTPUT_DIR
        / "10_lag_rolling_feature_samples.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n[LAG / ROLLING FEATURE SUMMARY]")
    print(summary.to_string(index=False))

    print("\n[LAG / ROLLING FEATURE SAMPLE]")
    print(samples.to_string(index=False))


if __name__ == "__main__":
    main()
