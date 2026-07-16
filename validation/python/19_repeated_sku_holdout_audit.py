from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


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

TARGET_COLUMN = "target_demand_next_1w"
TEST_WEEKS = 8
HOLDOUT_SKU_COUNT = 8
REPEAT_COUNT = 10
BASE_RANDOM_SEED = 42

CLEAN_HISTORY_FEATURES = [
    "sales_lag_1w",
    "sales_lag_4w",
    "sales_rolling_mean_4w",
    "sales_rolling_std_4w",
    "demand_volatility_index",
    "sales_history_weeks",
]

KNOWN_IDENTITY_FEATURES = [
    "sku_id",
    "channel_id",
    "center_id",
]

UNSEEN_IDENTITY_FEATURES = [
    "channel_id",
    "center_id",
]

COLD_START_NUMERIC_FEATURES = [
    "product_age_weeks",
    "new_product_flag",
    "unit_cost",
    "list_price",
]

COLD_START_CATEGORICAL_FEATURES = [
    "channel_id",
    "center_id",
    "category_l1",
    "category_l2",
    "brand",
]


def build_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    transformers = []

    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                    ),
                )
            ]
        )

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            )
        )

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="constant",
                        fill_value="UNKNOWN",
                    ),
                ),
                (
                    "one_hot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            )
        )

    if not transformers:
        raise ValueError(
            "At least one feature is required."
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    model = HistGradientBoostingRegressor(
        learning_rate=0.06,
        max_iter=250,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=BASE_RANDOM_SEED,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def prepare_features(
    data: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> pd.DataFrame:
    columns = (
        numeric_features
        + categorical_features
    )

    result = data[columns].copy()

    for column in numeric_features:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    for column in categorical_features:
        result[column] = (
            result[column]
            .fillna("UNKNOWN")
            .astype(str)
        )

    return result


def fit_and_predict(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> np.ndarray:
    pipeline = build_pipeline(
        numeric_features,
        categorical_features,
    )

    pipeline.fit(
        prepare_features(
            train_data,
            numeric_features,
            categorical_features,
        ),
        train_data[TARGET_COLUMN],
    )

    prediction = pipeline.predict(
        prepare_features(
            test_data,
            numeric_features,
            categorical_features,
        )
    )

    return np.clip(
        prediction,
        a_min=0,
        a_max=None,
    )


def calculate_metrics(
    actual: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    actual = np.asarray(
        actual,
        dtype=float,
    )

    prediction = np.asarray(
        prediction,
        dtype=float,
    )

    error = prediction - actual
    absolute_error = np.abs(error)
    denominator = actual.sum()

    return {
        "mae": float(
            absolute_error.mean()
        ),
        "rmse": float(
            np.sqrt(
                np.mean(
                    np.square(error)
                )
            )
        ),
        "wape": (
            float(
                absolute_error.sum()
                / denominator
            )
            if denominator > 0
            else np.nan
        ),
        "bias": (
            float(
                error.sum()
                / denominator
            )
            if denominator > 0
            else np.nan
        ),
    }


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            INPUT_PATH
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = pd.read_csv(
        INPUT_PATH,
        parse_dates=[
            "week_start_date",
        ],
    )

    data[TARGET_COLUMN] = pd.to_numeric(
        data[TARGET_COLUMN],
        errors="coerce",
    )

    data = data.loc[
        data[TARGET_COLUMN].notna()
    ].copy()

    all_skus = sorted(
        data["sku_id"]
        .dropna()
        .unique()
    )

    if len(all_skus) < HOLDOUT_SKU_COUNT:
        raise ValueError(
            "Not enough SKUs for holdout."
        )

    unique_weeks = sorted(
        data[
            "week_start_date"
        ].unique()
    )

    test_weeks = (
        unique_weeks[-TEST_WEEKS:]
    )

    test_start = pd.Timestamp(
        test_weeks[0]
    )

    data["label_horizon_end"] = (
        data["week_start_date"]
        + pd.Timedelta(weeks=1)
    )

    full_train = data.loc[
        data["label_horizon_end"]
        < test_start
    ].copy()

    full_test = data.loc[
        data["week_start_date"]
        .isin(test_weeks)
    ].copy()

    run_rows = []
    holdout_rows = []

    for repeat_index in range(
        REPEAT_COUNT
    ):
        seed = (
            BASE_RANDOM_SEED
            + repeat_index
        )

        rng = np.random.default_rng(
            seed
        )

        holdout_skus = sorted(
            rng.choice(
                all_skus,
                size=HOLDOUT_SKU_COUNT,
                replace=False,
            ).tolist()
        )

        unseen_train = full_train.loc[
            ~full_train["sku_id"].isin(
                holdout_skus
            )
        ].copy()

        holdout_test = full_test.loc[
            full_test["sku_id"].isin(
                holdout_skus
            )
        ].copy()

        if holdout_test.empty:
            raise ValueError(
                f"No test rows at repeat "
                f"{repeat_index + 1}."
            )

        actual = holdout_test[
            TARGET_COLUMN
        ].to_numpy(dtype=float)

        baseline_prediction = (
            pd.to_numeric(
                holdout_test[
                    "sales_rolling_mean_4w"
                ],
                errors="coerce",
            )
            .fillna(
                full_train[
                    TARGET_COLUMN
                ].median()
            )
            .clip(lower=0)
            .to_numpy(dtype=float)
        )

        known_prediction = (
            fit_and_predict(
                full_train,
                holdout_test,
                CLEAN_HISTORY_FEATURES,
                KNOWN_IDENTITY_FEATURES,
            )
        )

        unseen_history_prediction = (
            fit_and_predict(
                unseen_train,
                holdout_test,
                CLEAN_HISTORY_FEATURES,
                UNSEEN_IDENTITY_FEATURES,
            )
        )

        cold_start_prediction = (
            fit_and_predict(
                unseen_train,
                holdout_test,
                COLD_START_NUMERIC_FEATURES,
                COLD_START_CATEGORICAL_FEATURES,
            )
        )

        variants = [
            (
                "history_baseline_reference",
                baseline_prediction,
            ),
            (
                "known_sku_history_identity",
                known_prediction,
            ),
            (
                "unseen_sku_with_history",
                unseen_history_prediction,
            ),
            (
                "true_cold_start_context_only",
                cold_start_prediction,
            ),
        ]

        for (
            variant,
            prediction,
        ) in variants:
            run_rows.append(
                {
                    "repeat_number":
                        repeat_index + 1,
                    "random_seed": seed,
                    "variant": variant,
                    "holdout_skus":
                        " | ".join(
                            holdout_skus
                        ),
                    "holdout_sku_count":
                        len(holdout_skus),
                    "train_rows": (
                        len(full_train)
                        if variant in {
                            "history_baseline_reference",
                            "known_sku_history_identity",
                        }
                        else len(
                            unseen_train
                        )
                    ),
                    "test_rows":
                        len(holdout_test),
                    **calculate_metrics(
                        actual,
                        prediction,
                    ),
                }
            )

        for sku_id in holdout_skus:
            holdout_rows.append(
                {
                    "repeat_number":
                        repeat_index + 1,
                    "random_seed": seed,
                    "holdout_sku_id":
                        sku_id,
                }
            )

    runs = pd.DataFrame(
        run_rows
    )

    summary = (
        runs.groupby(
            "variant",
            as_index=False,
        )
        .agg(
            repeat_count=(
                "repeat_number",
                "nunique",
            ),
            mean_wape=(
                "wape",
                "mean",
            ),
            median_wape=(
                "wape",
                "median",
            ),
            std_wape=(
                "wape",
                "std",
            ),
            min_wape=(
                "wape",
                "min",
            ),
            max_wape=(
                "wape",
                "max",
            ),
            mean_bias=(
                "bias",
                "mean",
            ),
            std_bias=(
                "bias",
                "std",
            ),
            mean_mae=(
                "mae",
                "mean",
            ),
            mean_rmse=(
                "rmse",
                "mean",
            ),
        )
    )

    pivot_wape = runs.pivot(
        index="repeat_number",
        columns="variant",
        values="wape",
    )

    comparisons = pd.DataFrame(
        {
            "repeat_number":
                pivot_wape.index,
            "unseen_minus_known_wape":
                (
                    pivot_wape[
                        "unseen_sku_with_history"
                    ]
                    - pivot_wape[
                        "known_sku_history_identity"
                    ]
                ),
            "cold_start_minus_known_wape":
                (
                    pivot_wape[
                        "true_cold_start_context_only"
                    ]
                    - pivot_wape[
                        "known_sku_history_identity"
                    ]
                ),
            "unseen_improvement_vs_baseline":
                (
                    pivot_wape[
                        "history_baseline_reference"
                    ]
                    - pivot_wape[
                        "unseen_sku_with_history"
                    ]
                ),
        }
    ).reset_index(drop=True)

    comparison_summary = pd.DataFrame(
        [
            {
                "metric":
                    "mean_unseen_minus_known_wape",
                "value":
                    comparisons[
                        "unseen_minus_known_wape"
                    ].mean(),
            },
            {
                "metric":
                    "std_unseen_minus_known_wape",
                "value":
                    comparisons[
                        "unseen_minus_known_wape"
                    ].std(),
            },
            {
                "metric":
                    "mean_cold_start_minus_known_wape",
                "value":
                    comparisons[
                        "cold_start_minus_known_wape"
                    ].mean(),
            },
            {
                "metric":
                    "unseen_beats_baseline_repeat_count",
                "value":
                    int(
                        comparisons[
                            "unseen_improvement_vs_baseline"
                        ].gt(0).sum()
                    ),
            },
            {
                "metric":
                    "repeat_count",
                "value":
                    REPEAT_COUNT,
            },
        ]
    )

    holdout_table = pd.DataFrame(
        holdout_rows
    )

    sku_frequency = (
        holdout_table.groupby(
            "holdout_sku_id",
            as_index=False,
        )
        .agg(
            selected_repeat_count=(
                "repeat_number",
                "nunique",
            )
        )
        .sort_values(
            [
                "selected_repeat_count",
                "holdout_sku_id",
            ],
            ascending=[
                False,
                True,
            ],
        )
    )

    runs.to_csv(
        OUTPUT_DIR
        / "19_repeated_sku_holdout_runs.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        OUTPUT_DIR
        / "19_repeated_sku_holdout_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    comparisons.to_csv(
        OUTPUT_DIR
        / "19_repeated_sku_holdout_comparisons.csv",
        index=False,
        encoding="utf-8-sig",
    )

    comparison_summary.to_csv(
        OUTPUT_DIR
        / "19_repeated_sku_holdout_comparison_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    sku_frequency.to_csv(
        OUTPUT_DIR
        / "19_repeated_sku_holdout_sku_frequency.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n[REPEATED SKU HOLDOUT SUMMARY]"
    )
    print(
        summary.to_string(index=False)
    )

    print(
        "\n[REPEATED SKU HOLDOUT COMPARISON]"
    )
    print(
        comparison_summary.to_string(
            index=False
        )
    )

    print(
        "\n[PER-REPEAT COMPARISON]"
    )
    print(
        comparisons.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
