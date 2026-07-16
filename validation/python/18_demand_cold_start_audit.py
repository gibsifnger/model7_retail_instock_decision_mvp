import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = REPO_ROOT / "data" / "mart" / "final_modeling_table.csv"
MODEL_PATH = (
    REPO_ROOT
    / "src"
    / "models"
    / "train_demand_forecast_model.py"
)
OUTPUT_DIR = REPO_ROOT / "outputs" / "validation"

TARGET_COLUMN = "target_demand_next_1w"
TEST_WEEKS = 8
HOLDOUT_SKU_COUNT = 8
RANDOM_SEED = 42

CLEAN_HISTORY_FEATURES = [
    "sales_lag_1w",
    "sales_lag_4w",
    "sales_rolling_mean_4w",
    "sales_rolling_std_4w",
    "demand_volatility_index",
    "sales_history_weeks",
]

IDENTITY_FEATURES = [
    "sku_id",
    "channel_id",
    "center_id",
]

NON_SKU_IDENTITY_FEATURES = [
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
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median"
                            ),
                        )
                    ]
                ),
                numeric_features,
            )
        )

    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
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
                ),
                categorical_features,
            )
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
        random_state=RANDOM_SEED,
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
    columns = numeric_features + categorical_features
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


def calculate_metrics(
    actual: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)

    error = prediction - actual
    absolute_error = np.abs(error)
    denominator = actual.sum()

    return {
        "mae": float(absolute_error.mean()),
        "rmse": float(
            np.sqrt(np.mean(np.square(error)))
        ),
        "wape": (
            float(absolute_error.sum() / denominator)
            if denominator > 0
            else np.nan
        ),
        "bias": (
            float(error.sum() / denominator)
            if denominator > 0
            else np.nan
        ),
    }


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


def main() -> None:
    data = pd.read_csv(
        INPUT_PATH,
        parse_dates=["week_start_date"],
    )

    data[TARGET_COLUMN] = pd.to_numeric(
        data[TARGET_COLUMN],
        errors="coerce",
    )

    data = data.loc[
        data[TARGET_COLUMN].notna()
    ].copy()

    all_skus = sorted(
        data["sku_id"].dropna().unique()
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    holdout_skus = sorted(
        rng.choice(
            all_skus,
            size=HOLDOUT_SKU_COUNT,
            replace=False,
        ).tolist()
    )

    weeks = sorted(
        data["week_start_date"].unique()
    )

    test_weeks = weeks[-TEST_WEEKS:]
    test_start = pd.Timestamp(test_weeks[0])

    data["label_horizon_end"] = (
        data["week_start_date"]
        + pd.Timedelta(weeks=1)
    )

    full_train = data.loc[
        data["label_horizon_end"] < test_start
    ].copy()

    unseen_train = full_train.loc[
        ~full_train["sku_id"].isin(
            holdout_skus
        )
    ].copy()

    holdout_test = data.loc[
        data["week_start_date"].isin(
            test_weeks
        )
        & data["sku_id"].isin(
            holdout_skus
        )
    ].copy()

    actual = holdout_test[
        TARGET_COLUMN
    ].to_numpy(dtype=float)

    known_prediction = fit_and_predict(
        full_train,
        holdout_test,
        CLEAN_HISTORY_FEATURES,
        IDENTITY_FEATURES,
    )

    unseen_with_history_prediction = fit_and_predict(
        unseen_train,
        holdout_test,
        CLEAN_HISTORY_FEATURES,
        NON_SKU_IDENTITY_FEATURES,
    )

    true_cold_start_prediction = fit_and_predict(
        unseen_train,
        holdout_test,
        COLD_START_NUMERIC_FEATURES,
        COLD_START_CATEGORICAL_FEATURES,
    )

    history_baseline = (
        pd.to_numeric(
            holdout_test[
                "sales_rolling_mean_4w"
            ],
            errors="coerce",
        )
        .fillna(
            full_train[TARGET_COLUMN].median()
        )
        .clip(lower=0)
        .to_numpy(dtype=float)
    )

    variants = [
        (
            "history_baseline_reference",
            history_baseline,
            len(full_train),
            "과거 4주 평균이 존재하는 참고 기준",
        ),
        (
            "known_sku_history_identity",
            known_prediction,
            len(full_train),
            "해당 SKU의 학습이력과 SKU ID 사용",
        ),
        (
            "unseen_sku_with_history",
            unseen_with_history_prediction,
            len(unseen_train),
            "학습에서 제외된 SKU지만 자체 판매이력 사용",
        ),
        (
            "true_cold_start_context_only",
            true_cold_start_prediction,
            len(unseen_train),
            "SKU 판매이력 없이 상품·채널·센터 정보만 사용",
        ),
    ]

    rows = []

    for (
        variant,
        prediction,
        train_rows,
        description,
    ) in variants:
        rows.append(
            {
                "variant": variant,
                "description": description,
                "train_rows": train_rows,
                "test_rows": len(
                    holdout_test
                ),
                "holdout_sku_count":
                    len(holdout_skus),
                **calculate_metrics(
                    actual,
                    prediction,
                ),
            }
        )

    summary = pd.DataFrame(rows)

    known_wape = float(
        summary.loc[
            summary["variant"].eq(
                "known_sku_history_identity"
            ),
            "wape",
        ].iloc[0]
    )

    summary[
        "wape_change_vs_known_sku"
    ] = (
        summary["wape"]
        - known_wape
    )

    holdout_table = pd.DataFrame(
        {
            "holdout_sku_id":
                holdout_skus
        }
    )

    predictions = holdout_test[
        [
            "sku_id",
            "channel_id",
            "center_id",
            "week_start_date",
            TARGET_COLUMN,
        ]
    ].copy()

    predictions[
        "history_baseline_prediction"
    ] = history_baseline

    predictions[
        "known_sku_prediction"
    ] = known_prediction

    predictions[
        "unseen_sku_with_history_prediction"
    ] = unseen_with_history_prediction

    predictions[
        "true_cold_start_prediction"
    ] = true_cold_start_prediction

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        OUTPUT_DIR
        / "18_demand_cold_start_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    holdout_table.to_csv(
        OUTPUT_DIR
        / "18_demand_cold_start_holdout_skus.csv",
        index=False,
        encoding="utf-8-sig",
    )

    predictions.to_csv(
        OUTPUT_DIR
        / "18_demand_cold_start_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n[DEMAND COLD-START SUMMARY]"
    )
    print(
        summary.to_string(index=False)
    )

    print(
        "\n[HOLDOUT SKU LIST]"
    )
    print(
        holdout_table.to_string(index=False)
    )


if __name__ == "__main__":
    main()
