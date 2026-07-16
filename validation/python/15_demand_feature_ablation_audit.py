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
MODEL_FILE = REPO_ROOT / "src" / "models" / "train_demand_forecast_model.py"
OUTPUT_DIR = REPO_ROOT / "outputs" / "validation"

TEST_WEEKS = 8
TARGET_COLUMN = "target_demand_next_1w"

LEAKY_CURRENT_WEEK_FEATURES = [
    "observed_sales_qty",
    "demand_gap_qty",
    "sales_censored_flag",
    "stockout_adjusted_sales",
]


def load_original_module():
    spec = importlib.util.spec_from_file_location(
        "train_demand_forecast_model",
        MODEL_FILE,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {MODEL_FILE}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            )
        ]
    )

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

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    model = HistGradientBoostingRegressor(
        learning_rate=0.06,
        max_iter=250,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42,
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
    features = data[
        numeric_features + categorical_features
    ].copy()

    for column in numeric_features:
        features[column] = pd.to_numeric(
            features[column],
            errors="coerce",
        )

    for column in categorical_features:
        features[column] = (
            features[column]
            .fillna("UNKNOWN")
            .astype(str)
        )

    return features


def calculate_metrics(
    actual: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)

    error = prediction - actual
    absolute_error = np.abs(error)

    actual_sum = actual.sum()

    mae = float(absolute_error.mean())
    rmse = float(np.sqrt(np.mean(np.square(error))))

    if actual_sum == 0:
        wape = np.nan
        bias = np.nan
        forecast_accuracy = np.nan
    else:
        wape = float(absolute_error.sum() / actual_sum)
        bias = float(error.sum() / actual_sum)
        forecast_accuracy = float(1 - wape)

    return {
        "mae": mae,
        "rmse": rmse,
        "wape": wape,
        "bias": bias,
        "wape_based_accuracy": forecast_accuracy,
    }


def fit_and_predict(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> np.ndarray:
    train_features = prepare_features(
        train_data,
        numeric_features,
        categorical_features,
    )

    test_features = prepare_features(
        test_data,
        numeric_features,
        categorical_features,
    )

    pipeline = build_pipeline(
        numeric_features,
        categorical_features,
    )

    pipeline.fit(
        train_features,
        train_data[TARGET_COLUMN],
    )

    prediction = pipeline.predict(test_features)

    return np.clip(
        prediction,
        a_min=0,
        a_max=None,
    )


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    if not MODEL_FILE.exists():
        raise FileNotFoundError(MODEL_FILE)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    original_module = load_original_module()

    full_numeric_features = list(
        original_module.NUMERIC_FEATURES
    )

    categorical_features = list(
        original_module.CATEGORICAL_FEATURES
    )

    safe_numeric_features = [
        feature
        for feature in full_numeric_features
        if feature not in LEAKY_CURRENT_WEEK_FEATURES
    ]

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

    unique_weeks = sorted(
        data["week_start_date"].unique()
    )

    test_weeks = unique_weeks[-TEST_WEEKS:]
    test_start = pd.Timestamp(test_weeks[0])

    test_data = data.loc[
        data["week_start_date"].isin(test_weeks)
    ].copy()

    # 기존 분할
    original_train = data.loc[
        data["week_start_date"] < test_start
    ].copy()

    # 1주 Demand Horizon이 Test 기간에 닿지 않도록 Purge
    data["label_horizon_end"] = (
        data["week_start_date"]
        + pd.Timedelta(weeks=1)
    )

    purged_train = data.loc[
        data["label_horizon_end"] < test_start
    ].copy()

    actual = test_data[TARGET_COLUMN].to_numpy(dtype=float)

    # 기존 전체 Feature + 기존 Split
    full_original_prediction = fit_and_predict(
        original_train,
        test_data,
        full_numeric_features,
        categorical_features,
    )

    # 전체 Feature + Purged Split
    full_purged_prediction = fit_and_predict(
        purged_train,
        test_data,
        full_numeric_features,
        categorical_features,
    )

    # 현재 주 실적 Feature 제거 + Purged Split
    safe_purged_prediction = fit_and_predict(
        purged_train,
        test_data,
        safe_numeric_features,
        categorical_features,
    )

    # 동일 Test에 대한 4주 평균 Baseline
    fallback = float(
        purged_train[TARGET_COLUMN].median()
    )

    baseline_prediction = (
        pd.to_numeric(
            test_data["sales_rolling_mean_4w"],
            errors="coerce",
        )
        .fillna(fallback)
        .clip(lower=0)
        .to_numpy(dtype=float)
    )

    variants = [
        (
            "baseline_4w_mean",
            baseline_prediction,
            len(purged_train),
            1,
            "purged",
        ),
        (
            "full_original_split",
            full_original_prediction,
            len(original_train),
            len(full_numeric_features)
            + len(categorical_features),
            "original",
        ),
        (
            "full_purged",
            full_purged_prediction,
            len(purged_train),
            len(full_numeric_features)
            + len(categorical_features),
            "purged",
        ),
        (
            "safe_no_current_week_realized_purged",
            safe_purged_prediction,
            len(purged_train),
            len(safe_numeric_features)
            + len(categorical_features),
            "purged",
        ),
    ]

    summary_rows = []

    for (
        variant,
        prediction,
        train_rows,
        feature_count,
        split_type,
    ) in variants:
        metrics = calculate_metrics(
            actual,
            prediction,
        )

        summary_rows.append(
            {
                "variant": variant,
                "split_type": split_type,
                "train_rows": train_rows,
                "test_rows": len(test_data),
                "feature_count": feature_count,
                **metrics,
            }
        )

    summary = pd.DataFrame(summary_rows)

    baseline_wape = float(
        summary.loc[
            summary["variant"].eq("baseline_4w_mean"),
            "wape",
        ].iloc[0]
    )

    full_purged_wape = float(
        summary.loc[
            summary["variant"].eq("full_purged"),
            "wape",
        ].iloc[0]
    )

    summary["wape_improvement_vs_baseline"] = (
        baseline_wape - summary["wape"]
    )

    summary["wape_change_vs_full_purged"] = (
        summary["wape"] - full_purged_wape
    )

    predictions = test_data[
        [
            "sku_id",
            "channel_id",
            "center_id",
            "week_start_date",
            TARGET_COLUMN,
        ]
    ].copy()

    predictions["baseline_prediction"] = (
        baseline_prediction
    )

    predictions["full_original_prediction"] = (
        full_original_prediction
    )

    predictions["full_purged_prediction"] = (
        full_purged_prediction
    )

    predictions["safe_purged_prediction"] = (
        safe_purged_prediction
    )

    predictions["full_purged_abs_error"] = np.abs(
        full_purged_prediction - actual
    )

    predictions["safe_purged_abs_error"] = np.abs(
        safe_purged_prediction - actual
    )

    predictions["leakage_removal_error_change"] = (
        predictions["safe_purged_abs_error"]
        - predictions["full_purged_abs_error"]
    )

    removed_features = pd.DataFrame(
        {
            "removed_feature":
                LEAKY_CURRENT_WEEK_FEATURES,
            "reason": [
                "현재 주 실현 출고량",
                "현재 주 주문-출고 차이",
                "현재 주 부분출고 결과 포함",
                "현재 주 실현 주문수요",
            ],
        }
    )

    summary.to_csv(
        OUTPUT_DIR
        / "15_demand_feature_ablation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    predictions.to_csv(
        OUTPUT_DIR
        / "15_demand_feature_ablation_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    removed_features.to_csv(
        OUTPUT_DIR
        / "15_demand_removed_leakage_features.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n[DEMAND FEATURE ABLATION SUMMARY]")
    print(summary.to_string(index=False))

    print("\n[REMOVED CURRENT-WEEK FEATURES]")
    print(removed_features.to_string(index=False))


if __name__ == "__main__":
    main()
