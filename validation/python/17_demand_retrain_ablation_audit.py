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

CURRENT_WEEK_FEATURES = {
    "observed_sales_qty",
    "demand_gap_qty",
    "sales_censored_flag",
    "stockout_adjusted_sales",
}

IDENTITY_FEATURES = {
    "sku_id",
    "channel_id",
    "center_id",
}

INVENTORY_FEATURES = {
    "available_qty",
    "reserved_qty",
    "inbound_qty_next_1w",
    "inbound_qty_next_2w",
    "inbound_qty_next_4w",
    "inventory_position_qty",
    "inventory_cover_weeks",
    "safety_stock_qty",
    "open_po_qty",
    "overdue_po_qty",
}

SUSPECT_HISTORY_FEATURES = {
    "stockout_adjusted_sales_rolling_mean_4w",
    "stockout_days_last_4w",
    "in_stock_rate_4w",
}

CLEAN_HISTORY_FEATURES = {
    "sales_lag_1w",
    "sales_lag_4w",
    "sales_rolling_mean_4w",
    "sales_rolling_std_4w",
    "demand_volatility_index",
    "sales_history_weeks",
}


def load_model_module():
    spec = importlib.util.spec_from_file_location(
        "demand_model",
        MODEL_PATH,
    )

    if spec is None or spec.loader is None:
        raise ImportError(MODEL_PATH)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


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
                    SimpleImputer(strategy="median"),
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


def fit_variant(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> np.ndarray:
    pipeline = build_pipeline(
        numeric_features,
        categorical_features,
    )

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

    pipeline.fit(
        train_features,
        train_data[TARGET_COLUMN],
    )

    return np.clip(
        pipeline.predict(test_features),
        a_min=0,
        a_max=None,
    )


def main() -> None:
    module = load_model_module()

    full_numeric = [
        feature
        for feature in module.NUMERIC_FEATURES
        if feature not in CURRENT_WEEK_FEATURES
    ]

    full_categorical = list(
        module.CATEGORICAL_FEATURES
    )

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

    weeks = sorted(
        data["week_start_date"].unique()
    )

    test_weeks = weeks[-TEST_WEEKS:]
    test_start = pd.Timestamp(test_weeks[0])

    data["label_horizon_end"] = (
        data["week_start_date"]
        + pd.Timedelta(weeks=1)
    )

    train_data = data.loc[
        data["label_horizon_end"] < test_start
    ].copy()

    test_data = data.loc[
        data["week_start_date"].isin(test_weeks)
    ].copy()

    variants = {
        "full_safe": (
            full_numeric,
            full_categorical,
        ),
        "full_safe_minus_identity": (
            full_numeric,
            [
                feature
                for feature in full_categorical
                if feature not in IDENTITY_FEATURES
            ],
        ),
        "full_safe_minus_inventory": (
            [
                feature
                for feature in full_numeric
                if feature not in INVENTORY_FEATURES
            ],
            full_categorical,
        ),
        "full_safe_minus_suspect_history": (
            [
                feature
                for feature in full_numeric
                if feature
                not in SUSPECT_HISTORY_FEATURES
            ],
            full_categorical,
        ),
        "clean_history_only": (
            [
                feature
                for feature in full_numeric
                if feature in CLEAN_HISTORY_FEATURES
            ],
            [],
        ),
        "clean_history_plus_identity": (
            [
                feature
                for feature in full_numeric
                if feature in CLEAN_HISTORY_FEATURES
            ],
            [
                feature
                for feature in full_categorical
                if feature in IDENTITY_FEATURES
            ],
        ),
    }

    actual = test_data[
        TARGET_COLUMN
    ].to_numpy(dtype=float)

    baseline_prediction = (
        pd.to_numeric(
            test_data["sales_rolling_mean_4w"],
            errors="coerce",
        )
        .fillna(
            train_data[TARGET_COLUMN].median()
        )
        .clip(lower=0)
        .to_numpy(dtype=float)
    )

    rows = [
        {
            "variant": "baseline_4w_mean",
            "numeric_feature_count": 1,
            "categorical_feature_count": 0,
            **calculate_metrics(
                actual,
                baseline_prediction,
            ),
        }
    ]

    for (
        variant,
        (
            numeric_features,
            categorical_features,
        ),
    ) in variants.items():
        prediction = fit_variant(
            train_data,
            test_data,
            numeric_features,
            categorical_features,
        )

        rows.append(
            {
                "variant": variant,
                "numeric_feature_count":
                    len(numeric_features),
                "categorical_feature_count":
                    len(categorical_features),
                **calculate_metrics(
                    actual,
                    prediction,
                ),
            }
        )

    summary = pd.DataFrame(rows)

    full_safe_wape = float(
        summary.loc[
            summary["variant"].eq("full_safe"),
            "wape",
        ].iloc[0]
    )

    baseline_wape = float(
        summary.loc[
            summary["variant"].eq(
                "baseline_4w_mean"
            ),
            "wape",
        ].iloc[0]
    )

    summary[
        "wape_change_vs_full_safe"
    ] = (
        summary["wape"] - full_safe_wape
    )

    summary[
        "wape_improvement_vs_baseline"
    ] = (
        baseline_wape - summary["wape"]
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        OUTPUT_DIR
        / "17_demand_retrain_ablation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n[DEMAND RETRAIN ABLATION SUMMARY]"
    )

    print(
        summary.to_string(index=False)
    )


if __name__ == "__main__":
    main()
