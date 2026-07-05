"""Train and evaluate the MVP next-week demand forecasting model."""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


RANDOM_SEED = 42
TEST_WEEKS = 8

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = REPO_ROOT / "data" / "mart" / "final_modeling_table.csv"
PREDICTION_PATH = (
    REPO_ROOT / "outputs" / "predictions" / "demand_forecast_result.csv"
)
METRICS_PATH = REPO_ROOT / "outputs" / "metrics" / "demand_model_metrics.csv"
IMPORTANCE_PATH = (
    REPO_ROOT / "outputs" / "figures" / "feature_importance_demand.png"
)

TARGET_COLUMN = "target_demand_next_1w"
KEY_COLUMNS = ["sku_id", "channel_id", "center_id", "week_start_date"]

NUMERIC_FEATURES = [
    "observed_sales_qty",
    "demand_gap_qty",
    "sales_censored_flag",
    "stockout_adjusted_sales",
    "sales_lag_1w",
    "sales_lag_4w",
    "sales_rolling_mean_4w",
    "sales_rolling_std_4w",
    "demand_volatility_index",
    "stockout_adjusted_sales_rolling_mean_4w",
    "stockout_days_last_4w",
    "in_stock_rate_4w",
    "sales_history_weeks",
    "promo_flag",
    "promo_depth",
    "promo_duration_days",
    "promo_days_in_week",
    "promo_day_index",
    "historical_promo_uplift",
    "available_qty",
    "reserved_qty",
    "inbound_qty_next_1w",
    "inbound_qty_next_2w",
    "inbound_qty_next_4w",
    "inventory_position_qty",
    "inventory_cover_weeks",
    "safety_stock_qty",
    "standard_lead_time_days",
    "vendor_avg_lead_time",
    "vendor_lead_time_std",
    "po_fill_rate",
    "on_time_delivery_rate",
    "open_po_qty",
    "overdue_po_qty",
    "moq_qty",
    "order_multiple",
    "min_order_amount",
    "unit_cost",
    "list_price",
    "product_age_weeks",
    "new_product_flag",
    "active_flag",
    "order_block_flag",
    "manual_override_flag",
]

CATEGORICAL_FEATURES = [
    "sku_id",
    "channel_id",
    "center_id",
    "category_l1",
    "category_l2",
    "brand",
    "default_vendor_id",
    "promo_type",
    "vendor_country",
    "import_flag",
    "reliability_tier",
    "lead_time_profile",
    "fill_rate_profile",
]

MODEL_NAME = "HistGradientBoostingRegressor"
BASELINE_NAME = "sales_rolling_mean_4w_baseline"


def load_modeling_table() -> pd.DataFrame:
    """Load the feature table and validate the explicit modeling contract."""
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(
            f"Modeling table not found: {INPUT_PATH}\n"
            "Run src/features/build_final_feature_table.py first."
        )

    data = pd.read_csv(INPUT_PATH)
    required_columns = set(
        KEY_COLUMNS
        + NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
        + [TARGET_COLUMN, "stockout_flag"]
    )
    missing_columns = sorted(required_columns.difference(data.columns))
    if missing_columns:
        raise ValueError(
            "Modeling table is missing required columns: "
            + ", ".join(missing_columns)
        )

    data["week_start_date"] = pd.to_datetime(
        data["week_start_date"], errors="raise"
    )
    data[TARGET_COLUMN] = pd.to_numeric(data[TARGET_COLUMN], errors="coerce")
    data = data.loc[data[TARGET_COLUMN].notna()].copy()
    if data.empty:
        raise ValueError(f"No non-null rows available for target: {TARGET_COLUMN}")

    duplicate_count = data.duplicated(KEY_COLUMNS).sum()
    if duplicate_count:
        raise ValueError(
            f"Modeling data contains {duplicate_count} duplicate decision keys."
        )

    # Cap the known long-tail inventory ratio before train/test preprocessing.
    data["inventory_cover_weeks"] = pd.to_numeric(
        data["inventory_cover_weeks"], errors="coerce"
    ).clip(upper=52)

    for column in NUMERIC_FEATURES:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    for column in CATEGORICAL_FEATURES:
        data[column] = data[column].fillna("UNKNOWN").astype(str)

    return data.sort_values(KEY_COLUMNS, kind="stable").reset_index(drop=True)


def time_based_split(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[pd.Timestamp]]:
    """Use the final eight available feature weeks as the holdout test set."""
    unique_weeks = sorted(data["week_start_date"].unique())
    if len(unique_weeks) <= TEST_WEEKS:
        raise ValueError(
            f"At least {TEST_WEEKS + 1} weeks are required for a time split; "
            f"found {len(unique_weeks)}."
        )

    test_weeks = unique_weeks[-TEST_WEEKS:]
    test_start = test_weeks[0]
    train_data = data.loc[data["week_start_date"].lt(test_start)].copy()
    test_data = data.loc[data["week_start_date"].isin(test_weeks)].copy()

    if train_data.empty or test_data.empty:
        raise ValueError("Time-based split produced an empty train or test set.")

    return train_data, test_data, test_weeks


def build_model_pipeline() -> Pipeline:
    """Build a dense one-hot preprocessing and gradient boosting pipeline."""
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="UNKNOWN"),
            ),
            (
                "one_hot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    model = HistGradientBoostingRegressor(
        learning_rate=0.06,
        max_iter=250,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=RANDOM_SEED,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def calculate_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """Calculate the five required aggregate demand forecast metrics."""
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
        forecast_accuracy = float(1.0 - wape)

    return {
        "mae": mae,
        "rmse": rmse,
        "wape": wape,
        "bias": bias,
        "forecast_accuracy": forecast_accuracy,
    }


def build_metrics_table(
    model_metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> pd.DataFrame:
    """Create comparable model and baseline metric rows."""
    metadata = {
        "split": "time_based_test_last_8_weeks",
        "train_rows": len(train_data),
        "test_rows": len(test_data),
        "train_week_min": train_data["week_start_date"].min().strftime("%Y-%m-%d"),
        "train_week_max": train_data["week_start_date"].max().strftime("%Y-%m-%d"),
        "test_week_min": test_data["week_start_date"].min().strftime("%Y-%m-%d"),
        "test_week_max": test_data["week_start_date"].max().strftime("%Y-%m-%d"),
    }
    rows = [
        {"model_name": MODEL_NAME, **metadata, **model_metrics},
        {"model_name": BASELINE_NAME, **metadata, **baseline_metrics},
    ]
    column_order = [
        "model_name",
        "split",
        "mae",
        "rmse",
        "wape",
        "bias",
        "forecast_accuracy",
        "train_rows",
        "test_rows",
        "train_week_min",
        "train_week_max",
        "test_week_min",
        "test_week_max",
    ]
    return pd.DataFrame(rows)[column_order]


def build_prediction_output(
    test_data: pd.DataFrame,
    baseline_prediction: np.ndarray,
    model_prediction: np.ndarray,
) -> pd.DataFrame:
    """Create the requested grain-level test prediction output."""
    output_columns = KEY_COLUMNS + [
        TARGET_COLUMN,
        "promo_flag",
        "stockout_flag",
        "inventory_cover_weeks",
    ]
    result = test_data[output_columns].copy()
    result["baseline_pred"] = baseline_prediction
    result["demand_pred"] = model_prediction
    result["error"] = result["demand_pred"] - result[TARGET_COLUMN]
    result["abs_error"] = result["error"].abs()
    result["ape"] = safe_ape(
        result[TARGET_COLUMN].to_numpy(dtype=float),
        result["abs_error"].to_numpy(dtype=float),
    )
    result["week_start_date"] = result["week_start_date"].dt.strftime("%Y-%m-%d")

    requested_order = [
        "sku_id",
        "channel_id",
        "center_id",
        "week_start_date",
        TARGET_COLUMN,
        "baseline_pred",
        "demand_pred",
        "abs_error",
        "ape",
        "error",
        "promo_flag",
        "stockout_flag",
        "inventory_cover_weeks",
    ]
    return result[requested_order]


def safe_ape(actual: np.ndarray, absolute_error: np.ndarray) -> np.ndarray:
    """Calculate row APE while leaving zero-actual rows as NaN."""
    result = np.full(len(actual), np.nan, dtype=float)
    nonzero_actual = actual != 0
    np.divide(
        absolute_error,
        np.abs(actual),
        out=result,
        where=nonzero_actual,
    )
    return result


def save_feature_importance(
    pipeline: Pipeline,
    test_features: pd.DataFrame,
    test_target: pd.Series,
) -> pd.DataFrame:
    """Save a top-20 raw-feature permutation importance plot."""
    importance = permutation_importance(
        pipeline,
        test_features,
        test_target,
        scoring="neg_mean_absolute_error",
        n_repeats=3,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    feature_names = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    importance_table = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance.importances_mean,
        }
    ).sort_values("importance", ascending=False)
    top_features = importance_table.head(20).sort_values("importance")

    figure, axis = plt.subplots(figsize=(10, 7))
    axis.barh(top_features["feature"], top_features["importance"], color="#377eb8")
    axis.set_title("Demand Forecast Feature Importance (Permutation, Top 20)")
    axis.set_xlabel("Increase in MAE after permutation")
    axis.set_ylabel("Feature")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(IMPORTANCE_PATH, dpi=150, bbox_inches="tight")
    plt.close(figure)

    return importance_table


def ensure_output_directories() -> None:
    """Create all requested output directories if they do not exist."""
    for path in [PREDICTION_PATH, METRICS_PATH, IMPORTANCE_PATH]:
        path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """Train the demand model, compare the baseline, and save outputs."""
    data = load_modeling_table()
    train_data, test_data, _ = time_based_split(data)
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    train_features = train_data[feature_columns]
    train_target = train_data[TARGET_COLUMN]
    test_features = test_data[feature_columns]
    test_target = test_data[TARGET_COLUMN]

    print(f"Train rows: {len(train_data)}")
    print(
        "Train week range: "
        f"{train_data['week_start_date'].min():%Y-%m-%d} to "
        f"{train_data['week_start_date'].max():%Y-%m-%d}"
    )
    print(f"Test rows: {len(test_data)}")
    print(
        "Test week range: "
        f"{test_data['week_start_date'].min():%Y-%m-%d} to "
        f"{test_data['week_start_date'].max():%Y-%m-%d}"
    )

    pipeline = build_model_pipeline()
    pipeline.fit(train_features, train_target)
    model_prediction = np.clip(pipeline.predict(test_features), a_min=0, a_max=None)

    baseline_fallback = float(train_target.median())
    baseline_prediction = (
        pd.to_numeric(test_data["sales_rolling_mean_4w"], errors="coerce")
        .fillna(baseline_fallback)
        .clip(lower=0)
        .to_numpy(dtype=float)
    )

    model_metrics = calculate_metrics(test_target.to_numpy(), model_prediction)
    baseline_metrics = calculate_metrics(test_target.to_numpy(), baseline_prediction)
    metrics_table = build_metrics_table(
        model_metrics, baseline_metrics, train_data, test_data
    )
    prediction_output = build_prediction_output(
        test_data, baseline_prediction, model_prediction
    )

    ensure_output_directories()
    prediction_output.to_csv(PREDICTION_PATH, index=False, encoding="utf-8")
    metrics_table.to_csv(METRICS_PATH, index=False, encoding="utf-8")
    save_feature_importance(pipeline, test_features, test_target)

    for row in metrics_table.itertuples(index=False):
        print(
            f"{row.model_name}: "
            f"MAE={row.mae:.4f}, RMSE={row.rmse:.4f}, "
            f"WAPE={row.wape:.4f}, Bias={row.bias:.4f}, "
            f"Forecast Accuracy={row.forecast_accuracy:.4f}"
        )
    print(f"Saved predictions to: {PREDICTION_PATH}")
    print(f"Saved metrics to: {METRICS_PATH}")
    print(f"Saved feature importance to: {IMPORTANCE_PATH}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"[ERROR] {error}", file=sys.stderr, flush=True)
        raise SystemExit(1) from error
