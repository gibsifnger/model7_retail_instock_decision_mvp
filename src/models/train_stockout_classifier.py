"""Train and evaluate the MVP two-week stockout risk classifier."""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


RANDOM_SEED = 42
TEST_WEEKS = 8
DEFAULT_THRESHOLD = 0.5
FINAL_THRESHOLD = 0.4

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = REPO_ROOT / "data" / "mart" / "final_modeling_table.csv"
PREDICTION_PATH = (
    REPO_ROOT / "outputs" / "predictions" / "stockout_risk_result.csv"
)
METRICS_PATH = REPO_ROOT / "outputs" / "metrics" / "stockout_model_metrics.csv"
IMPORTANCE_PATH = (
    REPO_ROOT / "outputs" / "figures" / "feature_importance_stockout.png"
)

TARGET_COLUMN = "target_stockout_risk_next_2w"
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

MODEL_NAME = "HistGradientBoostingClassifier"
BASELINE_NAME = "inventory_cover_stockout_rule_baseline"


def load_modeling_table() -> pd.DataFrame:
    """Load and validate the explicit stockout classification feature set."""
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(
            f"Modeling table not found: {INPUT_PATH}\n"
            "Run src/features/build_final_feature_table.py first."
        )

    data = pd.read_csv(INPUT_PATH)
    required_columns = set(
        KEY_COLUMNS + NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN]
    )
    missing_columns = sorted(required_columns.difference(data.columns))
    if missing_columns:
        raise ValueError(
            "Modeling table is missing required columns: "
            + ", ".join(missing_columns)
        )

    # Features are selected only from the fixed allowlist above. In particular,
    # target_demand_next_1w, demand_pred, and all future labels are excluded.
    forbidden_features = {
        "target_demand_next_1w",
        "target_stockout_risk_next_2w",
        "demand_pred",
    }
    selected_features = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    if selected_features.intersection(forbidden_features):
        raise ValueError("A future label or demand prediction entered the feature set.")

    data["week_start_date"] = pd.to_datetime(
        data["week_start_date"], errors="raise"
    )
    data[TARGET_COLUMN] = pd.to_numeric(data[TARGET_COLUMN], errors="coerce")
    data = data.loc[data[TARGET_COLUMN].notna()].copy()
    if data.empty:
        raise ValueError(f"No non-null rows available for target: {TARGET_COLUMN}")
    if not set(data[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise ValueError(f"Target must be binary: {TARGET_COLUMN}")
    data[TARGET_COLUMN] = data[TARGET_COLUMN].astype(int)

    duplicate_count = data.duplicated(KEY_COLUMNS).sum()
    if duplicate_count:
        raise ValueError(
            f"Modeling data contains {duplicate_count} duplicate decision keys."
        )

    data["inventory_cover_weeks"] = pd.to_numeric(
        data["inventory_cover_weeks"], errors="coerce"
    ).clip(upper=52)
    for column in NUMERIC_FEATURES:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    for column in CATEGORICAL_FEATURES:
        data[column] = data[column].fillna("UNKNOWN").astype(str)

    return data.sort_values(
        ["week_start_date", "sku_id", "channel_id", "center_id"],
        kind="stable",
    ).reset_index(drop=True)


def time_based_split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reserve the final eight labeled weeks as the holdout test set."""
    unique_weeks = sorted(data["week_start_date"].unique())
    if len(unique_weeks) <= TEST_WEEKS:
        raise ValueError(
            f"At least {TEST_WEEKS + 1} labeled weeks are required; "
            f"found {len(unique_weeks)}."
        )

    test_weeks = unique_weeks[-TEST_WEEKS:]
    train_data = data.loc[data["week_start_date"].lt(test_weeks[0])].copy()
    test_data = data.loc[data["week_start_date"].isin(test_weeks)].copy()
    if train_data.empty or test_data.empty:
        raise ValueError("Time-based split produced an empty train or test set.")
    if train_data[TARGET_COLUMN].nunique() < 2:
        raise ValueError("Training target must contain both binary classes.")

    return train_data, test_data


def build_model_pipeline() -> Pipeline:
    """Create one-hot preprocessing and a class-balanced gradient booster."""
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
    classifier = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=250,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )
    return Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", classifier)]
    )


def calculate_classification_metrics(
    actual: np.ndarray,
    predicted_label: np.ndarray,
    risk_score: np.ndarray,
) -> dict[str, float | int]:
    """Calculate classification, ranking, and confusion-matrix metrics."""
    actual = np.asarray(actual, dtype=int)
    predicted_label = np.asarray(predicted_label, dtype=int)
    risk_score = np.asarray(risk_score, dtype=float)
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        actual, predicted_label, labels=[0, 1]
    ).ravel()

    roc_auc = (
        float(roc_auc_score(actual, risk_score))
        if np.unique(actual).size == 2
        else np.nan
    )
    pr_auc = (
        float(average_precision_score(actual, risk_score))
        if np.any(actual == 1)
        else np.nan
    )
    return {
        "accuracy": float(accuracy_score(actual, predicted_label)),
        "precision": float(precision_score(actual, predicted_label, zero_division=0)),
        "recall": float(recall_score(actual, predicted_label, zero_division=0)),
        "f1": float(f1_score(actual, predicted_label, zero_division=0)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "positive_rate_actual": float(actual.mean()),
        "positive_rate_pred": float(predicted_label.mean()),
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "true_positive": int(true_positive),
    }


def build_metrics_table(
    actual: np.ndarray,
    model_probability: np.ndarray,
    baseline_prediction: np.ndarray,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> pd.DataFrame:
    """Compare model thresholds 0.5/0.4 with the fixed rule baseline."""
    metadata = {
        "split": "time_based_test_last_8_weeks",
        "train_rows": len(train_data),
        "test_rows": len(test_data),
        "train_week_min": train_data["week_start_date"].min().strftime("%Y-%m-%d"),
        "train_week_max": train_data["week_start_date"].max().strftime("%Y-%m-%d"),
        "test_week_min": test_data["week_start_date"].min().strftime("%Y-%m-%d"),
        "test_week_max": test_data["week_start_date"].max().strftime("%Y-%m-%d"),
    }
    rows: list[dict[str, object]] = []
    for threshold in [DEFAULT_THRESHOLD, FINAL_THRESHOLD]:
        predicted_label = (model_probability >= threshold).astype(int)
        rows.append(
            {
                "model_name": MODEL_NAME,
                "threshold": threshold,
                **metadata,
                **calculate_classification_metrics(
                    actual, predicted_label, model_probability
                ),
            }
        )

    rows.append(
        {
            "model_name": BASELINE_NAME,
            "threshold": np.nan,
            **metadata,
            **calculate_classification_metrics(
                actual, baseline_prediction, baseline_prediction
            ),
        }
    )
    column_order = [
        "model_name",
        "split",
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "positive_rate_actual",
        "positive_rate_pred",
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
        "train_rows",
        "test_rows",
        "train_week_min",
        "train_week_max",
        "test_week_min",
        "test_week_max",
    ]
    return pd.DataFrame(rows)[column_order]


def build_baseline_prediction(test_data: pd.DataFrame) -> np.ndarray:
    """Apply the specified low-cover or recent-stockout baseline rule."""
    inventory_cover = pd.to_numeric(
        test_data["inventory_cover_weeks"], errors="coerce"
    ).fillna(np.inf)
    recent_stockout = pd.to_numeric(
        test_data["stockout_days_last_4w"], errors="coerce"
    ).fillna(0)
    return ((inventory_cover < 2) | (recent_stockout > 0)).astype(int).to_numpy()


def build_prediction_output(
    test_data: pd.DataFrame,
    model_probability: np.ndarray,
    model_label: np.ndarray,
    baseline_prediction: np.ndarray,
) -> pd.DataFrame:
    """Create the requested decision-grain risk prediction output."""
    columns = KEY_COLUMNS + [
        TARGET_COLUMN,
        "inventory_cover_weeks",
        "stockout_days_last_4w",
        "available_qty",
        "inbound_qty_next_2w",
        "promo_flag",
        "overdue_po_qty",
    ]
    result = test_data[columns].copy()
    result["stockout_risk_pred_proba"] = model_probability
    result["stockout_risk_pred_label"] = model_label
    result["baseline_risk_pred"] = baseline_prediction
    result["week_start_date"] = result["week_start_date"].dt.strftime("%Y-%m-%d")

    requested_order = [
        "sku_id",
        "channel_id",
        "center_id",
        "week_start_date",
        TARGET_COLUMN,
        "stockout_risk_pred_proba",
        "stockout_risk_pred_label",
        "baseline_risk_pred",
        "inventory_cover_weeks",
        "stockout_days_last_4w",
        "available_qty",
        "inbound_qty_next_2w",
        "promo_flag",
        "overdue_po_qty",
    ]
    return result[requested_order]


def save_feature_importance(
    pipeline: Pipeline,
    test_features: pd.DataFrame,
    test_target: pd.Series,
) -> None:
    """Save a top-20 raw-feature permutation importance plot using PR-AUC."""
    importance = permutation_importance(
        pipeline,
        test_features,
        test_target,
        scoring="average_precision",
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
    axis.barh(top_features["feature"], top_features["importance"], color="#e6550d")
    axis.set_title("Stockout Risk Feature Importance (Permutation, Top 20)")
    axis.set_xlabel("Decrease in PR-AUC after permutation")
    axis.set_ylabel("Feature")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(IMPORTANCE_PATH, dpi=150, bbox_inches="tight")
    plt.close(figure)


def ensure_output_directories() -> None:
    """Create all requested output directories."""
    for path in [PREDICTION_PATH, METRICS_PATH, IMPORTANCE_PATH]:
        path.parent.mkdir(parents=True, exist_ok=True)


def print_split_summary(train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    """Print time ranges, row counts, and target positive rates."""
    print(f"Train rows: {len(train_data)}")
    print(
        "Train week range: "
        f"{train_data['week_start_date'].min():%Y-%m-%d} to "
        f"{train_data['week_start_date'].max():%Y-%m-%d}"
    )
    print(f"Train target positive rate: {train_data[TARGET_COLUMN].mean():.4f}")
    print(f"Test rows: {len(test_data)}")
    print(
        "Test week range: "
        f"{test_data['week_start_date'].min():%Y-%m-%d} to "
        f"{test_data['week_start_date'].max():%Y-%m-%d}"
    )
    print(f"Test target positive rate: {test_data[TARGET_COLUMN].mean():.4f}")


def main() -> None:
    """Train the classifier, compare its baseline, and save all outputs."""
    data = load_modeling_table()
    train_data, test_data = time_based_split(data)
    print_split_summary(train_data, test_data)

    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    train_features = train_data[feature_columns]
    train_target = train_data[TARGET_COLUMN]
    test_features = test_data[feature_columns]
    test_target = test_data[TARGET_COLUMN]

    pipeline = build_model_pipeline()
    pipeline.fit(train_features, train_target)
    model_probability = pipeline.predict_proba(test_features)[:, 1]
    model_label = (model_probability >= FINAL_THRESHOLD).astype(int)
    baseline_prediction = build_baseline_prediction(test_data)

    metrics_table = build_metrics_table(
        test_target.to_numpy(),
        model_probability,
        baseline_prediction,
        train_data,
        test_data,
    )
    prediction_output = build_prediction_output(
        test_data,
        model_probability,
        model_label,
        baseline_prediction,
    )

    ensure_output_directories()
    prediction_output.to_csv(PREDICTION_PATH, index=False, encoding="utf-8")
    metrics_table.to_csv(METRICS_PATH, index=False, encoding="utf-8")
    save_feature_importance(pipeline, test_features, test_target)

    for row in metrics_table.itertuples(index=False):
        threshold_text = "rule" if pd.isna(row.threshold) else f"{row.threshold:.1f}"
        print(
            f"{row.model_name} (threshold={threshold_text}): "
            f"Accuracy={row.accuracy:.4f}, Precision={row.precision:.4f}, "
            f"Recall={row.recall:.4f}, F1={row.f1:.4f}, "
            f"ROC-AUC={row.roc_auc:.4f}, PR-AUC={row.pr_auc:.4f}, "
            f"CM=[[{row.true_negative}, {row.false_positive}], "
            f"[{row.false_negative}, {row.true_positive}]]"
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
