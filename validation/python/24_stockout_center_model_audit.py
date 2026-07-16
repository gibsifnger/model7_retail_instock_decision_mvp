from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
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


REPO_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    REPO_ROOT
    / "outputs"
    / "validation"
    / "23_stockout_center_grain_candidate.csv"
)

OUTPUT_DIR = (
    REPO_ROOT
    / "outputs"
    / "validation"
)

TARGET = "target_stockout_risk_next_2w"

KEY_COLUMNS = [
    "sku_id",
    "center_id",
    "week_start_date",
]

TEST_WEEKS = 8
VALIDATION_WINDOW_WEEKS = 8
LABEL_HORIZON_WEEKS = 2
RANDOM_SEED = 42

NUMERIC_FEATURES = [
    "center_sales_lag_1w",
    "center_sales_lag_4w",
    "center_sales_rolling_mean_4w",
    "center_sales_rolling_std_4w",
    "center_demand_volatility_index",
    "sales_history_weeks",
    "current_stockout_flag",
    "stockout_snapshots_last_4w",
    "in_stock_snapshot_rate_4w",
    "available_qty",
    "reserved_qty",
    "inbound_qty_next_1w",
    "inbound_qty_next_2w",
    "inbound_qty_next_4w",
    "inventory_position_next_2w",
    "inventory_cover_next_2w_basis",
    "open_po_qty",
    "overdue_po_qty",
    "standard_lead_time_days",
    "vendor_avg_lead_time",
    "vendor_lead_time_std",
    "po_fill_rate",
    "on_time_delivery_rate",
    "moq_qty",
    "order_multiple",
    "min_order_amount",
    "unit_cost",
    "list_price",
    "product_age_weeks",
    "new_product_flag",
    "active_flag",
    "order_block_flag",
]

CATEGORICAL_FEATURES = [
    "sku_id",
    "center_id",
    "category_l1",
    "category_l2",
    "brand",
    "default_vendor_id",
    "vendor_country",
    "import_flag",
    "reliability_tier",
    "lead_time_profile",
    "fill_rate_profile",
]


def build_pipeline() -> Pipeline:
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
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
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
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def prepare_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    result = data[
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
    ].copy()

    for column in NUMERIC_FEATURES:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    for column in CATEGORICAL_FEATURES:
        result[column] = (
            result[column]
            .fillna("UNKNOWN")
            .astype(str)
        )

    return result


def calculate_metrics(
    actual: np.ndarray,
    prediction: np.ndarray,
    probability: np.ndarray | None = None,
) -> dict[str, float | int]:
    actual = np.asarray(actual, dtype=int)
    prediction = np.asarray(prediction, dtype=int)

    tn, fp, fn, tp = confusion_matrix(
        actual,
        prediction,
        labels=[0, 1],
    ).ravel()

    result = {
        "rows": len(actual),
        "actual_positive_rate": float(actual.mean()),
        "predicted_positive_rate": float(prediction.mean()),
        "accuracy": float(
            accuracy_score(actual, prediction)
        ),
        "precision": float(
            precision_score(
                actual,
                prediction,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                actual,
                prediction,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                actual,
                prediction,
                zero_division=0,
            )
        ),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }

    if (
        probability is not None
        and np.unique(actual).size == 2
    ):
        probability = np.asarray(
            probability,
            dtype=float,
        )

        result["roc_auc"] = float(
            roc_auc_score(
                actual,
                probability,
            )
        )

        result["pr_auc"] = float(
            average_precision_score(
                actual,
                probability,
            )
        )
    else:
        result["roc_auc"] = np.nan
        result["pr_auc"] = np.nan

    return result


def build_baseline(
    data: pd.DataFrame,
) -> np.ndarray:
    cover = pd.to_numeric(
        data["inventory_cover_next_2w_basis"],
        errors="coerce",
    ).fillna(np.inf)

    recent_stockout = pd.to_numeric(
        data["stockout_snapshots_last_4w"],
        errors="coerce",
    ).fillna(0)

    return (
        (cover < 2)
        | (recent_stockout > 0)
    ).astype(int).to_numpy()


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = pd.read_csv(
        INPUT_PATH,
        parse_dates=["week_start_date"],
    )

    required = set(
        KEY_COLUMNS
        + NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
        + [TARGET]
    )

    missing = sorted(
        required.difference(data.columns)
    )

    if missing:
        raise ValueError(
            "Missing columns: "
            + ", ".join(missing)
        )

    data[TARGET] = pd.to_numeric(
        data[TARGET],
        errors="coerce",
    )

    data = data.loc[
        data[TARGET].notna()
    ].copy()

    data[TARGET] = data[TARGET].astype(int)

    duplicate_keys = int(
        data.duplicated(KEY_COLUMNS).sum()
    )

    if duplicate_keys:
        raise ValueError(
            f"Duplicate center keys: {duplicate_keys}"
        )

    unique_weeks = sorted(
        data["week_start_date"].unique()
    )

    required_weeks = (
        TEST_WEEKS
        + VALIDATION_WINDOW_WEEKS
        + LABEL_HORIZON_WEEKS
        + 1
    )

    if len(unique_weeks) < required_weeks:
        raise ValueError(
            "Not enough labeled weeks."
        )

    test_weeks = unique_weeks[-TEST_WEEKS:]
    test_start = pd.Timestamp(test_weeks[0])

    validation_window_start = pd.Timestamp(
        unique_weeks[
            -(
                TEST_WEEKS
                + VALIDATION_WINDOW_WEEKS
            )
        ]
    )

    data["label_horizon_end"] = (
        data["week_start_date"]
        + pd.Timedelta(
            weeks=LABEL_HORIZON_WEEKS
        )
    )

    train_data = data.loc[
        data["label_horizon_end"]
        < validation_window_start
    ].copy()

    validation_data = data.loc[
        data["week_start_date"]
        .ge(validation_window_start)
        & data["label_horizon_end"]
        .lt(test_start)
    ].copy()

    test_data = data.loc[
        data["week_start_date"].isin(
            test_weeks
        )
    ].copy()

    for name, table in [
        ("train", train_data),
        ("validation", validation_data),
        ("test", test_data),
    ]:
        if table.empty:
            raise ValueError(
                f"{name} data is empty."
            )

        if table[TARGET].nunique() < 2:
            raise ValueError(
                f"{name} target has only one class."
            )

    pipeline = build_pipeline()

    pipeline.fit(
        prepare_features(train_data),
        train_data[TARGET],
    )

    validation_probability = (
        pipeline.predict_proba(
            prepare_features(validation_data)
        )[:, 1]
    )

    threshold_rows = []

    for threshold in np.arange(
        0.05,
        0.951,
        0.05,
    ):
        validation_prediction = (
            validation_probability
            >= threshold
        ).astype(int)

        metrics = calculate_metrics(
            validation_data[TARGET],
            validation_prediction,
            validation_probability,
        )

        threshold_rows.append(
            {
                "threshold": round(
                    float(threshold),
                    2,
                ),
                **metrics,
            }
        )

    threshold_table = pd.DataFrame(
        threshold_rows
    )

    selected_row = (
        threshold_table.sort_values(
            [
                "f1",
                "recall",
                "precision",
                "threshold",
            ],
            ascending=[
                False,
                False,
                False,
                True,
            ],
        )
        .iloc[0]
    )

    selected_threshold = float(
        selected_row["threshold"]
    )

    test_probability = (
        pipeline.predict_proba(
            prepare_features(test_data)
        )[:, 1]
    )

    test_prediction = (
        test_probability
        >= selected_threshold
    ).astype(int)

    baseline_prediction = build_baseline(
        test_data
    )

    metrics_rows = [
        {
            "evaluation_set": "validation",
            "method":
                "model_selected_threshold",
            "threshold":
                selected_threshold,
            **calculate_metrics(
                validation_data[TARGET],
                (
                    validation_probability
                    >= selected_threshold
                ).astype(int),
                validation_probability,
            ),
        },
        {
            "evaluation_set": "test",
            "method":
                "model_locked_threshold",
            "threshold":
                selected_threshold,
            **calculate_metrics(
                test_data[TARGET],
                test_prediction,
                test_probability,
            ),
        },
        {
            "evaluation_set": "test",
            "method":
                "model_reference_threshold_0_4",
            "threshold": 0.4,
            **calculate_metrics(
                test_data[TARGET],
                (
                    test_probability >= 0.4
                ).astype(int),
                test_probability,
            ),
        },
        {
            "evaluation_set": "test",
            "method":
                "model_reference_threshold_0_5",
            "threshold": 0.5,
            **calculate_metrics(
                test_data[TARGET],
                (
                    test_probability >= 0.5
                ).astype(int),
                test_probability,
            ),
        },
        {
            "evaluation_set": "test",
            "method":
                "center_rule_baseline",
            "threshold": np.nan,
            **calculate_metrics(
                test_data[TARGET],
                baseline_prediction,
            ),
        },
    ]

    metrics = pd.DataFrame(
        metrics_rows
    )

    split_summary = pd.DataFrame(
        [
            {
                "split": "train",
                "rows": len(train_data),
                "week_min":
                    train_data[
                        "week_start_date"
                    ].min(),
                "week_max":
                    train_data[
                        "week_start_date"
                    ].max(),
                "positive_rows":
                    int(train_data[TARGET].sum()),
                "positive_rate":
                    float(train_data[TARGET].mean()),
            },
            {
                "split": "validation",
                "rows": len(validation_data),
                "week_min":
                    validation_data[
                        "week_start_date"
                    ].min(),
                "week_max":
                    validation_data[
                        "week_start_date"
                    ].max(),
                "positive_rows":
                    int(
                        validation_data[
                            TARGET
                        ].sum()
                    ),
                "positive_rate":
                    float(
                        validation_data[
                            TARGET
                        ].mean()
                    ),
            },
            {
                "split": "test",
                "rows": len(test_data),
                "week_min":
                    test_data[
                        "week_start_date"
                    ].min(),
                "week_max":
                    test_data[
                        "week_start_date"
                    ].max(),
                "positive_rows":
                    int(test_data[TARGET].sum()),
                "positive_rate":
                    float(test_data[TARGET].mean()),
            },
        ]
    )

    predictions = test_data[
        KEY_COLUMNS + [TARGET]
    ].copy()

    predictions[
        "stockout_probability"
    ] = test_probability

    predictions[
        "selected_threshold"
    ] = selected_threshold

    predictions[
        "model_prediction"
    ] = test_prediction

    predictions[
        "baseline_prediction"
    ] = baseline_prediction

    threshold_table.to_csv(
        OUTPUT_DIR
        / "24_stockout_center_threshold_validation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metrics.to_csv(
        OUTPUT_DIR
        / "24_stockout_center_model_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    split_summary.to_csv(
        OUTPUT_DIR
        / "24_stockout_center_split_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    predictions.to_csv(
        OUTPUT_DIR
        / "24_stockout_center_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n[STOCKOUT CENTER SPLIT SUMMARY]"
    )
    print(
        split_summary.to_string(
            index=False
        )
    )

    print(
        "\n[SELECTED TECHNICAL THRESHOLD]"
    )
    print(
        f"{selected_threshold:.2f}"
    )

    print(
        "\n[STOCKOUT CENTER MODEL METRICS]"
    )
    print(
        metrics.to_string(
            index=False
        )
    )

    print(
        "\n[TOP VALIDATION THRESHOLDS]"
    )
    print(
        threshold_table.sort_values(
            [
                "f1",
                "recall",
                "precision",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
