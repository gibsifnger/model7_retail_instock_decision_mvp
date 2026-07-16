from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
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
EFFECTIVE_VALIDATION_WEEKS = 8
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


def prepare_features(data: pd.DataFrame) -> pd.DataFrame:
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
    probability: np.ndarray,
) -> dict[str, float | int]:
    actual = np.asarray(actual, dtype=int)
    prediction = np.asarray(prediction, dtype=int)
    probability = np.asarray(probability, dtype=float)

    tn, fp, fn, tp = confusion_matrix(
        actual,
        prediction,
        labels=[0, 1],
    ).ravel()

    return {
        "rows": len(actual),
        "actual_positive_rate": float(actual.mean()),
        "predicted_positive_rate": float(prediction.mean()),
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
        "roc_auc": float(
            roc_auc_score(
                actual,
                probability,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                actual,
                probability,
            )
        ),
    }


def build_baseline(data: pd.DataFrame) -> np.ndarray:
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


def baseline_metrics(
    actual: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float | int]:
    actual = np.asarray(actual, dtype=int)
    prediction = np.asarray(prediction, dtype=int)

    tn, fp, fn, tp = confusion_matrix(
        actual,
        prediction,
        labels=[0, 1],
    ).ravel()

    return {
        "rows": len(actual),
        "actual_positive_rate": float(actual.mean()),
        "predicted_positive_rate": float(prediction.mean()),
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


def main() -> None:
    data = pd.read_csv(
        INPUT_PATH,
        parse_dates=["week_start_date"],
    )

    data[TARGET] = pd.to_numeric(
        data[TARGET],
        errors="coerce",
    )

    data = data.loc[
        data[TARGET].notna()
    ].copy()

    data[TARGET] = data[TARGET].astype(int)

    data["label_horizon_end"] = (
        data["week_start_date"]
        + pd.Timedelta(
            weeks=LABEL_HORIZON_WEEKS
        )
    )

    all_weeks = sorted(
        data["week_start_date"].unique()
    )

    test_weeks = all_weeks[-TEST_WEEKS:]
    test_start = pd.Timestamp(test_weeks[0])

    eligible_validation_weeks = [
        pd.Timestamp(week)
        for week in all_weeks
        if (
            pd.Timestamp(week) < test_start
            and pd.Timestamp(week)
            + pd.Timedelta(
                weeks=LABEL_HORIZON_WEEKS
            )
            < test_start
        )
    ]

    validation_weeks = (
        eligible_validation_weeks[
            -EFFECTIVE_VALIDATION_WEEKS:
        ]
    )

    if (
        len(validation_weeks)
        != EFFECTIVE_VALIDATION_WEEKS
    ):
        raise ValueError(
            "Could not construct eight effective "
            "validation weeks."
        )

    validation_start = pd.Timestamp(
        validation_weeks[0]
    )

    train_data = data.loc[
        data["label_horizon_end"]
        < validation_start
    ].copy()

    validation_data = data.loc[
        data["week_start_date"].isin(
            validation_weeks
        )
    ].copy()

    test_data = data.loc[
        data["week_start_date"].isin(
            test_weeks
        )
    ].copy()

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

    test_probability = (
        pipeline.predict_proba(
            prepare_features(test_data)
        )[:, 1]
    )

    threshold_rows = []

    for threshold in np.arange(
        0.05,
        0.951,
        0.05,
    ):
        threshold = round(
            float(threshold),
            2,
        )

        validation_prediction = (
            validation_probability
            >= threshold
        ).astype(int)

        test_prediction = (
            test_probability
            >= threshold
        ).astype(int)

        validation_metrics = (
            calculate_metrics(
                validation_data[TARGET],
                validation_prediction,
                validation_probability,
            )
        )

        test_metrics = calculate_metrics(
            test_data[TARGET],
            test_prediction,
            test_probability,
        )

        threshold_rows.append(
            {
                "threshold": threshold,
                "validation_precision":
                    validation_metrics["precision"],
                "validation_recall":
                    validation_metrics["recall"],
                "validation_f1":
                    validation_metrics["f1"],
                "validation_false_positive":
                    validation_metrics[
                        "false_positive"
                    ],
                "validation_false_negative":
                    validation_metrics[
                        "false_negative"
                    ],
                "test_precision":
                    test_metrics["precision"],
                "test_recall":
                    test_metrics["recall"],
                "test_f1":
                    test_metrics["f1"],
                "test_false_positive":
                    test_metrics[
                        "false_positive"
                    ],
                "test_false_negative":
                    test_metrics[
                        "false_negative"
                    ],
                "f1_transfer_gap":
                    test_metrics["f1"]
                    - validation_metrics["f1"],
            }
        )

    threshold_table = pd.DataFrame(
        threshold_rows
    )

    selected_row = (
        threshold_table.sort_values(
            [
                "validation_f1",
                "validation_recall",
                "validation_precision",
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

    selected_test_prediction = (
        test_probability
        >= selected_threshold
    ).astype(int)

    baseline_prediction = build_baseline(
        test_data
    )

    selected_metrics = pd.DataFrame(
        [
            {
                "method":
                    "model_validation_selected",
                "threshold":
                    selected_threshold,
                **calculate_metrics(
                    test_data[TARGET],
                    selected_test_prediction,
                    test_probability,
                ),
            },
            {
                "method":
                    "center_rule_baseline",
                "threshold":
                    np.nan,
                **baseline_metrics(
                    test_data[TARGET],
                    baseline_prediction,
                ),
                "roc_auc": np.nan,
                "pr_auc": np.nan,
            },
        ]
    )

    split_summary = pd.DataFrame(
        [
            {
                "split": "train",
                "rows": len(train_data),
                "week_count":
                    train_data[
                        "week_start_date"
                    ].nunique(),
                "week_min":
                    train_data[
                        "week_start_date"
                    ].min(),
                "week_max":
                    train_data[
                        "week_start_date"
                    ].max(),
                "positive_rate":
                    train_data[TARGET].mean(),
            },
            {
                "split": "validation",
                "rows":
                    len(validation_data),
                "week_count":
                    validation_data[
                        "week_start_date"
                    ].nunique(),
                "week_min":
                    validation_data[
                        "week_start_date"
                    ].min(),
                "week_max":
                    validation_data[
                        "week_start_date"
                    ].max(),
                "positive_rate":
                    validation_data[
                        TARGET
                    ].mean(),
            },
            {
                "split": "test",
                "rows": len(test_data),
                "week_count":
                    test_data[
                        "week_start_date"
                    ].nunique(),
                "week_min":
                    test_data[
                        "week_start_date"
                    ].min(),
                "week_max":
                    test_data[
                        "week_start_date"
                    ].max(),
                "positive_rate":
                    test_data[TARGET].mean(),
            },
        ]
    )

    weekly_rows = []

    for week, week_data in test_data.groupby(
        "week_start_date"
    ):
        week_index = week_data.index

        week_probability = test_probability[
            test_data.index.get_indexer(
                week_index
            )
        ]

        week_prediction = (
            week_probability
            >= selected_threshold
        ).astype(int)

        weekly_rows.append(
            {
                "week_start_date": week,
                **calculate_metrics(
                    week_data[TARGET],
                    week_prediction,
                    week_probability,
                ),
            }
        )

    weekly_metrics = pd.DataFrame(
        weekly_rows
    )

    cost_rows = []

    model_metrics = selected_metrics.loc[
        selected_metrics["method"].eq(
            "model_validation_selected"
        )
    ].iloc[0]

    baseline_metric_row = (
        selected_metrics.loc[
            selected_metrics["method"].eq(
                "center_rule_baseline"
            )
        ].iloc[0]
    )

    for missed_cost_ratio in [
        1,
        3,
        5,
        6,
        10,
        20,
    ]:
        model_cost = (
            model_metrics["false_positive"]
            + missed_cost_ratio
            * model_metrics[
                "false_negative"
            ]
        )

        baseline_cost = (
            baseline_metric_row[
                "false_positive"
            ]
            + missed_cost_ratio
            * baseline_metric_row[
                "false_negative"
            ]
        )

        cost_rows.append(
            {
                "missed_stockout_cost_ratio":
                    missed_cost_ratio,
                "model_cost": model_cost,
                "baseline_cost": baseline_cost,
                "preferred_method": (
                    "MODEL"
                    if model_cost
                    < baseline_cost
                    else "BASELINE"
                ),
            }
        )

    cost_comparison = pd.DataFrame(
        cost_rows
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    split_summary.to_csv(
        OUTPUT_DIR
        / "25_stockout_threshold_split_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    threshold_table.to_csv(
        OUTPUT_DIR
        / "25_stockout_threshold_transfer.csv",
        index=False,
        encoding="utf-8-sig",
    )

    selected_metrics.to_csv(
        OUTPUT_DIR
        / "25_stockout_selected_threshold_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    weekly_metrics.to_csv(
        OUTPUT_DIR
        / "25_stockout_weekly_test_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    cost_comparison.to_csv(
        OUTPUT_DIR
        / "25_stockout_cost_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n[CORRECTED STOCKOUT SPLIT SUMMARY]"
    )
    print(
        split_summary.to_string(index=False)
    )

    print(
        "\n[SELECTED VALIDATION THRESHOLD]"
    )
    print(
        f"{selected_threshold:.2f}"
    )

    print(
        "\n[SELECTED THRESHOLD TEST METRICS]"
    )
    print(
        selected_metrics.to_string(index=False)
    )

    print(
        "\n[TOP THRESHOLD TRANSFER RESULTS]"
    )
    print(
        threshold_table.sort_values(
            [
                "validation_f1",
                "validation_recall",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .head(10)
        .to_string(index=False)
    )

    print(
        "\n[WEEKLY TEST METRICS]"
    )
    print(
        weekly_metrics.to_string(index=False)
    )

    print(
        "\n[COST COMPARISON]"
    )
    print(
        cost_comparison.to_string(index=False)
    )


if __name__ == "__main__":
    main()
