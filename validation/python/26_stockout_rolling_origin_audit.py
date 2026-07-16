from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
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

TEST_WEEKS = 8
VALIDATION_WEEKS = 8
LABEL_HORIZON_WEEKS = 2

FOLD_COUNT = 4
FOLD_STEP_WEEKS = 4
MIN_TRAIN_WEEKS = 18

RANDOM_SEED = 42

THRESHOLDS = [
    round(float(value), 2)
    for value in np.arange(
        0.05,
        0.951,
        0.05,
    )
]

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


def build_pipeline(
    random_seed: int,
) -> Pipeline:
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
        random_state=random_seed,
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


def classification_metrics(
    actual: np.ndarray,
    prediction: np.ndarray,
    probability: np.ndarray,
) -> dict[str, float | int]:
    actual = np.asarray(actual, dtype=int)
    prediction = np.asarray(
        prediction,
        dtype=int,
    )
    probability = np.asarray(
        probability,
        dtype=float,
    )

    tn, fp, fn, tp = confusion_matrix(
        actual,
        prediction,
        labels=[0, 1],
    ).ravel()

    return {
        "rows": len(actual),
        "actual_positive_rate": float(
            actual.mean()
        ),
        "mean_probability": float(
            probability.mean()
        ),
        "calibration_gap": float(
            probability.mean()
            - actual.mean()
        ),
        "brier_score": float(
            brier_score_loss(
                actual,
                probability,
            )
        ),
        "predicted_positive_rate": float(
            prediction.mean()
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
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "true_negative": int(tn),
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


def select_threshold(
    actual: np.ndarray,
    probability: np.ndarray,
) -> tuple[float, pd.DataFrame]:
    rows = []

    for threshold in THRESHOLDS:
        prediction = (
            probability >= threshold
        ).astype(int)

        metrics = classification_metrics(
            actual,
            prediction,
            probability,
        )

        rows.append(
            {
                "threshold": threshold,
                **metrics,
            }
        )

    table = pd.DataFrame(rows)

    selected = (
        table.sort_values(
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

    return (
        float(selected["threshold"]),
        table,
    )


def build_folds(
    data: pd.DataFrame,
) -> list[dict[str, object]]:
    weeks = [
        pd.Timestamp(value)
        for value in sorted(
            data[
                "week_start_date"
            ].unique()
        )
    ]

    latest_end = len(weeks)

    test_end_positions = [
        latest_end
        - FOLD_STEP_WEEKS
        * (
            FOLD_COUNT
            - 1
            - index
        )
        for index in range(FOLD_COUNT)
    ]

    folds = []

    for (
        fold_index,
        test_end,
    ) in enumerate(
        test_end_positions,
        start=1,
    ):
        test_start_index = (
            test_end - TEST_WEEKS
        )

        if test_start_index < 0:
            raise ValueError(
                "Not enough weeks for test."
            )

        test_weeks = weeks[
            test_start_index:test_end
        ]

        test_start = test_weeks[0]

        eligible_validation_weeks = [
            week
            for week in weeks[
                :test_start_index
            ]
            if (
                week
                + pd.Timedelta(
                    weeks=
                        LABEL_HORIZON_WEEKS
                )
                < test_start
            )
        ]

        validation_weeks = (
            eligible_validation_weeks[
                -VALIDATION_WEEKS:
            ]
        )

        if (
            len(validation_weeks)
            != VALIDATION_WEEKS
        ):
            raise ValueError(
                f"Fold {fold_index}: "
                "validation weeks missing."
            )

        validation_start = (
            validation_weeks[0]
        )

        train_data = data.loc[
            (
                data[
                    "week_start_date"
                ]
                + pd.Timedelta(
                    weeks=
                        LABEL_HORIZON_WEEKS
                )
            )
            < validation_start
        ].copy()

        validation_data = data.loc[
            data[
                "week_start_date"
            ].isin(validation_weeks)
        ].copy()

        test_data = data.loc[
            data[
                "week_start_date"
            ].isin(test_weeks)
        ].copy()

        train_week_count = (
            train_data[
                "week_start_date"
            ].nunique()
        )

        if (
            train_week_count
            < MIN_TRAIN_WEEKS
        ):
            raise ValueError(
                f"Fold {fold_index}: "
                f"only {train_week_count} "
                "training weeks."
            )

        for (
            split_name,
            split_data,
        ) in [
            ("train", train_data),
            (
                "validation",
                validation_data,
            ),
            ("test", test_data),
        ]:
            if (
                split_data.empty
                or split_data[
                    TARGET
                ].nunique()
                < 2
            ):
                raise ValueError(
                    f"Fold {fold_index}: "
                    f"invalid {split_name}."
                )

        folds.append(
            {
                "fold_number":
                    fold_index,
                "train":
                    train_data,
                "validation":
                    validation_data,
                "test":
                    test_data,
            }
        )

    return folds


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

    data[TARGET] = pd.to_numeric(
        data[TARGET],
        errors="coerce",
    )

    data = data.loc[
        data[TARGET].notna()
    ].copy()

    data[TARGET] = (
        data[TARGET].astype(int)
    )

    folds = build_folds(data)

    fold_rows = []
    threshold_rows = []
    fixed_threshold_rows = []

    for fold in folds:
        fold_number = int(
            fold["fold_number"]
        )

        train_data = fold["train"]
        validation_data = (
            fold["validation"]
        )
        test_data = fold["test"]

        pipeline = build_pipeline(
            RANDOM_SEED
            + fold_number
        )

        pipeline.fit(
            prepare_features(
                train_data
            ),
            train_data[TARGET],
        )

        validation_probability = (
            pipeline.predict_proba(
                prepare_features(
                    validation_data
                )
            )[:, 1]
        )

        test_probability = (
            pipeline.predict_proba(
                prepare_features(
                    test_data
                )
            )[:, 1]
        )

        (
            selected_threshold,
            validation_threshold_table,
        ) = select_threshold(
            validation_data[TARGET]
            .to_numpy(dtype=int),
            validation_probability,
        )

        selected_test_prediction = (
            test_probability
            >= selected_threshold
        ).astype(int)

        validation_prediction = (
            validation_probability
            >= selected_threshold
        ).astype(int)

        validation_metrics = (
            classification_metrics(
                validation_data[TARGET]
                .to_numpy(dtype=int),
                validation_prediction,
                validation_probability,
            )
        )

        test_metrics = (
            classification_metrics(
                test_data[TARGET]
                .to_numpy(dtype=int),
                selected_test_prediction,
                test_probability,
            )
        )

        fold_rows.append(
            {
                "fold_number":
                    fold_number,
                "selected_threshold":
                    selected_threshold,
                "train_week_count":
                    train_data[
                        "week_start_date"
                    ].nunique(),
                "train_week_min":
                    train_data[
                        "week_start_date"
                    ].min(),
                "train_week_max":
                    train_data[
                        "week_start_date"
                    ].max(),
                "validation_week_min":
                    validation_data[
                        "week_start_date"
                    ].min(),
                "validation_week_max":
                    validation_data[
                        "week_start_date"
                    ].max(),
                "test_week_min":
                    test_data[
                        "week_start_date"
                    ].min(),
                "test_week_max":
                    test_data[
                        "week_start_date"
                    ].max(),
                "validation_positive_rate":
                    validation_metrics[
                        "actual_positive_rate"
                    ],
                "test_positive_rate":
                    test_metrics[
                        "actual_positive_rate"
                    ],
                "validation_mean_probability":
                    validation_metrics[
                        "mean_probability"
                    ],
                "test_mean_probability":
                    test_metrics[
                        "mean_probability"
                    ],
                "validation_calibration_gap":
                    validation_metrics[
                        "calibration_gap"
                    ],
                "test_calibration_gap":
                    test_metrics[
                        "calibration_gap"
                    ],
                "validation_brier_score":
                    validation_metrics[
                        "brier_score"
                    ],
                "test_brier_score":
                    test_metrics[
                        "brier_score"
                    ],
                "validation_precision":
                    validation_metrics[
                        "precision"
                    ],
                "validation_recall":
                    validation_metrics[
                        "recall"
                    ],
                "validation_f1":
                    validation_metrics[
                        "f1"
                    ],
                "test_precision":
                    test_metrics[
                        "precision"
                    ],
                "test_recall":
                    test_metrics[
                        "recall"
                    ],
                "test_f1":
                    test_metrics[
                        "f1"
                    ],
                "f1_transfer_gap":
                    test_metrics["f1"]
                    - validation_metrics["f1"],
                "test_roc_auc":
                    test_metrics[
                        "roc_auc"
                    ],
                "test_pr_auc":
                    test_metrics[
                        "pr_auc"
                    ],
                "test_false_positive":
                    test_metrics[
                        "false_positive"
                    ],
                "test_false_negative":
                    test_metrics[
                        "false_negative"
                    ],
            }
        )

        validation_threshold_table[
            "fold_number"
        ] = fold_number

        threshold_rows.append(
            validation_threshold_table
        )

        for threshold in THRESHOLDS:
            test_prediction = (
                test_probability
                >= threshold
            ).astype(int)

            metrics = (
                classification_metrics(
                    test_data[TARGET]
                    .to_numpy(dtype=int),
                    test_prediction,
                    test_probability,
                )
            )

            fixed_threshold_rows.append(
                {
                    "fold_number":
                        fold_number,
                    "threshold":
                        threshold,
                    "test_precision":
                        metrics[
                            "precision"
                        ],
                    "test_recall":
                        metrics[
                            "recall"
                        ],
                    "test_f1":
                        metrics["f1"],
                    "test_false_positive":
                        metrics[
                            "false_positive"
                        ],
                    "test_false_negative":
                        metrics[
                            "false_negative"
                        ],
                }
            )

    fold_results = pd.DataFrame(
        fold_rows
    )

    validation_threshold_results = (
        pd.concat(
            threshold_rows,
            ignore_index=True,
        )
    )

    fixed_threshold_results = (
        pd.DataFrame(
            fixed_threshold_rows
        )
    )

    selected_threshold_summary = (
        pd.DataFrame(
            [
                {
                    "fold_count":
                        len(fold_results),
                    "mean_selected_threshold":
                        fold_results[
                            "selected_threshold"
                        ].mean(),
                    "std_selected_threshold":
                        fold_results[
                            "selected_threshold"
                        ].std(),
                    "min_selected_threshold":
                        fold_results[
                            "selected_threshold"
                        ].min(),
                    "max_selected_threshold":
                        fold_results[
                            "selected_threshold"
                        ].max(),
                    "mean_validation_f1":
                        fold_results[
                            "validation_f1"
                        ].mean(),
                    "mean_test_f1":
                        fold_results[
                            "test_f1"
                        ].mean(),
                    "std_test_f1":
                        fold_results[
                            "test_f1"
                        ].std(),
                    "mean_f1_transfer_gap":
                        fold_results[
                            "f1_transfer_gap"
                        ].mean(),
                    "mean_test_calibration_gap":
                        fold_results[
                            "test_calibration_gap"
                        ].mean(),
                    "mean_test_brier_score":
                        fold_results[
                            "test_brier_score"
                        ].mean(),
                }
            ]
        )
    )

    fixed_threshold_summary = (
        fixed_threshold_results
        .groupby(
            "threshold",
            as_index=False,
        )
        .agg(
            mean_test_precision=(
                "test_precision",
                "mean",
            ),
            mean_test_recall=(
                "test_recall",
                "mean",
            ),
            mean_test_f1=(
                "test_f1",
                "mean",
            ),
            std_test_f1=(
                "test_f1",
                "std",
            ),
            mean_false_positive=(
                "test_false_positive",
                "mean",
            ),
            mean_false_negative=(
                "test_false_negative",
                "mean",
            ),
        )
        .sort_values(
            [
                "mean_test_f1",
                "mean_test_recall",
            ],
            ascending=[
                False,
                False,
            ],
        )
    )

    fold_results.to_csv(
        OUTPUT_DIR
        / "26_stockout_rolling_fold_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    selected_threshold_summary.to_csv(
        OUTPUT_DIR
        / "26_stockout_rolling_threshold_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    validation_threshold_results.to_csv(
        OUTPUT_DIR
        / "26_stockout_rolling_validation_thresholds.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fixed_threshold_results.to_csv(
        OUTPUT_DIR
        / "26_stockout_fixed_threshold_fold_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fixed_threshold_summary.to_csv(
        OUTPUT_DIR
        / "26_stockout_fixed_threshold_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n[ROLLING ORIGIN FOLD RESULTS]"
    )
    print(
        fold_results.to_string(
            index=False
        )
    )

    print(
        "\n[SELECTED THRESHOLD STABILITY]"
    )
    print(
        selected_threshold_summary
        .to_string(index=False)
    )

    print(
        "\n[FIXED THRESHOLD TEST SUMMARY]"
    )
    print(
        fixed_threshold_summary
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
