import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]

PREDICTION_PATH = (
    REPO_ROOT
    / "outputs"
    / "predictions"
    / "stockout_risk_result.csv"
)

FEATURE_PATH = (
    REPO_ROOT
    / "data"
    / "mart"
    / "final_modeling_table.csv"
)

MODEL_PATH = (
    REPO_ROOT
    / "src"
    / "models"
    / "train_stockout_classifier.py"
)

OUTPUT_DIR = (
    REPO_ROOT
    / "outputs"
    / "validation"
)

TARGET = "target_stockout_risk_next_2w"
PROBABILITY = "stockout_risk_pred_proba"
THRESHOLD = 0.4

ROW_KEYS = [
    "sku_id",
    "channel_id",
    "center_id",
    "week_start_date",
]

EVENT_KEYS = [
    "sku_id",
    "center_id",
    "week_start_date",
]

IMPORTANT_DETAIL_FEATURES = [
    "observed_sales_qty",
    "demand_gap_qty",
    "sales_censored_flag",
    "stockout_adjusted_sales",
    "sales_lag_1w",
    "sales_lag_4w",
    "sales_rolling_mean_4w",
    "sales_rolling_std_4w",
    "stockout_adjusted_sales_rolling_mean_4w",
    "stockout_days_last_4w",
    "in_stock_rate_4w",
    "promo_flag",
    "promo_depth",
    "promo_type",
    "available_qty",
    "reserved_qty",
    "inbound_qty_next_1w",
    "inbound_qty_next_2w",
    "inventory_position_qty",
    "inventory_cover_weeks",
    "safety_stock_qty",
    "open_po_qty",
    "overdue_po_qty",
]


def load_model_module():
    spec = importlib.util.spec_from_file_location(
        "stockout_model",
        MODEL_PATH,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Cannot load model module: {MODEL_PATH}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def safe_spearman(
    left: pd.Series,
    right: pd.Series,
) -> float:
    valid = pd.DataFrame(
        {
            "left": pd.to_numeric(
                left,
                errors="coerce",
            ),
            "right": pd.to_numeric(
                right,
                errors="coerce",
            ),
        }
    ).dropna()

    if (
        len(valid) < 3
        or valid["left"].nunique() < 2
        or valid["right"].nunique() < 2
    ):
        return np.nan

    return float(
        valid["left"].corr(
            valid["right"],
            method="spearman",
        )
    )


def main() -> None:
    for path in [
        PREDICTION_PATH,
        FEATURE_PATH,
        MODEL_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_module = load_model_module()

    numeric_features = list(
        model_module.NUMERIC_FEATURES
    )

    categorical_features = list(
        model_module.CATEGORICAL_FEATURES
    )

    predictions = pd.read_csv(
        PREDICTION_PATH,
    )

    features = pd.read_csv(
        FEATURE_PATH,
    )

    for table in [predictions, features]:
        table["week_start_date"] = pd.to_datetime(
            table["week_start_date"],
            errors="raise",
        )

    duplicate_feature_keys = int(
        features.duplicated(
            ROW_KEYS
        ).sum()
    )

    if duplicate_feature_keys:
        raise ValueError(
            "Feature table contains duplicate row keys: "
            f"{duplicate_feature_keys}"
        )

    missing_feature_columns = [
        feature
        for feature in (
            numeric_features
            + categorical_features
        )
        if feature not in predictions.columns
        and feature in features.columns
    ]

    feature_join = features[
        ROW_KEYS
        + missing_feature_columns
    ].copy()

    data = predictions.merge(
        feature_join,
        on=ROW_KEYS,
        how="left",
        validate="one_to_one",
    )

    required = {
        TARGET,
        PROBABILITY,
    }

    missing_required = sorted(
        required.difference(
            data.columns
        )
    )

    if missing_required:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_required)
        )

    data[TARGET] = pd.to_numeric(
        data[TARGET],
        errors="raise",
    ).astype(int)

    data[PROBABILITY] = pd.to_numeric(
        data[PROBABILITY],
        errors="raise",
    )

    data["predicted_label"] = (
        data[PROBABILITY] >= THRESHOLD
    ).astype(int)

    event_summary = (
        data.groupby(
            EVENT_KEYS,
            as_index=False,
        )
        .agg(
            channel_count=(
                "channel_id",
                "nunique",
            ),
            actual_target=(
                TARGET,
                "max",
            ),
            probability_min=(
                PROBABILITY,
                "min",
            ),
            probability_mean=(
                PROBABILITY,
                "mean",
            ),
            probability_max=(
                PROBABILITY,
                "max",
            ),
            predicted_label_nunique=(
                "predicted_label",
                "nunique",
            ),
            predicted_positive_channels=(
                "predicted_label",
                "sum",
            ),
        )
    )

    event_summary["probability_range"] = (
        event_summary["probability_max"]
        - event_summary["probability_min"]
    )

    event_summary["label_conflict_flag"] = (
        event_summary[
            "predicted_label_nunique"
        ] > 1
    ).astype(int)

    event_summary = event_summary.sort_values(
        [
            "label_conflict_flag",
            "probability_range",
        ],
        ascending=[
            False,
            False,
        ],
    ).reset_index(drop=True)

    conflict_events = event_summary.loc[
        event_summary[
            "label_conflict_flag"
        ].eq(1)
    ].copy()

    conflict_rows = data.merge(
        conflict_events[
            EVENT_KEYS
            + [
                "probability_range",
                "predicted_positive_channels",
            ]
        ],
        on=EVENT_KEYS,
        how="inner",
        validate="many_to_one",
    )

    conflict_rows["probability_rank_in_event"] = (
        conflict_rows.groupby(
            EVENT_KEYS
        )[PROBABILITY]
        .rank(
            method="first",
            ascending=False,
        )
        .astype(int)
    )

    detail_columns = (
        ROW_KEYS
        + [
            TARGET,
            PROBABILITY,
            "predicted_label",
            "probability_rank_in_event",
            "probability_range",
            "predicted_positive_channels",
        ]
        + [
            column
            for column in IMPORTANT_DETAIL_FEATURES
            if column in conflict_rows.columns
        ]
    )

    conflict_detail = conflict_rows[
        detail_columns
    ].sort_values(
        EVENT_KEYS
        + [
            PROBABILITY,
        ],
        ascending=[
            True,
            True,
            True,
            False,
        ],
    )

    numeric_rows = []

    for feature in numeric_features:
        if feature not in data.columns:
            continue

        data[feature] = pd.to_numeric(
            data[feature],
            errors="coerce",
        )

        feature_event = (
            data.groupby(
                EVENT_KEYS,
                as_index=False,
            )[feature]
            .agg(
                feature_min="min",
                feature_mean="mean",
                feature_max="max",
                feature_std="std",
                feature_nunique="nunique",
            )
        )

        feature_event["feature_range"] = (
            feature_event["feature_max"]
            - feature_event["feature_min"]
        )

        analysis = event_summary[
            EVENT_KEYS
            + [
                "probability_range",
                "label_conflict_flag",
            ]
        ].merge(
            feature_event,
            on=EVENT_KEYS,
            how="left",
            validate="one_to_one",
        )

        conflict_part = analysis.loc[
            analysis[
                "label_conflict_flag"
            ].eq(1)
        ]

        nonconflict_part = analysis.loc[
            analysis[
                "label_conflict_flag"
            ].eq(0)
        ]

        conflict_mean_range = float(
            conflict_part[
                "feature_range"
            ].mean()
        )

        nonconflict_mean_range = float(
            nonconflict_part[
                "feature_range"
            ].mean()
        )

        if nonconflict_mean_range > 0:
            range_ratio = (
                conflict_mean_range
                / nonconflict_mean_range
            )
        else:
            range_ratio = np.nan

        numeric_rows.append(
            {
                "feature": feature,
                "all_event_mean_range": float(
                    analysis[
                        "feature_range"
                    ].mean()
                ),
                "conflict_event_mean_range":
                    conflict_mean_range,
                "nonconflict_event_mean_range":
                    nonconflict_mean_range,
                "conflict_to_nonconflict_range_ratio":
                    range_ratio,
                "conflict_variation_rate": float(
                    conflict_part[
                        "feature_range"
                    ].gt(1e-12).mean()
                ),
                "nonconflict_variation_rate": float(
                    nonconflict_part[
                        "feature_range"
                    ].gt(1e-12).mean()
                ),
                "spearman_with_probability_range":
                    safe_spearman(
                        analysis["feature_range"],
                        analysis["probability_range"],
                    ),
            }
        )

    numeric_summary = pd.DataFrame(
        numeric_rows
    )

    if not numeric_summary.empty:
        numeric_summary[
            "absolute_spearman"
        ] = numeric_summary[
            "spearman_with_probability_range"
        ].abs()

        numeric_summary = numeric_summary.sort_values(
            [
                "absolute_spearman",
                "conflict_variation_rate",
            ],
            ascending=[
                False,
                False,
            ],
        ).reset_index(drop=True)

    categorical_rows = []

    ignored_identity_columns = {
        "sku_id",
        "center_id",
        "channel_id",
    }

    for feature in categorical_features:
        if (
            feature not in data.columns
            or feature in ignored_identity_columns
        ):
            continue

        feature_event = (
            data.groupby(
                EVENT_KEYS,
                as_index=False,
            )[feature]
            .agg(
                feature_nunique="nunique",
            )
        )

        feature_event[
            "feature_variation_flag"
        ] = (
            feature_event[
                "feature_nunique"
            ] > 1
        ).astype(int)

        analysis = event_summary[
            EVENT_KEYS
            + [
                "label_conflict_flag",
                "probability_range",
            ]
        ].merge(
            feature_event,
            on=EVENT_KEYS,
            how="left",
            validate="one_to_one",
        )

        conflict_part = analysis.loc[
            analysis[
                "label_conflict_flag"
            ].eq(1)
        ]

        nonconflict_part = analysis.loc[
            analysis[
                "label_conflict_flag"
            ].eq(0)
        ]

        categorical_rows.append(
            {
                "feature": feature,
                "conflict_variation_rate": float(
                    conflict_part[
                        "feature_variation_flag"
                    ].mean()
                ),
                "nonconflict_variation_rate": float(
                    nonconflict_part[
                        "feature_variation_flag"
                    ].mean()
                ),
                "variation_rate_difference": float(
                    conflict_part[
                        "feature_variation_flag"
                    ].mean()
                    - nonconflict_part[
                        "feature_variation_flag"
                    ].mean()
                ),
                "mean_probability_range_when_varied": float(
                    analysis.loc[
                        analysis[
                            "feature_variation_flag"
                        ].eq(1),
                        "probability_range",
                    ].mean()
                ),
            }
        )

    categorical_summary = pd.DataFrame(
        categorical_rows
    )

    if not categorical_summary.empty:
        categorical_summary = (
            categorical_summary.sort_values(
                [
                    "variation_rate_difference",
                    "conflict_variation_rate",
                ],
                ascending=[
                    False,
                    False,
                ],
            )
            .reset_index(drop=True)
        )

    high_level_summary = pd.DataFrame(
        [
            {
                "metric": "center_event_count",
                "value": len(
                    event_summary
                ),
            },
            {
                "metric": "label_conflict_event_count",
                "value": len(
                    conflict_events
                ),
            },
            {
                "metric": "label_conflict_event_rate",
                "value": (
                    len(conflict_events)
                    / len(event_summary)
                ),
            },
            {
                "metric": "mean_probability_range_all_events",
                "value": float(
                    event_summary[
                        "probability_range"
                    ].mean()
                ),
            },
            {
                "metric": "mean_probability_range_conflict_events",
                "value": float(
                    conflict_events[
                        "probability_range"
                    ].mean()
                ),
            },
            {
                "metric": "max_probability_range",
                "value": float(
                    event_summary[
                        "probability_range"
                    ].max()
                ),
            },
        ]
    )

    high_level_summary.to_csv(
        OUTPUT_DIR
        / "21_stockout_conflict_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    event_summary.to_csv(
        OUTPUT_DIR
        / "21_stockout_event_probability_range.csv",
        index=False,
        encoding="utf-8-sig",
    )

    conflict_detail.to_csv(
        OUTPUT_DIR
        / "21_stockout_conflict_event_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )

    numeric_summary.to_csv(
        OUTPUT_DIR
        / "21_stockout_numeric_feature_association.csv",
        index=False,
        encoding="utf-8-sig",
    )

    categorical_summary.to_csv(
        OUTPUT_DIR
        / "21_stockout_categorical_feature_association.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n[STOCKOUT CONFLICT SUMMARY]"
    )
    print(
        high_level_summary.to_string(
            index=False
        )
    )

    print(
        "\n[TOP NUMERIC FEATURE ASSOCIATIONS]"
    )
    print(
        numeric_summary.head(15).to_string(
            index=False
        )
    )

    print(
        "\n[CATEGORICAL FEATURE ASSOCIATIONS]"
    )
    print(
        categorical_summary.to_string(
            index=False
        )
    )

    print(
        "\n[TOP CONFLICT EVENTS]"
    )
    print(
        event_summary.head(15).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
