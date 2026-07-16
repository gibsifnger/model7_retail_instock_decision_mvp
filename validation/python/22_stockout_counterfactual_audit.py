import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]

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
PROBABILITY_THRESHOLD = 0.4

EVENT_KEYS = [
    "sku_id",
    "center_id",
    "week_start_date",
]

CURRENT_WEEK_FEATURES = [
    "observed_sales_qty",
    "demand_gap_qty",
    "sales_censored_flag",
    "stockout_adjusted_sales",
]

INVENTORY_DERIVED_FEATURES = [
    "inventory_cover_weeks",
    "safety_stock_qty",
]

DEMAND_HISTORY_FEATURES = [
    "sales_lag_1w",
    "sales_lag_4w",
    "sales_rolling_mean_4w",
    "sales_rolling_std_4w",
    "demand_volatility_index",
    "stockout_adjusted_sales_rolling_mean_4w",
    "stockout_days_last_4w",
    "in_stock_rate_4w",
    "sales_history_weeks",
]


def load_model_module():
    spec = importlib.util.spec_from_file_location(
        "stockout_model",
        MODEL_PATH,
    )

    if spec is None or spec.loader is None:
        raise ImportError(MODEL_PATH)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def equalize_numeric_within_event(
    data: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    result = data.copy()

    valid_columns = [
        column
        for column in columns
        if column in result.columns
    ]

    if not valid_columns:
        return result

    event_means = (
        result.groupby(EVENT_KEYS)[valid_columns]
        .transform("mean")
    )

    result[valid_columns] = event_means

    return result


def fix_channel_id(
    data: pd.DataFrame,
) -> pd.DataFrame:
    result = data.copy()

    reference_channel = sorted(
        result["channel_id"].dropna().unique()
    )[0]

    result["channel_id"] = reference_channel

    return result


def predict_probability(
    pipeline,
    data: pd.DataFrame,
    feature_columns: list[str],
) -> np.ndarray:
    return pipeline.predict_proba(
        data[feature_columns]
    )[:, 1]


def build_event_profile(
    data: pd.DataFrame,
    probability: np.ndarray,
) -> pd.DataFrame:
    result = data[
        EVENT_KEYS + [TARGET]
    ].copy()

    result["probability"] = probability

    result["predicted_label"] = (
        result["probability"]
        >= PROBABILITY_THRESHOLD
    ).astype(int)

    event = (
        result.groupby(
            EVENT_KEYS,
            as_index=False,
        )
        .agg(
            actual_target=(
                TARGET,
                "max",
            ),
            probability_min=(
                "probability",
                "min",
            ),
            probability_mean=(
                "probability",
                "mean",
            ),
            probability_max=(
                "probability",
                "max",
            ),
            label_nunique=(
                "predicted_label",
                "nunique",
            ),
        )
    )

    event["probability_range"] = (
        event["probability_max"]
        - event["probability_min"]
    )

    event["label_conflict_flag"] = (
        event["label_nunique"] > 1
    ).astype(int)

    return event


def main() -> None:
    module = load_model_module()

    data = module.load_modeling_table()

    train_data, test_data = (
        module.time_based_split(data)
    )

    feature_columns = (
        list(module.NUMERIC_FEATURES)
        + list(module.CATEGORICAL_FEATURES)
    )

    pipeline = module.build_model_pipeline()

    pipeline.fit(
        train_data[feature_columns],
        train_data[TARGET],
    )

    varying_numeric_features = []

    for feature in module.NUMERIC_FEATURES:
        if (
            test_data.groupby(EVENT_KEYS)[feature]
            .nunique(dropna=False)
            .gt(1)
            .any()
        ):
            varying_numeric_features.append(feature)

    scenarios = {
        "original": test_data.copy(),

        "channel_id_fixed_only":
            fix_channel_id(test_data),

        "inventory_derived_equalized":
            equalize_numeric_within_event(
                test_data,
                INVENTORY_DERIVED_FEATURES,
            ),

        "current_week_realized_equalized":
            equalize_numeric_within_event(
                test_data,
                CURRENT_WEEK_FEATURES,
            ),

        "demand_history_equalized":
            equalize_numeric_within_event(
                test_data,
                DEMAND_HISTORY_FEATURES,
            ),

        "inventory_and_current_week_equalized":
            equalize_numeric_within_event(
                equalize_numeric_within_event(
                    test_data,
                    INVENTORY_DERIVED_FEATURES,
                ),
                CURRENT_WEEK_FEATURES,
            ),

        "all_varying_numeric_equalized_keep_channel":
            equalize_numeric_within_event(
                test_data,
                varying_numeric_features,
            ),

        "all_channel_variation_equalized":
            fix_channel_id(
                equalize_numeric_within_event(
                    test_data,
                    varying_numeric_features,
                )
            ),
    }

    event_tables = {}
    summary_rows = []

    for scenario_name, scenario_data in scenarios.items():
        probability = predict_probability(
            pipeline,
            scenario_data,
            feature_columns,
        )

        event_profile = build_event_profile(
            scenario_data,
            probability,
        )

        event_tables[scenario_name] = event_profile

        summary_rows.append(
            {
                "scenario": scenario_name,
                "event_count": len(
                    event_profile
                ),
                "conflict_event_count": int(
                    event_profile[
                        "label_conflict_flag"
                    ].sum()
                ),
                "conflict_event_rate": float(
                    event_profile[
                        "label_conflict_flag"
                    ].mean()
                ),
                "mean_probability_range": float(
                    event_profile[
                        "probability_range"
                    ].mean()
                ),
                "median_probability_range": float(
                    event_profile[
                        "probability_range"
                    ].median()
                ),
                "max_probability_range": float(
                    event_profile[
                        "probability_range"
                    ].max()
                ),
            }
        )

    original_events = event_tables[
        "original"
    ][
        EVENT_KEYS
        + [
            "label_conflict_flag",
            "probability_range",
        ]
    ].rename(
        columns={
            "label_conflict_flag":
                "original_conflict_flag",
            "probability_range":
                "original_probability_range",
        }
    )

    comparison_rows = []

    for scenario_name, event_profile in (
        event_tables.items()
    ):
        comparison = event_profile.merge(
            original_events,
            on=EVENT_KEYS,
            how="left",
            validate="one_to_one",
        )

        original_conflicts = comparison.loc[
            comparison[
                "original_conflict_flag"
            ].eq(1)
        ]

        comparison_rows.append(
            {
                "scenario": scenario_name,
                "original_conflict_events": int(
                    comparison[
                        "original_conflict_flag"
                    ].sum()
                ),
                "resolved_original_conflicts": int(
                    (
                        original_conflicts[
                            "label_conflict_flag"
                        ].eq(0)
                    ).sum()
                ),
                "remaining_original_conflicts": int(
                    (
                        original_conflicts[
                            "label_conflict_flag"
                        ].eq(1)
                    ).sum()
                ),
                "new_conflicts_created": int(
                    (
                        comparison[
                            "original_conflict_flag"
                        ].eq(0)
                        & comparison[
                            "label_conflict_flag"
                        ].eq(1)
                    ).sum()
                ),
                "mean_range_on_original_conflicts":
                    float(
                        original_conflicts[
                            "probability_range"
                        ].mean()
                    ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    comparison_summary = pd.DataFrame(
        comparison_rows
    )

    varying_features = pd.DataFrame(
        {
            "varying_numeric_feature":
                varying_numeric_features
        }
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        OUTPUT_DIR
        / "22_stockout_counterfactual_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    comparison_summary.to_csv(
        OUTPUT_DIR
        / "22_stockout_counterfactual_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    varying_features.to_csv(
        OUTPUT_DIR
        / "22_stockout_varying_numeric_features.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n[STOCKOUT COUNTERFACTUAL SUMMARY]"
    )
    print(
        summary.to_string(index=False)
    )

    print(
        "\n[STOCKOUT COUNTERFACTUAL COMPARISON]"
    )
    print(
        comparison_summary.to_string(
            index=False
        )
    )

    print(
        "\n[VARYING NUMERIC FEATURES]"
    )
    print(
        varying_features.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
