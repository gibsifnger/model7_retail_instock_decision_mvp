import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


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
PERMUTATION_REPEATS = 5
RANDOM_SEED = 42

REMOVED_CURRENT_WEEK_FEATURES = {
    "observed_sales_qty",
    "demand_gap_qty",
    "sales_censored_flag",
    "stockout_adjusted_sales",
}


def load_model_module():
    spec = importlib.util.spec_from_file_location(
        "demand_model",
        MODEL_PATH,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Cannot load model module: {MODEL_PATH}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def calculate_wape(
    actual: np.ndarray,
    prediction: np.ndarray,
) -> float:
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)

    denominator = actual.sum()

    if denominator == 0:
        return np.nan

    return float(
        np.abs(prediction - actual).sum()
        / denominator
    )


def assign_feature_group(
    feature: str,
    categorical_features: set[str],
) -> str:
    name = feature.lower()

    if (
        name.startswith("sales_lag_")
        or name.startswith("sales_rolling_")
        or "demand_volatility" in name
        or "stockout_adjusted_sales_rolling" in name
        or "stockout_days_last" in name
        or "in_stock_rate" in name
        or "sales_history" in name
    ):
        return "demand_history"

    if (
        name.startswith("promo_")
        or "historical_promo" in name
    ):
        return "promotion"

    if (
        "available_qty" in name
        or "reserved_qty" in name
        or name.startswith("inbound_qty")
        or name.startswith("inventory_")
        or "safety_stock" in name
        or "open_po" in name
        or "overdue_po" in name
    ):
        return "inventory_supply"

    if (
        "vendor" in name
        or "lead_time" in name
        or "fill_rate" in name
        or "on_time_delivery" in name
        or "reliability" in name
        or "import_flag" in name
        or "moq" in name
    ):
        return "vendor_leadtime"

    if "override" in name:
        return "manual_override"

    if (
        "week_of_year" in name
        or name in {"year", "month", "quarter"}
        or "season" in name
        or name.endswith("_sin")
        or name.endswith("_cos")
    ):
        return "calendar"

    if name in {
        "sku_id",
        "channel_id",
        "center_id",
    }:
        return "identity_structure"

    if (
        "category" in name
        or "brand" in name
        or "product" in name
        or "abc" in name
        or "xyz" in name
        or "shelf_life" in name
    ):
        return "product_static"

    if feature in categorical_features:
        return "other_categorical_context"

    return "other_numeric_context"


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    module = load_model_module()

    safe_numeric_features = [
        feature
        for feature in module.NUMERIC_FEATURES
        if feature
        not in REMOVED_CURRENT_WEEK_FEATURES
    ]

    categorical_features = list(
        module.CATEGORICAL_FEATURES
    )

    feature_columns = (
        safe_numeric_features
        + categorical_features
    )

    # 모델 Pipeline이 Safe Feature 목록을 사용하도록 설정
    module.NUMERIC_FEATURES = safe_numeric_features
    module.CATEGORICAL_FEATURES = (
        categorical_features
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

    unique_weeks = sorted(
        data["week_start_date"].unique()
    )

    test_weeks = unique_weeks[-TEST_WEEKS:]
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

    train_features = train_data[
        feature_columns
    ].copy()

    test_features = test_data[
        feature_columns
    ].copy()

    train_target = train_data[TARGET_COLUMN]
    test_target = test_data[
        TARGET_COLUMN
    ].to_numpy(dtype=float)

    pipeline = module.build_model_pipeline()

    pipeline.fit(
        train_features,
        train_target,
    )

    base_prediction = np.clip(
        pipeline.predict(test_features),
        a_min=0,
        a_max=None,
    )

    base_wape = calculate_wape(
        test_target,
        base_prediction,
    )

    categorical_set = set(
        categorical_features
    )

    feature_map = pd.DataFrame(
        {
            "feature": feature_columns,
            "feature_type": [
                (
                    "categorical"
                    if feature in categorical_set
                    else "numeric"
                )
                for feature in feature_columns
            ],
            "feature_group": [
                assign_feature_group(
                    feature,
                    categorical_set,
                )
                for feature in feature_columns
            ],
        }
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    result_rows = []

    for group_name, group_table in (
        feature_map.groupby(
            "feature_group",
            sort=True,
        )
    ):
        group_features = (
            group_table["feature"].tolist()
        )

        repeated_wapes = []

        for _ in range(
            PERMUTATION_REPEATS
        ):
            permutation = rng.permutation(
                len(test_features)
            )

            permuted_features = (
                test_features.copy()
            )

            # 그룹 내부 Feature 간 관계는 유지하고
            # Target과의 행 대응만 끊는다.
            for feature in group_features:
                permuted_features[feature] = (
                    test_features[feature]
                    .iloc[permutation]
                    .to_numpy()
                )

            prediction = np.clip(
                pipeline.predict(
                    permuted_features
                ),
                a_min=0,
                a_max=None,
            )

            repeated_wapes.append(
                calculate_wape(
                    test_target,
                    prediction,
                )
            )

        mean_permuted_wape = float(
            np.mean(repeated_wapes)
        )

        std_permuted_wape = float(
            np.std(repeated_wapes)
        )

        wape_increase = (
            mean_permuted_wape
            - base_wape
        )

        result_rows.append(
            {
                "feature_group": group_name,
                "feature_count": len(
                    group_features
                ),
                "features": " | ".join(
                    group_features
                ),
                "base_wape": base_wape,
                "permuted_wape_mean":
                    mean_permuted_wape,
                "permuted_wape_std":
                    std_permuted_wape,
                "wape_increase":
                    wape_increase,
                "relative_wape_degradation":
                    (
                        wape_increase
                        / base_wape
                        if base_wape > 0
                        else np.nan
                    ),
            }
        )

    summary = (
        pd.DataFrame(result_rows)
        .sort_values(
            "wape_increase",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    summary.to_csv(
        OUTPUT_DIR
        / "16_demand_group_permutation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    feature_map.to_csv(
        OUTPUT_DIR
        / "16_demand_feature_group_map.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n[DEMAND GROUP PERMUTATION SUMMARY]"
    )

    print(
        summary[
            [
                "feature_group",
                "feature_count",
                "base_wape",
                "permuted_wape_mean",
                "permuted_wape_std",
                "wape_increase",
                "relative_wape_degradation",
            ]
        ].to_string(index=False)
    )

    print(
        "\n[FEATURE GROUP MAP]"
    )

    print(
        feature_map.to_string(index=False)
    )


if __name__ == "__main__":
    main()
