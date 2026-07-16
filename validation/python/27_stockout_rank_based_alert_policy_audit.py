import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


REPO_ROOT = Path(__file__).resolve().parents[2]

ROLLING_AUDIT_PATH = (
    REPO_ROOT
    / "validation"
    / "python"
    / "26_stockout_rolling_origin_audit.py"
)

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


def load_rolling_audit_module():
    spec = importlib.util.spec_from_file_location(
        "rolling_stockout_audit",
        ROLLING_AUDIT_PATH,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Cannot load: {ROLLING_AUDIT_PATH}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def build_rule_baseline(
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


def build_rank_prediction(
    data: pd.DataFrame,
    probability: np.ndarray,
    group_columns: list[str],
    top_percent: float | None = None,
    top_count: int | None = None,
) -> np.ndarray:
    if (
        (top_percent is None and top_count is None)
        or (
            top_percent is not None
            and top_count is not None
        )
    ):
        raise ValueError(
            "Specify exactly one of "
            "top_percent or top_count."
        )

    identity_columns = list(
        dict.fromkeys(
            group_columns
            + [
                "sku_id",
                "center_id",
            ]
        )
    )

    work = data[
        identity_columns
    ].copy()

    work["row_position"] = np.arange(
        len(work)
    )

    work["probability"] = np.asarray(
        probability,
        dtype=float,
    )

    prediction = np.zeros(
        len(work),
        dtype=int,
    )

    grouped = work.groupby(
        group_columns,
        sort=True,
        dropna=False,
    )

    for _, group in grouped:
        if top_percent is not None:
            selected_count = max(
                1,
                int(
                    math.ceil(
                        len(group)
                        * top_percent
                    )
                ),
            )
        else:
            selected_count = min(
                int(top_count),
                len(group),
            )

        selected_rows = (
            group.sort_values(
                [
                    "probability",
                    "sku_id",
                    "center_id",
                ],
                ascending=[
                    False,
                    True,
                    True,
                ],
                kind="stable",
            )
            .head(selected_count)[
                "row_position"
            ]
            .to_numpy(dtype=int)
        )

        prediction[selected_rows] = 1

    return prediction


def calculate_policy_metrics(
    actual: np.ndarray,
    prediction: np.ndarray,
    data: pd.DataFrame,
) -> dict[str, float | int]:
    actual = np.asarray(
        actual,
        dtype=int,
    )

    prediction = np.asarray(
        prediction,
        dtype=int,
    )

    tn, fp, fn, tp = confusion_matrix(
        actual,
        prediction,
        labels=[0, 1],
    ).ravel()

    weekly_alerts = (
        pd.DataFrame(
            {
                "week_start_date":
                    data[
                        "week_start_date"
                    ].to_numpy(),
                "alert": prediction,
            }
        )
        .groupby(
            "week_start_date"
        )["alert"]
        .sum()
    )

    center_week_alerts = (
        pd.DataFrame(
            {
                "week_start_date":
                    data[
                        "week_start_date"
                    ].to_numpy(),
                "center_id":
                    data[
                        "center_id"
                    ].to_numpy(),
                "alert": prediction,
            }
        )
        .groupby(
            [
                "week_start_date",
                "center_id",
            ]
        )["alert"]
        .sum()
    )

    actual_positive_rate = float(
        actual.mean()
    )

    precision = float(
        precision_score(
            actual,
            prediction,
            zero_division=0,
        )
    )

    return {
        "rows": len(actual),
        "actual_positive_rate":
            actual_positive_rate,
        "alert_count": int(
            prediction.sum()
        ),
        "average_alerts_per_week": float(
            weekly_alerts.mean()
        ),
        "std_alerts_per_week": float(
            weekly_alerts.std(ddof=0)
        ),
        "min_alerts_per_week": int(
            weekly_alerts.min()
        ),
        "max_alerts_per_week": int(
            weekly_alerts.max()
        ),
        "max_alerts_per_center_week": int(
            center_week_alerts.max()
        ),
        "precision": precision,
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
        "precision_lift_vs_base_rate": (
            precision
            / actual_positive_rate
            if actual_positive_rate > 0
            else np.nan
        ),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_negative": int(tn),
        "alerts_per_true_positive": (
            float(
                prediction.sum() / tp
            )
            if tp > 0
            else np.nan
        ),
    }


def build_weekly_metrics(
    fold_number: int,
    policy_name: str,
    data: pd.DataFrame,
    actual: np.ndarray,
    prediction: np.ndarray,
) -> list[dict[str, object]]:
    rows = []

    actual_series = pd.Series(
        actual,
        index=data.index,
    )

    prediction_series = pd.Series(
        prediction,
        index=data.index,
    )

    for week, week_data in data.groupby(
        "week_start_date",
        sort=True,
    ):
        index = week_data.index

        week_actual = actual_series.loc[
            index
        ].to_numpy(dtype=int)

        week_prediction = (
            prediction_series.loc[
                index
            ].to_numpy(dtype=int)
        )

        tn, fp, fn, tp = confusion_matrix(
            week_actual,
            week_prediction,
            labels=[0, 1],
        ).ravel()

        rows.append(
            {
                "fold_number":
                    fold_number,
                "policy":
                    policy_name,
                "week_start_date":
                    week,
                "actual_positive":
                    int(
                        week_actual.sum()
                    ),
                "alert_count":
                    int(
                        week_prediction.sum()
                    ),
                "precision":
                    float(
                        precision_score(
                            week_actual,
                            week_prediction,
                            zero_division=0,
                        )
                    ),
                "recall":
                    float(
                        recall_score(
                            week_actual,
                            week_prediction,
                            zero_division=0,
                        )
                    ),
                "f1":
                    float(
                        f1_score(
                            week_actual,
                            week_prediction,
                            zero_division=0,
                        )
                    ),
                "true_positive":
                    int(tp),
                "false_positive":
                    int(fp),
                "false_negative":
                    int(fn),
                "true_negative":
                    int(tn),
            }
        )

    return rows


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            INPUT_PATH
        )

    if not ROLLING_AUDIT_PATH.exists():
        raise FileNotFoundError(
            ROLLING_AUDIT_PATH
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    module = load_rolling_audit_module()

    target = module.TARGET

    data = pd.read_csv(
        INPUT_PATH,
        parse_dates=[
            "week_start_date",
        ],
    )

    data[target] = pd.to_numeric(
        data[target],
        errors="coerce",
    )

    data = data.loc[
        data[target].notna()
    ].copy()

    data[target] = (
        data[target].astype(int)
    )

    folds = module.build_folds(data)

    fold_rows = []
    weekly_rows = []
    selected_threshold_rows = []

    for fold in folds:
        fold_number = int(
            fold["fold_number"]
        )

        train_data = (
            fold["train"]
            .reset_index(drop=True)
        )

        validation_data = (
            fold["validation"]
            .reset_index(drop=True)
        )

        test_data = (
            fold["test"]
            .reset_index(drop=True)
        )

        pipeline = module.build_pipeline(
            module.RANDOM_SEED
            + fold_number
        )

        pipeline.fit(
            module.prepare_features(
                train_data
            ),
            train_data[target],
        )

        validation_probability = (
            pipeline.predict_proba(
                module.prepare_features(
                    validation_data
                )
            )[:, 1]
        )

        test_probability = (
            pipeline.predict_proba(
                module.prepare_features(
                    test_data
                )
            )[:, 1]
        )

        (
            selected_threshold,
            _,
        ) = module.select_threshold(
            validation_data[target]
            .to_numpy(dtype=int),
            validation_probability,
        )

        selected_threshold_rows.append(
            {
                "fold_number":
                    fold_number,
                "selected_threshold":
                    selected_threshold,
                "test_week_min":
                    test_data[
                        "week_start_date"
                    ].min(),
                "test_week_max":
                    test_data[
                        "week_start_date"
                    ].max(),
            }
        )

        policies = {
            "validation_selected_threshold":
                (
                    test_probability
                    >= selected_threshold
                ).astype(int),

            "fixed_threshold_0_05":
                (
                    test_probability
                    >= 0.05
                ).astype(int),

            "fixed_threshold_0_10":
                (
                    test_probability
                    >= 0.10
                ).astype(int),

            "fixed_threshold_0_15":
                (
                    test_probability
                    >= 0.15
                ).astype(int),

            "fixed_threshold_0_20":
                (
                    test_probability
                    >= 0.20
                ).astype(int),

            "global_weekly_top_05pct":
                build_rank_prediction(
                    test_data,
                    test_probability,
                    group_columns=[
                        "week_start_date",
                    ],
                    top_percent=0.05,
                ),

            "global_weekly_top_10pct":
                build_rank_prediction(
                    test_data,
                    test_probability,
                    group_columns=[
                        "week_start_date",
                    ],
                    top_percent=0.10,
                ),

            "global_weekly_top_15pct":
                build_rank_prediction(
                    test_data,
                    test_probability,
                    group_columns=[
                        "week_start_date",
                    ],
                    top_percent=0.15,
                ),

            "center_weekly_top_05pct":
                build_rank_prediction(
                    test_data,
                    test_probability,
                    group_columns=[
                        "week_start_date",
                        "center_id",
                    ],
                    top_percent=0.05,
                ),

            "center_weekly_top_10pct":
                build_rank_prediction(
                    test_data,
                    test_probability,
                    group_columns=[
                        "week_start_date",
                        "center_id",
                    ],
                    top_percent=0.10,
                ),

            "center_weekly_top_15pct":
                build_rank_prediction(
                    test_data,
                    test_probability,
                    group_columns=[
                        "week_start_date",
                        "center_id",
                    ],
                    top_percent=0.15,
                ),

            "global_weekly_top_10":
                build_rank_prediction(
                    test_data,
                    test_probability,
                    group_columns=[
                        "week_start_date",
                    ],
                    top_count=10,
                ),

            "global_weekly_top_20":
                build_rank_prediction(
                    test_data,
                    test_probability,
                    group_columns=[
                        "week_start_date",
                    ],
                    top_count=20,
                ),

            "global_weekly_top_30":
                build_rank_prediction(
                    test_data,
                    test_probability,
                    group_columns=[
                        "week_start_date",
                    ],
                    top_count=30,
                ),

            "center_rule_baseline":
                build_rule_baseline(
                    test_data
                ),
        }

        actual = test_data[
            target
        ].to_numpy(dtype=int)

        for (
            policy_name,
            prediction,
        ) in policies.items():
            metrics = (
                calculate_policy_metrics(
                    actual,
                    prediction,
                    test_data,
                )
            )

            fold_rows.append(
                {
                    "fold_number":
                        fold_number,
                    "policy":
                        policy_name,
                    "selected_threshold":
                        (
                            selected_threshold
                            if policy_name
                            == "validation_selected_threshold"
                            else np.nan
                        ),
                    "test_week_min":
                        test_data[
                            "week_start_date"
                        ].min(),
                    "test_week_max":
                        test_data[
                            "week_start_date"
                        ].max(),
                    **metrics,
                }
            )

            weekly_rows.extend(
                build_weekly_metrics(
                    fold_number,
                    policy_name,
                    test_data,
                    actual,
                    prediction,
                )
            )

    fold_results = pd.DataFrame(
        fold_rows
    )

    weekly_results = pd.DataFrame(
        weekly_rows
    )

    selected_thresholds = pd.DataFrame(
        selected_threshold_rows
    )

    policy_summary = (
        fold_results.groupby(
            "policy",
            as_index=False,
        )
        .agg(
            fold_count=(
                "fold_number",
                "nunique",
            ),
            mean_precision=(
                "precision",
                "mean",
            ),
            mean_recall=(
                "recall",
                "mean",
            ),
            mean_f1=(
                "f1",
                "mean",
            ),
            std_f1=(
                "f1",
                "std",
            ),
            mean_precision_lift=(
                "precision_lift_vs_base_rate",
                "mean",
            ),
            mean_alerts_per_week=(
                "average_alerts_per_week",
                "mean",
            ),
            std_alerts_per_week=(
                "std_alerts_per_week",
                "mean",
            ),
            max_alerts_per_week=(
                "max_alerts_per_week",
                "max",
            ),
            max_alerts_per_center_week=(
                "max_alerts_per_center_week",
                "max",
            ),
            mean_false_positive=(
                "false_positive",
                "mean",
            ),
            mean_false_negative=(
                "false_negative",
                "mean",
            ),
            mean_alerts_per_true_positive=(
                "alerts_per_true_positive",
                "mean",
            ),
        )
        .sort_values(
            [
                "mean_f1",
                "std_f1",
                "mean_recall",
            ],
            ascending=[
                False,
                True,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    fold_results.to_csv(
        OUTPUT_DIR
        / "27_stockout_rank_policy_fold_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    policy_summary.to_csv(
        OUTPUT_DIR
        / "27_stockout_rank_policy_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    weekly_results.to_csv(
        OUTPUT_DIR
        / "27_stockout_rank_policy_weekly_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    selected_thresholds.to_csv(
        OUTPUT_DIR
        / "27_stockout_rank_policy_selected_thresholds.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n[RANK-BASED ALERT POLICY SUMMARY]"
    )
    print(
        policy_summary.to_string(
            index=False
        )
    )

    print(
        "\n[POLICY FOLD RESULTS]"
    )
    print(
        fold_results[
            [
                "fold_number",
                "policy",
                "precision",
                "recall",
                "f1",
                "average_alerts_per_week",
                "std_alerts_per_week",
                "false_positive",
                "false_negative",
            ]
        ]
        .sort_values(
            [
                "fold_number",
                "f1",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .to_string(index=False)
    )

    print(
        "\n[SELECTED THRESHOLDS]"
    )
    print(
        selected_thresholds.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
