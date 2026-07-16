from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


REPO_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    REPO_ROOT
    / "outputs"
    / "predictions"
    / "stockout_risk_result.csv"
)

OUTPUT_DIR = (
    REPO_ROOT
    / "outputs"
    / "validation"
)

TARGET = "target_stockout_risk_next_2w"
PROBABILITY = "stockout_risk_pred_proba"
BASELINE = "baseline_risk_pred"

ROW_KEYS = [
    "sku_id",
    "channel_id",
    "center_id",
    "week_start_date",
]

CENTER_EVENT_KEYS = [
    "sku_id",
    "center_id",
    "week_start_date",
]

THRESHOLDS = [0.4, 0.5]


def classification_metrics(
    actual: np.ndarray,
    prediction: np.ndarray,
    score: np.ndarray | None = None,
) -> dict[str, float | int]:
    actual = np.asarray(actual, dtype=int)
    prediction = np.asarray(prediction, dtype=int)

    tn, fp, fn, tp = confusion_matrix(
        actual,
        prediction,
        labels=[0, 1],
    ).ravel()

    result = {
        "rows_or_events": len(actual),
        "actual_positive_rate": float(actual.mean()),
        "predicted_positive_rate": float(prediction.mean()),
        "accuracy": float(accuracy_score(actual, prediction)),
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

    if score is not None and np.unique(actual).size == 2:
        score = np.asarray(score, dtype=float)

        result["roc_auc"] = float(
            roc_auc_score(actual, score)
        )

        result["pr_auc"] = float(
            average_precision_score(actual, score)
        )
    else:
        result["roc_auc"] = np.nan
        result["pr_auc"] = np.nan

    return result


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = pd.read_csv(INPUT_PATH)

    required = set(
        ROW_KEYS
        + [
            TARGET,
            PROBABILITY,
            BASELINE,
        ]
    )

    missing = sorted(required.difference(data.columns))

    if missing:
        raise ValueError(
            "Missing columns: "
            + ", ".join(missing)
        )

    data["week_start_date"] = pd.to_datetime(
        data["week_start_date"],
        errors="raise",
    )

    data[TARGET] = pd.to_numeric(
        data[TARGET],
        errors="raise",
    ).astype(int)

    data[PROBABILITY] = pd.to_numeric(
        data[PROBABILITY],
        errors="raise",
    )

    data[BASELINE] = pd.to_numeric(
        data[BASELINE],
        errors="raise",
    ).astype(int)

    duplicate_rows = int(
        data.duplicated(ROW_KEYS).sum()
    )

    event_detail = (
        data.groupby(
            CENTER_EVENT_KEYS,
            as_index=False,
        )
        .agg(
            channel_count=(
                "channel_id",
                "nunique",
            ),
            target_nunique=(
                TARGET,
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
            baseline_min=(
                BASELINE,
                "min",
            ),
            baseline_mean=(
                BASELINE,
                "mean",
            ),
            baseline_max=(
                BASELINE,
                "max",
            ),
            baseline_nunique=(
                BASELINE,
                "nunique",
            ),
        )
    )

    event_detail["probability_range"] = (
        event_detail["probability_max"]
        - event_detail["probability_min"]
    )

    for threshold in THRESHOLDS:
        column = (
            f"pred_label_{str(threshold).replace('.', '_')}"
        )

        data[column] = (
            data[PROBABILITY] >= threshold
        ).astype(int)

        disagreement = (
            data.groupby(CENTER_EVENT_KEYS)[column]
            .nunique()
            .rename(
                f"{column}_nunique"
            )
            .reset_index()
        )

        event_detail = event_detail.merge(
            disagreement,
            on=CENTER_EVENT_KEYS,
            how="left",
            validate="one_to_one",
        )

        event_detail[
            f"probability_mean_label_{str(threshold).replace('.', '_')}"
        ] = (
            event_detail["probability_mean"]
            >= threshold
        ).astype(int)

        event_detail[
            f"probability_max_label_{str(threshold).replace('.', '_')}"
        ] = (
            event_detail["probability_max"]
            >= threshold
        ).astype(int)

    event_detail["baseline_any_label"] = (
        event_detail["baseline_max"] > 0
    ).astype(int)

    event_detail["baseline_majority_label"] = (
        event_detail["baseline_mean"] >= 0.5
    ).astype(int)

    summary_rows = [
        {
            "metric": "row_count",
            "value": len(data),
        },
        {
            "metric": "duplicate_row_keys",
            "value": duplicate_rows,
        },
        {
            "metric": "center_event_count",
            "value": len(event_detail),
        },
        {
            "metric": "row_to_center_event_ratio",
            "value": (
                len(data) / len(event_detail)
            ),
        },
        {
            "metric": "events_not_three_channels",
            "value": int(
                event_detail[
                    "channel_count"
                ].ne(3).sum()
            ),
        },
        {
            "metric": "events_with_target_disagreement",
            "value": int(
                event_detail[
                    "target_nunique"
                ].gt(1).sum()
            ),
        },
        {
            "metric": "events_with_probability_difference",
            "value": int(
                event_detail[
                    "probability_range"
                ].gt(1e-12).sum()
            ),
        },
        {
            "metric": "mean_probability_range",
            "value": float(
                event_detail[
                    "probability_range"
                ].mean()
            ),
        },
        {
            "metric": "max_probability_range",
            "value": float(
                event_detail[
                    "probability_range"
                ].max()
            ),
        },
        {
            "metric": "events_with_label_disagreement_threshold_0_4",
            "value": int(
                event_detail[
                    "pred_label_0_4_nunique"
                ].gt(1).sum()
            ),
        },
        {
            "metric": "events_with_label_disagreement_threshold_0_5",
            "value": int(
                event_detail[
                    "pred_label_0_5_nunique"
                ].gt(1).sum()
            ),
        },
        {
            "metric": "events_with_baseline_disagreement",
            "value": int(
                event_detail[
                    "baseline_nunique"
                ].gt(1).sum()
            ),
        },
        {
            "metric": "actual_positive_rows",
            "value": int(data[TARGET].sum()),
        },
        {
            "metric": "actual_positive_center_events",
            "value": int(
                event_detail[
                    "actual_target"
                ].sum()
            ),
        },
    ]

    summary = pd.DataFrame(summary_rows)

    metric_rows = []

    actual_row = data[TARGET].to_numpy(dtype=int)

    for threshold in THRESHOLDS:
        label_column = (
            f"pred_label_{str(threshold).replace('.', '_')}"
        )

        metric_rows.append(
            {
                "evaluation_grain": "channel_row",
                "prediction_method":
                    f"model_threshold_{threshold}",
                **classification_metrics(
                    actual_row,
                    data[label_column].to_numpy(dtype=int),
                    data[PROBABILITY].to_numpy(dtype=float),
                ),
            }
        )

    metric_rows.append(
        {
            "evaluation_grain": "channel_row",
            "prediction_method": "baseline_rule",
            **classification_metrics(
                actual_row,
                data[BASELINE].to_numpy(dtype=int),
            ),
        }
    )

    actual_event = event_detail[
        "actual_target"
    ].to_numpy(dtype=int)

    for threshold in THRESHOLDS:
        threshold_name = str(threshold).replace(
            ".",
            "_",
        )

        for aggregation in ["mean", "max"]:
            score_column = (
                f"probability_{aggregation}"
            )

            label_column = (
                f"probability_{aggregation}"
                f"_label_{threshold_name}"
            )

            metric_rows.append(
                {
                    "evaluation_grain": "center_event",
                    "prediction_method":
                        f"model_{aggregation}_probability"
                        f"_threshold_{threshold}",
                    **classification_metrics(
                        actual_event,
                        event_detail[
                            label_column
                        ].to_numpy(dtype=int),
                        event_detail[
                            score_column
                        ].to_numpy(dtype=float),
                    ),
                }
            )

    for baseline_method in [
        "baseline_any_label",
        "baseline_majority_label",
    ]:
        metric_rows.append(
            {
                "evaluation_grain": "center_event",
                "prediction_method": baseline_method,
                **classification_metrics(
                    actual_event,
                    event_detail[
                        baseline_method
                    ].to_numpy(dtype=int),
                ),
            }
        )

    metrics = pd.DataFrame(metric_rows)

    channel_profile = (
        data.groupby(
            "channel_id",
            as_index=False,
        )
        .agg(
            rows=(
                TARGET,
                "size",
            ),
            actual_positive_rate=(
                TARGET,
                "mean",
            ),
            mean_probability=(
                PROBABILITY,
                "mean",
            ),
            predicted_positive_rate_0_4=(
                "pred_label_0_4",
                "mean",
            ),
            predicted_positive_rate_0_5=(
                "pred_label_0_5",
                "mean",
            ),
            baseline_positive_rate=(
                BASELINE,
                "mean",
            ),
        )
    )

    summary.to_csv(
        OUTPUT_DIR
        / "20_stockout_output_grain_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metrics.to_csv(
        OUTPUT_DIR
        / "20_stockout_output_grain_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    channel_profile.to_csv(
        OUTPUT_DIR
        / "20_stockout_channel_profile.csv",
        index=False,
        encoding="utf-8-sig",
    )

    event_detail.to_csv(
        OUTPUT_DIR
        / "20_stockout_center_event_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n[STOCKOUT OUTPUT GRAIN SUMMARY]"
    )
    print(summary.to_string(index=False))

    print(
        "\n[STOCKOUT OUTPUT GRAIN METRICS]"
    )
    print(metrics.to_string(index=False))

    print(
        "\n[STOCKOUT CHANNEL PROFILE]"
    )
    print(channel_profile.to_string(index=False))


if __name__ == "__main__":
    main()
