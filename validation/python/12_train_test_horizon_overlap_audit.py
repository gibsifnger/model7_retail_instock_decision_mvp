from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/mart/final_modeling_table.csv")
OUTPUT_DIR = Path("outputs/validation")
TEST_WEEKS = 8


def audit_task(
    data: pd.DataFrame,
    task_name: str,
    target_column: str,
    horizon_weeks: int,
) -> tuple[list[tuple[str, object]], pd.DataFrame]:
    labeled = data.loc[
        data[target_column].notna()
    ].copy()

    unique_weeks = sorted(
        labeled["week_start_date"].unique()
    )

    test_weeks = unique_weeks[-TEST_WEEKS:]
    test_start = pd.Timestamp(test_weeks[0])
    test_end = pd.Timestamp(test_weeks[-1])

    train = labeled.loc[
        labeled["week_start_date"] < test_start
    ].copy()

    test = labeled.loc[
        labeled["week_start_date"].isin(test_weeks)
    ].copy()

    train["label_horizon_start"] = (
        train["week_start_date"]
        + pd.Timedelta(weeks=1)
    )

    train["label_horizon_end"] = (
        train["week_start_date"]
        + pd.Timedelta(weeks=horizon_weeks)
    )

    overlap = train.loc[
        train["label_horizon_end"] >= test_start
    ].copy()

    purged_train = train.loc[
        train["label_horizon_end"] < test_start
    ].copy()

    overlap_center_keys = (
        overlap[
            [
                "sku_id",
                "center_id",
                "week_start_date",
            ]
        ]
        .drop_duplicates()
        .shape[0]
    )

    summary = [
        (f"{task_name}__labeled_rows", len(labeled)),
        (f"{task_name}__unique_labeled_weeks", len(unique_weeks)),
        (f"{task_name}__test_start", test_start.date()),
        (f"{task_name}__test_end", test_end.date()),
        (f"{task_name}__train_rows_before_purge", len(train)),
        (f"{task_name}__test_rows", len(test)),
        (
            f"{task_name}__overlap_train_rows",
            len(overlap),
        ),
        (
            f"{task_name}__overlap_feature_weeks",
            overlap["week_start_date"].nunique(),
        ),
        (
            f"{task_name}__overlap_center_week_keys",
            overlap_center_keys,
        ),
        (
            f"{task_name}__train_rows_after_purge",
            len(purged_train),
        ),
        (
            f"{task_name}__required_purge_weeks",
            horizon_weeks,
        ),
    ]

    overlap_sample = overlap[
        [
            "sku_id",
            "channel_id",
            "center_id",
            "week_start_date",
            target_column,
            "label_horizon_start",
            "label_horizon_end",
        ]
    ].head(20)

    overlap_sample.insert(0, "task", task_name)

    return summary, overlap_sample


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(
        INPUT_PATH,
        parse_dates=["week_start_date"],
    )

    demand_summary, demand_sample = audit_task(
        data=data,
        task_name="demand",
        target_column="target_demand_next_1w",
        horizon_weeks=1,
    )

    stockout_summary, stockout_sample = audit_task(
        data=data,
        task_name="stockout",
        target_column="target_stockout_risk_next_2w",
        horizon_weeks=2,
    )

    summary = pd.DataFrame(
        demand_summary + stockout_summary,
        columns=["metric", "value"],
    )

    samples = pd.concat(
        [demand_sample, stockout_sample],
        ignore_index=True,
    )

    summary.to_csv(
        OUTPUT_DIR
        / "12_train_test_horizon_overlap_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    samples.to_csv(
        OUTPUT_DIR
        / "12_train_test_horizon_overlap_samples.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n[TRAIN / TEST HORIZON OVERLAP SUMMARY]")
    print(summary.to_string(index=False))

    print("\n[OVERLAP SAMPLE]")
    print(samples.to_string(index=False))


if __name__ == "__main__":
    main()
