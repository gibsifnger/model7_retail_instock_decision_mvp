from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

KEYS = ["sku_id", "channel_id", "center_id", "week_start_date"]

mart_path = Path("data/mart/final_modeling_table.csv")
pred_path = Path("outputs/predictions/stockout_risk_result.csv")
code_path = Path("src/models/train_stockout_classifier.py")

mart = pd.read_csv(mart_path)
pred = pd.read_csv(pred_path)

cols = KEYS + ["stockout_flag", "sales_censored_flag", "stockout_days_last_4w", "in_stock_rate_4w"]
audit = pred.merge(mart[cols], on=KEYS, how="left")

target_col = "target_stockout_risk_next_2w"
y_true = audit[target_col].astype(int)

print("=== 1. Prediction rows ===")
print("rows:", len(audit))
print("actual positive rate:", round(y_true.mean(), 4))

print("\n=== 2. Current stockout_flag distribution in prediction set ===")
print(audit["stockout_flag"].value_counts(dropna=False))

print("\n=== 3. Target positive rate by current stockout_flag ===")
print(audit.groupby("stockout_flag")[target_col].mean().round(4))

print("\n=== 4. Current stockout_flag alone as predictor ===")
y_stockout = audit["stockout_flag"].fillna(0).astype(int)

cm = confusion_matrix(y_true, y_stockout)
print("accuracy:", round(accuracy_score(y_true, y_stockout), 4))
print("precision:", round(precision_score(y_true, y_stockout, zero_division=0), 4))
print("recall:", round(recall_score(y_true, y_stockout, zero_division=0), 4))
print("f1:", round(f1_score(y_true, y_stockout, zero_division=0), 4))
print("roc_auc:", round(roc_auc_score(y_true, y_stockout), 4))
print("pr_auc:", round(average_precision_score(y_true, y_stockout), 4))
print("confusion_matrix:", cm.tolist())

print("\n=== 5. Model result from saved prediction ===")
y_model = audit["stockout_risk_pred_label"].astype(int)
y_proba = audit["stockout_risk_pred_proba"].astype(float)
cm_model = confusion_matrix(y_true, y_model)
print("accuracy:", round(accuracy_score(y_true, y_model), 4))
print("precision:", round(precision_score(y_true, y_model, zero_division=0), 4))
print("recall:", round(recall_score(y_true, y_model, zero_division=0), 4))
print("f1:", round(f1_score(y_true, y_model, zero_division=0), 4))
print("roc_auc:", round(roc_auc_score(y_true, y_proba), 4))
print("pr_auc:", round(average_precision_score(y_true, y_proba), 4))
print("confusion_matrix:", cm_model.tolist())

print("\n=== 6. Code keyword scan ===")
code = code_path.read_text(encoding="utf-8")
for token in [
    "target_stockout_risk_next_2w",
    "target_demand_next_1w",
    "demand_pred",
    "stockout_flag",
]:
    print(f"{token}: {'FOUND' if token in code else 'NOT FOUND'}")

print("\nManual check needed:")
print("- If target columns appear only as target/exclude/output, OK.")
print("- If target columns appear in numeric_features/categorical_features, FIX.")
print("- If stockout_flag appears in feature list, consider removing it or document why it is allowed.")
