# ─────────────────────────────────────────────────────────────
# PatentLens — Evaluation
# Calculates F1 scores from prediction CSV files
# ─────────────────────────────────────────────────────────────

import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)
import sys
import os

CATEGORIES = [
    "CLEAN_ENERGY", "AGRITECH", "MEDTECH",
    "MANUFACTURING", "DIGITAL_INFRA", "ENVIROTECH"
]


def evaluate(experiment_name):
    """
    Load predictions CSV and calculate all metrics.
    """
    pred_path = f"results/{experiment_name}_predictions.csv"

    if not os.path.exists(pred_path):
        print(f"File not found: {pred_path}")
        print("Run the experiment first:")
        print(f"  python src/llm_classify.py {experiment_name}")
        return

    print(f"\n{'='*55}")
    print(f"Evaluation: {experiment_name.upper()}")
    print(f"{'='*55}")

    df = pd.read_csv(pred_path)
    print(f"Total predictions: {len(df)}")

    # Remove UNKNOWN predictions
    unknown = df[df["predicted_label"] == "UNKNOWN"]
    if len(unknown) > 0:
        print(f"UNKNOWN predictions: {len(unknown)} "
              f"({len(unknown)/len(df)*100:.1f}%)")

    valid = df[df["predicted_label"] != "UNKNOWN"]

    true_labels = valid["true_label"].tolist()
    pred_labels = valid["predicted_label"].tolist()

    # Overall metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    macro_f1 = f1_score(
        true_labels, pred_labels,
        average="macro", zero_division=0
    )
    weighted_f1 = f1_score(
        true_labels, pred_labels,
        average="weighted", zero_division=0
    )

    print(f"\nOverall Results:")
    print(f"  Accuracy    : {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"  Macro-F1    : {macro_f1:.4f}")
    print(f"  Weighted-F1 : {weighted_f1:.4f}")

    # Per-category breakdown
    print(f"\nPer-Category F1 Scores:")
    report = classification_report(
        true_labels, pred_labels,
        labels=CATEGORIES,
        zero_division=0,
        output_dict=True
    )

    print(f"  {'Category':<20} {'F1':>6} {'Precision':>10} "
          f"{'Recall':>8} {'Support':>8}")
    print(f"  {'-'*55}")

    for cat in CATEGORIES:
        if cat in report:
            f1   = report[cat]["f1-score"]
            prec = report[cat]["precision"]
            rec  = report[cat]["recall"]
            sup  = int(report[cat]["support"])
            print(f"  {cat:<20} {f1:>6.3f} {prec:>10.3f} "
                  f"{rec:>8.3f} {sup:>8}")

    # Confusion matrix
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(true_labels, pred_labels, labels=CATEGORIES)
    header = f"  {'':>15}" + "".join(f"{c[:6]:>8}" for c in CATEGORIES)
    print(header)
    for i, row_label in enumerate(CATEGORIES):
        row = f"  {row_label:<15}" + "".join(f"{v:>8}" for v in cm[i])
        print(row)

    # Save results to text file
    result_path = f"results/{experiment_name}_results.txt"
    with open(result_path, "w") as f:
        f.write(f"PatentLens — {experiment_name.upper()} Results\n")
        f.write(f"{'='*55}\n")
        f.write(f"Accuracy    : {accuracy:.4f}\n")
        f.write(f"Macro-F1    : {macro_f1:.4f}\n")
        f.write(f"Weighted-F1 : {weighted_f1:.4f}\n\n")
        f.write(classification_report(
            true_labels, pred_labels,
            labels=CATEGORIES, zero_division=0
        ))

    print(f"\nResults saved: {result_path}")
    print(f"{'='*55}")

    return macro_f1


if __name__ == "__main__":
    experiment = sys.argv[1] if len(sys.argv) > 1 else "zeroshot"
    evaluate(experiment)