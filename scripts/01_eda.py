"""
01_eda.py

Exploratory Data Analysis for the UCI SECOM dataset.

This script highlights:
1. Dataset size: 1567 samples and 590 features
2. High-dimensional structure
3. Class imbalance: Pass vs Fail
4. Highly incomplete features
5. Overall missing rate
6. Low-information / low-variance features

Expected input:
    data/raw/uci-secom.csv

Outputs:
    outputs/eda/
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(path: Path) -> None:
    """Create directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def save_bar_chart(x, y, title, xlabel, ylabel, output_path: Path, figsize=(7, 5)) -> None:
    """Save a simple bar chart."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x, y)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_histogram(values, title, xlabel, ylabel, output_path: Path, bins=40, figsize=(8, 5)) -> None:
    """Save a histogram."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(values, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(input_path: Path, output_dir: Path) -> None:
    ensure_dir(output_dir)

    # =========================
    # 1. Read Data
    # =========================
    data_raw = pd.read_csv(input_path, encoding="utf-8")

    # Convert target label:
    # Original UCI SECOM label: -1 = Pass, 1 = Fail
    # Converted label: 0 = Pass, 1 = Fail
    if "Pass/Fail" not in data_raw.columns:
        raise ValueError("Column 'Pass/Fail' was not found in the dataset.")

    data_raw["Pass/Fail"] = data_raw["Pass/Fail"].replace(-1, 0)

    # Separate feature columns
    target_col = "Pass/Fail"
    time_col = "Time" if "Time" in data_raw.columns else None

    feature_cols = [col for col in data_raw.columns if col != target_col]
    numeric_feature_cols = [
        col for col in feature_cols
        if col != time_col and pd.api.types.is_numeric_dtype(data_raw[col])
    ]

    X = data_raw[numeric_feature_cols]
    y = data_raw[target_col]

    n_samples = data_raw.shape[0]
    n_total_columns = data_raw.shape[1]
    n_features = len(numeric_feature_cols)
    n_pass = int((y == 0).sum())
    n_fail = int((y == 1).sum())

    # =========================
    # 2. Dataset Overview
    # =========================
    overview = pd.DataFrame({
        "metric": [
            "n_samples",
            "n_total_columns",
            "n_numeric_features_excluding_time_and_target",
            "n_pass",
            "n_fail",
            "pass_rate",
            "fail_rate",
            "feature_to_sample_ratio",
            "overall_missing_rate"
        ],
        "value": [
            n_samples,
            n_total_columns,
            n_features,
            n_pass,
            n_fail,
            n_pass / n_samples,
            n_fail / n_samples,
            n_features / n_samples,
            X.isna().mean().mean()
        ]
    })

    overview.to_csv(output_dir / "eda_dataset_overview.csv", index=False)

    print("\n=== Dataset Overview ===")
    print(overview)

    # =========================
    # 3. Class Imbalance
    # =========================
    class_counts = y.value_counts().sort_index()
    class_summary = pd.DataFrame({
        "class_label": class_counts.index,
        "class_name": ["Pass" if label == 0 else "Fail" for label in class_counts.index],
        "count": class_counts.values,
        "percentage": class_counts.values / n_samples
    })

    class_summary.to_csv(output_dir / "eda_class_distribution.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 5))
    labels = ["Pass (0)", "Fail (1)"]
    values = [n_pass, n_fail]
    ax.bar(labels, values)
    ax.set_title("Class Distribution: Pass vs Fail")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")

    for i, value in enumerate(values):
        pct = value / n_samples * 100
        ax.text(i, value, f"{value}\n({pct:.1f}%)", ha="center", va="bottom")

    plt.tight_layout()
    fig.savefig(output_dir / "01_class_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Imbalance ratio
    imbalance_ratio = n_pass / n_fail if n_fail > 0 else np.nan
    imbalance_summary = pd.DataFrame({
        "metric": ["pass_to_fail_ratio"],
        "value": [imbalance_ratio]
    })
    imbalance_summary.to_csv(output_dir / "eda_imbalance_ratio.csv", index=False)

    # =========================
    # 4. High-Dimensional Data
    # =========================
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(["Samples", "Numeric Features"], [n_samples, n_features])
    ax.set_title("Dataset Size: Samples vs Features")
    ax.set_ylabel("Count")

    ax.text(0, n_samples, str(n_samples), ha="center", va="bottom")
    ax.text(1, n_features, str(n_features), ha="center", va="bottom")

    plt.tight_layout()
    fig.savefig(output_dir / "02_samples_vs_features.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # =========================
    # 5. Missing Values
    # =========================
    missing_rate_by_feature = X.isna().mean().sort_values(ascending=False)

    missing_summary = pd.DataFrame({
        "feature": missing_rate_by_feature.index,
        "missing_rate": missing_rate_by_feature.values,
        "missing_count": X[missing_rate_by_feature.index].isna().sum().values
    })

    missing_summary.to_csv(output_dir / "eda_missing_rate_by_feature.csv", index=False)

    # Top 30 missing features
    top_missing = missing_summary.head(30)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top_missing["feature"].astype(str), top_missing["missing_rate"])
    ax.invert_yaxis()
    ax.set_title("Top 30 Features by Missing Rate")
    ax.set_xlabel("Missing Rate")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    fig.savefig(output_dir / "03_top30_missing_rate_features.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Missing rate distribution
    save_histogram(
        missing_rate_by_feature.values,
        title="Distribution of Feature Missing Rates",
        xlabel="Missing Rate",
        ylabel="Number of Features",
        output_path=output_dir / "04_missing_rate_distribution.png",
        bins=40
    )

    # Missing rate threshold summary
    missing_thresholds = [0, 0.01, 0.05, 0.10, 0.20, 0.40, 0.50]
    missing_threshold_summary = pd.DataFrame({
        "missing_rate_greater_than": missing_thresholds,
        "n_features": [(missing_rate_by_feature > t).sum() for t in missing_thresholds]
    })
    missing_threshold_summary.to_csv(output_dir / "eda_missing_threshold_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        missing_threshold_summary["missing_rate_greater_than"].astype(str),
        missing_threshold_summary["n_features"]
    )
    ax.set_title("Number of Features Above Missing Rate Thresholds")
    ax.set_xlabel("Missing Rate Threshold")
    ax.set_ylabel("Number of Features")
    plt.tight_layout()
    fig.savefig(output_dir / "05_missing_threshold_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # =========================
    # 6. Low-Information Features: Variance
    # =========================
    # Variance is computed after simple median imputation for EDA only.
    # This is not the final preprocessing pipeline.
    X_median_imputed = X.copy()
    X_median_imputed = X_median_imputed.fillna(X_median_imputed.median(numeric_only=True))

    variance_by_feature = X_median_imputed.var(axis=0).sort_values()
    variance_summary = pd.DataFrame({
        "feature": variance_by_feature.index,
        "variance": variance_by_feature.values
    })
    variance_summary.to_csv(output_dir / "eda_variance_by_feature.csv", index=False)

    # Low variance threshold summary
    variance_thresholds = [0, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2]
    variance_threshold_summary = pd.DataFrame({
        "variance_less_than_or_equal_to": variance_thresholds,
        "n_features": [(variance_by_feature <= t).sum() for t in variance_thresholds]
    })
    variance_threshold_summary.to_csv(output_dir / "eda_variance_threshold_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        variance_threshold_summary["variance_less_than_or_equal_to"].astype(str),
        variance_threshold_summary["n_features"]
    )
    ax.set_title("Number of Low-Variance Features")
    ax.set_xlabel("Variance Threshold")
    ax.set_ylabel("Number of Features")
    plt.tight_layout()
    fig.savefig(output_dir / "06_low_variance_threshold_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Log variance distribution, excluding exact zero or negative values
    positive_variance = variance_by_feature[variance_by_feature > 0]
    log_variance = np.log10(positive_variance)

    save_histogram(
        log_variance.values,
        title="Distribution of Feature Variance (log10 scale)",
        xlabel="log10(Variance)",
        ylabel="Number of Features",
        output_path=output_dir / "07_log_variance_distribution.png",
        bins=40
    )

    # Zero-variance feature count
    zero_variance_count = int((variance_by_feature <= 0).sum())
    nonzero_variance_count = int((variance_by_feature > 0).sum())
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(
        ["Zero Variance", "Non-Zero Variance"],
        [zero_variance_count, nonzero_variance_count]
    )
    ax.set_title("Zero-Variance vs Non-Zero-Variance Features")
    ax.set_ylabel("Number of Features")
    
    ax.text(0, zero_variance_count, str(zero_variance_count), ha="center", va="bottom")
    ax.text(1, nonzero_variance_count, str(nonzero_variance_count), ha="center", va="bottom")
    
    plt.tight_layout()
    fig.savefig(output_dir / "08_zero_variance_feature_count.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    
    # Top 30 lowest non-zero variance features
    lowest_nonzero_variance = variance_summary[variance_summary["variance"] > 0].head(30)
    
    if not lowest_nonzero_variance.empty:
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(
            lowest_nonzero_variance["feature"].astype(str),
            lowest_nonzero_variance["variance"]
        )
        ax.invert_yaxis()
        ax.set_title("Top 30 Lowest Non-Zero Variance Features")
        ax.set_xlabel("Variance")
        ax.set_ylabel("Feature")
        plt.tight_layout()
        fig.savefig(
            output_dir / "09_top30_lowest_nonzero_variance_features.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close(fig)
    else:
        print("No non-zero variance features found.")

    # =========================
    # 7. Missingness by Class
    # =========================
    missing_rate_by_class = data_raw.groupby(target_col)[numeric_feature_cols].apply(
        lambda df: df.isna().mean().mean()
    )

    missing_by_class_summary = pd.DataFrame({
        "class_label": missing_rate_by_class.index,
        "class_name": ["Pass" if label == 0 else "Fail" for label in missing_rate_by_class.index],
        "average_missing_rate": missing_rate_by_class.values
    })
    missing_by_class_summary.to_csv(output_dir / "eda_missing_rate_by_class.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(missing_by_class_summary["class_name"], missing_by_class_summary["average_missing_rate"])
    ax.set_title("Average Missing Rate by Class")
    ax.set_xlabel("Class")
    ax.set_ylabel("Average Missing Rate")
    plt.tight_layout()
    fig.savefig(output_dir / "09_missing_rate_by_class.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # =========================
    # 8. One-page EDA Summary Text
    # =========================
    summary_text = f"""
SECOM EDA Summary
=================

Data source:
- UCI Machine Learning Repository
- Semiconductor manufacturing process data

Dataset shape:
- Samples: {n_samples}
- Total columns: {n_total_columns}
- Numeric features used for EDA, excluding Time and Pass/Fail: {n_features}

High-dimensional data:
- Feature-to-sample ratio: {n_features / n_samples:.3f}
- This indicates that the dataset contains many sensor/process variables relative to the number of observations.

Class imbalance:
- Pass: {n_pass} samples ({n_pass / n_samples:.2%})
- Fail: {n_fail} samples ({n_fail / n_samples:.2%})
- Pass-to-fail ratio: {imbalance_ratio:.2f}:1

Missing values:
- Overall missing rate: {X.isna().mean().mean():.2%}
- Number of features with missing rate > 1%: {(missing_rate_by_feature > 0.01).sum()}
- Number of features with missing rate > 5%: {(missing_rate_by_feature > 0.05).sum()}
- Number of features with missing rate > 40%: {(missing_rate_by_feature > 0.40).sum()}

Low-information features:
- Number of exact zero-variance features: {(variance_by_feature <= 0).sum()}
- Number of features with variance <= 1e-6: {(variance_by_feature <= 1e-6).sum()}
- Number of features with variance <= 1e-4: {(variance_by_feature <= 1e-4).sum()}

Generated outputs:
- CSV summary files
- EDA charts saved as PNG files
"""

    with open(output_dir / "eda_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text.strip())

    print("\n=== EDA completed ===")
    print(f"Outputs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EDA for the UCI SECOM dataset.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/uci-secom.csv"),
        help="Path to input CSV file. Default: data/raw/uci-secom.csv"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/eda"),
        help="Directory to save EDA outputs. Default: outputs/eda"
    )
    args = parser.parse_args()

    main(input_path=args.input, output_dir=args.output)
