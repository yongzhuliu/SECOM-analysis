"""
02_preprocess_ipynb_exact.py

Preprocessing script converted from the original notebook logic.

Goal:
    Reproduce the notebook preprocessing results as closely as possible.

Main difference from the cleaned/projectized version:
    - Keeps the same feature-selection logic as the ipynb.
    - Keeps XGB random_state=66, same as the ipynb.
    - Creates NA indicators from data_raw high-missing features, same as the ipynb.
    - Uses the same RFECV compact-selection rule:
        chosen_n_features = fewest features within best_score - 0.25 * SE

Expected input:
    data/raw/uci-secom.csv

Default outputs:
    data/processed/
    outputs/preprocessing/

Run:
    python scripts/02_preprocess.py

or:
    python scripts/02_preprocess.py --input uci-secom.csv
"""

from pathlib import Path
import argparse
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.feature_selection import VarianceThreshold, RFECV, RFE, SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier

from feature_engine.selection import SmartCorrelatedSelection
from xgboost import XGBClassifier


warnings.filterwarnings("ignore")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_series(series: pd.Series, path: Path) -> None:
    series.to_csv(path, index=False, encoding="utf-8-sig")


def plot_missing_rate_curve(mssr: pd.Series, output_path: Path) -> None:
    na_df = mssr.sort_values(ascending=False).reset_index()
    na_df.columns = ["feature", "missing rate"]
    na_df["n"] = np.arange(len(na_df)) + 1

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(na_df["n"], na_df["missing rate"], marker="o", markersize=3, linewidth=1)
    ax.axhline(y=0.5, linestyle="--", color="gray")
    ax.axhline(y=0.4, linestyle="--", color="red")
    ax.set_xlabel("Rank of Features Sorted by Missing Rate (Desc)")
    ax.set_ylabel("Missing Rate")
    ax.set_title("Missing Rate by Feature")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    na_df.to_csv(output_path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def plot_corr_heatmap(df: pd.DataFrame, output_path: Path, title: str, threshold: float = 0.7) -> None:
    corr = df.corr(numeric_only=True)
    mask = (corr.abs() >= threshold) & (corr.abs() < 1)
    selected = mask.any(axis=1)

    if selected.sum() < 2:
        print(f"Skip heatmap: less than 2 features with abs(corr) >= {threshold}.")
        return

    corr_sub = corr.loc[selected, selected]

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_sub, cmap="coolwarm", center=0)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def choose_compact_n_features(cv_result: pd.DataFrame, cv_folds: int = 5) -> dict:
    """
    Same selection logic as the ipynb:
        best_se = best_std / sqrt(cv_folds)
        threshold = best_score - best_se * 0.25
        choose the smallest n_features whose mean_score >= threshold.
    """
    best_idx = cv_result["mean_score"].idxmax()
    best_score = cv_result.loc[best_idx, "mean_score"]
    best_std = cv_result.loc[best_idx, "std_score"]
    best_n_features = cv_result.loc[best_idx, "n_features"]

    best_se = best_std / np.sqrt(cv_folds)
    threshold = best_score - best_se * 0.25

    candidate_rows = cv_result[cv_result["mean_score"] >= threshold]
    chosen_row = candidate_rows.sort_values("n_features").iloc[0]

    chosen_n_features = int(chosen_row["n_features"])
    chosen_score = float(chosen_row["mean_score"])

    return {
        "best_idx": int(best_idx),
        "best_score": float(best_score),
        "best_std": float(best_std),
        "best_n_features": int(best_n_features),
        "best_se": float(best_se),
        "threshold": float(threshold),
        "chosen_n_features": int(chosen_n_features),
        "chosen_score": float(chosen_score),
    }


def plot_rfecv_curve(cv_result: pd.DataFrame, choice: dict, output_path: Path, title: str) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(cv_result["n_features"], cv_result["mean_score"], linewidth=2)
    plt.fill_between(
        cv_result["n_features"],
        cv_result["mean_score"] - cv_result["std_score"],
        cv_result["mean_score"] + cv_result["std_score"],
        alpha=0.2,
    )

    plt.axvline(
        choice["chosen_n_features"],
        color="red",
        linestyle="--",
        label=f"Chosen Score: {choice['chosen_score']:.3f}, Features: {choice['chosen_n_features']}",
    )
    plt.axvline(
        choice["best_n_features"],
        color="gray",
        linestyle="--",
        label=f"Best Score: {choice['best_score']:.3f}, Features: {choice['best_n_features']}",
    )

    plt.xlabel("Number of Features")
    plt.ylabel("Average Precision")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main(
    input_path: Path,
    processed_dir: Path,
    output_dir: Path,
    model_dir: Path,
) -> None:
    ensure_dir(processed_dir)
    ensure_dir(output_dir)
    ensure_dir(model_dir)

    log = {
        "input_path": str(input_path),
        "note": "This script follows the original ipynb preprocessing logic as closely as possible.",
        "steps": [],
    }

    # =========================================================
    # Read Data
    # =========================================================
    data_raw = pd.read_csv(input_path, encoding="utf-8")

    print(data_raw.shape)
    print(data_raw.info())
    print(data_raw.head())
    data_raw.describe().T.to_csv(output_dir / "raw_describe_transposed.csv", encoding="utf-8-sig")

    # EDA target conversion, same as ipynb
    data_raw["Pass/Fail"] = data_raw["Pass/Fail"].replace(-1, 0)
    counts = data_raw["Pass/Fail"].value_counts()
    print(counts)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_title("Pass/Fail counts")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    plt.tight_layout()
    fig.savefig(output_dir / "pass_fail_counts.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # =========================================================
    # Split train/test set
    # =========================================================
    X_train, X_test, y_train, y_test = train_test_split(
        data_raw.drop(columns=["Pass/Fail"]),
        data_raw["Pass/Fail"],
        test_size=0.2,
        stratify=data_raw["Pass/Fail"],
        random_state=666,
        shuffle=True,
    )

    # Same filenames as ipynb, but saved under processed_dir.
    save_csv(X_train, processed_dir / "X_train.csv")
    save_csv(X_test, processed_dir / "X_test.csv")
    save_series(y_train, processed_dir / "y_train.csv")
    save_series(y_test, processed_dir / "y_test.csv")

    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)

    log["steps"].append({
        "step": "train_test_split",
        "X_train_shape": list(X_train.shape),
        "X_test_shape": list(X_test.shape),
    })

    # =========================================================
    # Data Cleaning
    # Drop "Time" feature.
    # =========================================================
    X_train.drop(columns=["Time"], inplace=True)
    X_test.drop(columns=["Time"], inplace=True)

    log["steps"].append({
        "step": "drop_time",
        "remaining_features": len(X_train.columns),
    })

    # =========================================================
    # Drop duplicated features.
    # =========================================================
    same = X_train.T.duplicated()
    duplicated_features = same[same].index.tolist()
    print(f"{sum(same)} features are are duplicated.")

    X_train.drop(columns=duplicated_features, inplace=True)
    X_test.drop(columns=duplicated_features, inplace=True)

    print(f"Remains {len(X_train.columns)} features.")

    pd.DataFrame({"feature": duplicated_features}).to_csv(
        output_dir / "duplicated_features_dropped.csv",
        index=False,
        encoding="utf-8-sig",
    )

    log["steps"].append({
        "step": "drop_duplicated_features",
        "n_dropped": len(duplicated_features),
        "remaining_features": len(X_train.columns),
    })

    # =========================================================
    # Missing rate plot
    # =========================================================
    mssr = X_train.isna().mean().sort_values(ascending=False)
    plot_missing_rate_curve(mssr, output_dir / "missing_rate_before_drop.png")

    # =========================================================
    # Drop features with missing rate > 40%.
    # =========================================================
    mssr = X_train.isna().mean()
    na_drop = mssr > 0.4
    high_missing_features_train = na_drop[na_drop].index.tolist()

    print(f"{sum(mssr > 0.4)} features has missing rate > 0.4.")

    X_train.drop(columns=high_missing_features_train, inplace=True)
    X_test.drop(columns=high_missing_features_train, inplace=True)

    print(f"Remain {len(X_train.columns)} features.")

    pd.DataFrame({"feature": high_missing_features_train}).to_csv(
        output_dir / "high_missing_features_dropped_from_train.csv",
        index=False,
        encoding="utf-8-sig",
    )

    log["steps"].append({
        "step": "drop_features_missing_rate_gt_0_4_based_on_X_train",
        "n_dropped": len(high_missing_features_train),
        "remaining_features": len(X_train.columns),
    })

    # =========================================================
    # Add NA indice for those features having missing rate > 40%.
    # IMPORTANT:
    # This intentionally follows the ipynb logic:
    #     mssr = data_raw.isna().mean()
    #     na_df = data_raw.loc[:, mssr[mssr>0.4].index].isna()
    #     pat = na_df.T.drop_duplicates().T
    #     X_train = pd.concat([X_train, pat.iloc[X_train.index, :]], axis=1)
    # =========================================================
    mssr_raw = data_raw.isna().mean()
    raw_high_missing_features = mssr_raw[mssr_raw > 0.4].index.tolist()

    if len(raw_high_missing_features) > 0:
        na_df = data_raw.loc[:, raw_high_missing_features].isna()
        pat = na_df.T.drop_duplicates().T
        pat.columns = [f"NA_{i}" for i in pat.columns]
    else:
        pat = pd.DataFrame(index=data_raw.index)

    idx = set(pat.columns)

    # Use iloc exactly like the ipynb.
    X_train = pd.concat([X_train, pat.iloc[X_train.index, :]], axis=1)
    X_test = pd.concat([X_test, pat.iloc[X_test.index, :]], axis=1)

    print(f"There are {pat.shape[1]} NA indice.")
    print(f"Remain {len(X_train.columns)} features.")

    pd.DataFrame({"feature": raw_high_missing_features}).to_csv(
        output_dir / "raw_high_missing_features_for_na_indicators.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame({"na_indicator": list(pat.columns)}).to_csv(
        output_dir / "na_indicator_features_added.csv",
        index=False,
        encoding="utf-8-sig",
    )

    log["steps"].append({
        "step": "add_na_indicators_from_data_raw_missing_rate_gt_0_4",
        "n_raw_high_missing_features": len(raw_high_missing_features),
        "n_na_indicators": pat.shape[1],
        "remaining_features": len(X_train.columns),
    })

    # =========================================================
    # Variance threshold = 0
    # =========================================================
    var_th = VarianceThreshold(threshold=0)
    var_th.fit(X_train)

    zero_var_features = X_train.columns[~var_th.get_support()].tolist()

    print(f"There are {sum(~var_th.get_support())} features having var = 0.")
    print(f"Remains {sum(var_th.get_support())} features.")

    keep_cols = X_train.columns[var_th.get_support()]

    X_train = pd.DataFrame(
        var_th.transform(X_train),
        columns=keep_cols,
        index=X_train.index,
    )
    X_test = pd.DataFrame(
        var_th.transform(X_test),
        columns=keep_cols,
        index=X_test.index,
    )

    pd.DataFrame({"feature": zero_var_features}).to_csv(
        output_dir / "zero_variance_features_dropped.csv",
        index=False,
        encoding="utf-8-sig",
    )

    log["steps"].append({
        "step": "variance_threshold_zero",
        "n_dropped": len(zero_var_features),
        "remaining_features": len(X_train.columns),
    })

    # =========================================================
    # Drop features with constant ratio > 99%
    # =========================================================
    def drop_near_constant_columns(df, tol=0.99):
        dominant_ratio = df.apply(lambda s: s.value_counts(normalize=True, dropna=False).iloc[0])
        keep_cols_inner = dominant_ratio[dominant_ratio < tol].index
        return keep_cols_inner, dominant_ratio.sort_values(ascending=False)

    keep_cols, dominant_ratio = drop_near_constant_columns(X_train, tol=0.99)

    print(dominant_ratio.head())
    print(f"Remains {len(keep_cols)} features.")

    near_constant_features = [c for c in X_train.columns if c not in keep_cols]

    X_train = pd.DataFrame(
        X_train.loc[:, keep_cols],
        columns=keep_cols,
        index=X_train.index,
    )
    X_test = pd.DataFrame(
        X_test.loc[:, keep_cols],
        columns=keep_cols,
        index=X_test.index,
    )

    dominant_ratio.to_frame("dominant_ratio").to_csv(
        output_dir / "dominant_ratio_by_feature.csv",
        encoding="utf-8-sig",
    )
    pd.DataFrame({"feature": near_constant_features}).to_csv(
        output_dir / "near_constant_features_dropped.csv",
        index=False,
        encoding="utf-8-sig",
    )

    log["steps"].append({
        "step": "drop_near_constant_columns_tol_0_99",
        "n_dropped": len(near_constant_features),
        "remaining_features": len(X_train.columns),
    })

    # =========================================================
    # Standardize
    # =========================================================
    scaler = StandardScaler()
    idx_cols = [c for c in X_train.columns if c in idx]
    scale_cols = [c for c in X_train.columns if c not in idx_cols]

    X_train.loc[:, scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_test.loc[:, scale_cols] = scaler.transform(X_test[scale_cols])

    joblib.dump(scaler, model_dir / "standard_scaler_ipynb_exact.joblib")

    log["steps"].append({
        "step": "standardize",
        "n_scaled_features": len(scale_cols),
        "n_na_indicator_features_not_scaled": len(idx_cols),
    })

    # =========================================================
    # Imputation: KNN
    # IMPORTANT:
    # Notebook uses X_train.drop(columns=idx).
    # To avoid errors if any NA indicator was removed earlier, use the existing idx_cols
    # but keep the same intended behavior: exclude NA indicators from KNN imputation.
    # =========================================================
    imputer = KNNImputer(n_neighbors=10, weights="distance")

    X_train_knn = pd.DataFrame(
        imputer.fit_transform(X_train.drop(columns=idx_cols)),
        columns=X_train.drop(columns=idx_cols).columns,
        index=X_train.index,
    )
    X_test_knn = pd.DataFrame(
        imputer.transform(X_test.drop(columns=idx_cols)),
        columns=X_test.drop(columns=idx_cols).columns,
        index=X_test.index,
    )

    X_train_knn = pd.concat([X_train_knn, X_train[idx_cols]], axis=1)
    X_test_knn = pd.concat([X_test_knn, X_test[idx_cols]], axis=1)

    joblib.dump(imputer, model_dir / "knn_imputer_ipynb_exact.joblib")

    save_csv(X_train_knn, processed_dir / "X_train_knn.csv")
    save_csv(X_test_knn, processed_dir / "X_test_knn.csv")

    log["steps"].append({
        "step": "knn_imputation",
        "X_train_knn_shape": list(X_train_knn.shape),
        "X_test_knn_shape": list(X_test_knn.shape),
    })

    # =========================================================
    # Feature Selection
    # 1. correlation selection
    # =========================================================
    plot_corr_heatmap(
        X_train_knn,
        output_dir / "correlation_heatmap_before_selection.png",
        title="Correlation Heatmap Before Selection",
        threshold=0.7,
    )

    corr_rf = SmartCorrelatedSelection(
        method="pearson",
        threshold=0.9,
        selection_method="model_performance",
        estimator=RandomForestClassifier(random_state=666),
        scoring="average_precision",
        cv=3,
    )

    X_train_corr_rf = corr_rf.fit_transform(X_train_knn, y_train)
    X_test_corr_rf = corr_rf.transform(X_test_knn)

    X_train_base, X_test_base = (X_train_corr_rf.copy(), X_test_corr_rf.copy())

    print(corr_rf.correlated_feature_sets_)
    print(corr_rf.features_to_drop_)
    print(len(X_train_corr_rf.columns))

    pd.DataFrame({"feature": list(corr_rf.features_to_drop_)}).to_csv(
        output_dir / "correlated_features_dropped.csv",
        index=False,
        encoding="utf-8-sig",
    )

    try:
        correlated_sets = [list(s) for s in corr_rf.correlated_feature_sets_]
    except Exception:
        correlated_sets = []

    with open(output_dir / "correlated_feature_sets.json", "w", encoding="utf-8") as f:
        json.dump(correlated_sets, f, ensure_ascii=False, indent=2)

    plot_corr_heatmap(
        X_train_corr_rf,
        output_dir / "correlation_heatmap_after_selection.png",
        title="Correlation Heatmap After Selection",
        threshold=0.7,
    )

    log["steps"].append({
        "step": "smart_correlated_selection",
        "n_dropped": len(list(corr_rf.features_to_drop_)),
        "X_train_base_shape": list(X_train_base.shape),
        "X_test_base_shape": list(X_test_base.shape),
    })

    # =========================================================
    # 2-1. RF-RFE
    # =========================================================
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=666,
        n_jobs=1,
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=666,
    )

    rfecv_rf = RFECV(
        estimator=rf,
        step=1,
        cv=cv,
        scoring="average_precision",
        n_jobs=-1,
    )

    rfecv_rf.fit(X_train_corr_rf, y_train)

    cv_result_rf = pd.DataFrame({
        "n_features": rfecv_rf.cv_results_["n_features"],
        "mean_score": rfecv_rf.cv_results_["mean_test_score"],
        "std_score": rfecv_rf.cv_results_["std_test_score"],
    })

    print(cv_result_rf.sort_values(by="mean_score", ascending=False).head(10))

    cv_result_rf.to_csv(output_dir / "rf_rfe_cv_result.csv", index=False, encoding="utf-8-sig")

    rf_choice = choose_compact_n_features(cv_result_rf, cv_folds=5)
    chosen_n_features = rf_choice["chosen_n_features"]

    plot_rfecv_curve(
        cv_result_rf,
        rf_choice,
        output_dir / "rf_rfe_cv_curve.png",
        title="Recursive Feature Elimination (RF)",
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=666,
        n_jobs=1,
    )

    rfe_rf = RFE(
        estimator=rf,
        n_features_to_select=chosen_n_features,
        step=1,
    )

    rfe_rf.fit(X_train_corr_rf, y_train)

    selected_features = X_train_corr_rf.columns[rfe_rf.support_].tolist()
    X_train_rferf = X_train_corr_rf[selected_features].copy()
    X_test_rferf = X_test_corr_rf[selected_features].copy()

    print("selected n_features:", rfe_rf.n_features_)
    print("selected features:", selected_features)
    print("ranking:", dict(zip(X_train_corr_rf.columns, rfe_rf.ranking_)))

    pd.DataFrame({
        "feature": X_train_corr_rf.columns,
        "ranking": rfe_rf.ranking_,
        "selected": rfe_rf.support_,
    }).to_csv(output_dir / "rf_rfe_feature_ranking.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame({"feature": selected_features}).to_csv(
        output_dir / "rf_rfe_selected_features.csv",
        index=False,
        encoding="utf-8-sig",
    )

    log["steps"].append({
        "step": "rf_rfe",
        "choice": rf_choice,
        "X_train_rferf_shape": list(X_train_rferf.shape),
        "X_test_rferf_shape": list(X_test_rferf.shape),
    })

    # =========================================================
    # 2-2. XGB-RFE
    # IMPORTANT:
    # Keep random_state=66 exactly as the ipynb.
    # =========================================================
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=66,
        scale_pos_weight=neg / pos,
        n_jobs=1,
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=666,
    )

    rfecv_xgb = RFECV(
        estimator=xgb,
        step=1,
        cv=cv,
        scoring="average_precision",
        n_jobs=-1,
    )

    rfecv_xgb.fit(X_train_corr_rf, y_train)

    cv_result_xgb = pd.DataFrame({
        "n_features": rfecv_xgb.cv_results_["n_features"],
        "mean_score": rfecv_xgb.cv_results_["mean_test_score"],
        "std_score": rfecv_xgb.cv_results_["std_test_score"],
    })

    print(cv_result_xgb.sort_values(by="mean_score", ascending=False).head(10))

    cv_result_xgb.to_csv(output_dir / "xgb_rfe_cv_result.csv", index=False, encoding="utf-8-sig")

    xgb_choice = choose_compact_n_features(cv_result_xgb, cv_folds=5)
    chosen_n_features = xgb_choice["chosen_n_features"]

    plot_rfecv_curve(
        cv_result_xgb,
        xgb_choice,
        output_dir / "xgb_rfe_cv_curve.png",
        title="Recursive Feature Elimination (XGB)",
    )

    rfe_xgb = RFE(
        estimator=xgb,
        n_features_to_select=chosen_n_features,
        step=1,
    )

    rfe_xgb.fit(X_train_corr_rf, y_train)

    selected_features = X_train_corr_rf.columns[rfe_xgb.support_].tolist()
    X_train_rfexgb = X_train_corr_rf[selected_features].copy()
    X_test_rfexgb = X_test_corr_rf[selected_features].copy()

    print("selected n_features:", rfe_xgb.n_features_)
    print("selected features:", selected_features)
    print("ranking:", dict(zip(X_train_corr_rf.columns, rfe_xgb.ranking_)))

    pd.DataFrame({
        "feature": X_train_corr_rf.columns,
        "ranking": rfe_xgb.ranking_,
        "selected": rfe_xgb.support_,
    }).to_csv(output_dir / "xgb_rfe_feature_ranking.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame({"feature": selected_features}).to_csv(
        output_dir / "xgb_rfe_selected_features.csv",
        index=False,
        encoding="utf-8-sig",
    )

    log["steps"].append({
        "step": "xgb_rfe",
        "choice": xgb_choice,
        "X_train_rfexgb_shape": list(X_train_rfexgb.shape),
        "X_test_rfexgb_shape": list(X_test_rfexgb.shape),
    })

    # =========================================================
    # 2-3. ANOVA F-test / SelectKBest (K=50)
    # =========================================================
    anova_k = min(50, X_train_corr_rf.shape[1])

    anova_selector = SelectKBest(score_func=f_classif, k=anova_k)
    X_train_anova_arr = anova_selector.fit_transform(X_train_corr_rf, y_train)
    X_test_anova_arr = anova_selector.transform(X_test_base)

    anova_selected_features = X_train_corr_rf.columns[anova_selector.get_support()].tolist()

    X_train_anova = pd.DataFrame(
        X_train_anova_arr,
        columns=anova_selected_features,
        index=X_train_corr_rf.index,
    )
    X_test_anova = pd.DataFrame(
        X_test_anova_arr,
        columns=anova_selected_features,
        index=X_test_base.index,
    )

    print("selected n_features:", X_train_anova.shape[1])
    print("selected features:", anova_selected_features)

    pd.DataFrame({
        "feature": X_train_corr_rf.columns,
        "anova_score": anova_selector.scores_,
        "p_value": anova_selector.pvalues_,
        "selected": anova_selector.get_support(),
    }).sort_values("anova_score", ascending=False).to_csv(
        output_dir / "anova_feature_scores.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame({"feature": anova_selected_features}).to_csv(
        output_dir / "anova_selected_features.csv",
        index=False,
        encoding="utf-8-sig",
    )

    joblib.dump(anova_selector, model_dir / "anova_selector_ipynb_exact.joblib")

    log["steps"].append({
        "step": "anova_select_k_best",
        "anova_k": int(anova_k),
        "X_train_anova_shape": list(X_train_anova.shape),
        "X_test_anova_shape": list(X_test_anova.shape),
    })

    # =========================================================
    # Save final processed datasets
    # Same filenames as ipynb, but under processed_dir.
    # =========================================================
    save_csv(X_train_base, processed_dir / "X_train_base.csv")
    save_csv(X_train_rferf, processed_dir / "X_train_rferf.csv")
    save_csv(X_train_rfexgb, processed_dir / "X_train_rfexgb.csv")
    save_csv(X_train_anova, processed_dir / "X_train_anova.csv")

    save_csv(X_test_base, processed_dir / "X_test_base.csv")
    save_csv(X_test_rferf, processed_dir / "X_test_rferf.csv")
    save_csv(X_test_rfexgb, processed_dir / "X_test_rfexgb.csv")
    save_csv(X_test_anova, processed_dir / "X_test_anova.csv")

    feature_sets = {
        "base": X_train_base.columns.tolist(),
        "rferf": X_train_rferf.columns.tolist(),
        "rfexgb": X_train_rfexgb.columns.tolist(),
        "anova": X_train_anova.columns.tolist(),
    }

    with open(processed_dir / "feature_sets.json", "w", encoding="utf-8") as f:
        json.dump(feature_sets, f, ensure_ascii=False, indent=2)

    final_shapes = {
        "X_train_base": list(X_train_base.shape),
        "X_test_base": list(X_test_base.shape),
        "X_train_rferf": list(X_train_rferf.shape),
        "X_test_rferf": list(X_test_rferf.shape),
        "X_train_rfexgb": list(X_train_rfexgb.shape),
        "X_test_rfexgb": list(X_test_rfexgb.shape),
        "X_train_anova": list(X_train_anova.shape),
        "X_test_anova": list(X_test_anova.shape),
    }

    log["final_shapes"] = final_shapes

    with open(output_dir / "preprocessing_log_ipynb_exact.json", "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    print("X_train_base:", X_train_base.shape)
    print("X_train_rferf:", X_train_rferf.shape)
    print("X_train_rfexgb:", X_train_rfexgb.shape)
    print("X_train_anova:", X_train_anova.shape)

    print("\nFinal shapes:")
    for name, shape in final_shapes.items():
        print(f"{name}: {shape}")

    print("\nSaved processed datasets to:", processed_dir.resolve())
    print("Saved preprocessing reports to:", output_dir.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run notebook-exact preprocessing for the UCI SECOM dataset."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/uci-secom.csv"),
        help="Path to input CSV. Default: data/raw/uci-secom.csv",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for processed CSV outputs. Default: data/processed",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/preprocessing"),
        help="Directory for preprocessing plots/logs. Default: outputs/preprocessing",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models"),
        help="Directory for fitted preprocessing objects. Default: models",
    )

    args = parser.parse_args()

    main(
        input_path=args.input,
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
    )
