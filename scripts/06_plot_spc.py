"""
06_plot_spc.py

SPC plot script converted from the original notebook block.

Purpose:
    Run after 05_spc_values.py to plot SPC charts from exported SPC value tables.

Expected inputs:
    outputs/spc_values/
    outputs/shap/spc_list__main_signal.csv
    outputs/shap/spc_list__auxiliary_diagnostic.csv
    outputs/shap/spc_list__final_deployment.csv or final_monitor_candidates.csv
    data/processed/X_test.csv
    data/processed/y_test.csv

Default output:
    outputs/spc_plots/

Run:
    python scripts/06_plot_spc.py

or:
    python scripts/06_plot_spc.py \
        --shap-output-dir outputs/shap \
        --spc-values-root outputs/spc_values \
        --output-dir outputs/spc_plots \
        --processed-dir data/processed
"""

# =========================================================
# SPC plot block
# 接在前一個 SPC values export block 後面直接執行
# ---------------------------------------------------------
# 輸出結構:
#   outputs/spc_plots/
#     main_signal/
#         tp_like/
#           shewhart_3sigma/
#           moving_range/
#           ewma/
#           cusum/
#           p_chart/
#           g_chart/
#           multivariate_t2/
#           pca_t2/
#           pca_spe/
#           pc1_pc2_scatter/
#         fp_like/
#         test_holdout/
#         split_stable/
#       auxiliary_diagnostic/
#         missingness/
#           p_chart/
#           g_chart/
#           shewhart_3sigma/
#           moving_range/
#           ewma/
#           cusum/
#           multivariate_t2/
#           pca_t2/
#           pca_spe/
#           pc1_pc2_scatter/
#         interaction/
#           pair_distance/
#           pair_hotelling_t2/
#           pair_scatter/
#           multivariate_t2/
#           pca_t2/
#           pca_spe/
#           pc1_pc2_scatter/
#         distribution_separation/
#           shewhart_3sigma/
#           moving_range/
#           ewma/
#           cusum/
#           p_chart/
#           g_chart/
#           multivariate_t2/
#           pca_t2/
#           pca_spe/
#           pc1_pc2_scatter/
#       final_deployment/
#         Tier_A/
#         Tier_B/
#
# 檔名格式:
#   rank_特徵名_分類名_圖名.png
# =========================================================

import os
import re
import io
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Config
# ---------------------------
SPC_PLOT_ROOT_NAME = ""
FIG_DPI = 160
FIGSIZE_WIDE = (14, 5)
FIGSIZE_STD = (12, 5)

# ---------------------------
# Project path config
# ---------------------------
# This script is intended to run after 05_spc_values.py.
# Default project structure:
#   outputs/shap/        : SHAP candidate lists from 04_shap_analysis.py
#   outputs/spc_values/  : SPC values from 05_spc_values.py
#   outputs/spc_plots/   : SPC plots from this script
SHAP_OUTPUT_DIR = "outputs/shap"
SPC_VALUES_ROOT = "outputs/spc_values"
SPC_PLOT_OUTPUT_DIR = "outputs/spc_plots"

# Backward-compatible alias used by the original notebook code.
OUT_DIR = SHAP_OUTPUT_DIR

PROCESSED_DIR = "data/processed"
USE_TEST_AS_MONITOR = True
SUBGROUP_SIZE = 25


if "OUT_DIR_USE" not in globals():
    OUT_DIR_USE = SHAP_OUTPUT_DIR if "SHAP_OUTPUT_DIR" in globals() else "outputs/shap"

# Read SPC values from outputs/spc_values by default.
if "SPC_EXPORT_ROOT" not in globals():
    SPC_EXPORT_ROOT = SPC_VALUES_ROOT if SPC_VALUES_ROOT else "outputs/spc_values"

# Write SPC plots to outputs/spc_plots by default.
SPC_PLOT_ROOT = SPC_PLOT_OUTPUT_DIR if "SPC_PLOT_OUTPUT_DIR" in globals() else "outputs/spc_plots"

# Safety guard: never use the old nested folders in project mode.
if str(SPC_EXPORT_ROOT).replace(chr(92), "/").endswith("outputs/shap/spc_values_grouped"):
    SPC_EXPORT_ROOT = "outputs/spc_values"
if str(SPC_PLOT_ROOT).replace(chr(92), "/").endswith("outputs/shap/spc_plots_grouped"):
    SPC_PLOT_ROOT = "outputs/spc_plots"

Path(SPC_PLOT_ROOT).mkdir(parents=True, exist_ok=True)

# ---------------------------
# Basic utils
# ---------------------------
def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def read_csv_flex_local(path: str) -> pd.DataFrame:
    """
    Robust CSV reader for SPC value files and SHAP list files.
    """
    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")

    encodings = [
        "utf-8-sig",
        "utf-8",
        "utf-16",
        "utf-16-le",
        "utf-16-be",
        "cp950",
        "big5",
        "latin1",
    ]

    errors = []
    for enc in encodings:
        try:
            return pd.read_csv(
                path,
                encoding=enc,
                encoding_errors="replace",
                engine="python",
            )
        except Exception as e:
            errors.append(f"{enc}: {type(e).__name__}: {e}")

    try:
        raw = Path(path).read_bytes()
        text = raw.decode("utf-8", errors="replace")
        return pd.read_csv(io.StringIO(text), engine="python")
    except Exception as e:
        errors.append(f"manual utf-8 replace: {type(e).__name__}: {e}")

    raise RuntimeError(
        f"Failed to read CSV file: {path}\n"
        f"Tried encodings/parsers:\n" + "\n".join(errors[-8:])
    )


def save_and_close(fig, out_path: str):
    ensure_dir(Path(out_path).parent)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def safe_filename_part(s):
    if pd.isna(s):
        return "NA"
    s = str(s).strip()
    s = re.sub(r'[\\/:*?"<>|]+', "_", s)
    s = re.sub(r"\s+", "_", s)
    return s if s else "NA"


def make_x(df: pd.DataFrame):
    if df is None or df.empty:
        return pd.Series(dtype=float)
    if "time_index" in df.columns:
        return pd.Series(df["time_index"]).reset_index(drop=True)
    return pd.Series(np.arange(len(df)))


def safe_col(df: pd.DataFrame, col: str):
    if df is None or df.empty or col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce").reset_index(drop=True)


def get_summary_df(summary_path: str):
    if not os.path.exists(summary_path):
        return pd.DataFrame()
    df = read_csv_flex_local(summary_path)
    return df if df is not None else pd.DataFrame()


def first_non_null(summary_df: pd.DataFrame, cols, default=None):
    if summary_df is None or summary_df.empty:
        return default
    for c in cols:
        if c in summary_df.columns:
            v = summary_df.iloc[0][c]
            if pd.notna(v):
                return v
    return default


def get_rank_str(summary_df: pd.DataFrame):
    rank_candidates = ["spc_export_rank", "export_rank", "anchor_rank", "rank"]
    v = first_non_null(summary_df, rank_candidates, default="999")
    try:
        return f"{int(float(v)):03d}"
    except Exception:
        return safe_filename_part(v)


def get_class_name(summary_df: pd.DataFrame, fallback_subtype: str):
    cls_candidates = [
        "list_subtype", "spc_selection_tier", "anchor_recommendation",
        "list_subtype__meta", "spc_selection_tier__meta", "anchor_recommendation__meta"
    ]
    v = first_non_null(summary_df, cls_candidates, default=fallback_subtype)
    return safe_filename_part(v)


def get_feature_name(summary_df: pd.DataFrame, fallback_name: str):
    v = first_non_null(summary_df, ["feature"], default=fallback_name)
    return safe_filename_part(v)


def get_pair_name(summary_df: pd.DataFrame, fallback_name: str):
    fa = first_non_null(summary_df, ["feature_a"], default=None)
    fb = first_non_null(summary_df, ["feature_b"], default=None)
    if fa is not None and fb is not None:
        return safe_filename_part(f"{fa}__{fb}")
    return safe_filename_part(fallback_name)


def build_plot_filename(rank_str: str, feature_name: str, class_name: str, chart_name: str):
    return f"{rank_str}_{feature_name}_{class_name}_{chart_name}.png"


def short_name_from_dir(path: str):
    return Path(path).name


# ---------------------------
# Canonical key helpers
# ---------------------------
def canonical_feature_key(x):
    if pd.isna(x):
        return None

    s = str(x).replace("\ufeff", "").strip()
    s = re.sub(r"\s+", " ", s)

    candidates = [s]

    try:
        xf = float(s)
        candidates.append(str(xf))
        if abs(xf - round(xf)) < 1e-9:
            candidates.append(str(int(round(xf))))
            candidates.append(f"{int(round(xf))}.0")
    except Exception:
        pass

    if re.fullmatch(r"-?\d+\.0+", s):
        try:
            candidates.append(str(int(float(s))))
        except Exception:
            pass

    candidates.append(s.lower())
    candidates.append(s.replace(" ", ""))

    seen = set()
    out = []
    for c in candidates:
        if c is None:
            continue
        cc = str(c).strip()
        if cc not in seen:
            seen.add(cc)
            out.append(cc)
    return out


def build_pair_key(a, b):
    a = safe_filename_part(a)
    b = safe_filename_part(b)
    parts = sorted([a, b])
    return "||".join(parts)


def resolve_route_by_feature(feature_name, feature_route_map):
    keys = canonical_feature_key(feature_name)
    if keys is None:
        return None
    for k in keys:
        if k in feature_route_map:
            return feature_route_map[k]
    return None



# ---------------------------
# CLI arguments
# ---------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot SPC charts from SPC values exported by 05_spc_values.py."
    )
    parser.add_argument(
        "--shap-output-dir",
        default="outputs/shap",
        help="Directory containing SHAP SPC list files. Default: outputs/shap",
    )
    parser.add_argument(
        "--spc-values-root",
        default="outputs/spc_values",
        help="SPC values root from 05_spc_values.py. Default: outputs/spc_values",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/spc_plots",
        help="Directory to save SPC plot outputs. Default: outputs/spc_plots",
    )
    parser.add_argument(
        "--processed-dir",
        default="data/processed",
        help="Directory containing X_test.csv/y_test.csv for anomaly markers. Default: data/processed",
    )
    parser.add_argument(
        "--fig-dpi",
        type=int,
        default=160,
        help="Figure DPI. Default: 160",
    )
    parser.add_argument(
        "--subgroup-size",
        type=int,
        default=25,
        help="Subgroup size used by p-chart. Should match 05_spc_values.py. Default: 25",
    )
    parser.add_argument(
        "--use-train-as-monitor",
        action="store_true",
        help="Use train data labels for anomaly markers instead of test labels.",
    )

    args = parser.parse_args()

    SHAP_OUTPUT_DIR = args.shap_output_dir
    OUT_DIR_USE = SHAP_OUTPUT_DIR
    OUT_DIR = SHAP_OUTPUT_DIR

    SPC_EXPORT_ROOT = args.spc_values_root
    SPC_PLOT_ROOT = args.output_dir

    PROCESSED_DIR = args.processed_dir
    FIG_DPI = args.fig_dpi
    SUBGROUP_SIZE = args.subgroup_size
    USE_TEST_AS_MONITOR = not args.use_train_as_monitor

    # Safety guard: never use the old nested folders in project mode.
    if str(SPC_EXPORT_ROOT).replace(chr(92), "/").endswith("outputs/shap/spc_values_grouped"):
        SPC_EXPORT_ROOT = "outputs/spc_values"
    if str(SPC_PLOT_ROOT).replace(chr(92), "/").endswith("outputs/shap/spc_plots_grouped"):
        SPC_PLOT_ROOT = "outputs/spc_plots"

    Path(SPC_PLOT_ROOT).mkdir(parents=True, exist_ok=True)



# ---------------------------
# Load list tables for routing
# ---------------------------
def load_optional_output(out_dir: str, candidates):
    for name in candidates:
        p = os.path.join(out_dir, name)
        if os.path.exists(p):
            return read_csv_flex_local(p), p
    return pd.DataFrame(), None


main_signal_df, _ = load_optional_output(
    OUT_DIR_USE,
    ["spc_list__main_signal.csv"]
)
aux_diag_df, _ = load_optional_output(
    OUT_DIR_USE,
    ["spc_list__auxiliary_diagnostic.csv"]
)
final_deploy_df, _ = load_optional_output(
    OUT_DIR_USE,
    ["spc_list__final_deployment.csv", "final_monitor_candidates.csv"]
)


def normalize_list_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    for c in ["feature", "feature_a", "feature_b", "list_subtype", "spc_selection_tier", "anchor_recommendation"]:
        if c in out.columns:
            out[c] = out[c].astype("object")
    return out


main_signal_df = normalize_list_df(main_signal_df)
aux_diag_df = normalize_list_df(aux_diag_df)
final_deploy_df = normalize_list_df(final_deploy_df)


def build_feature_route_map(df: pd.DataFrame, subtype_col: str):
    route_map = {}
    if df is None or df.empty or "feature" not in df.columns or subtype_col not in df.columns:
        return route_map

    for _, row in df.iterrows():
        feat = row.get("feature")
        subtype = row.get(subtype_col)
        if pd.isna(feat) or pd.isna(subtype):
            continue
        keys = canonical_feature_key(feat)
        if keys is None:
            continue
        for k in keys:
            route_map[k] = safe_filename_part(subtype)
    return route_map


def build_pair_route_map(df: pd.DataFrame, subtype_col: str):
    route_map = {}
    if df is None or df.empty or "feature_a" not in df.columns or "feature_b" not in df.columns or subtype_col not in df.columns:
        return route_map

    for _, row in df.iterrows():
        fa = row.get("feature_a")
        fb = row.get("feature_b")
        subtype = row.get(subtype_col)
        if pd.isna(fa) or pd.isna(fb) or pd.isna(subtype):
            continue
        key = build_pair_key(fa, fb)
        route_map[key] = safe_filename_part(subtype)
    return route_map


MAIN_FEATURE_ROUTE = build_feature_route_map(main_signal_df, "list_subtype")
AUX_FEATURE_ROUTE = build_feature_route_map(aux_diag_df, "list_subtype")
AUX_PAIR_ROUTE = build_pair_route_map(aux_diag_df, "list_subtype")

if "spc_selection_tier" in final_deploy_df.columns:
    FINAL_FEATURE_ROUTE = build_feature_route_map(final_deploy_df, "spc_selection_tier")
else:
    if "anchor_recommendation" in final_deploy_df.columns:
        tmp = final_deploy_df.copy()
        tmp["spc_selection_tier"] = tmp["anchor_recommendation"].map({
            "primary_anchor": "Tier_A",
            "review_anchor": "Tier_B",
        }).fillna("Tier_B")
        FINAL_FEATURE_ROUTE = build_feature_route_map(tmp, "spc_selection_tier")
    else:
        FINAL_FEATURE_ROUTE = {}

# ---------------------------
# Proactively create subtype dirs
# ---------------------------
def ensure_expected_subtype_dirs():
    expected = {
        "main_signal": sorted(set(MAIN_FEATURE_ROUTE.values())) if MAIN_FEATURE_ROUTE else ["tp_like", "fp_like", "test_holdout", "split_stable"],
        "auxiliary_diagnostic": sorted(set(AUX_FEATURE_ROUTE.values()) | set(AUX_PAIR_ROUTE.values())) if (AUX_FEATURE_ROUTE or AUX_PAIR_ROUTE) else ["missingness", "interaction", "distribution_separation"],
        "final_deployment": sorted(set(FINAL_FEATURE_ROUTE.values())) if FINAL_FEATURE_ROUTE else ["Tier_A", "Tier_B"],
    }

    for category, subtypes in expected.items():
        for subtype in subtypes:
            ensure_dir(os.path.join(SPC_PLOT_ROOT, category, safe_filename_part(subtype)))

ensure_expected_subtype_dirs()


# ---------------------------
# Optional monitor data loader
# ---------------------------
def load_label_simple_for_plot(paths):
    for p in paths:
        if p and os.path.exists(p):
            y_df = read_csv_flex_local(p)
            y_df = y_df.loc[:, ~y_df.columns.astype(str).str.contains(r"^Unnamed")]
            if y_df.shape[1] == 0:
                continue
            y = y_df.iloc[:, 0] if y_df.shape[1] == 1 else y_df[y_df.columns[0]]
            y = pd.Series(y).reset_index(drop=True)

            if y.dtype == bool:
                return y.astype(int)
            if str(y.dtype).startswith("int") or str(y.dtype).startswith("float"):
                return pd.to_numeric(y, errors="coerce").fillna(0).astype(int)

            mapping = {
                "normal": 0, "anomaly": 1, "abnormal": 1,
                "yes": 1, "no": 0, "true": 1, "false": 0,
                "positive": 1, "negative": 0, "pass": 0, "fail": 1,
                "-1": 0, "1": 1, "0": 0,
            }
            y_str = y.astype(str).str.strip().str.lower()
            if y_str.isin(mapping.keys()).all():
                return y_str.map(mapping).astype(int)
            return pd.to_numeric(y, errors="coerce").fillna(0).astype(int)
    return None


def load_monitor_context_if_needed():
    """
    The original notebook version used X_monitor_raw / y_test from the previous cell.
    In .py form, load them from data/processed if they are not already defined.
    """
    global X_monitor_raw, y_train, y_test

    if "X_monitor_raw" not in globals():
        test_path = os.path.join(PROCESSED_DIR, "X_test.csv")
        train_path = os.path.join(PROCESSED_DIR, "X_train.csv")

        if USE_TEST_AS_MONITOR and os.path.exists(test_path):
            X_monitor_raw = read_csv_flex_local(test_path)
        elif os.path.exists(train_path):
            X_monitor_raw = read_csv_flex_local(train_path)
        else:
            X_monitor_raw = pd.DataFrame()

        if not X_monitor_raw.empty:
            X_monitor_raw = X_monitor_raw.loc[:, ~X_monitor_raw.columns.astype(str).str.contains(r"^Unnamed")]

    if "y_test" not in globals():
        y_test = load_label_simple_for_plot([
            os.path.join(PROCESSED_DIR, "y_test.csv"),
            os.path.join(PROCESSED_DIR, "y_test_base.csv"),
        ])

    if "y_train" not in globals():
        y_train = load_label_simple_for_plot([
            os.path.join(PROCESSED_DIR, "y_train.csv"),
            os.path.join(PROCESSED_DIR, "y_train_base.csv"),
        ])


load_monitor_context_if_needed()



# ---------------------------
# Monitor anomaly mask
# ---------------------------
def get_monitor_anomaly_mask():
    if "X_monitor_raw" not in globals():
        return pd.Series(dtype=int)

    n = len(X_monitor_raw)

    if globals().get("USE_TEST_AS_MONITOR", True):
        if "y_test" in globals() and y_test is not None and len(y_test) == n:
            yy = pd.to_numeric(pd.Series(y_test), errors="coerce").fillna(0).astype(int)
            return yy.reset_index(drop=True)

    if "y_train" in globals() and y_train is not None and len(y_train) == n:
        yy = pd.to_numeric(pd.Series(y_train), errors="coerce").fillna(0).astype(int)
        return yy.reset_index(drop=True)

    return pd.Series(np.zeros(n, dtype=int))


MONITOR_ANOMALY_MASK = get_monitor_anomaly_mask()


def subgroup_any_mask(point_mask: pd.Series, subgroup_size: int):
    point_mask = pd.Series(point_mask).fillna(0).astype(int).reset_index(drop=True)
    if len(point_mask) == 0:
        return pd.Series(dtype=int)

    subgroup_size = max(1, subgroup_size)
    subgroup_id = np.arange(len(point_mask)) // subgroup_size
    out = pd.DataFrame({"mask": point_mask, "subgroup_id": subgroup_id})
    out = out.groupby("subgroup_id", as_index=False)["mask"].max()
    return out["mask"].astype(int)


def g_chart_anomaly_mask_from_feature(feature_name: str):
    if "X_monitor_raw" not in globals() or feature_name not in X_monitor_raw.columns:
        return pd.Series(dtype=int)

    event_monitor = X_monitor_raw[feature_name].isna().astype(int).reset_index(drop=True)
    anomaly_mask = pd.Series(MONITOR_ANOMALY_MASK).fillna(0).astype(int).reset_index(drop=True)
    event_pos = np.where(event_monitor.values == 1)[0]
    if len(event_pos) <= 1:
        return pd.Series(dtype=int)

    end_event_pos = event_pos[1:]
    return anomaly_mask.iloc[end_event_pos].reset_index(drop=True)


# ---------------------------
# Plot symbols
# ---------------------------
def add_anomaly_x(ax, x, y, anomaly_mask, label="Anomaly sample"):
    if len(x) == 0 or len(y) == 0:
        return
    x = pd.Series(x).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    m = pd.Series(anomaly_mask).fillna(0).astype(int).reset_index(drop=True)
    sel = (m == 1) & y.notna()
    if sel.sum() > 0:
        ax.scatter(
            x[sel], y[sel],
            marker="x", s=42, linewidths=1.4,
            c="red", label=label, zorder=6
        )


def add_alarm_circle(ax, x, y, alarm_mask, label="Alarm"):
    if len(x) == 0 or len(y) == 0:
        return
    x = pd.Series(x).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    m = pd.Series(alarm_mask).fillna(0).astype(int).reset_index(drop=True)
    sel = (m == 1) & y.notna()
    if sel.sum() > 0:
        ax.scatter(
            x[sel], y[sel],
            marker="o", s=46,
            facecolors="none", edgecolors="orange", linewidths=1.6,
            label=label, zorder=7
        )


def set_common_axis_style(ax, title: str, xlabel: str = "Index", ylabel: str = ""):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")


# ---------------------------
# subtype resolver
# 先用 list routing，再 fallback
# ---------------------------
def infer_subtype_name(category_name: str, summary_df: pd.DataFrame, current_path: str):
    # 1) summary 直接指定
    if summary_df is not None and not summary_df.empty:
        if category_name == "main_signal":
            v = first_non_null(summary_df, ["list_subtype", "list_subtype__meta"], default=None)
            if v is not None:
                return safe_filename_part(v)
        elif category_name == "auxiliary_diagnostic":
            # pair
            fa = first_non_null(summary_df, ["feature_a"], default=None)
            fb = first_non_null(summary_df, ["feature_b"], default=None)
            if fa is not None and fb is not None:
                route = AUX_PAIR_ROUTE.get(build_pair_key(fa, fb))
                if route is not None:
                    return safe_filename_part(route)

            # feature
            feat = first_non_null(summary_df, ["feature"], default=None)
            if feat is not None:
                route = resolve_route_by_feature(feat, AUX_FEATURE_ROUTE)
                if route is not None:
                    return safe_filename_part(route)

            v = first_non_null(summary_df, ["list_subtype", "list_subtype__meta"], default=None)
            if v is not None:
                return safe_filename_part(v)

        elif category_name == "final_deployment":
            feat = first_non_null(summary_df, ["feature"], default=None)
            if feat is not None:
                route = resolve_route_by_feature(feat, FINAL_FEATURE_ROUTE)
                if route is not None:
                    return safe_filename_part(route)

            v = first_non_null(summary_df, ["spc_selection_tier", "spc_selection_tier__meta", "anchor_recommendation"], default=None)
            if v is not None:
                if str(v) in ["primary_anchor", "Tier_A"]:
                    return "Tier_A"
                if str(v) in ["review_anchor", "Tier_B"]:
                    return "Tier_B"
                return safe_filename_part(v)

    # 2) path fallback
    p = str(current_path).replace("\\", "/").lower()

    if category_name == "auxiliary_diagnostic":
        if "/interaction_pairs/" in p:
            return "interaction"
        if "/missingness_features/" in p:
            return "missingness"
        return "distribution_separation"

    if category_name == "main_signal":
        for cand in ["tp_like", "fp_like", "test_holdout", "split_stable"]:
            if f"/{cand}/" in p:
                return cand
        return "main_signal"

    if category_name == "final_deployment":
        for cand in ["tier_a", "tier_b", "primary_anchor", "review_anchor"]:
            if f"/{cand}/" in p:
                if cand in ["tier_a", "primary_anchor"]:
                    return "Tier_A"
                if cand in ["tier_b", "review_anchor"]:
                    return "Tier_B"
        return "Tier_A"

    return "NA"


# ---------------------------
# Output path helper
# 結構: category / subtype / chart_type / file.png
# ---------------------------
def get_chart_output_path(category_name: str, subtype_name: str, chart_name: str, rank_str: str, feature_name: str, class_name: str):
    chart_dir = os.path.join(SPC_PLOT_ROOT, category_name, subtype_name, chart_name)
    ensure_dir(chart_dir)
    fname = build_plot_filename(rank_str, feature_name, class_name, chart_name)
    return os.path.join(chart_dir, fname)


# ---------------------------
# Plotters: continuous feature
# ---------------------------
def plot_shewhart(series_df: pd.DataFrame, category_name: str, subtype_name: str, feature_name: str, class_name: str, rank_str: str):
    if series_df.empty or "value" not in series_df.columns:
        return

    x = make_x(series_df)
    y = safe_col(series_df, "value")
    cl = safe_col(series_df, "CL")
    ucl = safe_col(series_df, "UCL")
    lcl = safe_col(series_df, "LCL")
    alarm_hi = safe_col(series_df, "beyond_UCL")
    alarm_lo = safe_col(series_df, "below_LCL")
    alarm_any = ((alarm_hi.fillna(0) == 1) | (alarm_lo.fillna(0) == 1)).astype(int)

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.plot(x, y, label="Value")
    if cl.notna().any():
        ax.plot(x, cl, linestyle="--", label="CL")
    if ucl.notna().any():
        ax.plot(x, ucl, linestyle="--", label="UCL (+3σ)")
    if lcl.notna().any():
        ax.plot(x, lcl, linestyle="--", label="LCL (-3σ)")

    add_anomaly_x(ax, x, y, MONITOR_ANOMALY_MASK, label="Anomaly sample")
    add_alarm_circle(ax, x, y, alarm_any, label="Alarm")

    set_common_axis_style(ax, f"{feature_name} - Shewhart 3σ", ylabel=feature_name)

    out_path = get_chart_output_path(category_name, subtype_name, "shewhart_3sigma", rank_str, feature_name, class_name)
    save_and_close(fig, out_path)


def plot_mr(series_df: pd.DataFrame, category_name: str, subtype_name: str, feature_name: str, class_name: str, rank_str: str):
    if series_df.empty or "MR" not in series_df.columns:
        return

    x = make_x(series_df)
    y = safe_col(series_df, "MR")
    cl = safe_col(series_df, "MR_CL")
    ucl = safe_col(series_df, "MR_UCL")
    lcl = safe_col(series_df, "MR_LCL")
    alarm = safe_col(series_df, "MR_beyond_UCL")

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.plot(x, y, label="Moving Range")
    if cl.notna().any():
        ax.plot(x, cl, linestyle="--", label="MR CL")
    if ucl.notna().any():
        ax.plot(x, ucl, linestyle="--", label="MR UCL")
    if lcl.notna().any():
        ax.plot(x, lcl, linestyle="--", label="MR LCL")

    add_anomaly_x(ax, x, y, MONITOR_ANOMALY_MASK, label="Anomaly sample")
    add_alarm_circle(ax, x, y, alarm, label="Alarm")

    set_common_axis_style(ax, f"{feature_name} - Moving Range", ylabel="MR")

    out_path = get_chart_output_path(category_name, subtype_name, "moving_range", rank_str, feature_name, class_name)
    save_and_close(fig, out_path)


def plot_ewma(series_df: pd.DataFrame, category_name: str, subtype_name: str, feature_name: str, class_name: str, rank_str: str):
    if series_df.empty or "EWMA" not in series_df.columns:
        return

    x = make_x(series_df)
    ewma = safe_col(series_df, "EWMA")
    cl = safe_col(series_df, "EWMA_CL")
    ucl = safe_col(series_df, "EWMA_UCL")
    lcl = safe_col(series_df, "EWMA_LCL")
    alarm = safe_col(series_df, "EWMA_beyond_any_limit")

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.plot(x, ewma, label="EWMA")
    if cl.notna().any():
        ax.plot(x, cl, linestyle="--", label="EWMA CL")
    if ucl.notna().any():
        ax.plot(x, ucl, linestyle="--", label="EWMA UCL")
    if lcl.notna().any():
        ax.plot(x, lcl, linestyle="--", label="EWMA LCL")

    add_anomaly_x(ax, x, ewma, MONITOR_ANOMALY_MASK, label="Anomaly sample")
    add_alarm_circle(ax, x, ewma, alarm, label="Alarm")

    set_common_axis_style(ax, f"{feature_name} - EWMA", ylabel="EWMA")

    out_path = get_chart_output_path(category_name, subtype_name, "ewma", rank_str, feature_name, class_name)
    save_and_close(fig, out_path)


def plot_cusum(series_df: pd.DataFrame, category_name: str, subtype_name: str, feature_name: str, class_name: str, rank_str: str):
    if series_df.empty or "CUSUM_CPLUS" not in series_df.columns:
        return

    x = make_x(series_df)
    cplus = safe_col(series_df, "CUSUM_CPLUS")
    cminus = safe_col(series_df, "CUSUM_CMINUS")
    hpos = safe_col(series_df, "CUSUM_H_POS")
    hneg = safe_col(series_df, "CUSUM_H_NEG")
    alarm_any = safe_col(series_df, "CUSUM_alarm_any")

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.plot(x, cplus, label="CUSUM C+")
    ax.plot(x, cminus, label="CUSUM C-")
    if hpos.notna().any():
        ax.plot(x, hpos, linestyle="--", label="H+")
    if hneg.notna().any():
        ax.plot(x, hneg, linestyle="--", label="H-")

    add_anomaly_x(ax, x, cplus, MONITOR_ANOMALY_MASK, label="Anomaly sample")
    add_alarm_circle(ax, x, cplus, alarm_any, label="Alarm")

    set_common_axis_style(ax, f"{feature_name} - CUSUM", ylabel="CUSUM")

    out_path = get_chart_output_path(category_name, subtype_name, "cusum", rank_str, feature_name, class_name)
    save_and_close(fig, out_path)


def plot_feature_folder(feature_dir: str, category_name: str):
    series_path = os.path.join(feature_dir, "series.csv")
    summary_path = os.path.join(feature_dir, "summary.csv")
    if not os.path.exists(series_path):
        return

    series_df = read_csv_flex_local(series_path)
    summary_df = get_summary_df(summary_path)

    subtype_name = infer_subtype_name(category_name, summary_df, feature_dir)
    rank_str = get_rank_str(summary_df)
    class_name = get_class_name(summary_df, fallback_subtype=subtype_name)
    feature_name = get_feature_name(summary_df, fallback_name=short_name_from_dir(feature_dir))

    plot_shewhart(series_df, category_name, subtype_name, feature_name, class_name, rank_str)
    plot_mr(series_df, category_name, subtype_name, feature_name, class_name, rank_str)
    plot_ewma(series_df, category_name, subtype_name, feature_name, class_name, rank_str)
    plot_cusum(series_df, category_name, subtype_name, feature_name, class_name, rank_str)


# ---------------------------
# Plotters: missingness
# ---------------------------
def plot_p_chart(p_df: pd.DataFrame, category_name: str, subtype_name: str, feature_name: str, class_name: str, rank_str: str):
    if p_df.empty or "event_rate" not in p_df.columns:
        return

    x = pd.Series(np.arange(len(p_df)))
    y = safe_col(p_df, "event_rate")
    cl = safe_col(p_df, "pbar")
    ucl = safe_col(p_df, "UCL")
    lcl = safe_col(p_df, "LCL")
    alarm_hi = safe_col(p_df, "beyond_UCL")
    alarm_lo = safe_col(p_df, "below_LCL")
    alarm_any = ((alarm_hi.fillna(0) == 1) | (alarm_lo.fillna(0) == 1)).astype(int)

    subgroup_mask = subgroup_any_mask(MONITOR_ANOMALY_MASK, subgroup_size=SUBGROUP_SIZE)

    fig, ax = plt.subplots(figsize=FIGSIZE_STD)
    ax.plot(x, y, marker="o", label="Event Rate")
    if cl.notna().any():
        ax.plot(x, cl, linestyle="--", label="p̄")
    if ucl.notna().any():
        ax.plot(x, ucl, linestyle="--", label="UCL")
    if lcl.notna().any():
        ax.plot(x, lcl, linestyle="--", label="LCL")

    add_anomaly_x(ax, x, y, subgroup_mask, label="Anomaly sample")
    add_alarm_circle(ax, x, y, alarm_any, label="Alarm")

    set_common_axis_style(ax, f"{feature_name} - p Chart", ylabel="Event Rate")

    out_path = get_chart_output_path(category_name, subtype_name, "p_chart", rank_str, feature_name, class_name)
    save_and_close(fig, out_path)


def plot_g_chart(g_df: pd.DataFrame, category_name: str, subtype_name: str, feature_name: str, class_name: str, rank_str: str):
    if g_df.empty or "gap" not in g_df.columns:
        return

    x = safe_col(g_df, "event_order")
    y = safe_col(g_df, "gap")
    cl = safe_col(g_df, "CL")
    ucl = safe_col(g_df, "UCL")
    lcl = safe_col(g_df, "LCL")
    alarm_hi = safe_col(g_df, "beyond_UCL")
    alarm_lo = safe_col(g_df, "below_LCL")
    alarm_any = ((alarm_hi.fillna(0) == 1) | (alarm_lo.fillna(0) == 1)).astype(int)

    event_anomaly_mask = g_chart_anomaly_mask_from_feature(feature_name)

    fig, ax = plt.subplots(figsize=FIGSIZE_STD)
    ax.plot(x, y, marker="o", label="Gap")
    if cl.notna().any():
        ax.plot(x, cl, linestyle="--", label="CL")
    if ucl.notna().any():
        ax.plot(x, ucl, linestyle="--", label="UCL")
    if lcl.notna().any():
        ax.plot(x, lcl, linestyle="--", label="LCL")

    add_anomaly_x(ax, x, y, event_anomaly_mask, label="Anomaly sample")
    add_alarm_circle(ax, x, y, alarm_any, label="Alarm")

    set_common_axis_style(ax, f"{feature_name} - g Chart", xlabel="Event Order", ylabel="Gap")

    out_path = get_chart_output_path(category_name, subtype_name, "g_chart", rank_str, feature_name, class_name)
    save_and_close(fig, out_path)


def plot_missingness_folder(feature_dir: str, category_name: str):
    p_path = os.path.join(feature_dir, "p_chart.csv")
    g_path = os.path.join(feature_dir, "g_chart.csv")
    summary_path = os.path.join(feature_dir, "summary.csv")

    summary_df = get_summary_df(summary_path)
    subtype_name = infer_subtype_name(category_name, summary_df, feature_dir)
    rank_str = get_rank_str(summary_df)
    class_name = get_class_name(summary_df, fallback_subtype=subtype_name)
    feature_name = get_feature_name(summary_df, fallback_name=short_name_from_dir(feature_dir))

    if os.path.exists(p_path):
        p_df = read_csv_flex_local(p_path)
        plot_p_chart(p_df, category_name, subtype_name, feature_name, class_name, rank_str)

    if os.path.exists(g_path):
        g_df = read_csv_flex_local(g_path)
        plot_g_chart(g_df, category_name, subtype_name, feature_name, class_name, rank_str)


# ---------------------------
# Plotters: interaction pairs
# ---------------------------
def plot_pair_distance(pair_df: pd.DataFrame, category_name: str, subtype_name: str, pair_name: str, class_name: str, rank_str: str):
    if pair_df.empty or "pair_z_distance" not in pair_df.columns:
        return

    x = make_x(pair_df)
    y = safe_col(pair_df, "pair_z_distance")
    cl = safe_col(pair_df, "pair_z_distance_CL")
    ucl = safe_col(pair_df, "pair_z_distance_UCL")
    alarm = safe_col(pair_df, "pair_z_distance_alarm")

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.plot(x, y, label="Pair z-distance")
    if cl.notna().any():
        ax.plot(x, cl, linestyle="--", label="CL")
    if ucl.notna().any():
        ax.plot(x, ucl, linestyle="--", label="UCL")

    add_anomaly_x(ax, x, y, MONITOR_ANOMALY_MASK, label="Anomaly sample")
    add_alarm_circle(ax, x, y, alarm, label="Alarm")

    set_common_axis_style(ax, f"{pair_name} - Pair Distance", ylabel="Distance")

    out_path = get_chart_output_path(category_name, subtype_name, "pair_distance", rank_str, pair_name, class_name)
    save_and_close(fig, out_path)


def plot_pair_t2(pair_df: pd.DataFrame, category_name: str, subtype_name: str, pair_name: str, class_name: str, rank_str: str):
    if pair_df.empty or "T2" not in pair_df.columns:
        return

    x = make_x(pair_df)
    y = safe_col(pair_df, "T2")
    cl = safe_col(pair_df, "T2_CL")
    ucl = safe_col(pair_df, "T2_UCL")
    alarm = safe_col(pair_df, "T2_beyond_UCL")

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.plot(x, y, label="Hotelling's T²")
    if cl.notna().any():
        ax.plot(x, cl, linestyle="--", label="T² CL")
    if ucl.notna().any():
        ax.plot(x, ucl, linestyle="--", label="T² UCL")

    add_anomaly_x(ax, x, y, MONITOR_ANOMALY_MASK, label="Anomaly sample")
    add_alarm_circle(ax, x, y, alarm, label="Alarm")

    set_common_axis_style(ax, f"{pair_name} - Hotelling's T²", ylabel="T²")

    out_path = get_chart_output_path(category_name, subtype_name, "pair_hotelling_t2", rank_str, pair_name, class_name)
    save_and_close(fig, out_path)


def plot_pair_scatter(pair_df: pd.DataFrame, category_name: str, subtype_name: str, pair_name: str, class_name: str, rank_str: str):
    if pair_df.empty or "feature_a" not in pair_df.columns or "feature_b" not in pair_df.columns:
        return

    x = safe_col(pair_df, "feature_a")
    y = safe_col(pair_df, "feature_b")
    alarm = safe_col(pair_df, "T2_beyond_UCL")
    if alarm.empty:
        alarm = safe_col(pair_df, "pair_z_distance_alarm")
    anomaly = pd.Series(MONITOR_ANOMALY_MASK).fillna(0).astype(int).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=FIGSIZE_STD)

    normal_mask = (anomaly != 1) & (alarm.fillna(0).astype(int) != 1)
    anomaly_only_mask = (anomaly == 1)
    alarm_mask = (alarm.fillna(0).astype(int) == 1)

    ax.scatter(x[normal_mask], y[normal_mask], s=16, alpha=0.7, label="Normal sample")
    if anomaly_only_mask.sum() > 0:
        ax.scatter(x[anomaly_only_mask], y[anomaly_only_mask], marker="x", s=42, linewidths=1.4, c="red", label="Anomaly sample")
    if alarm_mask.sum() > 0:
        ax.scatter(x[alarm_mask], y[alarm_mask], marker="o", s=46, facecolors="none", edgecolors="orange", linewidths=1.6, label="Alarm")

    set_common_axis_style(ax, f"{pair_name} - Pair Scatter", xlabel="Feature A", ylabel="Feature B")

    out_path = get_chart_output_path(category_name, subtype_name, "pair_scatter", rank_str, pair_name, class_name)
    save_and_close(fig, out_path)


def plot_interaction_folder(pair_dir: str, category_name: str):
    pair_path = os.path.join(pair_dir, "pair_series.csv")
    summary_path = os.path.join(pair_dir, "summary.csv")
    if not os.path.exists(pair_path):
        return

    pair_df = read_csv_flex_local(pair_path)
    summary_df = get_summary_df(summary_path)

    subtype_name = infer_subtype_name(category_name, summary_df, pair_dir)
    rank_str = get_rank_str(summary_df)
    class_name = get_class_name(summary_df, fallback_subtype=subtype_name)
    pair_name = get_pair_name(summary_df, fallback_name=short_name_from_dir(pair_dir))

    plot_pair_distance(pair_df, category_name, subtype_name, pair_name, class_name, rank_str)
    plot_pair_t2(pair_df, category_name, subtype_name, pair_name, class_name, rank_str)
    plot_pair_scatter(pair_df, category_name, subtype_name, pair_name, class_name, rank_str)


# ---------------------------
# Plotters: multivariate bundle
# ---------------------------
def plot_multivar_t2(series_df: pd.DataFrame, category_name: str, subtype_name: str, bundle_name: str, class_name: str):
    if series_df.empty or "T2" not in series_df.columns:
        return

    x = make_x(series_df)
    y = safe_col(series_df, "T2")
    cl = safe_col(series_df, "T2_CL")
    ucl = safe_col(series_df, "T2_UCL")
    alarm = safe_col(series_df, "T2_beyond_UCL")

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.plot(x, y, label="Hotelling's T²")
    if cl.notna().any():
        ax.plot(x, cl, linestyle="--", label="T² CL")
    if ucl.notna().any():
        ax.plot(x, ucl, linestyle="--", label="T² UCL")

    add_anomaly_x(ax, x, y, MONITOR_ANOMALY_MASK, label="Anomaly sample")
    add_alarm_circle(ax, x, y, alarm, label="Alarm")

    set_common_axis_style(ax, f"{bundle_name} - Multivariate T²", ylabel="T²")

    out_path = get_chart_output_path(category_name, subtype_name, "multivariate_t2", "000", bundle_name, class_name)
    save_and_close(fig, out_path)


def plot_pca_t2(scores_df: pd.DataFrame, category_name: str, subtype_name: str, bundle_name: str, class_name: str):
    if scores_df.empty or "PCA_T2" not in scores_df.columns:
        return

    x = make_x(scores_df)
    y = safe_col(scores_df, "PCA_T2")
    cl = safe_col(scores_df, "PCA_T2_CL")
    ucl = safe_col(scores_df, "PCA_T2_UCL")
    alarm = safe_col(scores_df, "PCA_T2_beyond_UCL")

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.plot(x, y, label="PCA T²")
    if cl.notna().any():
        ax.plot(x, cl, linestyle="--", label="PCA T² CL")
    if ucl.notna().any():
        ax.plot(x, ucl, linestyle="--", label="PCA T² UCL")

    add_anomaly_x(ax, x, y, MONITOR_ANOMALY_MASK, label="Anomaly sample")
    add_alarm_circle(ax, x, y, alarm, label="Alarm")

    set_common_axis_style(ax, f"{bundle_name} - PCA T²", ylabel="PCA T²")

    out_path = get_chart_output_path(category_name, subtype_name, "pca_t2", "000", bundle_name, class_name)
    save_and_close(fig, out_path)


def plot_pca_spe(spe_df: pd.DataFrame, category_name: str, subtype_name: str, bundle_name: str, class_name: str):
    if spe_df.empty or "PCA_SPE" not in spe_df.columns:
        return

    x = make_x(spe_df)
    y = safe_col(spe_df, "PCA_SPE")
    cl = safe_col(spe_df, "PCA_SPE_CL")
    ucl = safe_col(spe_df, "PCA_SPE_UCL")
    alarm = safe_col(spe_df, "PCA_SPE_beyond_UCL")

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.plot(x, y, label="PCA SPE / Q")
    if cl.notna().any():
        ax.plot(x, cl, linestyle="--", label="SPE CL")
    if ucl.notna().any():
        ax.plot(x, ucl, linestyle="--", label="SPE UCL")

    add_anomaly_x(ax, x, y, MONITOR_ANOMALY_MASK, label="Anomaly sample")
    add_alarm_circle(ax, x, y, alarm, label="Alarm")

    set_common_axis_style(ax, f"{bundle_name} - PCA SPE", ylabel="SPE")

    out_path = get_chart_output_path(category_name, subtype_name, "pca_spe", "000", bundle_name, class_name)
    save_and_close(fig, out_path)


def plot_pc_scatter(scores_df: pd.DataFrame, category_name: str, subtype_name: str, bundle_name: str, class_name: str):
    if scores_df.empty or "PC1" not in scores_df.columns or "PC2" not in scores_df.columns:
        return

    x = safe_col(scores_df, "PC1")
    y = safe_col(scores_df, "PC2")
    alarm = safe_col(scores_df, "PCA_T2_beyond_UCL")
    anomaly = pd.Series(MONITOR_ANOMALY_MASK).fillna(0).astype(int).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=FIGSIZE_STD)

    normal_mask = (anomaly != 1) & (alarm.fillna(0).astype(int) != 1)
    anomaly_only_mask = (anomaly == 1)
    alarm_mask = (alarm.fillna(0).astype(int) == 1)

    ax.scatter(x[normal_mask], y[normal_mask], s=16, alpha=0.7, label="Normal sample")
    if anomaly_only_mask.sum() > 0:
        ax.scatter(x[anomaly_only_mask], y[anomaly_only_mask], marker="x", s=42, linewidths=1.4, c="red", label="Anomaly sample")
    if alarm_mask.sum() > 0:
        ax.scatter(x[alarm_mask], y[alarm_mask], marker="o", s=46, facecolors="none", edgecolors="orange", linewidths=1.6, label="Alarm")

    set_common_axis_style(ax, f"{bundle_name} - PC1 vs PC2", xlabel="PC1", ylabel="PC2")

    out_path = get_chart_output_path(category_name, subtype_name, "pc1_pc2_scatter", "000", bundle_name, class_name)
    save_and_close(fig, out_path)


def plot_multivariate_folder(bundle_dir: str, category_name: str):
    t2_path = os.path.join(bundle_dir, "hotelling_t2_series.csv")
    pca_scores_path = os.path.join(bundle_dir, "pca_scores_series.csv")
    pca_spe_path = os.path.join(bundle_dir, "pca_spe_series.csv")

    summary_df = pd.DataFrame()
    subtype_name = infer_subtype_name(category_name, summary_df, bundle_dir)
    bundle_name = safe_filename_part(short_name_from_dir(bundle_dir))
    class_name = subtype_name

    if os.path.exists(t2_path):
        t2_df = read_csv_flex_local(t2_path)
        plot_multivar_t2(t2_df, category_name, subtype_name, bundle_name, class_name)

    if os.path.exists(pca_scores_path):
        pca_scores_df = read_csv_flex_local(pca_scores_path)
        plot_pca_t2(pca_scores_df, category_name, subtype_name, bundle_name, class_name)
        plot_pc_scatter(pca_scores_df, category_name, subtype_name, bundle_name, class_name)

    if os.path.exists(pca_spe_path):
        pca_spe_df = read_csv_flex_local(pca_spe_path)
        plot_pca_spe(pca_spe_df, category_name, subtype_name, bundle_name, class_name)


# ---------------------------
# Recursive scan
# ---------------------------
def scan_and_plot_category(category_name: str):
    category_root = os.path.join(SPC_EXPORT_ROOT, category_name)
    if not os.path.exists(category_root):
        return

    for dirpath, dirnames, filenames in os.walk(category_root):
        # feature folder
        if "summary.csv" in filenames and "series.csv" in filenames:
            plot_feature_folder(dirpath, category_name)
            continue

        # missingness folder
        if "summary.csv" in filenames and ("p_chart.csv" in filenames or "g_chart.csv" in filenames):
            plot_missingness_folder(dirpath, category_name)
            continue

        # interaction pair folder
        if "summary.csv" in filenames and "pair_series.csv" in filenames:
            plot_interaction_folder(dirpath, category_name)
            continue

        # multivariate bundle folder
        if (
            "hotelling_t2_series.csv" in filenames
            or "pca_scores_series.csv" in filenames
            or "pca_spe_series.csv" in filenames
        ):
            plot_multivariate_folder(dirpath, category_name)
            continue


# ---------------------------
# Run
# ---------------------------
for category in ["main_signal", "auxiliary_diagnostic", "final_deployment"]:
    scan_and_plot_category(category)

print("=" * 100)
print("SPC plots exported")
print("=" * 100)
print(f"Input SPC values root: {SPC_EXPORT_ROOT}")
print(f"Output plot root    : {SPC_PLOT_ROOT}")
print("- anomaly sample: red x")
print("- alarm        : circle")
print("- folder order : category -> subtype -> chart_type")
print("- filename     : rank_feature_class_chart.png")
print("Done.")