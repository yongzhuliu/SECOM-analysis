"""
05_spc_values.py

SPC values export script converted from the original notebook block.

Purpose:
    Run after 04_shap_analysis.py to export SPC-ready values grouped by:
    - main_signal
    - auxiliary_diagnostic
    - final_deployment

Expected inputs:
    data/processed/X_train.csv
    data/processed/X_test.csv
    data/processed/y_train.csv
    data/processed/y_test.csv
    outputs/shap/run_manifest.json
    outputs/shap/spc_list__main_signal.csv
    outputs/shap/spc_list__auxiliary_diagnostic.csv
    outputs/shap/spc_list__final_deployment.csv or final_monitor_candidates.csv
    outputs/shap/spc_monitoring_prep_pack__final.csv as fallback
    outputs/shap/spc_monitoring_prep_pack__final.csv or spc_monitoring_prep_pack.csv

Default output:
    outputs/spc_values/

Run:
    python scripts/05_spc_values.py

or:
    python scripts/05_spc_values.py \
        --shap-output-dir outputs/shap \
        --output-dir outputs/spc_values \
        --processed-dir data/processed
"""

# =========================================================
# SPC values export block
# 接在前面 attribution pipeline block 後面直接執行
# ---------------------------------------------------------
# 目的:
# - 依 main_signal / auxiliary_diagnostic / final_deployment 分資料夾輸出 SPC 數值
# - 修正 list feature / raw column 名稱型別不一致造成的對不到欄位問題
# - 連續特徵: 3-sigma / EWMA / CUSUM / Moving Range
# - 缺失型特徵: p-chart / g-chart
# - interaction pair: 2維 Hotelling's T² proxy + pair distance
# - 每個子資料夾都會存 summary 與逐特徵(或逐pair) time-series CSV
# - 每個 subtype 的連續特徵群組，再輸出 multivariate T² 與 PCA SPC
#
# 輸出:
#   outputs/spc_values/
#     main_signal/
#       tp_like/
#       fp_like/
#       test_holdout/
#       split_stable/
#     auxiliary_diagnostic/
#       missingness/
#       interaction/
#       distribution_separation/
#     final_deployment/
#       Tier_A/
#       Tier_B/
# =========================================================

import os
import re
import json
import math
import io
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------
# Config
# ---------------------------
# Old notebook used SPC_EXPORT_ROOT_NAME = "".
# In project form, SPC values should be written directly to outputs/spc_values/.
SPC_EXPORT_ROOT_NAME = ""
SPC_OUTPUT_DIR = "outputs/spc_values"
SUBGROUP_SIZE = 25
EWMA_LAMBDA = 0.20
CUSUM_K = 0.50
CUSUM_H = 5.00
PCA_EXPLAINED_VAR = 0.90
MIN_STD_EPS = 1e-12
USE_TEST_AS_MONITOR = True
TIME_COL = None  # 若有時間欄位可指定，例如 "timestamp"

# ---------------------------
# Project path config
# ---------------------------
# This script is intended to run after 04_shap_analysis.py.
# Default project structure:
#   data/processed/
#   outputs/shap/
# SHAP_OUTPUT_DIR: input folder from 04_shap_analysis.py
SHAP_OUTPUT_DIR = "outputs/shap"

# SPC_OUTPUT_DIR: output folder for this script
SPC_OUTPUT_DIR = "outputs/spc_values"

# Backward-compatible alias used by original notebook code
OUT_DIR = SHAP_OUTPUT_DIR

PROCESSED_DIR = "data/processed"
RAW_TRAIN_CSV_PATH = "data/processed/X_train.csv"
RAW_TEST_CSV_PATH = "data/processed/X_test.csv"


# ---------------------------
# Basic utils
# ---------------------------
def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))


def read_csv_flex_local(path: str) -> pd.DataFrame:
    """
    Robust CSV reader for project outputs.

    Why this is needed:
    - Some files may be utf-8-sig, utf-8, cp950/big5, utf-16, or contain a few invalid bytes.
    - Pandas may raise UnicodeDecodeError or ParserError depending on the file.
    - This function tries multiple encodings and finally falls back to replacement decoding.
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

    # Final fallback: manually decode bytes with replacement.
    try:
        raw = Path(path).read_bytes()
        text = raw.decode("utf-8", errors="replace")
        return pd.read_csv(io.StringIO(text), engine="python")
    except Exception as e:
        errors.append(f"manual utf-8 replace: {type(e).__name__}: {e}")

    error_msg = "\n".join(errors[-8:])
    raise RuntimeError(f"Failed to read CSV file: {path}\nTried encodings/parsers:\n{error_msg}")


def write_df_or_empty_local(df: pd.DataFrame, path: str):
    ensure_dir(Path(path).parent)
    if df is None:
        df = pd.DataFrame()
    df.to_csv(path, index=False, encoding="utf-8-sig")


def load_first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def load_label_simple(paths):
    label_path = load_first_existing(paths)
    if label_path is None:
        return None, None

    y_df = read_csv_flex_local(label_path)
    y_df = y_df.loc[:, ~y_df.columns.astype(str).str.contains(r"^Unnamed")]
    if y_df.shape[1] == 0:
        return None, label_path

    if y_df.shape[1] == 1:
        y = y_df.iloc[:, 0]
    else:
        label_col_candidates = ["target", "label", "y", "is_anomaly", "anomaly"]
        chosen = None
        for c in label_col_candidates:
            if c in y_df.columns:
                chosen = c
                break
        if chosen is None:
            chosen = y_df.columns[0]
        y = y_df[chosen]

    y = pd.Series(y).reset_index(drop=True)

    if y.dtype == bool:
        y = y.astype(int)
    elif str(y.dtype).startswith("int") or str(y.dtype).startswith("float"):
        y = pd.to_numeric(y, errors="coerce")
    else:
        mapping = {
            "normal": 0, "anomaly": 1, "abnormal": 1,
            "yes": 1, "no": 0, "true": 1, "false": 0,
            "positive": 1, "negative": 0, "pass": 0, "fail": 1,
            "-1": 0, "1": 1, "0": 0,
        }
        y_str = y.astype(str).str.strip().str.lower()
        if y_str.isin(mapping.keys()).all():
            y = y_str.map(mapping).astype(int)
        else:
            y = pd.to_numeric(y, errors="coerce")

    return y, label_path


def choose_existing_output(out_dir: str, candidates):
    for name in candidates:
        p = os.path.join(out_dir, name)
        if os.path.exists(p):
            return p
    return None


def load_optional_output(out_dir: str, candidates):
    p = choose_existing_output(out_dir, candidates)
    if p is None:
        return pd.DataFrame(), None
    return read_csv_flex_local(p), p


def get_time_index(df: pd.DataFrame):
    if TIME_COL and TIME_COL in df.columns:
        return pd.Series(df[TIME_COL]).reset_index(drop=True)
    return pd.RangeIndex(start=0, stop=len(df), step=1)


def numeric_series(s):
    return pd.to_numeric(s, errors="coerce")


def safe_std(s):
    s = numeric_series(s).dropna()
    if len(s) <= 1:
        return np.nan
    return float(s.std(ddof=1))


def safe_mean(s):
    s = numeric_series(s).dropna()
    if len(s) == 0:
        return np.nan
    return float(s.mean())


def safe_quantile(s, q):
    s = numeric_series(s).dropna()
    if len(s) == 0:
        return np.nan
    return float(s.quantile(q))


# ---------------------------
# Feature name resolver
# ---------------------------
def canonical_feature_key(x):
    if pd.isna(x):
        return None

    s = str(x).replace("\ufeff", "").strip()
    s = re.sub(r"\s+", " ", s)

    # 原樣
    candidates = [s]

    # 數值型別常見變形，例如 123 / 123.0 / "123"
    try:
        xf = float(s)
        candidates.append(str(xf))
        if abs(xf - round(xf)) < 1e-9:
            candidates.append(str(int(round(xf))))
            candidates.append(f"{int(round(xf))}.0")
    except Exception:
        pass

    # 去除整數 .0
    if re.fullmatch(r"-?\d+\.0+", s):
        try:
            candidates.append(str(int(float(s))))
        except Exception:
            pass

    # 字串再 lower 一份
    candidates.append(s.lower())

    # 去掉空白版
    candidates.append(s.replace(" ", ""))

    # unique，保持順序
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


def build_column_resolver(columns):
    resolver = {}
    actual_cols = list(columns)

    for col in actual_cols:
        keys = canonical_feature_key(col)
        for k in keys:
            if k not in resolver:
                resolver[k] = col
    return resolver


def resolve_feature_name(x, resolver):
    keys = canonical_feature_key(x)
    if keys is None:
        return None
    for k in keys:
        if k in resolver:
            return resolver[k]
    return None


def normalize_list_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    for c in ["feature", "feature_a", "feature_b", "pair_key", "list_subtype", "spc_selection_tier", "anchor_recommendation"]:
        if c in out.columns:
            out[c] = out[c].astype("object")
    return out


# ---------------------------
# SPC calculators
# ---------------------------
def calc_shewhart_3sigma(x_monitor: pd.Series, center: float, sigma: float):
    x = numeric_series(x_monitor)
    sigma = float(sigma) if pd.notna(sigma) else np.nan
    if pd.isna(sigma) or abs(sigma) < MIN_STD_EPS:
        sigma = np.nan

    out = pd.DataFrame({"value": x})
    out["CL"] = center
    out["UCL"] = center + 3 * sigma if pd.notna(sigma) else np.nan
    out["LCL"] = center - 3 * sigma if pd.notna(sigma) else np.nan
    out["beyond_UCL"] = ((out["value"] > out["UCL"]) & out["UCL"].notna()).astype(int)
    out["below_LCL"] = ((out["value"] < out["LCL"]) & out["LCL"].notna()).astype(int)
    out["beyond_any_limit"] = ((out["beyond_UCL"] == 1) | (out["below_LCL"] == 1)).astype(int)
    return out


def calc_moving_range(x_monitor: pd.Series):
    x = numeric_series(x_monitor)
    mr = x.diff().abs()
    mrbar = float(mr.dropna().mean()) if mr.dropna().shape[0] > 0 else np.nan
    out = pd.DataFrame({
        "value": x,
        "MR": mr,
        "MR_CL": mrbar,
        "MR_UCL": (3.267 * mrbar) if pd.notna(mrbar) else np.nan,
        "MR_LCL": 0.0 if pd.notna(mrbar) else np.nan,
    })
    out["MR_beyond_UCL"] = ((out["MR"] > out["MR_UCL"]) & out["MR_UCL"].notna()).astype(int)
    return out


def calc_ewma(x_monitor: pd.Series, center: float, sigma: float, lam: float = EWMA_LAMBDA):
    x = numeric_series(x_monitor)
    z = []
    prev = center
    for val in x:
        if pd.isna(val):
            z.append(prev)
        else:
            prev = lam * float(val) + (1 - lam) * prev
            z.append(prev)
    z = pd.Series(z, index=x.index, dtype=float)

    t = np.arange(1, len(x) + 1)
    if pd.notna(sigma):
        sigma_z = sigma * np.sqrt(lam / (2 - lam) * (1 - (1 - lam) ** (2 * t)))
    else:
        sigma_z = np.full(len(x), np.nan)

    out = pd.DataFrame({
        "value": x,
        "EWMA": z,
        "EWMA_CL": center,
        "EWMA_UCL": center + 3 * sigma_z,
        "EWMA_LCL": center - 3 * sigma_z,
    }, index=x.index)
    out["EWMA_beyond_UCL"] = ((out["EWMA"] > out["EWMA_UCL"]) & out["EWMA_UCL"].notna()).astype(int)
    out["EWMA_below_LCL"] = ((out["EWMA"] < out["EWMA_LCL"]) & out["EWMA_LCL"].notna()).astype(int)
    out["EWMA_beyond_any_limit"] = ((out["EWMA_beyond_UCL"] == 1) | (out["EWMA_below_LCL"] == 1)).astype(int)
    return out


def calc_cusum(x_monitor: pd.Series, center: float, sigma: float, k: float = CUSUM_K, h: float = CUSUM_H):
    x = numeric_series(x_monitor)
    if pd.isna(sigma) or abs(sigma) < MIN_STD_EPS:
        sigma = np.nan

    cplus = []
    cminus = []
    s_pos = 0.0
    s_neg = 0.0

    for val in x:
        if pd.isna(val) or pd.isna(sigma):
            cplus.append(np.nan)
            cminus.append(np.nan)
            continue
        z = (float(val) - center) / sigma
        s_pos = max(0.0, s_pos + z - k)
        s_neg = min(0.0, s_neg + z + k)
        cplus.append(s_pos)
        cminus.append(s_neg)

    out = pd.DataFrame({
        "value": x,
        "CUSUM_CPLUS": cplus,
        "CUSUM_CMINUS": cminus,
        "CUSUM_H_POS": h,
        "CUSUM_H_NEG": -h,
    }, index=x.index)
    out["CUSUM_alarm_pos"] = ((out["CUSUM_CPLUS"] > h) & out["CUSUM_CPLUS"].notna()).astype(int)
    out["CUSUM_alarm_neg"] = ((out["CUSUM_CMINUS"] < -h) & out["CUSUM_CMINUS"].notna()).astype(int)
    out["CUSUM_alarm_any"] = ((out["CUSUM_alarm_pos"] == 1) | (out["CUSUM_alarm_neg"] == 1)).astype(int)
    return out


def build_subgroups(event_series: pd.Series, subgroup_size: int = SUBGROUP_SIZE):
    s = numeric_series(event_series).fillna(0).astype(int).reset_index(drop=True)
    subgroup_size = max(1, min(subgroup_size, len(s))) if len(s) > 0 else subgroup_size
    if len(s) == 0:
        return pd.DataFrame(columns=["subgroup_id", "n", "event_n", "event_rate"])

    subgroup_id = np.arange(len(s)) // subgroup_size
    out = pd.DataFrame({"event": s, "subgroup_id": subgroup_id})
    out = out.groupby("subgroup_id", as_index=False).agg(
        n=("event", "size"),
        event_n=("event", "sum")
    )
    out["event_rate"] = out["event_n"] / out["n"]
    return out


def calc_p_chart(event_train: pd.Series, event_monitor: pd.Series, subgroup_size: int = SUBGROUP_SIZE):
    train_s = numeric_series(event_train).fillna(0).astype(int)
    pbar = float(train_s.mean()) if len(train_s) > 0 else np.nan

    monitor_grp = build_subgroups(event_monitor, subgroup_size=subgroup_size)
    if monitor_grp.empty:
        return monitor_grp

    if pd.notna(pbar):
        sigma = np.sqrt(np.maximum(pbar * (1 - pbar) / monitor_grp["n"], 0.0))
    else:
        sigma = np.nan

    monitor_grp["pbar"] = pbar
    monitor_grp["UCL"] = np.minimum(1.0, pbar + 3 * sigma) if pd.notna(pbar) else np.nan
    monitor_grp["LCL"] = np.maximum(0.0, pbar - 3 * sigma) if pd.notna(pbar) else np.nan
    monitor_grp["beyond_UCL"] = ((monitor_grp["event_rate"] > monitor_grp["UCL"]) & monitor_grp["UCL"].notna()).astype(int)
    monitor_grp["below_LCL"] = ((monitor_grp["event_rate"] < monitor_grp["LCL"]) & monitor_grp["LCL"].notna()).astype(int)
    monitor_grp["beyond_any_limit"] = ((monitor_grp["beyond_UCL"] == 1) | (monitor_grp["below_LCL"] == 1)).astype(int)
    return monitor_grp


def calc_g_chart(event_train: pd.Series, event_monitor: pd.Series):
    train_s = numeric_series(event_train).fillna(0).astype(int)
    p = float(train_s.mean()) if len(train_s) > 0 else np.nan

    mon = numeric_series(event_monitor).fillna(0).astype(int).reset_index(drop=True)
    event_pos = np.where(mon.values == 1)[0]
    if len(event_pos) <= 1 or pd.isna(p) or p <= 0:
        return pd.DataFrame(columns=["event_order", "gap", "CL", "UCL", "LCL", "beyond_UCL"])

    gaps = np.diff(event_pos)
    cl = 1.0 / p
    sigma_g = math.sqrt((1 - p) / (p ** 2))
    ucl = cl + 3 * sigma_g
    lcl = max(0.0, cl - 3 * sigma_g)

    out = pd.DataFrame({
        "event_order": np.arange(1, len(gaps) + 1),
        "gap": gaps.astype(float),
        "CL": cl,
        "UCL": ucl,
        "LCL": lcl,
    })
    out["beyond_UCL"] = (out["gap"] > out["UCL"]).astype(int)
    out["below_LCL"] = (out["gap"] < out["LCL"]).astype(int)
    out["beyond_any_limit"] = ((out["beyond_UCL"] == 1) | (out["below_LCL"] == 1)).astype(int)
    return out


def prepare_matrix(df: pd.DataFrame):
    X = df.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    med = X.median(numeric_only=True)
    X = X.fillna(med)
    return X


def calc_hotelling_t2(train_df: pd.DataFrame, monitor_df: pd.DataFrame):
    if train_df is None or monitor_df is None or train_df.empty or monitor_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    X_train = prepare_matrix(train_df)
    X_mon = prepare_matrix(monitor_df)
    common_cols = [c for c in X_train.columns if c in X_mon.columns]
    if len(common_cols) < 2:
        return pd.DataFrame(), pd.DataFrame()

    X_train = X_train[common_cols].copy()
    X_mon = X_mon[common_cols].copy()

    mu = X_train.mean(axis=0)
    cov = np.cov(X_train.values, rowvar=False)
    if np.ndim(cov) == 0:
        return pd.DataFrame(), pd.DataFrame()

    cov = cov + np.eye(cov.shape[0]) * 1e-6
    inv_cov = np.linalg.pinv(cov)

    def mahal_sq(X):
        diff = X - mu.values
        return np.einsum("ij,jk,ik->i", diff, inv_cov, diff)

    t2_train = mahal_sq(X_train.values)
    t2_mon = mahal_sq(X_mon.values)

    cl = float(np.mean(t2_train))
    sd = float(np.std(t2_train, ddof=1)) if len(t2_train) > 1 else np.nan
    ucl = cl + 3 * sd if pd.notna(sd) else np.nan
    lcl = max(0.0, cl - 3 * sd) if pd.notna(sd) else np.nan

    summary = pd.DataFrame([{
        "n_features": len(common_cols),
        "features": "|".join(map(str, common_cols)),
        "T2_CL": cl,
        "T2_UCL": ucl,
        "T2_LCL": lcl,
    }])
    series = pd.DataFrame({
        "T2": t2_mon,
        "T2_CL": cl,
        "T2_UCL": ucl,
        "T2_LCL": lcl,
        "T2_beyond_UCL": ((t2_mon > ucl) if pd.notna(ucl) else np.zeros(len(t2_mon))).astype(int)
    })
    return summary, series


def calc_pca_spc(train_df: pd.DataFrame, monitor_df: pd.DataFrame, explained_var: float = PCA_EXPLAINED_VAR):
    if train_df is None or monitor_df is None or train_df.empty or monitor_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    X_train = prepare_matrix(train_df)
    X_mon = prepare_matrix(monitor_df)
    common_cols = [c for c in X_train.columns if c in X_mon.columns]
    if len(common_cols) < 2:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    X_train = X_train[common_cols].copy()
    X_mon = X_mon[common_cols].copy()

    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0, ddof=1).replace(0, 1.0).fillna(1.0)

    Z_train = (X_train - mu) / sd
    Z_mon = (X_mon - mu) / sd

    cov = np.cov(Z_train.values, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    eigvals = np.where(eigvals < 1e-12, 1e-12, eigvals)
    ratio = eigvals / eigvals.sum()
    cum_ratio = np.cumsum(ratio)
    n_comp = int(np.searchsorted(cum_ratio, explained_var) + 1)
    n_comp = max(1, min(n_comp, len(common_cols)))

    P = eigvecs[:, :n_comp]
    L = eigvals[:n_comp]

    scores_train = Z_train.values @ P
    scores_mon = Z_mon.values @ P

    t2_train = np.sum((scores_train ** 2) / L, axis=1)
    t2_mon = np.sum((scores_mon ** 2) / L, axis=1)

    recon_train = scores_train @ P.T
    recon_mon = scores_mon @ P.T
    spe_train = np.sum((Z_train.values - recon_train) ** 2, axis=1)
    spe_mon = np.sum((Z_mon.values - recon_mon) ** 2, axis=1)

    t2_cl = float(np.mean(t2_train))
    t2_sd = float(np.std(t2_train, ddof=1)) if len(t2_train) > 1 else np.nan
    t2_ucl = t2_cl + 3 * t2_sd if pd.notna(t2_sd) else np.nan

    spe_cl = float(np.mean(spe_train))
    spe_sd = float(np.std(spe_train, ddof=1)) if len(spe_train) > 1 else np.nan
    spe_ucl = spe_cl + 3 * spe_sd if pd.notna(spe_sd) else np.nan

    summary = pd.DataFrame([{
        "n_features": len(common_cols),
        "features": "|".join(map(str, common_cols)),
        "n_components": n_comp,
        "explained_variance_sum": float(cum_ratio[n_comp - 1]),
        "PCA_T2_CL": t2_cl,
        "PCA_T2_UCL": t2_ucl,
        "PCA_SPE_CL": spe_cl,
        "PCA_SPE_UCL": spe_ucl,
    }])

    score_cols = {f"PC{i+1}": scores_mon[:, i] for i in range(n_comp)}
    scores_df = pd.DataFrame(score_cols)
    scores_df["PCA_T2"] = t2_mon
    scores_df["PCA_T2_CL"] = t2_cl
    scores_df["PCA_T2_UCL"] = t2_ucl
    scores_df["PCA_T2_beyond_UCL"] = ((t2_mon > t2_ucl) if pd.notna(t2_ucl) else np.zeros(len(t2_mon))).astype(int)

    spe_df = pd.DataFrame({
        "PCA_SPE": spe_mon,
        "PCA_SPE_CL": spe_cl,
        "PCA_SPE_UCL": spe_ucl,
        "PCA_SPE_beyond_UCL": ((spe_mon > spe_ucl) if pd.notna(spe_ucl) else np.zeros(len(spe_mon))).astype(int)
    })

    return summary, scores_df, spe_df



# ---------------------------
# CLI arguments
# ---------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Export SPC value tables from SHAP/root-cause candidate lists."
    )
    parser.add_argument(
        "--shap-output-dir",
        default="outputs/shap",
        help="Directory produced by 04_shap_analysis.py. Default: outputs/shap",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/spc_values",
        help="Directory to save SPC value outputs. Default: outputs/spc_values",
    )
    parser.add_argument(
        "--processed-dir",
        default="data/processed",
        help="Directory containing X_train.csv, X_test.csv, y_train.csv, y_test.csv. Default: data/processed",
    )
    parser.add_argument(
        "--raw-train",
        default=None,
        help="Optional path to raw train CSV. Default: <processed-dir>/X_train.csv",
    )
    parser.add_argument(
        "--raw-test",
        default=None,
        help="Optional path to raw test CSV. Default: <processed-dir>/X_test.csv",
    )
    parser.add_argument(
        "--subgroup-size",
        type=int,
        default=25,
        help="Subgroup size for p-chart. Default: 25",
    )
    parser.add_argument(
        "--ewma-lambda",
        type=float,
        default=0.20,
        help="EWMA lambda. Default: 0.20",
    )
    parser.add_argument(
        "--cusum-k",
        type=float,
        default=0.50,
        help="CUSUM k. Default: 0.50",
    )
    parser.add_argument(
        "--cusum-h",
        type=float,
        default=5.00,
        help="CUSUM h. Default: 5.00",
    )
    parser.add_argument(
        "--pca-explained-var",
        type=float,
        default=0.90,
        help="Explained variance target for PCA SPC. Default: 0.90",
    )
    parser.add_argument(
        "--use-train-as-monitor",
        action="store_true",
        help="Use train data as monitor series instead of test data.",
    )
    parser.add_argument(
        "--time-col",
        default=None,
        help="Optional time column name for time_index.",
    )

    args = parser.parse_args()

    SHAP_OUTPUT_DIR = args.shap_output_dir
    SPC_OUTPUT_DIR = args.output_dir
    OUT_DIR = SHAP_OUTPUT_DIR
    PROCESSED_DIR = args.processed_dir
    RAW_TRAIN_CSV_PATH = args.raw_train or os.path.join(PROCESSED_DIR, "X_train.csv")
    RAW_TEST_CSV_PATH = args.raw_test or os.path.join(PROCESSED_DIR, "X_test.csv")

    SUBGROUP_SIZE = args.subgroup_size
    EWMA_LAMBDA = args.ewma_lambda
    CUSUM_K = args.cusum_k
    CUSUM_H = args.cusum_h
    PCA_EXPLAINED_VAR = args.pca_explained_var
    USE_TEST_AS_MONITOR = not args.use_train_as_monitor
    TIME_COL = args.time_col

    TRAIN_LABEL_CANDIDATES_SIMPLE = [
        os.path.join(PROCESSED_DIR, "y_train.csv"),
        os.path.join(PROCESSED_DIR, "y_train_base.csv"),
    ]
    TEST_LABEL_CANDIDATES_SIMPLE = [
        os.path.join(PROCESSED_DIR, "y_test.csv"),
        os.path.join(PROCESSED_DIR, "y_test_base.csv"),
    ]



# ---------------------------
# Load manifest / paths
# ---------------------------
run_manifest_path = os.path.join(SHAP_OUTPUT_DIR, "run_manifest.json") if "SHAP_OUTPUT_DIR" in globals() else "outputs/shap/run_manifest.json"
run_manifest = {}
if os.path.exists(run_manifest_path):
    with open(run_manifest_path, "r", encoding="utf-8") as f:
        run_manifest = json.load(f)

OUT_DIR_USE = SHAP_OUTPUT_DIR if "SHAP_OUTPUT_DIR" in globals() else run_manifest.get("OUT_DIR", "outputs/shap")
RAW_TRAIN_PATH = globals().get("RAW_TRAIN_CSV_PATH", run_manifest.get("RAW_TRAIN_CSV_PATH", "X_train.csv"))
RAW_TEST_PATH = globals().get("RAW_TEST_CSV_PATH", run_manifest.get("RAW_TEST_CSV_PATH", "X_test.csv"))

TRAIN_LABEL_CANDIDATES_SIMPLE = [os.path.join(PROCESSED_DIR, "y_train.csv"), os.path.join(PROCESSED_DIR, "y_train_base.csv")]
TEST_LABEL_CANDIDATES_SIMPLE = [os.path.join(PROCESSED_DIR, "y_test.csv"), os.path.join(PROCESSED_DIR, "y_test_base.csv")]

# ---------------------------
# Load grouped lists
# ---------------------------
main_signal_df, main_signal_path = load_optional_output(
    OUT_DIR_USE,
    ["spc_list__main_signal.csv"]
)
aux_diag_df, aux_diag_path = load_optional_output(
    OUT_DIR_USE,
    ["spc_list__auxiliary_diagnostic.csv"]
)
final_deploy_df, final_deploy_path = load_optional_output(
    OUT_DIR_USE,
    [
        "spc_list__final_deployment.csv",
        "final_monitor_candidates.csv",
        "spc_monitoring_prep_pack__final.csv",
        "spc_monitoring_prep_pack.csv",
    ]
)
spc_prep_final_df, spc_prep_path = load_optional_output(
    OUT_DIR_USE,
    ["spc_monitoring_prep_pack__final.csv", "spc_monitoring_prep_pack.csv"]
)

main_signal_df = normalize_list_df(main_signal_df)
aux_diag_df = normalize_list_df(aux_diag_df)
final_deploy_df = normalize_list_df(final_deploy_df)
spc_prep_final_df = normalize_list_df(spc_prep_final_df)

# ---------------------------
# Load raw data
# ---------------------------
X_train_raw = read_csv_flex_local(RAW_TRAIN_PATH) if os.path.exists(RAW_TRAIN_PATH) else pd.DataFrame()
X_test_raw = read_csv_flex_local(RAW_TEST_PATH) if os.path.exists(RAW_TEST_PATH) else pd.DataFrame()

if not X_train_raw.empty:
    X_train_raw = X_train_raw.loc[:, ~X_train_raw.columns.astype(str).str.contains(r"^Unnamed")]
if not X_test_raw.empty:
    X_test_raw = X_test_raw.loc[:, ~X_test_raw.columns.astype(str).str.contains(r"^Unnamed")]

y_train, y_train_path = load_label_simple(TRAIN_LABEL_CANDIDATES_SIMPLE)
y_test, y_test_path = load_label_simple(TEST_LABEL_CANDIDATES_SIMPLE)

# 監控序列優先用 test，沒有則退回 train
X_monitor_raw = X_test_raw.copy() if (USE_TEST_AS_MONITOR and not X_test_raw.empty) else X_train_raw.copy()
monitor_time = get_time_index(X_monitor_raw)

# baseline normal：優先 train 正常樣本，否則 train 全體
baseline_mask = None
if y_train is not None and not X_train_raw.empty and len(y_train) == len(X_train_raw):
    baseline_mask = (pd.to_numeric(y_train, errors="coerce").fillna(0).astype(int) == 0)
else:
    baseline_mask = pd.Series([True] * len(X_train_raw)) if not X_train_raw.empty else pd.Series(dtype=bool)

# 建 resolver
TRAIN_COL_RESOLVER = build_column_resolver(X_train_raw.columns)
MONITOR_COL_RESOLVER = build_column_resolver(X_monitor_raw.columns)

# ---------------------------
# Enrich meta
# ---------------------------
def enrich_with_spc_meta(list_df: pd.DataFrame):
    if list_df is None or list_df.empty:
        return pd.DataFrame()

    out = list_df.copy()
    if spc_prep_final_df is None or spc_prep_final_df.empty:
        return out
    if "feature" not in out.columns or "feature" not in spc_prep_final_df.columns:
        return out

    meta_cols = [c for c in [
        "feature",
        "dominant_monitor_direction_all",
        "mean_suggested_threshold",
        "normal_mean",
        "normal_std",
        "normal_p01",
        "normal_p99",
        "normal_missing_rate",
        "anchor_recommendation",
        "spc_selection_tier",
    ] if c in spc_prep_final_df.columns]

    meta = spc_prep_final_df[meta_cols].drop_duplicates(subset=["feature"]).copy()

    rename_map = {}
    for c in meta.columns:
        if c != "feature" and c in out.columns:
            rename_map[c] = f"{c}__meta"
    if rename_map:
        meta = meta.rename(columns=rename_map)

    out = out.merge(meta, on="feature", how="left")
    return out


main_signal_df = enrich_with_spc_meta(main_signal_df)
aux_diag_df = enrich_with_spc_meta(aux_diag_df)
final_deploy_df = enrich_with_spc_meta(final_deploy_df)

# ---------------------------
# Output root
# ---------------------------
SPC_EXPORT_ROOT = SPC_OUTPUT_DIR if "SPC_OUTPUT_DIR" in globals() else "outputs/spc_values"
# Safety guard: never write SPC values under outputs/shap/spc_values_grouped in project mode.
if str(SPC_EXPORT_ROOT).replace("\\", "/").endswith("outputs/shap/spc_values_grouped"):
    SPC_EXPORT_ROOT = SPC_OUTPUT_DIR if "SPC_OUTPUT_DIR" in globals() else "outputs/spc_values"

ensure_dir(SPC_EXPORT_ROOT)

run_info = {
    "SHAP_OUTPUT_DIR": OUT_DIR_USE,
    "SPC_OUTPUT_DIR": SPC_EXPORT_ROOT,
    "RAW_TRAIN_PATH": RAW_TRAIN_PATH,
    "RAW_TEST_PATH": RAW_TEST_PATH,
    "main_signal_path": main_signal_path,
    "aux_diag_path": aux_diag_path,
    "final_deploy_path": final_deploy_path,
    "spc_prep_path": spc_prep_path,
    "y_train_path": y_train_path,
    "y_test_path": y_test_path,
    "SUBGROUP_SIZE": SUBGROUP_SIZE,
    "EWMA_LAMBDA": EWMA_LAMBDA,
    "CUSUM_K": CUSUM_K,
    "CUSUM_H": CUSUM_H,
    "PCA_EXPLAINED_VAR": PCA_EXPLAINED_VAR,
    "n_train_cols": int(len(X_train_raw.columns)),
    "n_monitor_cols": int(len(X_monitor_raw.columns)),
}
with open(os.path.join(SPC_EXPORT_ROOT, "spc_export_manifest.json"), "w", encoding="utf-8") as f:
    json.dump(run_info, f, ensure_ascii=False, indent=2)


# ---------------------------
# Core data access
# ---------------------------
def resolve_feature_for_export(feature_name):
    train_col = resolve_feature_name(feature_name, TRAIN_COL_RESOLVER)
    monitor_col = resolve_feature_name(feature_name, MONITOR_COL_RESOLVER)

    if train_col is None and monitor_col is None:
        return None

    # 若一邊找不到，盡量用另一邊
    if train_col is None:
        train_col = monitor_col
    if monitor_col is None:
        monitor_col = train_col

    # 兩邊欄名不同但其實是同義時，用各自實際欄名
    return {
        "requested_feature": feature_name,
        "train_col": train_col,
        "monitor_col": monitor_col,
    }


def get_baseline_series_from_resolved(resolved_feature: dict) -> pd.Series:
    if resolved_feature is None:
        return pd.Series(dtype=float)

    train_col = resolved_feature["train_col"]
    if train_col not in X_train_raw.columns:
        return pd.Series(dtype=float)

    s = numeric_series(X_train_raw[train_col])
    if baseline_mask is not None and len(baseline_mask) == len(s):
        s = s[baseline_mask.values]
    return s.dropna()


def get_monitor_series_from_resolved(resolved_feature: dict) -> pd.Series:
    if resolved_feature is None:
        return pd.Series(dtype=float)

    monitor_col = resolved_feature["monitor_col"]
    if monitor_col not in X_monitor_raw.columns:
        return pd.Series(dtype=float)

    return numeric_series(X_monitor_raw[monitor_col]).reset_index(drop=True)


# ---------------------------
# Exporters
# ---------------------------
def export_continuous_feature_spc(feature_name: str, out_dir: str, row_meta: dict | None = None):
    resolved = resolve_feature_for_export(feature_name)
    if resolved is None:
        return None, {"requested_feature": feature_name, "reason": "feature_not_found_in_train_and_monitor"}

    base = get_baseline_series_from_resolved(resolved)
    mon = get_monitor_series_from_resolved(resolved)

    if len(base.dropna()) < 3:
        return None, {"requested_feature": feature_name, "resolved_feature": resolved["train_col"], "reason": "baseline_too_small"}
    if len(mon.dropna()) == 0:
        return None, {"requested_feature": feature_name, "resolved_feature": resolved["monitor_col"], "reason": "monitor_all_null"}

    center = safe_mean(base)
    sigma = safe_std(base)

    shew = calc_shewhart_3sigma(mon, center=center, sigma=sigma)
    mr = calc_moving_range(mon)
    ewma = calc_ewma(mon, center=center, sigma=sigma, lam=EWMA_LAMBDA)
    cusum = calc_cusum(mon, center=center, sigma=sigma, k=CUSUM_K, h=CUSUM_H)

    out = pd.DataFrame({"time_index": monitor_time})
    out = pd.concat([
        out.reset_index(drop=True),
        shew.reset_index(drop=True),
        mr[["MR", "MR_CL", "MR_UCL", "MR_LCL", "MR_beyond_UCL"]].reset_index(drop=True),
        ewma[["EWMA", "EWMA_CL", "EWMA_UCL", "EWMA_LCL", "EWMA_beyond_any_limit"]].reset_index(drop=True),
        cusum[["CUSUM_CPLUS", "CUSUM_CMINUS", "CUSUM_H_POS", "CUSUM_H_NEG", "CUSUM_alarm_any"]].reset_index(drop=True),
    ], axis=1)

    summary = {
        "feature": feature_name,
        "resolved_train_feature": resolved["train_col"],
        "resolved_monitor_feature": resolved["monitor_col"],
        "n_baseline": int(base.notna().sum()),
        "n_monitor": int(mon.notna().sum()),
        "center_mean": center,
        "sigma_train": sigma,
        "normal_p01": safe_quantile(base, 0.01),
        "normal_p99": safe_quantile(base, 0.99),
        "n_beyond_3sigma": int(out["beyond_any_limit"].sum()),
        "n_ewma_alarm": int(out["EWMA_beyond_any_limit"].sum()),
        "n_cusum_alarm": int(out["CUSUM_alarm_any"].sum()),
    }
    if row_meta:
        summary.update(row_meta)

    feature_dir = os.path.join(out_dir, safe_filename(feature_name))
    ensure_dir(feature_dir)
    write_df_or_empty_local(pd.DataFrame([summary]), os.path.join(feature_dir, "summary.csv"))
    write_df_or_empty_local(out, os.path.join(feature_dir, "series.csv"))
    return summary, None


def export_missingness_spc(feature_name: str, out_dir: str, row_meta: dict | None = None):
    resolved = resolve_feature_for_export(feature_name)
    if resolved is None:
        return None, {"requested_feature": feature_name, "reason": "feature_not_found_in_train_and_monitor"}

    train_col = resolved["train_col"]
    monitor_col = resolved["monitor_col"]

    if train_col not in X_train_raw.columns or monitor_col not in X_monitor_raw.columns:
        return None, {"requested_feature": feature_name, "reason": "resolved_column_missing_in_raw"}

    event_train = X_train_raw[train_col].isna().astype(int)
    event_monitor = X_monitor_raw[monitor_col].isna().astype(int)

    p_df = calc_p_chart(event_train, event_monitor, subgroup_size=SUBGROUP_SIZE)
    g_df = calc_g_chart(event_train, event_monitor)

    summary = {
        "feature": feature_name,
        "resolved_train_feature": train_col,
        "resolved_monitor_feature": monitor_col,
        "baseline_missing_rate": float(event_train.mean()) if len(event_train) > 0 else np.nan,
        "monitor_missing_rate": float(event_monitor.mean()) if len(event_monitor) > 0 else np.nan,
        "n_monitor_rows": int(len(event_monitor)),
        "n_pchart_alarm": int(p_df["beyond_any_limit"].sum()) if not p_df.empty else 0,
        "n_gchart_alarm": int(g_df["beyond_any_limit"].sum()) if not g_df.empty else 0,
    }
    if row_meta:
        summary.update(row_meta)

    feature_dir = os.path.join(out_dir, safe_filename(feature_name))
    ensure_dir(feature_dir)
    write_df_or_empty_local(pd.DataFrame([summary]), os.path.join(feature_dir, "summary.csv"))
    write_df_or_empty_local(p_df, os.path.join(feature_dir, "p_chart.csv"))
    write_df_or_empty_local(g_df, os.path.join(feature_dir, "g_chart.csv"))
    return summary, None


def export_interaction_pair_spc(feature_a: str, feature_b: str, out_dir: str, row_meta: dict | None = None):
    resolved_a = resolve_feature_for_export(feature_a)
    resolved_b = resolve_feature_for_export(feature_b)

    if resolved_a is None or resolved_b is None:
        return None, {
            "requested_feature_a": feature_a,
            "requested_feature_b": feature_b,
            "reason": "pair_feature_not_found"
        }

    train_a, train_b = resolved_a["train_col"], resolved_b["train_col"]
    mon_a, mon_b = resolved_a["monitor_col"], resolved_b["monitor_col"]

    if train_a not in X_train_raw.columns or train_b not in X_train_raw.columns:
        return None, {
            "requested_feature_a": feature_a,
            "requested_feature_b": feature_b,
            "reason": "resolved_train_pair_missing"
        }
    if mon_a not in X_monitor_raw.columns or mon_b not in X_monitor_raw.columns:
        return None, {
            "requested_feature_a": feature_a,
            "requested_feature_b": feature_b,
            "reason": "resolved_monitor_pair_missing"
        }

    train_pair = X_train_raw[[train_a, train_b]].copy()
    if baseline_mask is not None and len(baseline_mask) == len(train_pair):
        train_pair = train_pair.loc[baseline_mask.values].copy()

    mon_pair = X_monitor_raw[[mon_a, mon_b]].copy()

    t2_summary, t2_series = calc_hotelling_t2(train_pair, mon_pair)
    if t2_summary.empty or t2_series.empty:
        return None, {
            "requested_feature_a": feature_a,
            "requested_feature_b": feature_b,
            "resolved_feature_a": train_a,
            "resolved_feature_b": train_b,
            "reason": "pair_t2_not_available"
        }

    train_num = prepare_matrix(train_pair)
    mon_num = prepare_matrix(mon_pair)
    mu = train_num.mean(axis=0)
    sd = train_num.std(axis=0, ddof=1).replace(0, 1.0).fillna(1.0)
    z_dist_train = np.sqrt(np.sum(((train_num - mu) / sd) ** 2, axis=1))
    z_dist_mon = np.sqrt(np.sum(((mon_num - mu) / sd) ** 2, axis=1))
    z_cl = float(np.mean(z_dist_train))
    z_sd = float(np.std(z_dist_train, ddof=1)) if len(z_dist_train) > 1 else np.nan
    z_ucl = z_cl + 3 * z_sd if pd.notna(z_sd) else np.nan

    pair_series = pd.DataFrame({
        "time_index": monitor_time,
        "feature_a": numeric_series(X_monitor_raw[mon_a]).reset_index(drop=True),
        "feature_b": numeric_series(X_monitor_raw[mon_b]).reset_index(drop=True),
        "pair_z_distance": z_dist_mon,
        "pair_z_distance_CL": z_cl,
        "pair_z_distance_UCL": z_ucl,
        "pair_z_distance_alarm": ((z_dist_mon > z_ucl) if pd.notna(z_ucl) else np.zeros(len(z_dist_mon))).astype(int),
    })
    pair_series = pd.concat([pair_series.reset_index(drop=True), t2_series.reset_index(drop=True)], axis=1)

    summary = {
        "feature_a": feature_a,
        "feature_b": feature_b,
        "resolved_train_feature_a": train_a,
        "resolved_train_feature_b": train_b,
        "resolved_monitor_feature_a": mon_a,
        "resolved_monitor_feature_b": mon_b,
        "n_monitor_rows": int(len(pair_series)),
        "pair_z_distance_CL": z_cl,
        "pair_z_distance_UCL": z_ucl,
        "n_pair_z_distance_alarm": int(pair_series["pair_z_distance_alarm"].sum()),
        "T2_CL": t2_summary.iloc[0]["T2_CL"],
        "T2_UCL": t2_summary.iloc[0]["T2_UCL"],
        "n_T2_alarm": int(pair_series["T2_beyond_UCL"].sum()),
    }
    if row_meta:
        summary.update(row_meta)

    pair_name = f"{safe_filename(feature_a)}__{safe_filename(feature_b)}"
    pair_dir = os.path.join(out_dir, pair_name)
    ensure_dir(pair_dir)
    write_df_or_empty_local(pd.DataFrame([summary]), os.path.join(pair_dir, "summary.csv"))
    write_df_or_empty_local(pair_series, os.path.join(pair_dir, "pair_series.csv"))
    return summary, None


def export_multivariate_bundle(feature_list, out_dir, bundle_name="multivariate_bundle"):
    resolved_features = []
    unresolved = []

    for f in feature_list:
        r = resolve_feature_for_export(f)
        if r is None:
            unresolved.append({"requested_feature": f, "reason": "feature_not_found"})
            continue
        resolved_features.append((f, r["train_col"], r["monitor_col"]))

    train_cols = [t for _, t, _ in resolved_features if t in X_train_raw.columns]
    monitor_cols = [m for _, _, m in resolved_features if m in X_monitor_raw.columns]

    # multivariate 需要能在 train / monitor 都成立的共同欄位
    pairs = [(req, t, m) for req, t, m in resolved_features if t in X_train_raw.columns and m in X_monitor_raw.columns]
    if len(pairs) < 2:
        return pd.DataFrame(unresolved)

    train_df = X_train_raw[[t for _, t, _ in pairs]].copy()
    if baseline_mask is not None and len(baseline_mask) == len(train_df):
        train_df = train_df.loc[baseline_mask.values].copy()

    monitor_df = X_monitor_raw[[m for _, _, m in pairs]].copy()

    # 欄名對齊成 requested feature，方便解讀
    requested_names = [req for req, _, _ in pairs]
    train_df.columns = requested_names
    monitor_df.columns = requested_names

    ensure_dir(out_dir)

    t2_summary, t2_series = calc_hotelling_t2(train_df, monitor_df)
    pca_summary, pca_scores, pca_spe = calc_pca_spc(train_df, monitor_df, explained_var=PCA_EXPLAINED_VAR)

    bundle_dir = os.path.join(out_dir, safe_filename(bundle_name))
    ensure_dir(bundle_dir)

    write_df_or_empty_local(t2_summary, os.path.join(bundle_dir, "hotelling_t2_summary.csv"))
    if not t2_series.empty:
        t2_series = pd.concat([
            pd.DataFrame({"time_index": monitor_time}).reset_index(drop=True),
            t2_series.reset_index(drop=True)
        ], axis=1)
    write_df_or_empty_local(t2_series, os.path.join(bundle_dir, "hotelling_t2_series.csv"))

    write_df_or_empty_local(pca_summary, os.path.join(bundle_dir, "pca_summary.csv"))
    if not pca_scores.empty:
        pca_scores = pd.concat([
            pd.DataFrame({"time_index": monitor_time}).reset_index(drop=True),
            pca_scores.reset_index(drop=True)
        ], axis=1)
    if not pca_spe.empty:
        pca_spe = pd.concat([
            pd.DataFrame({"time_index": monitor_time}).reset_index(drop=True),
            pca_spe.reset_index(drop=True)
        ], axis=1)
    write_df_or_empty_local(pca_scores, os.path.join(bundle_dir, "pca_scores_series.csv"))
    write_df_or_empty_local(pca_spe, os.path.join(bundle_dir, "pca_spe_series.csv"))

    unresolved_df = pd.DataFrame(unresolved)
    write_df_or_empty_local(unresolved_df, os.path.join(bundle_dir, "unresolved_features.csv"))
    return unresolved_df


# ---------------------------
# Group processor
# ---------------------------
def clean_row_meta(row: pd.Series, exclude_cols):
    meta = {}
    for k, v in row.items():
        if k in exclude_cols:
            continue
        if pd.isna(v):
            continue
        meta[k] = v
    return meta


def process_feature_list_df(df: pd.DataFrame, category_name: str, subtype_col: str, default_subtype: str):
    if df is None or df.empty:
        return

    category_root = os.path.join(SPC_EXPORT_ROOT, category_name)
    ensure_dir(category_root)

    out_df = df.copy()
    if subtype_col not in out_df.columns:
        out_df[subtype_col] = default_subtype
    out_df[subtype_col] = out_df[subtype_col].fillna(default_subtype).astype(str)

    category_debug_rows = []

    for subtype, sub_df in out_df.groupby(subtype_col, dropna=False):
        subtype = str(subtype) if pd.notna(subtype) else default_subtype
        subtype_root = os.path.join(category_root, safe_filename(subtype))
        ensure_dir(subtype_root)

        continuous_summaries = []
        missing_summaries = []
        interaction_summaries = []
        unresolved_rows = []

        # 1) feature-based exports
        if "feature" in sub_df.columns:
            for _, row in sub_df.iterrows():
                feature_name = row.get("feature")
                if pd.isna(feature_name):
                    continue

                row_meta = clean_row_meta(row, exclude_cols=["feature"])

                cont_summary, cont_unresolved = export_continuous_feature_spc(
                    feature_name=feature_name,
                    out_dir=os.path.join(subtype_root, "continuous_features"),
                    row_meta=row_meta,
                )
                if cont_summary is not None:
                    continuous_summaries.append(cont_summary)
                if cont_unresolved is not None:
                    cont_unresolved.update({"export_type": "continuous"})
                    unresolved_rows.append(cont_unresolved)

                miss_summary, miss_unresolved = export_missingness_spc(
                    feature_name=feature_name,
                    out_dir=os.path.join(subtype_root, "missingness_features"),
                    row_meta=row_meta,
                )
                if miss_summary is not None:
                    missing_summaries.append(miss_summary)
                if miss_unresolved is not None:
                    miss_unresolved.update({"export_type": "missingness"})
                    unresolved_rows.append(miss_unresolved)

        # 2) interaction pair exports
        pair_cols_ready = ("feature_a" in sub_df.columns) and ("feature_b" in sub_df.columns)
        if pair_cols_ready:
            for _, row in sub_df.iterrows():
                fa = row.get("feature_a")
                fb = row.get("feature_b")
                if pd.isna(fa) or pd.isna(fb):
                    continue

                row_meta = clean_row_meta(row, exclude_cols=["feature_a", "feature_b"])
                pair_summary, pair_unresolved = export_interaction_pair_spc(
                    feature_a=fa,
                    feature_b=fb,
                    out_dir=os.path.join(subtype_root, "interaction_pairs"),
                    row_meta=row_meta,
                )
                if pair_summary is not None:
                    interaction_summaries.append(pair_summary)
                if pair_unresolved is not None:
                    pair_unresolved.update({"export_type": "interaction"})
                    unresolved_rows.append(pair_unresolved)

        # 3) multivariate bundle
        bundle_features = []
        if "feature" in sub_df.columns:
            bundle_features = [f for f in sub_df["feature"].dropna().tolist()]
        elif pair_cols_ready:
            feats = []
            feats.extend(sub_df["feature_a"].dropna().tolist())
            feats.extend(sub_df["feature_b"].dropna().tolist())
            bundle_features = sorted(set(feats))

        multi_unresolved_df = export_multivariate_bundle(
            feature_list=bundle_features,
            out_dir=os.path.join(subtype_root, "multivariate_spc"),
            bundle_name=f"{subtype}__bundle"
        )

        cont_df = pd.DataFrame(continuous_summaries)
        miss_df = pd.DataFrame(missing_summaries)
        inter_df = pd.DataFrame(interaction_summaries)
        unresolved_df = pd.DataFrame(unresolved_rows)

        write_df_or_empty_local(cont_df, os.path.join(subtype_root, "continuous_feature_summary__all.csv"))
        write_df_or_empty_local(miss_df, os.path.join(subtype_root, "missingness_summary__all.csv"))
        write_df_or_empty_local(inter_df, os.path.join(subtype_root, "interaction_pair_summary__all.csv"))
        write_df_or_empty_local(unresolved_df, os.path.join(subtype_root, "unresolved_items__all.csv"))

        category_debug_rows.append({
            "category": category_name,
            "subtype": subtype,
            "n_rows_in_list": int(len(sub_df)),
            "n_continuous_features_exported": int(len(cont_df)),
            "n_missingness_features_exported": int(len(miss_df)),
            "n_interaction_pairs_exported": int(len(inter_df)),
            "n_unresolved_items": int(len(unresolved_df)),
            "n_multivariate_unresolved_items": int(len(multi_unresolved_df)) if multi_unresolved_df is not None else 0,
        })

    write_df_or_empty_local(pd.DataFrame(category_debug_rows), os.path.join(category_root, "_category_summary.csv"))


# ---------------------------
# Run
# ---------------------------
process_feature_list_df(
    df=main_signal_df,
    category_name="main_signal",
    subtype_col="list_subtype",
    default_subtype="main_signal"
)

process_feature_list_df(
    df=aux_diag_df,
    category_name="auxiliary_diagnostic",
    subtype_col="list_subtype",
    default_subtype="auxiliary_diagnostic"
)

if final_deploy_df is not None and not final_deploy_df.empty:
    final_deploy_df = final_deploy_df.copy()
    if "spc_selection_tier" not in final_deploy_df.columns:
        if "anchor_recommendation" in final_deploy_df.columns:
            final_deploy_df["spc_selection_tier"] = final_deploy_df["anchor_recommendation"].map({
                "primary_anchor": "Tier_A",
                "review_anchor": "Tier_B",
            }).fillna("Tier_B")
        else:
            final_deploy_df["spc_selection_tier"] = "Tier_A"

process_feature_list_df(
    df=final_deploy_df,
    category_name="final_deployment",
    subtype_col="spc_selection_tier",
    default_subtype="Tier_A"
)

# ---------------------------
# Quick debug summary
# ---------------------------
def quick_match_report(list_df: pd.DataFrame, name: str):
    rows = []

    if list_df is not None and not list_df.empty and "feature" in list_df.columns:
        feats = list_df["feature"].dropna().tolist()
        resolved_train = [resolve_feature_name(f, TRAIN_COL_RESOLVER) for f in feats]
        resolved_monitor = [resolve_feature_name(f, MONITOR_COL_RESOLVER) for f in feats]
        rows.append({
            "list_name": name,
            "entity": "feature",
            "n_items": len(feats),
            "match_in_train": int(sum(x is not None for x in resolved_train)),
            "match_in_monitor": int(sum(x is not None for x in resolved_monitor)),
        })

    if list_df is not None and not list_df.empty and "feature_a" in list_df.columns and "feature_b" in list_df.columns:
        fa = list_df["feature_a"].dropna().tolist()
        fb = list_df["feature_b"].dropna().tolist()
        ra = [resolve_feature_name(f, TRAIN_COL_RESOLVER) for f in fa]
        rb = [resolve_feature_name(f, TRAIN_COL_RESOLVER) for f in fb]
        rows.append({
            "list_name": name,
            "entity": "pair_feature_a",
            "n_items": len(fa),
            "match_in_train": int(sum(x is not None for x in ra)),
            "match_in_monitor": int(sum(resolve_feature_name(f, MONITOR_COL_RESOLVER) is not None for f in fa)),
        })
        rows.append({
            "list_name": name,
            "entity": "pair_feature_b",
            "n_items": len(fb),
            "match_in_train": int(sum(x is not None for x in rb)),
            "match_in_monitor": int(sum(resolve_feature_name(f, MONITOR_COL_RESOLVER) is not None for f in fb)),
        })

    return pd.DataFrame(rows)


debug_match_df = pd.concat([
    quick_match_report(main_signal_df, "main_signal"),
    quick_match_report(aux_diag_df, "auxiliary_diagnostic"),
    quick_match_report(final_deploy_df, "final_deployment"),
], axis=0, ignore_index=True)

write_df_or_empty_local(debug_match_df, os.path.join(SPC_EXPORT_ROOT, "_debug_feature_match_report.csv"))

print("=" * 100)
print("SPC values exported")
print("=" * 100)
print(f"Root folder: {SPC_EXPORT_ROOT}")
print("- main_signal/")
print("- auxiliary_diagnostic/")
print("- final_deployment/")
print("另外已輸出:")
print("- _debug_feature_match_report.csv")
print("- 各 subtype 的 unresolved_items__all.csv")
print("Done.")