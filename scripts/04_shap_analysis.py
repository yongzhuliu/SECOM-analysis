"""
04_shap_analysis.py

SHAP / root-cause feature attribution script converted from the original notebook cell.

Expected inputs:
    data/processed/X_train_base.csv
    data/processed/X_train_rferf.csv
    data/processed/X_train_rfexgb.csv
    data/processed/X_train_anova.csv
    data/processed/y_train.csv
    outputs/training/all_model_evaluation_detail.csv
    models/all_best_models_and_thresholds.joblib

Default output:
    outputs/shap/

Run:
    python scripts/04_shap_analysis.py

or:
    python scripts/04_shap_analysis.py \
        --processed-dir data/processed \
        --training-output-dir outputs/training \
        --model-dir models \
        --output-dir outputs/shap
"""

# =========================================================
# Initial anomaly feature attribution pipeline
# Single-cell version for notebook:
# - 可直接 joblib.load 你先前存好的 artifacts
# - 先在同一格內補回訓練時用到的自訂 class / scorer
# - 讀取已存好的 csv / joblib
# - 輸出 TP/FP 分組 SHAP、統計摘要、候選特徵總表
# - 追加 missingness / interaction 候選 / 高相關伴隨特徵 /
#   正負類分布差異 / SPC 前置整合包
# =========================================================

import os
import re
import math
import json
import warnings
from collections import Counter
from itertools import combinations

import joblib
import numpy as np
import pandas as pd
import shap

try:
    from scipy.stats import ks_2samp as scipy_ks_2samp
    from scipy.stats import wasserstein_distance as scipy_wasserstein_distance
except Exception:
    scipy_ks_2samp = None
    scipy_wasserstein_distance = None

# ---- imports required for unpickling training artifacts ----
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    IsolationForest,
    AdaBoostClassifier,
    VotingClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier

warnings.filterwarnings("ignore")


# =========================================================
# Project path config
# =========================================================
# These defaults match the project structure:
#   data/processed/
#   outputs/training/
#   models/
#   outputs/shap/
#
# They can be overridden from command line arguments in __main__.
PROCESSED_DIR = "data/processed"
TRAINING_OUTPUT_DIR = "outputs/training"
MODEL_DIR = "models"
SHAP_OUTPUT_DIR = "outputs/shap"


def configure_project_paths(processed_dir, training_output_dir, model_dir, shap_output_dir):
    """Update global project paths before running the SHAP pipeline."""
    global PROCESSED_DIR, TRAINING_OUTPUT_DIR, MODEL_DIR, SHAP_OUTPUT_DIR
    global DETAIL_CSV_PATH, ARTIFACT_PATH, OUT_DIR
    global TRAIN_CSV_MAP, LABEL_CSV_CANDIDATES

    PROCESSED_DIR = processed_dir
    TRAINING_OUTPUT_DIR = training_output_dir
    MODEL_DIR = model_dir
    SHAP_OUTPUT_DIR = shap_output_dir

    DETAIL_CSV_PATH = os.path.join(TRAINING_OUTPUT_DIR, "all_model_evaluation_detail.csv")
    ARTIFACT_PATH = os.path.join(MODEL_DIR, "all_best_models_and_thresholds.joblib")
    OUT_DIR = SHAP_OUTPUT_DIR

    TRAIN_CSV_MAP = {
        "Base": os.path.join(PROCESSED_DIR, "X_train_base.csv"),
        "RFERF": os.path.join(PROCESSED_DIR, "X_train_rferf.csv"),
        "RFEXGB": os.path.join(PROCESSED_DIR, "X_train_rfexgb.csv"),
        "ANOVA": os.path.join(PROCESSED_DIR, "X_train_anova.csv"),
    }

    LABEL_CSV_CANDIDATES = {
        "Base": [os.path.join(PROCESSED_DIR, "y_train_base.csv"), os.path.join(PROCESSED_DIR, "y_train.csv")],
        "RFERF": [os.path.join(PROCESSED_DIR, "y_train_rferf.csv"), os.path.join(PROCESSED_DIR, "y_train.csv")],
        "RFEXGB": [os.path.join(PROCESSED_DIR, "y_train_rfexgb.csv"), os.path.join(PROCESSED_DIR, "y_train.csv")],
        "ANOVA": [os.path.join(PROCESSED_DIR, "y_train_anova.csv"), os.path.join(PROCESSED_DIR, "y_train.csv")],
    }


# =========================================================
# Training-time custom objects (must exist before joblib.load)
# =========================================================
RANDOM_STATE = 666
POS_LABEL = 1
NEG_LABEL = 0


def take_rows(X, idx):
    if hasattr(X, "iloc"):
        return X.iloc[idx]
    return X[idx]


def to_1d_numpy(y):
    if isinstance(y, pd.Series):
        return y.to_numpy()
    if isinstance(y, pd.DataFrame):
        return y.squeeze().to_numpy()
    return np.asarray(y)


class SampleWeightClassifier(ClassifierMixin, BaseEstimator):
    _estimator_type = "classifier"

    def __init__(self, base_estimator, class_weight=None):
        self.base_estimator = base_estimator
        self.class_weight = class_weight

    def fit(self, X, y):
        y_arr = to_1d_numpy(y)
        self.estimator_ = clone(self.base_estimator)

        fit_kwargs = {}
        if self.class_weight is not None:
            sample_weight = np.ones(len(y_arr), dtype=float)
            for cls, weight in self.class_weight.items():
                sample_weight[y_arr == cls] = float(weight)
            fit_kwargs["sample_weight"] = sample_weight

        self.estimator_.fit(X, y_arr, **fit_kwargs)
        self.classes_ = getattr(self.estimator_, "classes_", np.unique(y_arr))
        if hasattr(self.estimator_, "feature_names_in_"):
            self.feature_names_in_ = self.estimator_.feature_names_in_
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def decision_function(self, X):
        if hasattr(self.estimator_, "decision_function"):
            return self.estimator_.decision_function(X)
        if hasattr(self.estimator_, "predict_proba"):
            proba = self.estimator_.predict_proba(X)
            return proba[:, 1]
        raise AttributeError("Wrapped estimator has no decision_function or predict_proba")

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        return tags


class NormalOnlyAnomalyDetector(BaseEstimator):
    def __init__(
        self,
        model_name="IsolationForest",
        n_estimators=300,
        max_samples="auto",
        contamination=0.05,
        n_neighbors=20,
        random_state=RANDOM_STATE,
    ):
        self.model_name = model_name
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    def _build_model(self):
        if self.model_name == "IsolationForest":
            return IsolationForest(
                n_estimators=self.n_estimators,
                max_samples=self.max_samples,
                contamination=self.contamination,
                random_state=self.random_state,
                n_jobs=1,
            )

        if self.model_name == "LOF":
            return LocalOutlierFactor(
                n_neighbors=self.n_neighbors,
                contamination=self.contamination,
                novelty=True,
            )

        raise ValueError(f"Unknown anomaly model: {self.model_name}")

    def fit(self, X, y=None):
        if y is None:
            X_normal = X
        else:
            y_np = to_1d_numpy(y)
            normal_idx = np.where(y_np == NEG_LABEL)[0]
            X_normal = take_rows(X, normal_idx)

        self.scaler_ = StandardScaler()
        X_normal_scaled = self.scaler_.fit_transform(X_normal)

        self.model_ = self._build_model()
        self.model_.fit(X_normal_scaled)
        return self

    def decision_function(self, X):
        X_scaled = self.scaler_.transform(X)

        if hasattr(self.model_, "decision_function"):
            return np.asarray(-self.model_.decision_function(X_scaled)).ravel()

        if hasattr(self.model_, "score_samples"):
            return np.asarray(-self.model_.score_samples(X_scaled)).ravel()

        pred = np.asarray(self.model_.predict(X_scaled)).ravel()
        return (pred == -1).astype(float)

    def predict(self, X):
        scores = self.decision_function(X)
        return (scores >= 0.0).astype(int)


def anomaly_ap_scorer(estimator, X, y):
    y = np.asarray(y).ravel().astype(int)
    scores = np.asarray(estimator.decision_function(X)).ravel()
    return average_precision_score(y, scores)


# =========================================================
# Config
# =========================================================
DETAIL_CSV_PATH = os.path.join(TRAINING_OUTPUT_DIR, "all_model_evaluation_detail.csv")
ARTIFACT_PATH = os.path.join(MODEL_DIR, "all_best_models_and_thresholds.joblib")
OUT_DIR = SHAP_OUTPUT_DIR

TOP_K_MODELS = 10

TRAIN_CSV_MAP = {
    "Base": os.path.join(PROCESSED_DIR, "X_train_base.csv"),
    "RFERF": os.path.join(PROCESSED_DIR, "X_train_rferf.csv"),
    "RFEXGB": os.path.join(PROCESSED_DIR, "X_train_rfexgb.csv"),
    "ANOVA": os.path.join(PROCESSED_DIR, "X_train_anova.csv"),
}

# 先找 feature_set 專屬標籤；沒有就退回共用 y_train.csv
LABEL_CSV_CANDIDATES = {
    "Base": [os.path.join(PROCESSED_DIR, "y_train_base.csv"), os.path.join(PROCESSED_DIR, "y_train.csv")],
    "RFERF": [os.path.join(PROCESSED_DIR, "y_train_rferf.csv"), os.path.join(PROCESSED_DIR, "y_train.csv")],
    "RFEXGB": [os.path.join(PROCESSED_DIR, "y_train_rfexgb.csv"), os.path.join(PROCESSED_DIR, "y_train.csv")],
    "ANOVA": [os.path.join(PROCESSED_DIR, "y_train_anova.csv"), os.path.join(PROCESSED_DIR, "y_train.csv")],
}

LABEL_COLUMN_CANDIDATES = ["target", "label", "y", "is_anomaly", "anomaly"]

BACKGROUND_SIZE = 60
MAX_EXPLAIN_ROWS_PER_GROUP = 80
MAX_EVALS_CAP = 500
SHAP_RANDOM_STATE = 666

TOP_N_PER_MODEL_FOR_STABILITY = 10
CORR_THRESHOLD = 0.90
MIN_GROUP_SIZE_FOR_SHAP = 8

NORMAL_HIGH_QUANTILE = 0.99
NORMAL_LOW_QUANTILE = 0.01

# --- appended modules config (不改原邏輯，只新增輸出) ---
PAIR_TOP_FEATURES = 15
PAIR_MIN_SUPPORT = 10
PAIR_MIN_LIFT = 2.0
HIGH_CORR_COMPANION_THRESHOLD = 0.95

OOF_N_SPLITS = 5
MIN_FP_EVIDENCE_ROWS = 15
MIN_DIRECTIONAL_PUSH_RATE = 0.60
MIN_ABS_SMD_FOR_MONITOR = 0.30
MIN_TEST_MONITOR_SCORE = 0.60
OUTPUT_TOP_N = 10


# =========================================================
# Helpers
# =========================================================
def safe_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))


def ensure_out_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_csv_flex(path: str) -> pd.DataFrame:
    last_err = None
    for enc in ["utf-8-sig", "utf-8", "cp950", "big5"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err


def make_artifact_key(row: pd.Series) -> str:
    feature_set = str(row["feature_set"])
    model = str(row["model"])
    balance_method = row["balance_method"]

    if pd.isna(balance_method):
        return f"{feature_set}_{model}"

    balance_method = str(balance_method)

    if balance_method == "N/A":
        return f"{feature_set}_{model}"

    if model == "VotingEnsemble":
        base_balance = balance_method.replace("+NoResample", "")
        return f"{feature_set}_{model}_{base_balance}_plus_NoResample"

    return f"{feature_set}_{model}_{balance_method}"


def load_feature_train_df(feature_set: str) -> tuple[pd.DataFrame, str]:
    if feature_set not in TRAIN_CSV_MAP:
        raise KeyError(f"未知的 feature_set: {feature_set}")

    csv_path = TRAIN_CSV_MAP[feature_set]
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到 train csv: {csv_path}")

    df = read_csv_flex(csv_path)
    df = df.loc[:, ~df.columns.astype(str).str.contains(r"^Unnamed")]
    return df, csv_path


def should_skip_feature_attribution(row: pd.Series, artifact: dict) -> tuple[bool, str]:
    feature_set = str(row["feature_set"])
    model_name = str(row["model"])
    final_model = artifact.get("final_model")

    if feature_set == "Fusion":
        return True, "Fusion 沒有單一原始 feature space，歸因結果不直接對應可管制特徵"

    if model_name == "VotingEnsemble":
        return True, "VotingEnsemble 為 score-fusion ensemble，未實作 final-system attribution，先排除於最終歸因名單"

    if isinstance(final_model, dict) and final_model.get("type") == "score_fusion_ensemble":
        return True, "score_fusion_ensemble 不適用單一原始 feature space 的 SHAP 流程"

    return False, ""

def load_label_series(feature_set: str, expected_len: int) -> tuple[pd.Series | None, str | None]:
    candidates = LABEL_CSV_CANDIDATES.get(feature_set, ["y_train.csv"])

    label_path = None
    for p in candidates:
        if p and os.path.exists(p):
            label_path = p
            break

    if label_path is None:
        print(f"[Warn] no label csv found for {feature_set} -> will use pred_anomaly / pred_normal only")
        return None, None

    y_df = read_csv_flex(label_path)
    y_df = y_df.loc[:, ~y_df.columns.astype(str).str.contains(r"^Unnamed")]

    if y_df.shape[1] == 1:
        y = y_df.iloc[:, 0]
    else:
        col = None
        for c in LABEL_COLUMN_CANDIDATES:
            if c in y_df.columns:
                col = c
                break
        if col is None:
            col = y_df.columns[0]
        y = y_df[col]

    y = pd.Series(y).reset_index(drop=True)

    if len(y) != expected_len:
        raise ValueError(
            f"Label length mismatch for {feature_set}: len(y)={len(y)} != expected_len={expected_len}"
        )

    uniq = set(pd.Series(y).dropna().unique().tolist())
    if uniq <= {0, 1}:
        y = y.astype(int)
    elif uniq <= {False, True}:
        y = y.astype(int)
    else:
        mapping = {
            "normal": 0,
            "anomaly": 1,
            "abnormal": 1,
            "yes": 1,
            "no": 0,
            "true": 1,
            "false": 0,
            "positive": 1,
            "negative": 0,
            "pass": 0,
            "fail": 1,
            "-1": 0,
            "1": 1,
        }
        y_str = y.astype(str).str.strip().str.lower()
        if y_str.isin(mapping.keys()).all():
            y = y_str.map(mapping).astype(int)
        else:
            raise ValueError(f"Unsupported label values for {feature_set}: {sorted(list(uniq))[:10]}")

    return y, label_path


def ensure_dataframe(X, columns):
    if isinstance(X, pd.DataFrame):
        return X.loc[:, columns]
    return pd.DataFrame(X, columns=columns)


def get_expected_feature_columns(final_model) -> list[str] | None:
    if hasattr(final_model, "feature_names_in_"):
        return list(final_model.feature_names_in_)

    if hasattr(final_model, "named_steps"):
        for _, step in reversed(final_model.named_steps.items()):
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)

    if hasattr(final_model, "estimators_"):
        for est in final_model.estimators_:
            if hasattr(est, "feature_names_in_"):
                return list(est.feature_names_in_)
            if hasattr(est, "named_steps"):
                for _, step in reversed(est.named_steps.items()):
                    if hasattr(step, "feature_names_in_"):
                        return list(step.feature_names_in_)

    return None


def align_columns_for_model(X_df: pd.DataFrame, final_model) -> pd.DataFrame:
    X_df = X_df.copy()
    expected_cols = get_expected_feature_columns(final_model)

    if expected_cols is None:
        return X_df

    missing = [c for c in expected_cols if c not in X_df.columns]
    if missing:
        raise ValueError(f"train csv 缺少模型需要的欄位，缺少前幾個欄位: {missing[:10]}")

    return X_df.loc[:, expected_cols].copy()


def get_model_score_fn(final_model, category: str, feature_columns: list[str], prefer_margin: bool = False):
    def score_fn(X):
        X = ensure_dataframe(X, feature_columns)

        if category == "supervised":
            if prefer_margin and hasattr(final_model, "decision_function"):
                return np.asarray(final_model.decision_function(X)).ravel()

            if hasattr(final_model, "predict_proba"):
                proba = np.asarray(final_model.predict_proba(X))
                if proba.ndim == 2:
                    return proba[:, 1].ravel()
                return proba.ravel()

            if hasattr(final_model, "decision_function"):
                return np.asarray(final_model.decision_function(X)).ravel()

            raise ValueError("Supervised model has neither predict_proba nor decision_function")

        if category == "anomaly":
            if hasattr(final_model, "decision_function"):
                return np.asarray(final_model.decision_function(X)).ravel()

            if hasattr(final_model, "score_samples"):
                return np.asarray(final_model.score_samples(X)).ravel()

            pred = np.asarray(final_model.predict(X)).ravel()
            return (pred == 1).astype(float)

        raise ValueError(f"Unknown category: {category}")

    return score_fn

def to_binary_prediction(scores: np.ndarray, threshold: float) -> np.ndarray:
    return (np.asarray(scores).ravel() >= float(threshold)).astype(int)


def build_group_masks(y_true: pd.Series | None, y_pred: np.ndarray) -> dict[str, np.ndarray]:
    y_pred = np.asarray(y_pred).astype(int).ravel()

    if y_true is None:
        return {
            "pred_anomaly": y_pred == 1,
            "pred_normal": y_pred == 0,
        }

    y_true = np.asarray(y_true).astype(int).ravel()

    return {
        "TP": (y_true == 1) & (y_pred == 1),
        "FP": (y_true == 0) & (y_pred == 1),
        "TN": (y_true == 0) & (y_pred == 0),
        "FN": (y_true == 1) & (y_pred == 0),
        "pred_anomaly": y_pred == 1,
        "pred_normal": y_pred == 0,
        "actual_anomaly": y_true == 1,
        "actual_normal": y_true == 0,
    }


def pick_explain_subset_by_mask(
    X_df: pd.DataFrame,
    scores: np.ndarray,
    mask: np.ndarray,
    max_rows: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    scores = np.asarray(scores).ravel()
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return X_df.iloc[[]].copy(), idx

    idx = idx[np.argsort(scores[idx])[::-1]]
    idx = idx[:max_rows]
    return X_df.iloc[idx].copy(), idx


def get_background_df(
    X_train_for_model: pd.DataFrame,
    y_true: pd.Series | None = None,
    category: str = "supervised",
    n_background: int = BACKGROUND_SIZE,
) -> pd.DataFrame:
    use_df = X_train_for_model.copy()

    if category == "anomaly" and y_true is not None:
        y_arr = np.asarray(y_true).astype(int).ravel()
        normal_mask = y_arr == NEG_LABEL
        if normal_mask.sum() > 0:
            use_df = X_train_for_model.loc[normal_mask].copy()

    if len(use_df) <= n_background:
        return use_df.copy()

    return use_df.sample(
        n=n_background,
        random_state=SHAP_RANDOM_STATE,
        replace=False,
    ).copy()

def _unwrap_native_shap_model(final_model):
    model = final_model
    scaler = None
    use_transformed = False

    if hasattr(model, "named_steps"):
        steps = model.named_steps
        scaler = steps.get("scaler")
        model = steps.get("clf", model)
        if scaler is not None:
            use_transformed = True

    if isinstance(model, SampleWeightClassifier):
        model = getattr(model, "estimator_", getattr(model, "base_estimator", model))

    return model, scaler, use_transformed


def _coerce_shap_values(values):
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        if values.shape[-1] == 2:
            values = values[:, :, 1]
        else:
            values = np.squeeze(values)
    if values.ndim != 2:
        raise ValueError(f"Unexpected SHAP shape: {values.shape}")
    return values


def get_explanation_score_fn(final_model, category: str, feature_columns: list[str]):
    return get_model_score_fn(final_model, category, feature_columns, prefer_margin=(category == "supervised"))


def compute_shap_values(final_model, score_fn, background_df: pd.DataFrame, explain_df: pd.DataFrame, category: str) -> np.ndarray:
    if explain_df.empty:
        return np.empty((0, explain_df.shape[1]))

    model, scaler, use_transformed = _unwrap_native_shap_model(final_model)

    # Tree-based native SHAP on raw feature space when no scaler is involved.
    tree_like = (
        isinstance(model, (RandomForestClassifier, HistGradientBoostingClassifier, XGBClassifier, BalancedRandomForestClassifier))
    )
    if tree_like and not use_transformed:
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(explain_df)
            return _coerce_shap_values(shap_values)
        except Exception:
            pass

    # Linear SHAP for LR when scaler exists; keep original feature names, use transformed values.
    if isinstance(model, LogisticRegression):
        try:
            bg = scaler.transform(background_df) if scaler is not None else background_df.to_numpy()
            ex = scaler.transform(explain_df) if scaler is not None else explain_df.to_numpy()
            explainer = shap.LinearExplainer(model, bg)
            shap_values = explainer(ex).values if callable(explainer) else explainer.shap_values(ex)
            shap_values = _coerce_shap_values(shap_values)
            return shap_values
        except Exception:
            pass

    # Robust fallback.
    n_features = explain_df.shape[1]
    min_required_evals = 2 * n_features + 1
    max_evals = max(min_required_evals, min(MAX_EVALS_CAP, 10 * n_features + 1))

    explainer = shap.Explainer(score_fn, background_df, algorithm="permutation")
    shap_exp = explainer(
        explain_df,
        max_evals=max_evals,
        batch_size=20,
    )

    shap_values = np.asarray(shap_exp.values)
    return _coerce_shap_values(shap_values)

def summarize_shap(shap_values: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    if shap_values.size == 0:
        return pd.DataFrame({
            "rank": [],
            "feature": [],
            "mean_positive_shap": [],
            "mean_negative_shap_abs": [],
            "mean_shap": [],
            "mean_abs_shap": [],
            "positive_push_rate": [],
            "negative_push_rate": [],
        })

    values = np.asarray(shap_values)
    mean_positive = np.maximum(values, 0).mean(axis=0)
    mean_negative_abs = np.maximum(-values, 0).mean(axis=0)
    mean_signed = values.mean(axis=0)
    mean_abs = np.abs(values).mean(axis=0)
    positive_rate = (values > 0).mean(axis=0)
    negative_rate = (values < 0).mean(axis=0)

    out = pd.DataFrame({
        "feature": feature_names,
        "mean_positive_shap": mean_positive,
        "mean_negative_shap_abs": mean_negative_abs,
        "mean_shap": mean_signed,
        "mean_abs_shap": mean_abs,
        "positive_push_rate": positive_rate,
        "negative_push_rate": negative_rate,
    }).sort_values(
        by=["mean_positive_shap", "mean_abs_shap", "feature"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    out.insert(0, "rank", range(1, len(out) + 1))
    return out


def safe_quantile(series: pd.Series, q: float) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return np.nan
    return float(s.quantile(q))


def safe_mean(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return np.nan
    return float(s.mean())


def safe_std(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) <= 1:
        return np.nan
    return float(s.std(ddof=1))


def safe_median(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return np.nan
    return float(s.median())


def standardized_mean_diff(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan

    ma, mb = a.mean(), b.mean()
    sa, sb = a.std(ddof=1), b.std(ddof=1)
    pooled = math.sqrt(((len(a) - 1) * sa**2 + (len(b) - 1) * sb**2) / (len(a) + len(b) - 2))
    if pooled == 0 or np.isnan(pooled):
        return np.nan
    return float((ma - mb) / pooled)


def determine_monitor_direction(tp_series: pd.Series, tn_series: pd.Series) -> str:
    tp = pd.to_numeric(tp_series, errors="coerce").dropna()
    tn = pd.to_numeric(tn_series, errors="coerce").dropna()

    if len(tp) < 5 or len(tn) < 5:
        return "unknown"

    tp_med = tp.median()
    tn_med = tn.median()
    tn_q01 = tn.quantile(0.01)
    tn_q99 = tn.quantile(0.99)
    tp_q10 = tp.quantile(0.10)
    tp_q90 = tp.quantile(0.90)

    if tp_med > tn_q99:
        return "high"
    if tp_med < tn_q01:
        return "low"
    if (tp_q10 < tn_q01) and (tp_q90 > tn_q99):
        return "both"
    if tp_med > tn_med:
        return "high"
    if tp_med < tn_med:
        return "low"
    return "flat"


def suggested_monitor_threshold(direction: str, tn_series: pd.Series) -> float:
    tn = pd.to_numeric(tn_series, errors="coerce").dropna()
    if len(tn) == 0:
        return np.nan

    if direction == "high":
        return float(tn.quantile(NORMAL_HIGH_QUANTILE))
    if direction == "low":
        return float(tn.quantile(NORMAL_LOW_QUANTILE))
    return np.nan


def calc_trigger_rate(series: pd.Series, direction: str, threshold: float) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0 or np.isnan(threshold):
        return np.nan

    if direction == "high":
        return float((s >= threshold).mean())
    if direction == "low":
        return float((s <= threshold).mean())
    return np.nan


def build_group_distribution_summary(X_df: pd.DataFrame, group_masks: dict[str, np.ndarray]) -> pd.DataFrame:
    records = []
    groups_to_use = [g for g in ["TP", "FP", "TN", "FN", "pred_anomaly", "pred_normal"] if g in group_masks]

    for feature in X_df.columns:
        for group_name in groups_to_use:
            mask = group_masks[group_name]
            s = pd.to_numeric(X_df.loc[mask, feature], errors="coerce")

            records.append({
                "feature": feature,
                "group": group_name,
                "n": int(mask.sum()),
                "non_null_n": int(s.notna().sum()),
                "missing_rate": float(s.isna().mean()) if len(s) > 0 else np.nan,
                "mean": safe_mean(s),
                "median": safe_median(s),
                "std": safe_std(s),
                "p01": safe_quantile(s, 0.01),
                "p05": safe_quantile(s, 0.05),
                "p25": safe_quantile(s, 0.25),
                "p50": safe_quantile(s, 0.50),
                "p75": safe_quantile(s, 0.75),
                "p95": safe_quantile(s, 0.95),
                "p99": safe_quantile(s, 0.99),
            })

    return pd.DataFrame(records)


def add_model_meta(df: pd.DataFrame, row: pd.Series, artifact_key: str, threshold: float, feature_set: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    out = df.copy()
    out.insert(0, "artifact_key", artifact_key)
    out.insert(1, "feature_set", feature_set)
    out.insert(2, "category", str(row["category"]))
    out.insert(3, "model", str(row["model"]))
    out.insert(4, "balance_method", str(row["balance_method"]))
    out["threshold"] = float(threshold)
    return out


def merge_tp_fp_shap(tp_df: pd.DataFrame, fp_df: pd.DataFrame, min_fp_rows: int = MIN_FP_EVIDENCE_ROWS) -> pd.DataFrame:
    tp_use = tp_df[[
        "feature",
        "rank",
        "mean_positive_shap",
        "mean_negative_shap_abs",
        "mean_shap",
        "mean_abs_shap",
        "positive_push_rate",
        "negative_push_rate",
        "n_group_rows_total",
        "n_explained_rows",
    ]].rename(columns={
        "rank": "tp_rank",
        "mean_positive_shap": "tp_mean_positive_shap",
        "mean_negative_shap_abs": "tp_mean_negative_shap_abs",
        "mean_shap": "tp_mean_shap",
        "mean_abs_shap": "tp_mean_abs_shap",
        "positive_push_rate": "tp_positive_push_rate",
        "negative_push_rate": "tp_negative_push_rate",
        "n_group_rows_total": "tp_group_rows_total",
        "n_explained_rows": "tp_explained_rows",
    })

    fp_use = fp_df[[
        "feature",
        "rank",
        "mean_positive_shap",
        "mean_negative_shap_abs",
        "mean_shap",
        "mean_abs_shap",
        "positive_push_rate",
        "negative_push_rate",
        "n_group_rows_total",
        "n_explained_rows",
    ]].rename(columns={
        "rank": "fp_rank",
        "mean_positive_shap": "fp_mean_positive_shap",
        "mean_negative_shap_abs": "fp_mean_negative_shap_abs",
        "mean_shap": "fp_mean_shap",
        "mean_abs_shap": "fp_mean_abs_shap",
        "positive_push_rate": "fp_positive_push_rate",
        "negative_push_rate": "fp_negative_push_rate",
        "n_group_rows_total": "fp_group_rows_total",
        "n_explained_rows": "fp_explained_rows",
    })

    out = tp_use.merge(fp_use, on="feature", how="outer")

    out["tp_group_rows_total"] = pd.to_numeric(out.get("tp_group_rows_total"), errors="coerce")
    out["fp_group_rows_total"] = pd.to_numeric(out.get("fp_group_rows_total"), errors="coerce")
    out["fp_evidence_sufficient"] = (out["fp_group_rows_total"].fillna(0) >= min_fp_rows).astype(int)

    out["tp_fp_shap_gap"] = np.where(
        out["fp_evidence_sufficient"] == 1,
        pd.to_numeric(out["tp_mean_positive_shap"], errors="coerce") - pd.to_numeric(out["fp_mean_positive_shap"], errors="coerce"),
        np.nan,
    )
    out["tp_fp_rank_gap"] = np.where(
        out["fp_evidence_sufficient"] == 1,
        pd.to_numeric(out["fp_rank"], errors="coerce") - pd.to_numeric(out["tp_rank"], errors="coerce"),
        np.nan,
    )

    def classify(row):
        tp = pd.to_numeric(pd.Series([row.get("tp_mean_positive_shap")]), errors="coerce").iloc[0]
        fp = pd.to_numeric(pd.Series([row.get("fp_mean_positive_shap")]), errors="coerce").iloc[0]
        gap = pd.to_numeric(pd.Series([row.get("tp_fp_shap_gap")]), errors="coerce").iloc[0]
        fp_ok = int(row.get("fp_evidence_sufficient", 0)) == 1

        if pd.notna(tp) and tp > 0 and not fp_ok:
            return "insufficient_fp_evidence"
        if pd.notna(tp) and pd.notna(gap) and tp > 0 and gap > 0 and tp >= 1.5 * max(float(fp), 1e-12):
            return "true_anomaly_candidate"
        if fp_ok and pd.notna(fp) and pd.notna(tp) and fp > 0 and fp >= 1.2 * max(float(tp), 1e-12):
            return "false_positive_risk"
        if pd.notna(tp) and pd.notna(fp) and tp > 0 and fp > 0:
            return "shared_signal"
        return "weak_signal"

    out["shap_signal_type"] = out.apply(classify, axis=1)
    out["tp_signal_available"] = (pd.to_numeric(out["tp_mean_positive_shap"], errors="coerce").fillna(0) > 0).astype(int)

    gap_sort = pd.to_numeric(out["tp_fp_shap_gap"], errors="coerce").fillna(-1e9)
    tp_sort = pd.to_numeric(out["tp_mean_positive_shap"], errors="coerce").fillna(0)
    out = out.assign(_gap_sort=gap_sort, _tp_sort=tp_sort).sort_values(
        by=["fp_evidence_sufficient", "_gap_sort", "_tp_sort", "feature"],
        ascending=[False, False, False, True],
    ).drop(columns=["_gap_sort", "_tp_sort"]).reset_index(drop=True)

    return out

def build_feature_stats_comparison(X_df: pd.DataFrame, group_masks: dict[str, np.ndarray]) -> pd.DataFrame:
    records = []
    has_tp = "TP" in group_masks
    has_fp = "FP" in group_masks
    has_tn = "TN" in group_masks
    has_fn = "FN" in group_masks

    for feature in X_df.columns:
        s = pd.to_numeric(X_df[feature], errors="coerce")

        tp = s[group_masks["TP"]] if has_tp else pd.Series(dtype=float)
        fp = s[group_masks["FP"]] if has_fp else pd.Series(dtype=float)
        tn = s[group_masks["TN"]] if has_tn else pd.Series(dtype=float)
        fn = s[group_masks["FN"]] if has_fn else pd.Series(dtype=float)

        direction = determine_monitor_direction(tp, tn) if (has_tp and has_tn) else "unknown"
        threshold = suggested_monitor_threshold(direction, tn) if has_tn else np.nan
        tp_trigger = calc_trigger_rate(tp, direction, threshold) if has_tp else np.nan
        fp_trigger = calc_trigger_rate(fp, direction, threshold) if has_fp else np.nan
        tn_trigger = calc_trigger_rate(tn, direction, threshold) if has_tn else np.nan

        records.append({
            "feature": feature,
            "tp_n": int(group_masks["TP"].sum()) if has_tp else 0,
            "fp_n": int(group_masks["FP"].sum()) if has_fp else 0,
            "tn_n": int(group_masks["TN"].sum()) if has_tn else 0,
            "fn_n": int(group_masks["FN"].sum()) if has_fn else 0,
            "tp_mean": safe_mean(tp),
            "fp_mean": safe_mean(fp),
            "tn_mean": safe_mean(tn),
            "fn_mean": safe_mean(fn),
            "tp_median": safe_median(tp),
            "fp_median": safe_median(fp),
            "tn_median": safe_median(tn),
            "fn_median": safe_median(fn),
            "tp_vs_tn_smd": standardized_mean_diff(tp, tn),
            "tp_vs_fp_smd": standardized_mean_diff(tp, fp),
            "fp_vs_tn_smd": standardized_mean_diff(fp, tn),
            "monitor_direction": direction,
            "suggested_threshold": threshold,
            "tp_trigger_rate": tp_trigger,
            "fp_trigger_rate": fp_trigger,
            "tn_trigger_rate": tn_trigger,
        })

    return pd.DataFrame(records)


def aggregate_feature_stability(per_model_feature_tables: list[pd.DataFrame], top_n: int = 10) -> pd.DataFrame:
    if not per_model_feature_tables:
        return pd.DataFrame()

    rows = []
    for df in per_model_feature_tables:
        if df.empty:
            continue

        use = df.sort_values("rank").head(top_n).copy()
        for _, r in use.iterrows():
            rows.append({
                "artifact_key": r["artifact_key"],
                "feature": r["feature"],
                "rank": int(r["rank"]),
                "mean_positive_shap": float(r["mean_positive_shap"]),
                "category": r["category"],
                "model": r["model"],
                "feature_set": r["feature_set"],
            })

    if not rows:
        return pd.DataFrame()

    raw = pd.DataFrame(rows)

    agg = raw.groupby("feature").agg(
        appeared_in_topn_models=("artifact_key", "nunique"),
        avg_rank=("rank", "mean"),
        best_rank=("rank", "min"),
        worst_rank=("rank", "max"),
        avg_mean_positive_shap=("mean_positive_shap", "mean"),
        max_mean_positive_shap=("mean_positive_shap", "max"),
    ).reset_index()

    cat_counts = raw.groupby("feature")["category"].agg(lambda s: ",".join(sorted(set(s)))).reset_index()
    fs_counts = raw.groupby("feature")["feature_set"].agg(lambda s: ",".join(sorted(set(s)))).reset_index()

    agg = agg.merge(cat_counts, on="feature", how="left")
    agg = agg.merge(fs_counts, on="feature", how="left")
    agg = agg.rename(columns={
        "category": "appeared_categories",
        "feature_set": "appeared_feature_sets",
    })

    agg = agg.sort_values(
        by=["appeared_in_topn_models", "avg_rank", "avg_mean_positive_shap", "feature"],
        ascending=[False, True, False, True],
    ).reset_index(drop=True)

    return agg


def build_redundancy_clusters(X_df: pd.DataFrame, candidate_features: list[str], corr_threshold: float = 0.90) -> pd.DataFrame:
    if not candidate_features:
        return pd.DataFrame(columns=["feature", "redundancy_cluster", "cluster_size"])

    use = [c for c in candidate_features if c in X_df.columns]
    if len(use) == 0:
        return pd.DataFrame(columns=["feature", "redundancy_cluster", "cluster_size"])

    corr = X_df[use].apply(pd.to_numeric, errors="coerce").corr().abs().fillna(0.0)

    visited = set()
    clusters = []
    cluster_id = 0

    for feat in use:
        if feat in visited:
            continue

        cluster_id += 1
        stack = [feat]
        members = set()

        while stack:
            cur = stack.pop()
            if cur in members:
                continue
            members.add(cur)
            visited.add(cur)

            neigh = corr.index[(corr.loc[cur] >= corr_threshold)].tolist()
            for nb in neigh:
                if nb not in members:
                    stack.append(nb)

        members = sorted(members)
        for m in members:
            clusters.append({
                "feature": m,
                "redundancy_cluster": f"C{cluster_id:03d}",
                "cluster_size": len(members),
            })

    return pd.DataFrame(clusters)


def normalize_score(s: pd.Series, higher_better: bool = True) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").copy()
    valid = x.dropna()
    if len(valid) == 0:
        return pd.Series(np.nan, index=x.index)

    min_v = valid.min()
    max_v = valid.max()
    if max_v == min_v:
        out = pd.Series(1.0, index=x.index)
    else:
        out = (x - min_v) / (max_v - min_v)

    if not higher_better:
        out = 1 - out
    return out


def build_candidate_feature_summary(stability_df: pd.DataFrame, model_level_tables: list[pd.DataFrame]) -> pd.DataFrame:
    if not model_level_tables:
        return pd.DataFrame()

    big = pd.concat(model_level_tables, axis=0, ignore_index=True)

    agg = big.groupby("feature").agg(
        n_models=("artifact_key", "nunique"),
        mean_tp_mean_positive_shap=("tp_mean_positive_shap", "mean"),
        max_tp_mean_positive_shap=("tp_mean_positive_shap", "max"),
        mean_fp_mean_positive_shap=("fp_mean_positive_shap", "mean"),
        mean_tp_fp_shap_gap=("tp_fp_shap_gap", "mean"),
        mean_tp_positive_push_rate=("tp_positive_push_rate", "mean"),
        mean_fp_positive_push_rate=("fp_positive_push_rate", "mean"),
        mean_tp_vs_tn_smd=("tp_vs_tn_smd", "mean"),
        mean_tp_vs_fp_smd=("tp_vs_fp_smd", "mean"),
        avg_tp_trigger_rate=("tp_trigger_rate", "mean"),
        avg_fp_trigger_rate=("fp_trigger_rate", "mean"),
        avg_tn_trigger_rate=("tn_trigger_rate", "mean"),
        fp_evidence_coverage=("fp_evidence_sufficient", "mean"),
        dominant_monitor_direction=("monitor_direction", lambda s: Counter(s.dropna()).most_common(1)[0][0] if len(s.dropna()) > 0 else "unknown"),
    ).reset_index()

    if stability_df is not None and not stability_df.empty:
        agg = agg.merge(stability_df, on="feature", how="left")

    agg["mean_tp_mean_positive_shap"] = pd.to_numeric(agg["mean_tp_mean_positive_shap"], errors="coerce")
    agg["mean_fp_mean_positive_shap"] = pd.to_numeric(agg["mean_fp_mean_positive_shap"], errors="coerce")
    agg["mean_tp_fp_shap_gap"] = pd.to_numeric(agg["mean_tp_fp_shap_gap"], errors="coerce")
    agg["tp_fp_ratio"] = agg["mean_tp_mean_positive_shap"] / np.maximum(agg["mean_fp_mean_positive_shap"].fillna(0), 1e-6)

    agg["score_tp_shap"] = normalize_score(agg["mean_tp_mean_positive_shap"], higher_better=True).fillna(0)
    agg["score_gap"] = normalize_score(agg["mean_tp_fp_shap_gap"], higher_better=True).fillna(0)
    agg["score_ratio"] = normalize_score(np.log1p(agg["tp_fp_ratio"].clip(lower=0)), higher_better=True).fillna(0)
    agg["score_fp_penalty"] = normalize_score(agg["mean_fp_mean_positive_shap"], higher_better=False).fillna(0)

    agg["final_candidate_score"] = (
        0.40 * agg["score_tp_shap"] +
        0.30 * agg["score_gap"] +
        0.20 * agg["score_ratio"] +
        0.10 * agg["score_fp_penalty"]
    )

    agg["tp_signal_available"] = (agg["mean_tp_mean_positive_shap"].fillna(0) > 0).astype(int)
    agg["gap_positive"] = (agg["mean_tp_fp_shap_gap"].fillna(-np.inf) > 0).astype(int)
    agg["ratio_pass"] = (agg["tp_fp_ratio"].fillna(0) >= PRIORITY_TP_FP_RATIO_MIN).astype(int)
    agg["tp_fp_priority_candidate"] = (
        (agg["tp_signal_available"] == 1) &
        (agg["gap_positive"] == 1) &
        (agg["ratio_pass"] == 1)
    ).astype(int)

    def recommend(row):
        tp = float(row.get("mean_tp_mean_positive_shap", 0) or 0)
        fp = float(row.get("mean_fp_mean_positive_shap", 0) or 0)

        if tp <= 0:
            return "drop"
        if fp > 0 and fp >= tp:
            return "false_positive_risk"
        if row.get("tp_fp_priority_candidate", 0) == 1:
            return "monitor_first"
        return "review"

    agg["recommendation"] = agg.apply(recommend, axis=1)

    agg = agg.sort_values(
        by=["tp_fp_priority_candidate", "final_candidate_score", "mean_tp_fp_shap_gap", "mean_tp_mean_positive_shap", "feature"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)

    return agg

def build_model_feature_recommendation_table(shap_cmp_df: pd.DataFrame, stats_cmp_df: pd.DataFrame) -> pd.DataFrame:
    out = shap_cmp_df.merge(stats_cmp_df, on="feature", how="left")

    out["model_feature_score"] = (
        normalize_score(out["tp_mean_positive_shap"], higher_better=True).fillna(0) * 0.40 +
        normalize_score(out["tp_fp_shap_gap"], higher_better=True).fillna(0) * 0.30 +
        normalize_score(out["tp_vs_tn_smd"].abs(), higher_better=True).fillna(0) * 0.20 +
        normalize_score(out["fp_mean_positive_shap"], higher_better=False).fillna(0) * 0.10
    )

    out["hard_gate_tp_signal"] = (
        (pd.to_numeric(out["tp_mean_positive_shap"], errors="coerce").fillna(0) > 0) &
        (pd.to_numeric(out["tp_positive_push_rate"], errors="coerce").fillna(0) >= MIN_DIRECTIONAL_PUSH_RATE)
    ).astype(int)
    out["hard_gate_separation"] = (pd.to_numeric(out["tp_vs_tn_smd"], errors="coerce").abs().fillna(0) >= MIN_ABS_SMD_FOR_MONITOR).astype(int)
    out["hard_gate_gap"] = (
        (pd.to_numeric(out["tp_fp_shap_gap"], errors="coerce").fillna(-np.inf) > 0) &
        (pd.to_numeric(out.get("fp_evidence_sufficient"), errors="coerce").fillna(0) >= 1)
    ).astype(int)
    out["hard_gate_pass"] = ((out["hard_gate_tp_signal"] == 1) & (out["hard_gate_separation"] == 1)).astype(int)

    def recommend(row):
        if row.get("hard_gate_pass", 0) != 1:
            return "drop"
        if row.get("shap_signal_type") == "false_positive_risk":
            return "false_positive_risk"
        if row.get("fp_evidence_sufficient", 0) != 1:
            return "review"
        if (
            row.get("shap_signal_type") == "true_anomaly_candidate" and
            row.get("hard_gate_gap", 0) == 1 and
            row.get("model_feature_score", 0) >= MIN_TEST_MONITOR_SCORE
        ):
            return "monitor_first"
        if row.get("model_feature_score", 0) >= 0.40:
            return "review"
        return "drop"

    out["recommendation"] = out.apply(recommend, axis=1)
    out = out.sort_values(
        by=["hard_gate_pass", "model_feature_score", "tp_fp_shap_gap", "tp_mean_positive_shap", "feature"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)

    return out

def ks_statistic_fallback(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan

    if scipy_ks_2samp is not None:
        return float(scipy_ks_2samp(a, b).statistic)

    data_all = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(np.sort(a), data_all, side="right") / len(a)
    cdf_b = np.searchsorted(np.sort(b), data_all, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def wasserstein_distance_fallback(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan

    if scipy_wasserstein_distance is not None:
        return float(scipy_wasserstein_distance(a, b))

    qs = np.linspace(0.0, 1.0, 201)
    aq = np.quantile(a, qs)
    bq = np.quantile(b, qs)
    return float(np.mean(np.abs(aq - bq)))


def build_missingness_indicator_df(X_df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=X_df.index)
    for c in X_df.columns:
        out[f"{c}__is_missing"] = X_df[c].isna().astype(int)
    return out


def build_missingness_stats_comparison(X_df: pd.DataFrame, group_masks: dict[str, np.ndarray]) -> pd.DataFrame:
    X_miss = build_missingness_indicator_df(X_df)
    records = []

    has_tp = "TP" in group_masks
    has_fp = "FP" in group_masks
    has_tn = "TN" in group_masks
    has_fn = "FN" in group_masks

    for col in X_miss.columns:
        base_feature = col.replace("__is_missing", "")
        s = pd.to_numeric(X_miss[col], errors="coerce")

        tp = s[group_masks["TP"]] if has_tp else pd.Series(dtype=float)
        fp = s[group_masks["FP"]] if has_fp else pd.Series(dtype=float)
        tn = s[group_masks["TN"]] if has_tn else pd.Series(dtype=float)
        fn = s[group_masks["FN"]] if has_fn else pd.Series(dtype=float)

        tp_missing_rate = float(tp.mean()) if len(tp) else np.nan
        fp_missing_rate = float(fp.mean()) if len(fp) else np.nan
        tn_missing_rate = float(tn.mean()) if len(tn) else np.nan
        fn_missing_rate = float(fn.mean()) if len(fn) else np.nan

        records.append({
            "feature": base_feature,
            "tp_missing_rate": tp_missing_rate,
            "fp_missing_rate": fp_missing_rate,
            "tn_missing_rate": tn_missing_rate,
            "fn_missing_rate": fn_missing_rate,
            "missing_tp_tn_gap": tp_missing_rate - tn_missing_rate if len(tp) and len(tn) else np.nan,
            "missing_tp_fp_gap": tp_missing_rate - fp_missing_rate if len(tp) and len(fp) else np.nan,
            "overall_missing_rate": float(s.mean()) if len(s) else np.nan,
        })

    out = pd.DataFrame(records)
    if out.empty:
        return out

    def classify(row):
        if pd.notna(row["missing_tp_tn_gap"]) and pd.notna(row["missing_tp_fp_gap"]):
            if row["missing_tp_tn_gap"] > 0 and row["missing_tp_fp_gap"] > 0:
                return "tp_associated_missingness"
            if row["fp_missing_rate"] > row["tp_missing_rate"]:
                return "fp_risk_missingness"
        return "weak"

    out["missing_signal_type"] = out.apply(classify, axis=1)
    out = out.sort_values(
        by=["missing_tp_tn_gap", "missing_tp_fp_gap", "feature"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return out


def build_pair_rules(stats_cmp_df: pd.DataFrame, top_features: list[str]) -> pd.DataFrame:
    if stats_cmp_df is None or stats_cmp_df.empty or not top_features:
        return pd.DataFrame()

    use = stats_cmp_df[stats_cmp_df["feature"].isin(top_features)].copy()
    use = use[use["monitor_direction"].isin(["high", "low"])].copy()
    use = use[pd.notna(use["suggested_threshold"])].copy()

    rows = []
    feats = sorted(use["feature"].unique().tolist())
    for a, b in combinations(feats, 2):
        ra = use.loc[use["feature"] == a].iloc[0]
        rb = use.loc[use["feature"] == b].iloc[0]

        rows.append({
            "feature_a": a,
            "feature_b": b,
            "dir_a": ra["monitor_direction"],
            "dir_b": rb["monitor_direction"],
            "thr_a": float(ra["suggested_threshold"]),
            "thr_b": float(rb["suggested_threshold"]),
        })

    return pd.DataFrame(rows)

def evaluate_pair_rule_precision_lift(X_df: pd.DataFrame, y_true: pd.Series, pair_rules_df: pd.DataFrame) -> pd.DataFrame:
    if y_true is None or pair_rules_df is None or pair_rules_df.empty:
        return pd.DataFrame()

    y = np.asarray(y_true).astype(int).ravel()
    base_rate = float((y == 1).mean()) if len(y) else np.nan
    pos_n = int((y == 1).sum())
    neg_n = int((y == 0).sum())

    rows = []
    for _, r in pair_rules_df.iterrows():
        a, b = r["feature_a"], r["feature_b"]
        da, db = r["dir_a"], r["dir_b"]
        ta, tb = r["thr_a"], r["thr_b"]

        sa = pd.to_numeric(X_df[a], errors="coerce")
        sb = pd.to_numeric(X_df[b], errors="coerce")

        cond_a = (sa >= ta) if da == "high" else (sa <= ta)
        cond_b = (sb >= tb) if db == "high" else (sb <= tb)

        pair_hit = cond_a & cond_b & sa.notna() & sb.notna()
        support_n = int(pair_hit.sum())
        if support_n == 0:
            continue

        hit_arr = pair_hit.to_numpy()
        y_hit = y[hit_arr]
        tp = int((y_hit == 1).sum())
        fp = int((y_hit == 0).sum())

        precision = tp / support_n if support_n > 0 else np.nan
        recall = tp / pos_n if pos_n > 0 else np.nan
        fp_rate = fp / neg_n if neg_n > 0 else np.nan
        lift = precision / base_rate if base_rate and base_rate > 0 else np.nan

        rows.append({
            "feature_a": a,
            "feature_b": b,
            "pair_key": " || ".join(sorted([a, b])),
            "rule_desc": f"{a}({da}) AND {b}({db})",
            "support_n": support_n,
            "support_rate": support_n / len(y) if len(y) > 0 else np.nan,
            "pair_precision": precision,
            "pair_recall": recall,
            "pair_fp_rate": fp_rate,
            "pair_precision_lift": lift,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["recommendation"] = np.where(
        (out["pair_precision_lift"] >= PAIR_MIN_LIFT) & (out["support_n"] >= PAIR_MIN_SUPPORT),
        "interaction_candidate",
        "review",
    )
    out = out.sort_values(
        by=["pair_precision_lift", "pair_precision", "support_n", "pair_key"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    return out


def build_class_distribution_separation(X_df: pd.DataFrame, y_true: pd.Series | None) -> pd.DataFrame:
    if y_true is None:
        return pd.DataFrame()

    y = np.asarray(y_true).astype(int).ravel()
    rows = []

    for feature in X_df.columns:
        s = pd.to_numeric(X_df[feature], errors="coerce")
        pos = s[y == POS_LABEL].dropna()
        neg = s[y == NEG_LABEL].dropna()

        if len(pos) == 0 or len(neg) == 0:
            continue

        ks = ks_statistic_fallback(pos.to_numpy(), neg.to_numpy())
        wd = wasserstein_distance_fallback(pos.to_numpy(), neg.to_numpy())

        rows.append({
            "feature": feature,
            "pos_n": int(len(pos)),
            "neg_n": int(len(neg)),
            "pos_mean": float(pos.mean()),
            "neg_mean": float(neg.mean()),
            "pos_median": float(pos.median()),
            "neg_median": float(neg.median()),
            "pos_p95": float(pos.quantile(0.95)),
            "neg_p95": float(neg.quantile(0.95)),
            "mean_gap": float(pos.mean() - neg.mean()),
            "median_gap": float(pos.median() - neg.median()),
            "p95_gap": float(pos.quantile(0.95) - neg.quantile(0.95)),
            "ks_stat": ks,
            "wasserstein_distance": wd,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["distribution_sep_score"] = (
        normalize_score(out["ks_stat"], higher_better=True).fillna(0) * 0.60 +
        normalize_score(out["wasserstein_distance"], higher_better=True).fillna(0) * 0.40
    )
    out = out.sort_values(
        by=["distribution_sep_score", "ks_stat", "feature"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return out


def aggregate_missingness_across_models(tables: list[pd.DataFrame]) -> pd.DataFrame:
    if not tables:
        return pd.DataFrame()

    big = pd.concat([df for df in tables if df is not None and not df.empty], axis=0, ignore_index=True)
    if big.empty:
        return big

    agg = big.groupby("feature").agg(
        n_models=("artifact_key", "nunique"),
        mean_tp_missing_rate=("tp_missing_rate", "mean"),
        mean_fp_missing_rate=("fp_missing_rate", "mean"),
        mean_tn_missing_rate=("tn_missing_rate", "mean"),
        mean_missing_tp_tn_gap=("missing_tp_tn_gap", "mean"),
        mean_missing_tp_fp_gap=("missing_tp_fp_gap", "mean"),
        overall_missing_rate=("overall_missing_rate", "mean"),
        dominant_missing_signal_type=("missing_signal_type", lambda s: Counter(pd.Series(s).dropna()).most_common(1)[0][0] if len(pd.Series(s).dropna()) > 0 else "weak"),
    ).reset_index()

    agg["has_missing_signal"] = (
        (agg["mean_missing_tp_tn_gap"] > 0) &
        (agg["dominant_missing_signal_type"] == "tp_associated_missingness")
    ).astype(int)

    agg = agg.sort_values(
        by=["has_missing_signal", "mean_missing_tp_tn_gap", "feature"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return agg


def aggregate_class_distribution_across_models(tables: list[pd.DataFrame]) -> pd.DataFrame:
    if not tables:
        return pd.DataFrame()

    big = pd.concat([df for df in tables if df is not None and not df.empty], axis=0, ignore_index=True)
    if big.empty:
        return big

    agg = big.groupby("feature").agg(
        n_models=("artifact_key", "nunique"),
        mean_pos_mean=("pos_mean", "mean"),
        mean_neg_mean=("neg_mean", "mean"),
        mean_pos_median=("pos_median", "mean"),
        mean_neg_median=("neg_median", "mean"),
        mean_mean_gap=("mean_gap", "mean"),
        mean_median_gap=("median_gap", "mean"),
        mean_p95_gap=("p95_gap", "mean"),
        mean_ks_stat=("ks_stat", "mean"),
        max_ks_stat=("ks_stat", "max"),
        mean_wasserstein_distance=("wasserstein_distance", "mean"),
        mean_distribution_sep_score=("distribution_sep_score", "mean"),
    ).reset_index()

    agg = agg.sort_values(
        by=["mean_distribution_sep_score", "mean_ks_stat", "feature"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return agg


def aggregate_pair_rules_across_models(tables: list[pd.DataFrame]) -> pd.DataFrame:
    if not tables:
        return pd.DataFrame()

    big = pd.concat([df for df in tables if df is not None and not df.empty], axis=0, ignore_index=True)
    if big.empty:
        return big

    agg = big.groupby("pair_key").agg(
        n_models=("artifact_key", "nunique"),
        feature_a=("feature_a", "first"),
        feature_b=("feature_b", "first"),
        mean_support_n=("support_n", "mean"),
        max_support_n=("support_n", "max"),
        mean_pair_precision=("pair_precision", "mean"),
        max_pair_precision=("pair_precision", "max"),
        mean_pair_recall=("pair_recall", "mean"),
        mean_pair_fp_rate=("pair_fp_rate", "mean"),
        mean_pair_precision_lift=("pair_precision_lift", "mean"),
        max_pair_precision_lift=("pair_precision_lift", "max"),
        dominant_recommendation=("recommendation", lambda s: Counter(pd.Series(s).dropna()).most_common(1)[0][0] if len(pd.Series(s).dropna()) > 0 else "review"),
        representative_rule_desc=("rule_desc", "first"),
    ).reset_index()

    agg["is_interaction_candidate"] = (
        (agg["mean_pair_precision_lift"] >= PAIR_MIN_LIFT) &
        (agg["mean_support_n"] >= PAIR_MIN_SUPPORT)
    ).astype(int)

    agg = agg.sort_values(
        by=["is_interaction_candidate", "mean_pair_precision_lift", "mean_pair_precision", "pair_key"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    return agg


def build_priority_feature_corr_companions(
    X_df: pd.DataFrame,
    anchor_features: list[str],
    abs_corr_threshold: float = 0.95,
) -> pd.DataFrame:
    if X_df is None or X_df.empty or not anchor_features:
        return pd.DataFrame()

    use = X_df.apply(pd.to_numeric, errors="coerce")
    corr = use.corr().fillna(0.0)

    rows = []
    for anchor in anchor_features:
        if anchor not in corr.columns:
            continue

        s = corr[anchor].drop(labels=[anchor], errors="ignore")
        s = s[s.abs() >= abs_corr_threshold].sort_values(key=lambda x: x.abs(), ascending=False)

        for feat, val in s.items():
            rows.append({
                "anchor_feature": anchor,
                "companion_feature": feat,
                "corr": float(val),
                "abs_corr": float(abs(val)),
                "corr_sign": "positive" if val >= 0 else "negative",
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.sort_values(
        by=["anchor_feature", "abs_corr", "companion_feature"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    return out


def build_baseline_normal_stats(X_df: pd.DataFrame, y_true: pd.Series | None) -> pd.DataFrame:
    if y_true is None:
        return pd.DataFrame()

    y = np.asarray(y_true).astype(int).ravel()
    rows = []

    for feature in X_df.columns:
        s = pd.to_numeric(X_df[feature], errors="coerce")
        normal = s[y == NEG_LABEL]
        anomaly = s[y == POS_LABEL]

        rows.append({
            "feature": feature,
            "normal_n": int(normal.notna().sum()),
            "anomaly_n": int(anomaly.notna().sum()),
            "normal_missing_rate": float(normal.isna().mean()) if len(normal) else np.nan,
            "normal_mean": safe_mean(normal),
            "normal_std": safe_std(normal),
            "normal_median": safe_median(normal),
            "normal_p01": safe_quantile(normal, 0.01),
            "normal_p05": safe_quantile(normal, 0.05),
            "normal_p25": safe_quantile(normal, 0.25),
            "normal_p50": safe_quantile(normal, 0.50),
            "normal_p75": safe_quantile(normal, 0.75),
            "normal_p95": safe_quantile(normal, 0.95),
            "normal_p99": safe_quantile(normal, 0.99),
            "anomaly_mean": safe_mean(anomaly),
            "anomaly_median": safe_median(anomaly),
        })

    return pd.DataFrame(rows)


def build_primary_anchor_summary(
    candidate_summary_df: pd.DataFrame,
    all_stats_cmp_df: pd.DataFrame,
    class_sep_agg_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if candidate_summary_df is None or candidate_summary_df.empty:
        return pd.DataFrame()

    out = candidate_summary_df.copy()

    stats_source = all_stats_cmp_df.copy() if all_stats_cmp_df is not None and not all_stats_cmp_df.empty else pd.DataFrame(columns=["feature"])
    if not stats_source.empty:
        stats_agg = stats_source.groupby("feature").agg(
            dominant_monitor_direction_all=("monitor_direction", lambda s: Counter(pd.Series(s).dropna()).most_common(1)[0][0] if len(pd.Series(s).dropna()) > 0 else "unknown"),
            mean_suggested_threshold=("suggested_threshold", "mean"),
            threshold_defined_rate=("suggested_threshold", lambda s: float(pd.Series(s).notna().mean()) if len(pd.Series(s)) else np.nan),
            mean_tp_trigger_rate_all=("tp_trigger_rate", "mean"),
            mean_fp_trigger_rate_all=("fp_trigger_rate", "mean"),
            mean_tn_trigger_rate_all=("tn_trigger_rate", "mean"),
        ).reset_index()
        out = out.merge(stats_agg, on="feature", how="left")
    else:
        out["dominant_monitor_direction_all"] = out.get("dominant_monitor_direction", "unknown")
        out["mean_suggested_threshold"] = np.nan
        out["threshold_defined_rate"] = np.nan
        out["mean_tp_trigger_rate_all"] = np.nan
        out["mean_fp_trigger_rate_all"] = np.nan
        out["mean_tn_trigger_rate_all"] = np.nan

    if class_sep_agg_df is not None and not class_sep_agg_df.empty:
        class_sep_use = class_sep_agg_df[[
            "feature",
            "mean_ks_stat",
            "mean_wasserstein_distance",
            "mean_distribution_sep_score",
            "mean_median_gap",
            "mean_p95_gap",
        ]].copy()
        out = out.merge(class_sep_use, on="feature", how="left")
    else:
        out["mean_ks_stat"] = np.nan
        out["mean_wasserstein_distance"] = np.nan
        out["mean_distribution_sep_score"] = np.nan
        out["mean_median_gap"] = np.nan
        out["mean_p95_gap"] = np.nan

    out["score_model_signal"] = (
        0.40 * normalize_score(out["mean_tp_mean_positive_shap"], True).fillna(0) +
        0.30 * normalize_score(out["mean_tp_fp_shap_gap"], True).fillna(0) +
        0.20 * normalize_score(np.log1p(out["tp_fp_ratio"].clip(lower=0)), True).fillna(0) +
        0.10 * normalize_score(out["mean_fp_mean_positive_shap"], False).fillna(0)
    )
    out["score_distribution_sep"] = normalize_score(out["mean_distribution_sep_score"], True).fillna(0)
    out["score_stability_anchor"] = normalize_score(out["appeared_in_topn_models"], True).fillna(0)
    out["score_fp_penalty_anchor"] = normalize_score(out["mean_fp_mean_positive_shap"], False).fillna(0)

    # 簡化後：priority_control 的核心只看 TP/FP。
    out["overall_anchor_score"] = out["score_model_signal"]

    def anchor_recommend(row):
        if row.get("tp_fp_priority_candidate", 0) == 1:
            return "primary_anchor"
        if row.get("tp_signal_available", 0) == 1:
            return "review_anchor"
        return "drop"

    out["anchor_recommendation"] = out.apply(anchor_recommend, axis=1)
    out = out.sort_values(
        by=["tp_fp_priority_candidate", "overall_anchor_score", "mean_tp_fp_shap_gap", "mean_tp_mean_positive_shap", "feature"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)
    out.insert(0, "anchor_rank", range(1, len(out) + 1))
    return out

def build_spc_monitoring_prep_pack(
    primary_anchor_df: pd.DataFrame,
    baseline_stats_df: pd.DataFrame | None,
    missingness_agg_df: pd.DataFrame | None,
    pair_agg_df: pd.DataFrame | None,
    corr_companion_df: pd.DataFrame | None,
    redundancy_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if primary_anchor_df is None or primary_anchor_df.empty:
        return pd.DataFrame()

    out = primary_anchor_df.copy()

    if baseline_stats_df is not None and not baseline_stats_df.empty:
        out = out.merge(baseline_stats_df, on="feature", how="left")

    if missingness_agg_df is not None and not missingness_agg_df.empty:
        miss_cols = [
            "feature",
            "mean_tp_missing_rate",
            "mean_fp_missing_rate",
            "mean_tn_missing_rate",
            "mean_missing_tp_tn_gap",
            "mean_missing_tp_fp_gap",
            "overall_missing_rate",
            "dominant_missing_signal_type",
            "has_missing_signal",
        ]
        out = out.merge(missingness_agg_df[miss_cols], on="feature", how="left")
    else:
        out["has_missing_signal"] = 0
        out["dominant_missing_signal_type"] = np.nan

    if corr_companion_df is not None and not corr_companion_df.empty:
        corr_cnt = corr_companion_df.groupby("anchor_feature").size().reset_index(name="n_corr_companions_095")
        corr_cnt = corr_cnt.rename(columns={"anchor_feature": "feature"})
        best_corr = corr_companion_df.sort_values(["anchor_feature", "abs_corr"], ascending=[True, False]).drop_duplicates("anchor_feature")
        best_corr = best_corr.rename(columns={
            "anchor_feature": "feature",
            "companion_feature": "top_corr_companion",
            "abs_corr": "top_corr_companion_abs_corr",
        })
        out = out.merge(corr_cnt, on="feature", how="left")
        out = out.merge(best_corr[["feature", "top_corr_companion", "top_corr_companion_abs_corr"]], on="feature", how="left")
    else:
        out["n_corr_companions_095"] = 0
        out["top_corr_companion"] = np.nan
        out["top_corr_companion_abs_corr"] = np.nan

    if pair_agg_df is not None and not pair_agg_df.empty:
        a_side = pair_agg_df.rename(columns={"feature_a": "feature", "feature_b": "pair_partner"})
        b_side = pair_agg_df.rename(columns={"feature_b": "feature", "feature_a": "pair_partner"})
        pair_long = pd.concat([a_side, b_side], axis=0, ignore_index=True)

        pair_flag = pair_long.groupby("feature").agg(
            has_interaction_support=("is_interaction_candidate", "max"),
            best_pair_precision_lift=("mean_pair_precision_lift", "max"),
        ).reset_index()

        pair_desc = pair_long.sort_values(
            by=["feature", "mean_pair_precision_lift", "mean_pair_precision"],
            ascending=[True, False, False],
        ).drop_duplicates("feature")
        pair_desc = pair_desc[["feature", "representative_rule_desc"]].rename(columns={"representative_rule_desc": "top_pair_rule_desc"})

        out = out.merge(pair_flag, on="feature", how="left")
        out = out.merge(pair_desc, on="feature", how="left")
    else:
        out["has_interaction_support"] = 0
        out["best_pair_precision_lift"] = np.nan
        out["top_pair_rule_desc"] = np.nan

    if redundancy_df is not None and not redundancy_df.empty:
        out = out.merge(redundancy_df, on="feature", how="left")

    out["has_missing_signal"] = out.get("has_missing_signal", 0)
    out["has_missing_signal"] = pd.Series(out["has_missing_signal"]).fillna(0).astype(int)
    out["has_interaction_support"] = out.get("has_interaction_support", 0)
    out["has_interaction_support"] = pd.Series(out["has_interaction_support"]).fillna(0).astype(int)
    out["n_corr_companions_095"] = out.get("n_corr_companions_095", 0)
    out["n_corr_companions_095"] = pd.Series(out["n_corr_companions_095"]).fillna(0).astype(int)
    out["monitor_tier"] = np.where(out["anchor_recommendation"] == "primary_anchor", "primary_anchor", "review_anchor")
    out["source_type"] = "raw_feature"

    keep_cols_first = [
        "anchor_rank",
        "feature",
        "monitor_tier",
        "source_type",
        "overall_anchor_score",
        "anchor_recommendation",
        "recommendation",
        "dominant_monitor_direction_all",
        "mean_suggested_threshold",
        "mean_tp_mean_positive_shap",
        "mean_fp_mean_positive_shap",
        "mean_tp_fp_shap_gap",
        "appeared_in_topn_models",
        "avg_rank",
        "mean_tp_vs_tn_smd",
        "mean_distribution_sep_score",
        "mean_ks_stat",
        "mean_wasserstein_distance",
        "mean_tp_trigger_rate_all",
        "mean_fp_trigger_rate_all",
        "mean_tn_trigger_rate_all",
        "has_missing_signal",
        "dominant_missing_signal_type",
        "n_corr_companions_095",
        "top_corr_companion",
        "top_corr_companion_abs_corr",
        "has_interaction_support",
        "best_pair_precision_lift",
        "top_pair_rule_desc",
        "redundancy_cluster",
        "cluster_size",
        "normal_mean",
        "normal_std",
        "normal_median",
        "normal_p01",
        "normal_p05",
        "normal_p25",
        "normal_p50",
        "normal_p75",
        "normal_p95",
        "normal_p99",
        "normal_missing_rate",
    ]

    default_fill_values = {
        "redundancy_cluster": np.nan,
        "cluster_size": np.nan,
        "dominant_missing_signal_type": np.nan,
        "top_corr_companion": np.nan,
        "top_corr_companion_abs_corr": np.nan,
        "best_pair_precision_lift": np.nan,
        "top_pair_rule_desc": np.nan,
        "normal_mean": np.nan,
        "normal_std": np.nan,
        "normal_median": np.nan,
        "normal_p01": np.nan,
        "normal_p05": np.nan,
        "normal_p25": np.nan,
        "normal_p50": np.nan,
        "normal_p75": np.nan,
        "normal_p95": np.nan,
        "normal_p99": np.nan,
        "normal_missing_rate": np.nan,
    }
    for c in keep_cols_first:
        if c not in out.columns:
            out[c] = default_fill_values.get(c, np.nan)

    rest_cols = [c for c in out.columns if c not in keep_cols_first]
    return out[keep_cols_first + rest_cols]


# =========================================================
# Main
# =========================================================


# =========================================================
# Additional config for pooled/train-test validation modules
# =========================================================
TEST_CSV_MAP = {
    "Base": "X_test_base.csv",
    "RFERF": "X_test_rferf.csv",
    "RFEXGB": "X_test_rfexgb.csv",
    "ANOVA": "X_test_anova.csv",
}

TEST_LABEL_CSV_CANDIDATES = {
    "Base": ["y_test_base.csv", "y_test.csv"],
    "RFERF": ["y_test_rferf.csv", "y_test.csv"],
    "RFEXGB": ["y_test_rfexgb.csv", "y_test.csv"],
    "ANOVA": ["y_test_anova.csv", "y_test.csv"],
}

RAW_TRAIN_CSV_PATH = "X_train.csv"
RAW_TEST_CSV_PATH = "X_test.csv"

POOLED_BACKGROUND_SIZE = 80
SPLIT_VALIDATION_TOP_FEATURES = 20
SPLIT_VALIDATION_MIN_SCORE = 0.50
SPLIT_VALIDATION_MIN_PASS_RATE = 0.50
PRIORITY_TP_FP_RATIO_MIN = 1.50
PRIORITY_TOP_N = 8


def load_feature_test_df(feature_set: str) -> tuple[pd.DataFrame | None, str | None]:
    if feature_set not in TEST_CSV_MAP:
        return None, None

    csv_path = TEST_CSV_MAP[feature_set]
    if not csv_path or not os.path.exists(csv_path):
        return None, None

    df = read_csv_flex(csv_path)
    df = df.loc[:, ~df.columns.astype(str).str.contains(r"^Unnamed")]
    return df, csv_path


def load_test_label_series(feature_set: str, expected_len: int) -> tuple[pd.Series | None, str | None]:
    candidates = TEST_LABEL_CSV_CANDIDATES.get(feature_set, ["y_test.csv"])

    label_path = None
    for p in candidates:
        if p and os.path.exists(p):
            label_path = p
            break

    if label_path is None:
        return None, None

    y_df = read_csv_flex(label_path)
    y_df = y_df.loc[:, ~y_df.columns.astype(str).str.contains(r"^Unnamed")]

    if y_df.shape[1] == 1:
        y = y_df.iloc[:, 0]
    else:
        col = None
        for c in LABEL_COLUMN_CANDIDATES:
            if c in y_df.columns:
                col = c
                break
        if col is None:
            col = y_df.columns[0]
        y = y_df[col]

    y = pd.Series(y).reset_index(drop=True)

    if len(y) != expected_len:
        raise ValueError(
            f"Test label length mismatch for {feature_set}: len(y)={len(y)} != expected_len={expected_len}"
        )

    uniq = set(pd.Series(y).dropna().unique().tolist())
    if uniq <= {0, 1}:
        y = y.astype(int)
    elif uniq <= {False, True}:
        y = y.astype(int)
    else:
        mapping = {
            "normal": 0,
            "anomaly": 1,
            "abnormal": 1,
            "yes": 1,
            "no": 0,
            "true": 1,
            "false": 0,
            "positive": 1,
            "negative": 0,
            "pass": 0,
            "fail": 1,
            "-1": 0,
            "1": 1,
        }
        y_str = y.astype(str).str.strip().str.lower()
        if y_str.isin(mapping.keys()).all():
            y = y_str.map(mapping).astype(int)
        else:
            raise ValueError(f"Unsupported test label values for {feature_set}: {sorted(list(uniq))[:10]}")

    return y, label_path


def load_raw_split_df(path: str) -> tuple[pd.DataFrame | None, str | None]:
    if not path or not os.path.exists(path):
        return None, None

    df = read_csv_flex(path)
    df = df.loc[:, ~df.columns.astype(str).str.contains(r"^Unnamed")]
    return df, path


def subset_columns_if_present(df: pd.DataFrame | None, columns: list[str]) -> pd.DataFrame | None:
    if df is None:
        return None

    missing = [c for c in columns if c not in df.columns]
    if missing:
        return None

    return df.loc[:, columns].copy()


def get_background_df_custom(X_df: pd.DataFrame, n_background: int) -> pd.DataFrame:
    if X_df is None or X_df.empty:
        return pd.DataFrame()
    if len(X_df) <= n_background:
        return X_df.copy()

    return X_df.sample(
        n=n_background,
        random_state=SHAP_RANDOM_STATE,
        replace=False,
    ).copy()


def strip_model_meta_cols(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    drop_cols = ["artifact_key", "feature_set", "category", "model", "balance_method", "threshold"]
    return df.drop(columns=drop_cols, errors="ignore").copy()


def safe_ratio_retention(train_value, test_value) -> float:
    if pd.isna(train_value) or pd.isna(test_value):
        return np.nan
    train_abs = abs(float(train_value))
    test_abs = abs(float(test_value))
    if train_abs < 1e-12 and test_abs < 1e-12:
        return 1.0
    if train_abs < 1e-12:
        return 0.0
    return float(min(test_abs / train_abs, 1.0))


def build_feature_split_stability_check(
    feature_list: list[str],
    train_stats_df: pd.DataFrame,
    test_stats_df: pd.DataFrame,
    train_class_sep_df: pd.DataFrame | None = None,
    test_class_sep_df: pd.DataFrame | None = None,
    pooled_reco_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if not feature_list:
        return pd.DataFrame()

    out = pd.DataFrame({"feature": list(dict.fromkeys(feature_list))})

    train_stats = strip_model_meta_cols(train_stats_df)
    test_stats = strip_model_meta_cols(test_stats_df)
    out = out.merge(
        train_stats[[
            "feature", "monitor_direction", "suggested_threshold",
            "tp_vs_tn_smd", "tp_trigger_rate", "fp_trigger_rate", "tn_trigger_rate"
        ]].rename(columns={
            "monitor_direction": "train_monitor_direction",
            "suggested_threshold": "train_suggested_threshold",
            "tp_vs_tn_smd": "train_tp_vs_tn_smd",
            "tp_trigger_rate": "train_tp_trigger_rate",
            "fp_trigger_rate": "train_fp_trigger_rate",
            "tn_trigger_rate": "train_tn_trigger_rate",
        }),
        on="feature",
        how="left",
    )
    out = out.merge(
        test_stats[[
            "feature", "monitor_direction", "suggested_threshold",
            "tp_vs_tn_smd", "tp_trigger_rate", "fp_trigger_rate", "tn_trigger_rate"
        ]].rename(columns={
            "monitor_direction": "test_monitor_direction",
            "suggested_threshold": "test_suggested_threshold",
            "tp_vs_tn_smd": "test_tp_vs_tn_smd",
            "tp_trigger_rate": "test_tp_trigger_rate",
            "fp_trigger_rate": "test_fp_trigger_rate",
            "tn_trigger_rate": "test_tn_trigger_rate",
        }),
        on="feature",
        how="left",
    )

    if train_class_sep_df is not None and not train_class_sep_df.empty:
        train_sep = strip_model_meta_cols(train_class_sep_df)
        out = out.merge(
            train_sep[["feature", "ks_stat", "wasserstein_distance", "distribution_sep_score"]].rename(columns={
                "ks_stat": "train_ks_stat",
                "wasserstein_distance": "train_wasserstein_distance",
                "distribution_sep_score": "train_distribution_sep_score",
            }),
            on="feature",
            how="left",
        )

    if test_class_sep_df is not None and not test_class_sep_df.empty:
        test_sep = strip_model_meta_cols(test_class_sep_df)
        out = out.merge(
            test_sep[["feature", "ks_stat", "wasserstein_distance", "distribution_sep_score"]].rename(columns={
                "ks_stat": "test_ks_stat",
                "wasserstein_distance": "test_wasserstein_distance",
                "distribution_sep_score": "test_distribution_sep_score",
            }),
            on="feature",
            how="left",
        )

    if pooled_reco_df is not None and not pooled_reco_df.empty:
        pooled_use = strip_model_meta_cols(pooled_reco_df)
        keep = [c for c in [
            "feature", "model_feature_score", "recommendation", "tp_mean_positive_shap",
            "fp_mean_positive_shap", "tp_fp_shap_gap"
        ] if c in pooled_use.columns]
        if keep:
            out = out.merge(
                pooled_use[keep].rename(columns={
                    "model_feature_score": "pooled_model_feature_score",
                    "recommendation": "pooled_recommendation",
                    "tp_mean_positive_shap": "pooled_tp_mean_positive_shap",
                    "fp_mean_positive_shap": "pooled_fp_mean_positive_shap",
                    "tp_fp_shap_gap": "pooled_tp_fp_shap_gap",
                }),
                on="feature",
                how="left",
            )

    out["direction_consistent"] = (
        out["train_monitor_direction"].isin(["high", "low"]) &
        (out["train_monitor_direction"] == out["test_monitor_direction"])
    ).astype(int)

    out["smd_retention_score"] = out.apply(
        lambda r: safe_ratio_retention(r.get("train_tp_vs_tn_smd"), r.get("test_tp_vs_tn_smd")),
        axis=1,
    )
    out["distribution_retention_score"] = out.apply(
        lambda r: safe_ratio_retention(r.get("train_distribution_sep_score"), r.get("test_distribution_sep_score")),
        axis=1,
    )
    out["tp_trigger_retention_score"] = out.apply(
        lambda r: safe_ratio_retention(r.get("train_tp_trigger_rate"), r.get("test_tp_trigger_rate")),
        axis=1,
    )

    def fp_penalty(row):
        train_fp = row.get("train_fp_trigger_rate")
        test_fp = row.get("test_fp_trigger_rate")
        if pd.isna(train_fp) or pd.isna(test_fp):
            return np.nan
        delta = float(test_fp) - float(train_fp)
        return float(np.clip(1.0 - max(delta, 0.0), 0.0, 1.0))

    out["fp_robustness_score"] = out.apply(fp_penalty, axis=1)

    out["mean_split_validation_score"] = (
        0.30 * out["direction_consistent"].fillna(0).astype(float) +
        0.25 * out["smd_retention_score"].fillna(0) +
        0.25 * out["distribution_retention_score"].fillna(0) +
        0.10 * out["tp_trigger_retention_score"].fillna(0) +
        0.10 * out["fp_robustness_score"].fillna(0)
    )

    out["validation_pass"] = (
        (out["direction_consistent"] == 1) &
        (
            (out["smd_retention_score"].fillna(0) >= 0.50) |
            (out["distribution_retention_score"].fillna(0) >= 0.50)
        )
    ).astype(int)

    out = out.sort_values(
        by=["mean_split_validation_score", "direction_consistent", "feature"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return out


def aggregate_split_stability(tables: list[pd.DataFrame]) -> pd.DataFrame:
    if not tables:
        return pd.DataFrame()

    big = pd.concat([df for df in tables if df is not None and not df.empty], axis=0, ignore_index=True)
    if big.empty:
        return big

    agg = big.groupby("feature").agg(
        n_models=("feature", "size"),
        direction_consistent_rate=("direction_consistent", "mean"),
        mean_split_validation_score=("mean_split_validation_score", "mean"),
        validation_pass_rate=("validation_pass", "mean"),
        mean_smd_retention_score=("smd_retention_score", "mean"),
        mean_distribution_retention_score=("distribution_retention_score", "mean"),
        mean_tp_trigger_retention_score=("tp_trigger_retention_score", "mean"),
        mean_fp_robustness_score=("fp_robustness_score", "mean"),
        dominant_train_direction=("train_monitor_direction", lambda s: Counter(pd.Series(s).dropna()).most_common(1)[0][0] if len(pd.Series(s).dropna()) > 0 else "unknown"),
        dominant_test_direction=("test_monitor_direction", lambda s: Counter(pd.Series(s).dropna()).most_common(1)[0][0] if len(pd.Series(s).dropna()) > 0 else "unknown"),
    ).reset_index()

    agg = agg.sort_values(
        by=["mean_split_validation_score", "validation_pass_rate", "feature"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return agg


def aggregate_recommendation_consistency(reco_tables: list[pd.DataFrame]) -> pd.DataFrame:
    if not reco_tables:
        return pd.DataFrame()

    big = pd.concat([df for df in reco_tables if df is not None and not df.empty], axis=0, ignore_index=True)
    if big.empty:
        return big

    base = big.groupby("feature").size().reset_index(name="n_models")
    pivot = big.pivot_table(index="feature", columns="recommendation", values="artifact_key", aggfunc="count", fill_value=0)
    pivot = pivot.reset_index()

    out = base.merge(pivot, on="feature", how="left")
    for col in ["monitor_first", "review", "drop", "false_positive_risk"]:
        if col not in out.columns:
            out[col] = 0

    count_cols = ["monitor_first", "review", "drop", "false_positive_risk"]
    out["recommendation_consistency_score"] = out[count_cols].max(axis=1) / out["n_models"].replace(0, np.nan)
    out = out.rename(columns={
        "monitor_first": "n_monitor_first",
        "review": "n_review",
        "drop": "n_drop",
        "false_positive_risk": "n_false_positive_risk",
    })
    return out


def add_pair_corr_metadata(pair_eval_df: pd.DataFrame, X_ref_df: pd.DataFrame | None, corr_threshold: float = HIGH_CORR_COMPANION_THRESHOLD) -> pd.DataFrame:
    if pair_eval_df is None or pair_eval_df.empty:
        return pair_eval_df
    if X_ref_df is None or X_ref_df.empty:
        out = pair_eval_df.copy()
        out["pair_abs_corr"] = np.nan
        out["redundant_pair_flag"] = 0
        return out

    num = X_ref_df.apply(pd.to_numeric, errors="coerce")
    corr = num.corr().fillna(0.0)

    out = pair_eval_df.copy()
    vals = []
    for _, r in out.iterrows():
        a = r["feature_a"]
        b = r["feature_b"]
        if a in corr.columns and b in corr.columns:
            vals.append(float(abs(corr.loc[a, b])))
        else:
            vals.append(np.nan)
    out["pair_abs_corr"] = vals
    out["redundant_pair_flag"] = (pd.Series(out["pair_abs_corr"]).fillna(0) >= corr_threshold).astype(int)
    return out
def build_validated_primary_anchor_summary(
    primary_anchor_df: pd.DataFrame,
    split_stability_agg_df: pd.DataFrame | None,
    reco_consistency_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if primary_anchor_df is None or primary_anchor_df.empty:
        return pd.DataFrame()

    out = primary_anchor_df.copy()

    if split_stability_agg_df is not None and not split_stability_agg_df.empty:
        out = out.merge(split_stability_agg_df, on="feature", how="left")
    else:
        out["mean_split_validation_score"] = np.nan
        out["validation_pass_rate"] = np.nan
        out["direction_consistent_rate"] = np.nan

    if reco_consistency_df is not None and not reco_consistency_df.empty:
        out = out.merge(reco_consistency_df, on="feature", how="left")
    else:
        out["recommendation_consistency_score"] = np.nan
        out["n_monitor_first"] = np.nan
        out["n_review"] = np.nan
        out["n_drop"] = np.nan
        out["n_false_positive_risk"] = np.nan

    # 確保必要欄位存在，避免 out.get(..., 0) 回傳 int 而不是 Series
    if "tp_fp_priority_candidate" not in out.columns:
        out["tp_fp_priority_candidate"] = 0
    if "tp_signal_available" not in out.columns:
        out["tp_signal_available"] = 0
    if "has_interaction_support" not in out.columns:
        out["has_interaction_support"] = 0

    out["tp_fp_priority_candidate"] = pd.to_numeric(out["tp_fp_priority_candidate"], errors="coerce").fillna(0).astype(int)
    out["tp_signal_available"] = pd.to_numeric(out["tp_signal_available"], errors="coerce").fillna(0).astype(int)
    out["has_interaction_support"] = pd.to_numeric(out["has_interaction_support"], errors="coerce").fillna(0).astype(int)

    out["validated_anchor_score"] = (
        0.85 * pd.to_numeric(out["overall_anchor_score"], errors="coerce").fillna(0) +
        0.15 * pd.to_numeric(out["mean_split_validation_score"], errors="coerce").fillna(0)
    )

    out["split_hard_gate_pass"] = (
        (pd.to_numeric(out["mean_split_validation_score"], errors="coerce").fillna(0) >= SPLIT_VALIDATION_MIN_SCORE) &
        (pd.to_numeric(out["validation_pass_rate"], errors="coerce").fillna(0) >= SPLIT_VALIDATION_MIN_PASS_RATE) &
        (pd.to_numeric(out["direction_consistent_rate"], errors="coerce").fillna(0) >= SPLIT_VALIDATION_MIN_PASS_RATE)
    ).astype(int)

    out = out.sort_values(
        by=["split_hard_gate_pass", "validated_anchor_score", "overall_anchor_score", "feature"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    primary_pool_mask = (
        (out["split_hard_gate_pass"] == 1) &
        (out["tp_fp_priority_candidate"] == 1)
    )

    out["primary_pool_rank"] = np.nan
    if primary_pool_mask.any():
        pool_order = out.loc[primary_pool_mask].sort_values(
            by=["validated_anchor_score", "mean_tp_fp_shap_gap", "tp_fp_ratio", "mean_tp_mean_positive_shap", "feature"],
            ascending=[False, False, False, False, True],
        ).index
        out.loc[pool_order, "primary_pool_rank"] = np.arange(1, len(pool_order) + 1)

    out["final_anchor_recommendation"] = "drop"

    # watchlist：只保留 TP 有訊號，但沒有被選進 priority 的特徵；
    # interaction_support 維持獨立，不混進 priority / watchlist。
    review_mask = (
        (out["split_hard_gate_pass"] == 1) &
        (out["tp_signal_available"] == 1) &
        (out["has_interaction_support"] != 1)
    )
    out.loc[review_mask, "final_anchor_recommendation"] = "review_anchor"

    top_priority_mask = (
        primary_pool_mask &
        (pd.to_numeric(out["primary_pool_rank"], errors="coerce").fillna(np.inf) <= PRIORITY_TOP_N)
    )
    out.loc[top_priority_mask, "final_anchor_recommendation"] = "primary_anchor"

    out = out.sort_values(
        by=["split_hard_gate_pass", "final_anchor_recommendation", "validated_anchor_score", "feature"],
        ascending=[False, True, False, True],
    ).reset_index(drop=True)

    out.insert(0, "validated_anchor_rank", range(1, len(out) + 1))
    return out

def _effective_oof_n_splits(y: pd.Series | np.ndarray | None, requested: int = OOF_N_SPLITS) -> int:
    if y is None:
        return 0
    y_arr = np.asarray(y).astype(int).ravel()
    uniq, counts = np.unique(y_arr, return_counts=True)
    if len(uniq) < 2:
        return 0
    return int(max(0, min(requested, int(counts.min()))))


def compute_oof_scores_from_final_model(final_model, X_df: pd.DataFrame, y_true: pd.Series | np.ndarray | None, category: str) -> np.ndarray | None:
    n_splits = _effective_oof_n_splits(y_true, OOF_N_SPLITS)
    if n_splits < 2:
        return None

    y_arr = np.asarray(y_true).astype(int).ravel()
    oof_scores = np.full(len(X_df), np.nan, dtype=float)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SHAP_RANDOM_STATE)
    feature_columns = list(X_df.columns)

    for tr_idx, va_idx in cv.split(X_df, y_arr):
        fold_model = clone(final_model)
        X_tr = X_df.iloc[tr_idx].copy()
        y_tr = y_arr[tr_idx]
        X_va = X_df.iloc[va_idx].copy()

        fold_model.fit(X_tr, y_tr)
        fold_score_fn = get_model_score_fn(fold_model, category, feature_columns, prefer_margin=False)
        oof_scores[va_idx] = np.asarray(fold_score_fn(X_va)).ravel()

    if np.isnan(oof_scores).any():
        return None
    return oof_scores


def write_df_or_empty(df: pd.DataFrame | None, path: str) -> None:
    if df is None or df.empty:
        pd.DataFrame().to_csv(path, index=False, encoding="utf-8-sig")
    else:
        df.to_csv(path, index=False, encoding="utf-8-sig")


def add_group_source(df: pd.DataFrame | None, group_source: str) -> pd.DataFrame | None:
    if df is None or df.empty:
        return df
    out = df.copy()
    out["group_source"] = group_source
    return out


def prefer_non_empty(*dfs) -> pd.DataFrame:
    for df in dfs:
        if df is not None and not df.empty:
            return df.copy()
    return pd.DataFrame()


def aggregate_cross_model_rank_table(rank_df: pd.DataFrame) -> pd.DataFrame:
    if rank_df is None or rank_df.empty:
        return pd.DataFrame()

    keep_cols = [
        c for c in [
            "feature", "rank", "mean_positive_shap", "artifact_key", "feature_set", "model", "group_source"
        ] if c in rank_df.columns
    ]
    out = rank_df[keep_cols].copy()
    if "artifact_key" not in out.columns:
        out["artifact_key"] = "unknown"

    agg = out.groupby("feature").agg(
        appeared_in_topn_models=("artifact_key", "nunique"),
        avg_rank=("rank", "mean"),
        best_rank=("rank", "min"),
        avg_mean_positive_shap=("mean_positive_shap", "mean"),
        max_mean_positive_shap=("mean_positive_shap", "max"),
    ).reset_index()

    if "feature_set" in out.columns:
        fs = out.groupby("feature")["feature_set"].agg(
            lambda s: ",".join(sorted(set(pd.Series(s).dropna().astype(str))))
        ).reset_index()
        agg = agg.merge(fs, on="feature", how="left")

    if "model" in out.columns:
        md = out.groupby("feature")["model"].agg(
            lambda s: ",".join(sorted(set(pd.Series(s).dropna().astype(str))))
        ).reset_index()
        agg = agg.merge(md, on="feature", how="left")

    agg = agg.sort_values(
        by=["appeared_in_topn_models", "avg_rank", "avg_mean_positive_shap", "feature"],
        ascending=[False, True, False, True],
    ).reset_index(drop=True)
    return agg


def take_top_n(df: pd.DataFrame, by: list[str], ascending: list[bool], n: int = OUTPUT_TOP_N) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    for c in by:
        if c not in out.columns:
            out[c] = np.nan

    out = out.sort_values(by=by, ascending=ascending).reset_index(drop=True).head(n).copy()
    out.insert(0, "export_rank", range(1, len(out) + 1))
    return out


def add_list_meta(df: pd.DataFrame, category: str, subtype: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    out.insert(0, "list_category", category)
    out.insert(1, "list_subtype", subtype)
    return out


def build_tiered_final_spc_list(spc_prep_df: pd.DataFrame, top_n: int = OUTPUT_TOP_N) -> pd.DataFrame:
    if spc_prep_df is None or spc_prep_df.empty:
        return pd.DataFrame()

    if "anchor_recommendation" not in spc_prep_df.columns:
        return pd.DataFrame()

    out = spc_prep_df.copy()
    for c in ["validated_anchor_score", "overall_anchor_score", "mean_tp_fp_shap_gap", "mean_tp_mean_positive_shap", "feature"]:
        if c not in out.columns:
            out[c] = np.nan

    tier_a = out[out["anchor_recommendation"] == "primary_anchor"].copy()
    tier_b = out[out["anchor_recommendation"] == "review_anchor"].copy()

    tier_a = tier_a.sort_values(
        by=["validated_anchor_score", "overall_anchor_score", "mean_tp_fp_shap_gap", "mean_tp_mean_positive_shap", "feature"],
        ascending=[False, False, False, False, True],
    ).copy()
    tier_a["spc_selection_tier"] = "Tier_A"

    tier_b = tier_b.sort_values(
        by=["validated_anchor_score", "overall_anchor_score", "mean_tp_fp_shap_gap", "mean_tp_mean_positive_shap", "feature"],
        ascending=[False, False, False, False, True],
    ).copy()
    tier_b["spc_selection_tier"] = "Tier_B"

    selected = pd.concat([tier_a, tier_b], axis=0, ignore_index=True).head(top_n).copy()
    if selected.empty:
        return selected

    selected.insert(0, "spc_export_rank", range(1, len(selected) + 1))
    selected["spc_export_selected"] = 1
    return selected


def build_grouped_output_lists(
    tp_like_rank_df: pd.DataFrame,
    fp_like_rank_df: pd.DataFrame,
    holdout_candidate_df: pd.DataFrame,
    split_stable_df: pd.DataFrame,
    missingness_df: pd.DataFrame,
    interaction_df: pd.DataFrame,
    distribution_df: pd.DataFrame,
    top_n: int = OUTPUT_TOP_N,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tp_like_top = take_top_n(
        tp_like_rank_df,
        by=["appeared_in_topn_models", "avg_rank", "avg_mean_positive_shap", "feature"],
        ascending=[False, True, False, True],
        n=top_n,
    )
    tp_like_top = add_list_meta(tp_like_top, "main_signal", "tp_like")

    fp_like_top = take_top_n(
        fp_like_rank_df,
        by=["appeared_in_topn_models", "avg_rank", "avg_mean_positive_shap", "feature"],
        ascending=[False, True, False, True],
        n=top_n,
    )
    fp_like_top = add_list_meta(fp_like_top, "main_signal", "fp_like")

    holdout_top = take_top_n(
        holdout_candidate_df,
        by=["tp_fp_priority_candidate", "final_candidate_score", "mean_tp_fp_shap_gap", "mean_tp_mean_positive_shap", "feature"],
        ascending=[False, False, False, False, True],
        n=top_n,
    )
    holdout_top = add_list_meta(holdout_top, "main_signal", "test_holdout")

    split_top = take_top_n(
        split_stable_df,
        by=["validation_pass_rate", "direction_consistent_rate", "mean_split_validation_score", "feature"],
        ascending=[False, False, False, True],
        n=top_n,
    )
    split_top = add_list_meta(split_top, "main_signal", "split_stable")

    main_signal_df = pd.concat([tp_like_top, fp_like_top, holdout_top, split_top], axis=0, ignore_index=True)

    missing_top = take_top_n(
        missingness_df,
        by=["has_missing_signal", "mean_missing_tp_tn_gap", "mean_missing_tp_fp_gap", "feature"],
        ascending=[False, False, False, True],
        n=top_n,
    )
    missing_top = add_list_meta(missing_top, "auxiliary_diagnostic", "missingness")

    interaction_top = take_top_n(
        interaction_df,
        by=["is_interaction_candidate", "mean_pair_precision_lift", "mean_pair_precision", "pair_key"],
        ascending=[False, False, False, True],
        n=top_n,
    )
    interaction_top = add_list_meta(interaction_top, "auxiliary_diagnostic", "interaction")

    dist_top = take_top_n(
        distribution_df,
        by=["mean_distribution_sep_score", "mean_ks_stat", "feature"],
        ascending=[False, False, True],
        n=top_n,
    )
    dist_top = add_list_meta(dist_top, "auxiliary_diagnostic", "distribution_separation")

    aux_df = pd.concat([missing_top, interaction_top, dist_top], axis=0, ignore_index=True)
    return main_signal_df, aux_df


def ordered_union_feature_list(*feature_lists, limit: int | None = None) -> list[str]:
    out = []
    seen = set()
    for fl in feature_lists:
        for feat in fl or []:
            if feat not in seen:
                out.append(feat)
                seen.add(feat)
            if limit is not None and len(out) >= limit:
                return out
    return out


def pick_pair_candidate_features(
    reco_df: pd.DataFrame | None,
    split_validation_df: pd.DataFrame | None = None,
    top_k: int = PAIR_TOP_FEATURES,
) -> list[str]:
    if reco_df is None or reco_df.empty:
        return []

    reco_core = strip_model_meta_cols(reco_df)
    use = reco_core[reco_core["recommendation"].isin(["monitor_first", "review"])].copy()
    if use.empty:
        use = reco_core.copy()

    candidate_list = use.head(top_k * 2)["feature"].tolist()
    if split_validation_df is not None and not split_validation_df.empty:
        sv = strip_model_meta_cols(split_validation_df)
        sv_pass = sv[(sv["validation_pass"] == 1) & (sv["direction_consistent"] == 1)]["feature"].tolist()
        candidate_list = [f for f in candidate_list if f in set(sv_pass)] or candidate_list

    return candidate_list[:top_k]


def maybe_merge_redundancy(candidate_summary_df: pd.DataFrame, X_ref_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if candidate_summary_df is None or candidate_summary_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    candidate_features = candidate_summary_df.loc[
        candidate_summary_df["recommendation"].isin(["monitor_first", "review"]),
        "feature",
    ].tolist()
    redundancy_df = build_redundancy_clusters(
        X_df=X_ref_df if X_ref_df is not None and not X_ref_df.empty else pd.DataFrame(),
        candidate_features=candidate_features,
        corr_threshold=CORR_THRESHOLD,
    )
    out = candidate_summary_df.copy()
    if redundancy_df is not None and not redundancy_df.empty:
        out = out.merge(redundancy_df, on="feature", how="left")
    else:
        out["redundancy_cluster"] = np.nan
        out["cluster_size"] = np.nan
    return out, redundancy_df


def build_rank_output(shap_values: np.ndarray, feature_names: list[str], group_name: str, n_group_rows_total: int, n_explained_rows: int, source_cols: dict[str, str]) -> pd.DataFrame:
    rank_df = summarize_shap(shap_values, feature_names)
    rank_df["group_name"] = group_name
    rank_df["n_group_rows_total"] = int(n_group_rows_total)
    rank_df["n_explained_rows"] = int(n_explained_rows)
    for k, v in source_cols.items():
        rank_df[k] = v
    return rank_df



def main():
    ensure_out_dir(OUT_DIR)

    print("[Load] detail csv")
    results_detail_df = read_csv_flex(DETAIL_CSV_PATH)
    results_detail_df.columns = [str(c) for c in results_detail_df.columns]

    print("[Load] artifacts joblib")
    all_artifacts = joblib.load(ARTIFACT_PATH)

    if "rank" in results_detail_df.columns:
        topk_df = (
            results_detail_df.sort_values(by=["rank"], ascending=True).head(TOP_K_MODELS).reset_index(drop=True)
        )
    else:
        topk_df = (
            results_detail_df.sort_values(
                by=["f1_score", "average_precision", "balanced_accuracy", "mcc"],
                ascending=[False, False, False, False],
            ).head(TOP_K_MODELS).reset_index(drop=True)
        )

    topk_df["artifact_key"] = topk_df.apply(make_artifact_key, axis=1)
    topk_df.to_csv(os.path.join(OUT_DIR, "top_models_for_feature_attribution.csv"), index=False, encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print("Top models for anomaly feature attribution")
    print("=" * 100)
    show_cols = ["feature_set", "category", "model", "balance_method", "f1_score", "artifact_key"]
    if "rank" in topk_df.columns:
        show_cols = ["rank"] + show_cols
    print(topk_df[show_cols].to_string(index=False))

    feature_train_cache = {}
    label_cache = {}
    feature_test_cache = {}
    label_test_cache = {}
    raw_train_cache = None
    raw_test_cache = None
    pooled_feature_space_cache = {}
    pooled_label_cache = {}

    run_metadata = []

    all_tp_rankings = []
    all_fp_rankings = []
    all_group_distribution_tables = []
    all_stats_cmp_tables = []
    all_model_feature_recommendation_tables = []
    model_level_candidate_rows = []
    per_model_top_feature_tables_for_stability = []
    all_missingness_tables = []
    all_pair_rule_tables = []
    all_class_sep_tables = []

    all_test_tp_rankings = []
    all_test_fp_rankings = []
    all_test_group_distribution_tables = []
    all_test_stats_cmp_tables = []
    all_test_model_feature_recommendation_tables = []
    test_model_level_candidate_rows = []
    per_model_test_top_feature_tables_for_stability = []
    all_test_class_sep_tables = []

    all_pooled_tp_rankings = []
    all_pooled_fp_rankings = []
    all_pooled_group_distribution_tables = []
    all_pooled_stats_cmp_tables = []
    all_pooled_model_feature_recommendation_tables = []
    pooled_model_level_candidate_rows = []
    per_model_pooled_top_feature_tables_for_stability = []
    all_pooled_class_sep_tables = []

    all_split_validation_tables = []
    all_raw_missingness_train_tables = []
    all_raw_missingness_test_tables = []
    all_raw_missingness_pooled_tables = []
    all_pair_rule_test_tables = []
    all_pair_rule_pooled_tables = []

    for i, row in topk_df.iterrows():
        feature_set = str(row["feature_set"])
        category = str(row["category"])
        model_name = str(row["model"])
        balance_method = str(row["balance_method"])
        artifact_key = str(row["artifact_key"])

        print("\n" + "-" * 100)
        print(f"[{i+1}/{len(topk_df)}] Start -> {artifact_key}")

        if artifact_key not in all_artifacts:
            print(f"[Skip] artifact not found: {artifact_key}")
            continue

        artifact = all_artifacts[artifact_key]
        final_model = artifact["final_model"]
        threshold = float(artifact["threshold"])

        should_skip, skip_reason = should_skip_feature_attribution(row, artifact)
        if should_skip:
            print(f"[Skip] {artifact_key}: {skip_reason}")
            run_metadata.append({
                "artifact_key": artifact_key,
                "feature_set": feature_set,
                "category": category,
                "model": model_name,
                "balance_method": balance_method,
                "threshold": threshold,
                "skipped": True,
                "skip_reason": skip_reason,
            })
            continue

        if feature_set not in feature_train_cache:
            train_df_raw, train_csv_path = load_feature_train_df(feature_set)
            feature_train_cache[feature_set] = (train_df_raw, train_csv_path)
        else:
            train_df_raw, train_csv_path = feature_train_cache[feature_set]

        if feature_set not in label_cache:
            y_train, label_csv_path = load_label_series(feature_set, expected_len=len(train_df_raw))
            label_cache[feature_set] = (y_train, label_csv_path)
        else:
            y_train, label_csv_path = label_cache[feature_set]

        X_train_for_model = align_columns_for_model(train_df_raw, final_model)
        feature_names = list(X_train_for_model.columns)
        pred_score_fn = get_model_score_fn(final_model, category, feature_names, prefer_margin=False)
        explain_score_fn = get_explanation_score_fn(final_model, category, feature_names)

        train_scores_final = pred_score_fn(X_train_for_model)
        train_oof_scores = compute_oof_scores_from_final_model(final_model, X_train_for_model, y_train, category)
        if train_oof_scores is not None:
            train_group_scores = train_oof_scores
            group_source = "train_oof"
        else:
            train_group_scores = train_scores_final
            group_source = "train_final_fallback"

        y_pred_train = to_binary_prediction(train_group_scores, threshold)
        group_masks = build_group_masks(y_train, y_pred_train)
        background_df = get_background_df(X_train_for_model, y_true=y_train, category=category, n_background=BACKGROUND_SIZE)

        group_count_info = {k: int(v.sum()) for k, v in group_masks.items()}
        print(f"feature_set={feature_set}, category={category}, model={model_name}, balance={balance_method}")
        print(f"train_csv={train_csv_path}")
        print(f"label_csv={label_csv_path}")
        print(f"X_train shape={X_train_for_model.shape}")
        print(f"threshold={threshold:.6f}")
        print(f"group_source={group_source}")
        print(f"group counts = {group_count_info}")

        dist_df = build_group_distribution_summary(X_train_for_model, group_masks)
        dist_df = add_group_source(dist_df, group_source)
        dist_df = add_model_meta(dist_df, row, artifact_key, threshold, feature_set)
        dist_path = os.path.join(OUT_DIR, f"group_distribution_summary__{safe_filename(artifact_key)}.csv")
        write_df_or_empty(dist_df, dist_path)
        if dist_df is not None and not dist_df.empty:
            all_group_distribution_tables.append(dist_df)

        stats_cmp_df = build_feature_stats_comparison(X_train_for_model, group_masks)
        stats_cmp_df = add_group_source(stats_cmp_df, group_source)
        stats_cmp_df = add_model_meta(stats_cmp_df, row, artifact_key, threshold, feature_set)
        stats_cmp_path = os.path.join(OUT_DIR, f"feature_stats_comparison__{safe_filename(artifact_key)}.csv")
        write_df_or_empty(stats_cmp_df, stats_cmp_path)
        if stats_cmp_df is not None and not stats_cmp_df.empty:
            all_stats_cmp_tables.append(stats_cmp_df)

        tp_group_name = "TP" if "TP" in group_masks else "pred_anomaly"
        fp_group_name = "FP" if "FP" in group_masks else "pred_normal"

        tp_explain_df, _ = pick_explain_subset_by_mask(X_train_for_model, train_group_scores, group_masks[tp_group_name], MAX_EXPLAIN_ROWS_PER_GROUP)
        fp_explain_df, _ = pick_explain_subset_by_mask(X_train_for_model, train_group_scores, group_masks[fp_group_name], MAX_EXPLAIN_ROWS_PER_GROUP)

        if len(tp_explain_df) < MIN_GROUP_SIZE_FOR_SHAP:
            print(f"[Warn] {tp_group_name} rows < {MIN_GROUP_SIZE_FOR_SHAP}, SHAP may be unstable")
        if len(fp_explain_df) < MIN_GROUP_SIZE_FOR_SHAP:
            print(f"[Warn] {fp_group_name} rows < {MIN_GROUP_SIZE_FOR_SHAP}, SHAP may be unstable")

        tp_shap_values = compute_shap_values(final_model, explain_score_fn, background_df, tp_explain_df, category) if len(tp_explain_df) > 0 else np.empty((0, len(feature_names)))
        fp_shap_values = compute_shap_values(final_model, explain_score_fn, background_df, fp_explain_df, category) if len(fp_explain_df) > 0 else np.empty((0, len(feature_names)))

        tp_rank_df = build_rank_output(
            tp_shap_values, feature_names, tp_group_name, int(group_masks[tp_group_name].sum()), int(len(tp_explain_df)),
            {"source_train_csv": train_csv_path, "source_label_csv": str(label_csv_path), "group_source": group_source},
        )
        tp_rank_df = add_model_meta(tp_rank_df, row, artifact_key, threshold, feature_set)
        tp_rank_path = os.path.join(OUT_DIR, f"shap_rank__{tp_group_name}__{safe_filename(artifact_key)}.csv")
        write_df_or_empty(tp_rank_df, tp_rank_path)
        if tp_rank_df is not None and not tp_rank_df.empty:
            all_tp_rankings.append(tp_rank_df)
            per_model_top_feature_tables_for_stability.append(tp_rank_df)

        fp_rank_df = build_rank_output(
            fp_shap_values, feature_names, fp_group_name, int(group_masks[fp_group_name].sum()), int(len(fp_explain_df)),
            {"source_train_csv": train_csv_path, "source_label_csv": str(label_csv_path), "group_source": group_source},
        )
        fp_rank_df = add_model_meta(fp_rank_df, row, artifact_key, threshold, feature_set)
        fp_rank_path = os.path.join(OUT_DIR, f"shap_rank__{fp_group_name}__{safe_filename(artifact_key)}.csv")
        write_df_or_empty(fp_rank_df, fp_rank_path)
        if fp_rank_df is not None and not fp_rank_df.empty:
            all_fp_rankings.append(fp_rank_df)

        shap_cmp_df = merge_tp_fp_shap(tp_rank_df, fp_rank_df)
        shap_cmp_df = add_group_source(shap_cmp_df, group_source)
        shap_cmp_df = add_model_meta(shap_cmp_df, row, artifact_key, threshold, feature_set)

        stats_cmp_core = strip_model_meta_cols(stats_cmp_df)
        model_feature_reco_df = build_model_feature_recommendation_table(strip_model_meta_cols(shap_cmp_df), stats_cmp_core)
        model_feature_reco_df = add_group_source(model_feature_reco_df, group_source)
        model_feature_reco_df = add_model_meta(model_feature_reco_df, row, artifact_key, threshold, feature_set)
        reco_path = os.path.join(OUT_DIR, f"monitor_feature_recommendation__{safe_filename(artifact_key)}.csv")
        write_df_or_empty(model_feature_reco_df, reco_path)
        if model_feature_reco_df is not None and not model_feature_reco_df.empty:
            all_model_feature_recommendation_tables.append(model_feature_reco_df)
            model_level_candidate_rows.append(model_feature_reco_df)

        missing_cmp_df = build_missingness_stats_comparison(X_train_for_model, group_masks)
        missing_cmp_df = add_group_source(missing_cmp_df, group_source)
        missing_cmp_df = add_model_meta(missing_cmp_df, row, artifact_key, threshold, feature_set)
        missing_path = os.path.join(OUT_DIR, f"missingness_feature_stats__{safe_filename(artifact_key)}.csv")
        write_df_or_empty(missing_cmp_df, missing_path)
        if missing_cmp_df is not None and not missing_cmp_df.empty:
            all_missingness_tables.append(missing_cmp_df)

        class_sep_df = build_class_distribution_separation(X_train_for_model, y_train)
        class_sep_df = add_model_meta(class_sep_df, row, artifact_key, threshold, feature_set) if class_sep_df is not None and not class_sep_df.empty else class_sep_df
        class_sep_path = os.path.join(OUT_DIR, f"class_distribution_separation__{safe_filename(artifact_key)}.csv")
        write_df_or_empty(class_sep_df, class_sep_path)
        if class_sep_df is not None and not class_sep_df.empty:
            all_class_sep_tables.append(class_sep_df)

        exploratory_pair_features = pick_pair_candidate_features(model_feature_reco_df, top_k=PAIR_TOP_FEATURES)
        pair_rules_df = build_pair_rules(stats_cmp_core, top_features=exploratory_pair_features)
        pair_eval_df = evaluate_pair_rule_precision_lift(X_train_for_model, y_train, pair_rules_df)
        pair_eval_df = add_pair_corr_metadata(pair_eval_df, X_train_for_model) if pair_eval_df is not None and not pair_eval_df.empty else pair_eval_df
        if pair_eval_df is not None and not pair_eval_df.empty:
            pair_eval_df["evidence_scope"] = "exploratory_train_oof_groups"
        pair_eval_df = add_model_meta(pair_eval_df, row, artifact_key, threshold, feature_set) if pair_eval_df is not None and not pair_eval_df.empty else pair_eval_df
        pair_path = os.path.join(OUT_DIR, f"pair_rule_candidates__{safe_filename(artifact_key)}.csv")
        write_df_or_empty(pair_eval_df, pair_path)
        if pair_eval_df is not None and not pair_eval_df.empty:
            all_pair_rule_tables.append(pair_eval_df)

        pooled_ready = False
        split_validation_df = pd.DataFrame()
        pooled_model_feature_reco_df = pd.DataFrame()
        test_model_feature_reco_df = pd.DataFrame()
        test_csv_path = None
        test_label_csv_path = None
        pooled_reco_path = None
        split_validation_path = None
        raw_missing_train_path = None
        raw_missing_test_path = None
        raw_missing_pooled_path = None
        pair_test_path = None
        pair_pooled_path = None

        if feature_set not in feature_test_cache:
            feature_test_cache[feature_set] = load_feature_test_df(feature_set)
        test_df_raw, test_csv_path = feature_test_cache[feature_set]

        if test_df_raw is not None:
            if feature_set not in label_test_cache:
                y_test_loaded, y_test_path = load_test_label_series(feature_set, expected_len=len(test_df_raw))
                label_test_cache[feature_set] = (y_test_loaded, y_test_path)
            y_test, test_label_csv_path = label_test_cache[feature_set]
        else:
            y_test, test_label_csv_path = None, None

        if test_df_raw is not None and y_test is not None:
            try:
                X_test_for_model = align_columns_for_model(test_df_raw, final_model)
                if len(X_test_for_model) != len(y_test):
                    raise ValueError("test feature / label length mismatch after alignment")

                test_scores = pred_score_fn(X_test_for_model)
                y_pred_test = to_binary_prediction(test_scores, threshold)
                group_masks_test = build_group_masks(y_test, y_pred_test)
                pooled_ready = True

                test_dist_df = build_group_distribution_summary(X_test_for_model, group_masks_test)
                test_dist_df = add_group_source(test_dist_df, "test_holdout")
                test_dist_df = add_model_meta(test_dist_df, row, artifact_key, threshold, feature_set)
                test_dist_path = os.path.join(OUT_DIR, f"group_distribution_summary__test__{safe_filename(artifact_key)}.csv")
                write_df_or_empty(test_dist_df, test_dist_path)
                if test_dist_df is not None and not test_dist_df.empty:
                    all_test_group_distribution_tables.append(test_dist_df)

                test_stats_cmp_df = build_feature_stats_comparison(X_test_for_model, group_masks_test)
                test_stats_cmp_df = add_group_source(test_stats_cmp_df, "test_holdout")
                test_stats_cmp_df = add_model_meta(test_stats_cmp_df, row, artifact_key, threshold, feature_set)
                test_stats_path = os.path.join(OUT_DIR, f"feature_stats_comparison__test__{safe_filename(artifact_key)}.csv")
                write_df_or_empty(test_stats_cmp_df, test_stats_path)
                if test_stats_cmp_df is not None and not test_stats_cmp_df.empty:
                    all_test_stats_cmp_tables.append(test_stats_cmp_df)

                test_class_sep_df = build_class_distribution_separation(X_test_for_model, y_test)
                test_class_sep_df = add_model_meta(test_class_sep_df, row, artifact_key, threshold, feature_set) if test_class_sep_df is not None and not test_class_sep_df.empty else test_class_sep_df
                test_class_sep_path = os.path.join(OUT_DIR, f"class_distribution_separation__test__{safe_filename(artifact_key)}.csv")
                write_df_or_empty(test_class_sep_df, test_class_sep_path)
                if test_class_sep_df is not None and not test_class_sep_df.empty:
                    all_test_class_sep_tables.append(test_class_sep_df)

                test_tp_group = "TP" if "TP" in group_masks_test else "pred_anomaly"
                test_fp_group = "FP" if "FP" in group_masks_test else "pred_normal"
                test_tp_explain_df, _ = pick_explain_subset_by_mask(X_test_for_model, test_scores, group_masks_test[test_tp_group], MAX_EXPLAIN_ROWS_PER_GROUP)
                test_fp_explain_df, _ = pick_explain_subset_by_mask(X_test_for_model, test_scores, group_masks_test[test_fp_group], MAX_EXPLAIN_ROWS_PER_GROUP)

                test_tp_shap_values = compute_shap_values(final_model, explain_score_fn, background_df, test_tp_explain_df, category) if len(test_tp_explain_df) > 0 else np.empty((0, len(feature_names)))
                test_fp_shap_values = compute_shap_values(final_model, explain_score_fn, background_df, test_fp_explain_df, category) if len(test_fp_explain_df) > 0 else np.empty((0, len(feature_names)))

                test_tp_rank_df = build_rank_output(
                    test_tp_shap_values, feature_names, test_tp_group, int(group_masks_test[test_tp_group].sum()), int(len(test_tp_explain_df)),
                    {"source_train_csv": train_csv_path, "source_test_csv": str(test_csv_path), "source_label_csv": str(test_label_csv_path), "group_source": "test_holdout"},
                )
                test_tp_rank_df = add_model_meta(test_tp_rank_df, row, artifact_key, threshold, feature_set)
                test_tp_rank_path = os.path.join(OUT_DIR, f"shap_rank__test__{test_tp_group}__{safe_filename(artifact_key)}.csv")
                write_df_or_empty(test_tp_rank_df, test_tp_rank_path)
                if test_tp_rank_df is not None and not test_tp_rank_df.empty:
                    all_test_tp_rankings.append(test_tp_rank_df)
                    per_model_test_top_feature_tables_for_stability.append(test_tp_rank_df)

                test_fp_rank_df = build_rank_output(
                    test_fp_shap_values, feature_names, test_fp_group, int(group_masks_test[test_fp_group].sum()), int(len(test_fp_explain_df)),
                    {"source_train_csv": train_csv_path, "source_test_csv": str(test_csv_path), "source_label_csv": str(test_label_csv_path), "group_source": "test_holdout"},
                )
                test_fp_rank_df = add_model_meta(test_fp_rank_df, row, artifact_key, threshold, feature_set)
                test_fp_rank_path = os.path.join(OUT_DIR, f"shap_rank__test__{test_fp_group}__{safe_filename(artifact_key)}.csv")
                write_df_or_empty(test_fp_rank_df, test_fp_rank_path)
                if test_fp_rank_df is not None and not test_fp_rank_df.empty:
                    all_test_fp_rankings.append(test_fp_rank_df)

                test_shap_cmp_df = merge_tp_fp_shap(test_tp_rank_df, test_fp_rank_df)
                test_shap_cmp_df = add_group_source(test_shap_cmp_df, "test_holdout")
                test_shap_cmp_df = add_model_meta(test_shap_cmp_df, row, artifact_key, threshold, feature_set)
                test_model_feature_reco_df = build_model_feature_recommendation_table(strip_model_meta_cols(test_shap_cmp_df), strip_model_meta_cols(test_stats_cmp_df))
                test_model_feature_reco_df = add_group_source(test_model_feature_reco_df, "test_holdout")
                test_model_feature_reco_df = add_model_meta(test_model_feature_reco_df, row, artifact_key, threshold, feature_set)
                test_reco_path = os.path.join(OUT_DIR, f"monitor_feature_recommendation__test__{safe_filename(artifact_key)}.csv")
                write_df_or_empty(test_model_feature_reco_df, test_reco_path)
                if test_model_feature_reco_df is not None and not test_model_feature_reco_df.empty:
                    all_test_model_feature_recommendation_tables.append(test_model_feature_reco_df)
                    test_model_level_candidate_rows.append(test_model_feature_reco_df)

                X_pooled = pd.concat([X_train_for_model, X_test_for_model], axis=0, ignore_index=True)
                y_pooled = pd.Series(pd.concat([pd.Series(y_train), pd.Series(y_test)], axis=0, ignore_index=True))
                pooled_scores = np.concatenate([np.asarray(train_scores_final).ravel(), np.asarray(test_scores).ravel()])
                y_pred_pooled = to_binary_prediction(pooled_scores, threshold)
                group_masks_pooled = build_group_masks(y_pooled, y_pred_pooled)

                if feature_set not in pooled_feature_space_cache:
                    pooled_feature_space_cache[feature_set] = X_pooled.copy()
                    pooled_label_cache[feature_set] = y_pooled.copy()

                pooled_background_df = get_background_df(X_pooled, y_true=y_pooled, category=category, n_background=POOLED_BACKGROUND_SIZE)

                pooled_dist_df = build_group_distribution_summary(X_pooled, group_masks_pooled)
                pooled_dist_df = add_group_source(pooled_dist_df, "pooled_auxiliary")
                pooled_dist_df = add_model_meta(pooled_dist_df, row, artifact_key, threshold, feature_set)
                pooled_dist_path = os.path.join(OUT_DIR, f"group_distribution_summary__pooled__{safe_filename(artifact_key)}.csv")
                write_df_or_empty(pooled_dist_df, pooled_dist_path)
                if pooled_dist_df is not None and not pooled_dist_df.empty:
                    all_pooled_group_distribution_tables.append(pooled_dist_df)

                pooled_stats_cmp_df = build_feature_stats_comparison(X_pooled, group_masks_pooled)
                pooled_stats_cmp_df = add_group_source(pooled_stats_cmp_df, "pooled_auxiliary")
                pooled_stats_cmp_df = add_model_meta(pooled_stats_cmp_df, row, artifact_key, threshold, feature_set)
                pooled_stats_path = os.path.join(OUT_DIR, f"feature_stats_comparison__pooled__{safe_filename(artifact_key)}.csv")
                write_df_or_empty(pooled_stats_cmp_df, pooled_stats_path)
                if pooled_stats_cmp_df is not None and not pooled_stats_cmp_df.empty:
                    all_pooled_stats_cmp_tables.append(pooled_stats_cmp_df)

                pooled_tp_group = "TP" if "TP" in group_masks_pooled else "pred_anomaly"
                pooled_fp_group = "FP" if "FP" in group_masks_pooled else "pred_normal"
                pooled_tp_explain_df, _ = pick_explain_subset_by_mask(X_pooled, pooled_scores, group_masks_pooled[pooled_tp_group], MAX_EXPLAIN_ROWS_PER_GROUP)
                pooled_fp_explain_df, _ = pick_explain_subset_by_mask(X_pooled, pooled_scores, group_masks_pooled[pooled_fp_group], MAX_EXPLAIN_ROWS_PER_GROUP)
                pooled_tp_shap_values = compute_shap_values(final_model, explain_score_fn, pooled_background_df, pooled_tp_explain_df, category) if len(pooled_tp_explain_df) > 0 else np.empty((0, len(feature_names)))
                pooled_fp_shap_values = compute_shap_values(final_model, explain_score_fn, pooled_background_df, pooled_fp_explain_df, category) if len(pooled_fp_explain_df) > 0 else np.empty((0, len(feature_names)))

                pooled_tp_rank_df = build_rank_output(
                    pooled_tp_shap_values, feature_names, pooled_tp_group, int(group_masks_pooled[pooled_tp_group].sum()), int(len(pooled_tp_explain_df)),
                    {"source_train_csv": train_csv_path, "source_test_csv": str(test_csv_path), "group_source": "pooled_auxiliary"},
                )
                pooled_tp_rank_df = add_model_meta(pooled_tp_rank_df, row, artifact_key, threshold, feature_set)
                pooled_tp_rank_path = os.path.join(OUT_DIR, f"shap_rank__pooled__{pooled_tp_group}__{safe_filename(artifact_key)}.csv")
                write_df_or_empty(pooled_tp_rank_df, pooled_tp_rank_path)
                if pooled_tp_rank_df is not None and not pooled_tp_rank_df.empty:
                    all_pooled_tp_rankings.append(pooled_tp_rank_df)
                    per_model_pooled_top_feature_tables_for_stability.append(pooled_tp_rank_df)

                pooled_fp_rank_df = build_rank_output(
                    pooled_fp_shap_values, feature_names, pooled_fp_group, int(group_masks_pooled[pooled_fp_group].sum()), int(len(pooled_fp_explain_df)),
                    {"source_train_csv": train_csv_path, "source_test_csv": str(test_csv_path), "group_source": "pooled_auxiliary"},
                )
                pooled_fp_rank_df = add_model_meta(pooled_fp_rank_df, row, artifact_key, threshold, feature_set)
                pooled_fp_rank_path = os.path.join(OUT_DIR, f"shap_rank__pooled__{pooled_fp_group}__{safe_filename(artifact_key)}.csv")
                write_df_or_empty(pooled_fp_rank_df, pooled_fp_rank_path)
                if pooled_fp_rank_df is not None and not pooled_fp_rank_df.empty:
                    all_pooled_fp_rankings.append(pooled_fp_rank_df)

                pooled_shap_cmp_df = merge_tp_fp_shap(pooled_tp_rank_df, pooled_fp_rank_df)
                pooled_shap_cmp_df = add_group_source(pooled_shap_cmp_df, "pooled_auxiliary")
                pooled_shap_cmp_df = add_model_meta(pooled_shap_cmp_df, row, artifact_key, threshold, feature_set)
                pooled_model_feature_reco_df = build_model_feature_recommendation_table(strip_model_meta_cols(pooled_shap_cmp_df), strip_model_meta_cols(pooled_stats_cmp_df))
                pooled_model_feature_reco_df = add_group_source(pooled_model_feature_reco_df, "pooled_auxiliary")
                pooled_model_feature_reco_df = add_model_meta(pooled_model_feature_reco_df, row, artifact_key, threshold, feature_set)
                pooled_reco_path = os.path.join(OUT_DIR, f"pooled_monitor_feature_recommendation__{safe_filename(artifact_key)}.csv")
                write_df_or_empty(pooled_model_feature_reco_df, pooled_reco_path)
                if pooled_model_feature_reco_df is not None and not pooled_model_feature_reco_df.empty:
                    all_pooled_model_feature_recommendation_tables.append(pooled_model_feature_reco_df)
                    pooled_model_level_candidate_rows.append(pooled_model_feature_reco_df)

                pooled_class_sep_df = build_class_distribution_separation(X_pooled, y_pooled)
                pooled_class_sep_df = add_model_meta(pooled_class_sep_df, row, artifact_key, threshold, feature_set) if pooled_class_sep_df is not None and not pooled_class_sep_df.empty else pooled_class_sep_df
                pooled_class_sep_path = os.path.join(OUT_DIR, f"class_distribution_separation__pooled__{safe_filename(artifact_key)}.csv")
                write_df_or_empty(pooled_class_sep_df, pooled_class_sep_path)
                if pooled_class_sep_df is not None and not pooled_class_sep_df.empty:
                    all_pooled_class_sep_tables.append(pooled_class_sep_df)

                split_feature_list = ordered_union_feature_list(
                    test_model_feature_reco_df.head(SPLIT_VALIDATION_TOP_FEATURES)["feature"].tolist() if test_model_feature_reco_df is not None and not test_model_feature_reco_df.empty else [],
                    model_feature_reco_df.head(SPLIT_VALIDATION_TOP_FEATURES)["feature"].tolist() if model_feature_reco_df is not None and not model_feature_reco_df.empty else [],
                    pooled_model_feature_reco_df.head(SPLIT_VALIDATION_TOP_FEATURES)["feature"].tolist() if pooled_model_feature_reco_df is not None and not pooled_model_feature_reco_df.empty else [],
                    limit=SPLIT_VALIDATION_TOP_FEATURES,
                )
                split_validation_df = build_feature_split_stability_check(
                    feature_list=split_feature_list,
                    train_stats_df=stats_cmp_df,
                    test_stats_df=test_stats_cmp_df,
                    train_class_sep_df=class_sep_df,
                    test_class_sep_df=test_class_sep_df,
                    pooled_reco_df=pooled_model_feature_reco_df,
                )
                split_validation_df = add_model_meta(split_validation_df, row, artifact_key, threshold, feature_set) if split_validation_df is not None and not split_validation_df.empty else split_validation_df
                split_validation_path = os.path.join(OUT_DIR, f"feature_split_stability_check__{safe_filename(artifact_key)}.csv")
                write_df_or_empty(split_validation_df, split_validation_path)
                if split_validation_df is not None and not split_validation_df.empty:
                    all_split_validation_tables.append(split_validation_df)

                if raw_train_cache is None:
                    raw_train_cache, _ = load_raw_split_df(RAW_TRAIN_CSV_PATH)
                if raw_test_cache is None:
                    raw_test_cache, _ = load_raw_split_df(RAW_TEST_CSV_PATH)
                raw_train_sel = subset_columns_if_present(raw_train_cache, feature_names)
                raw_test_sel = subset_columns_if_present(raw_test_cache, feature_names)
                if raw_train_sel is not None:
                    raw_missing_train_df = build_missingness_stats_comparison(raw_train_sel, group_masks)
                    raw_missing_train_df = add_group_source(raw_missing_train_df, group_source)
                    raw_missing_train_df = add_model_meta(raw_missing_train_df, row, artifact_key, threshold, feature_set)
                    raw_missing_train_path = os.path.join(OUT_DIR, f"raw_missingness_feature_stats__train_raw__{safe_filename(artifact_key)}.csv")
                    write_df_or_empty(raw_missing_train_df, raw_missing_train_path)
                    if raw_missing_train_df is not None and not raw_missing_train_df.empty:
                        all_raw_missingness_train_tables.append(raw_missing_train_df)
                if raw_test_sel is not None:
                    raw_missing_test_df = build_missingness_stats_comparison(raw_test_sel, group_masks_test)
                    raw_missing_test_df = add_group_source(raw_missing_test_df, "test_holdout")
                    raw_missing_test_df = add_model_meta(raw_missing_test_df, row, artifact_key, threshold, feature_set)
                    raw_missing_test_path = os.path.join(OUT_DIR, f"raw_missingness_feature_stats__test_raw__{safe_filename(artifact_key)}.csv")
                    write_df_or_empty(raw_missing_test_df, raw_missing_test_path)
                    if raw_missing_test_df is not None and not raw_missing_test_df.empty:
                        all_raw_missingness_test_tables.append(raw_missing_test_df)
                if raw_train_sel is not None and raw_test_sel is not None:
                    raw_pooled_sel = pd.concat([raw_train_sel, raw_test_sel], axis=0, ignore_index=True)
                    raw_missing_pooled_df = build_missingness_stats_comparison(raw_pooled_sel, group_masks_pooled)
                    raw_missing_pooled_df = add_group_source(raw_missing_pooled_df, "pooled_auxiliary")
                    raw_missing_pooled_df = add_model_meta(raw_missing_pooled_df, row, artifact_key, threshold, feature_set)
                    raw_missing_pooled_path = os.path.join(OUT_DIR, f"raw_missingness_feature_stats__pooled_raw__{safe_filename(artifact_key)}.csv")
                    write_df_or_empty(raw_missing_pooled_df, raw_missing_pooled_path)
                    if raw_missing_pooled_df is not None and not raw_missing_pooled_df.empty:
                        all_raw_missingness_pooled_tables.append(raw_missing_pooled_df)

                pair_top_features = pick_pair_candidate_features(test_model_feature_reco_df, split_validation_df, top_k=PAIR_TOP_FEATURES)
                pair_stats_source = strip_model_meta_cols(test_stats_cmp_df)
                validated_pair_rules_df = build_pair_rules(pair_stats_source, top_features=pair_top_features)

                pair_test_df = evaluate_pair_rule_precision_lift(X_test_for_model, y_test, validated_pair_rules_df)
                pair_test_df = add_pair_corr_metadata(pair_test_df, X_pooled) if pair_test_df is not None and not pair_test_df.empty else pair_test_df
                if pair_test_df is not None and not pair_test_df.empty:
                    pair_test_df["evidence_scope"] = "test_holdout"
                pair_test_df = add_model_meta(pair_test_df, row, artifact_key, threshold, feature_set) if pair_test_df is not None and not pair_test_df.empty else pair_test_df
                pair_test_path = os.path.join(OUT_DIR, f"pair_rule_candidates__test__{safe_filename(artifact_key)}.csv")
                write_df_or_empty(pair_test_df, pair_test_path)
                if pair_test_df is not None and not pair_test_df.empty:
                    all_pair_rule_test_tables.append(pair_test_df)

                pair_pooled_df = evaluate_pair_rule_precision_lift(X_pooled, y_pooled, validated_pair_rules_df)
                pair_pooled_df = add_pair_corr_metadata(pair_pooled_df, X_pooled) if pair_pooled_df is not None and not pair_pooled_df.empty else pair_pooled_df
                if pair_pooled_df is not None and not pair_pooled_df.empty:
                    pair_pooled_df["evidence_scope"] = "pooled_auxiliary"
                pair_pooled_df = add_model_meta(pair_pooled_df, row, artifact_key, threshold, feature_set) if pair_pooled_df is not None and not pair_pooled_df.empty else pair_pooled_df
                pair_pooled_path = os.path.join(OUT_DIR, f"pair_rule_candidates__pooled__{safe_filename(artifact_key)}.csv")
                write_df_or_empty(pair_pooled_df, pair_pooled_path)
                if pair_pooled_df is not None and not pair_pooled_df.empty:
                    all_pair_rule_pooled_tables.append(pair_pooled_df)

            except Exception as e:
                print(f"[Warn] skip pooled/test modules for {artifact_key}: {e}")

        print("\nTop 10 TP-like anomaly-push features:")
        print(tp_rank_df[["rank", "feature", "mean_positive_shap", "positive_push_rate"]].head(10).to_string(index=False))
        print("\nTop 10 FP-like anomaly-push features:")
        print(fp_rank_df[["rank", "feature", "mean_positive_shap", "positive_push_rate"]].head(10).to_string(index=False))
        print("\nTop 10 model recommendations:")
        print(model_feature_reco_df[["feature", "tp_mean_positive_shap", "fp_mean_positive_shap", "tp_fp_shap_gap", "monitor_direction", "recommendation"]].head(10).to_string(index=False))
        if test_model_feature_reco_df is not None and not test_model_feature_reco_df.empty:
            print("\nTop 10 test-holdout recommendations:")
            print(test_model_feature_reco_df[["feature", "tp_mean_positive_shap", "fp_mean_positive_shap", "tp_fp_shap_gap", "monitor_direction", "recommendation"]].head(10).to_string(index=False))
        if split_validation_df is not None and not split_validation_df.empty:
            print("\nTop 5 split-stable candidates:")
            print(split_validation_df[["feature", "mean_split_validation_score", "direction_consistent", "validation_pass"]].head(5).to_string(index=False))

        run_metadata.append({
            "artifact_key": artifact_key,
            "feature_set": feature_set,
            "category": category,
            "model": model_name,
            "balance_method": balance_method,
            "threshold": threshold,
            "group_source": group_source,
            "train_csv": train_csv_path,
            "label_csv": label_csv_path,
            "test_csv": test_csv_path,
            "test_label_csv": test_label_csv_path,
            "n_rows": int(len(X_train_for_model)),
            "n_features": int(X_train_for_model.shape[1]),
            **group_count_info,
            "tp_shap_csv": os.path.basename(tp_rank_path),
            "fp_shap_csv": os.path.basename(fp_rank_path),
            "feature_stats_csv": os.path.basename(stats_cmp_path),
            "group_distribution_csv": os.path.basename(dist_path),
            "recommendation_csv": os.path.basename(reco_path),
            "test_recommendation_csv": os.path.basename(test_reco_path) if 'test_reco_path' in locals() and test_df_raw is not None and y_test is not None else None,
            "missingness_csv": os.path.basename(missing_path),
            "class_distribution_csv": os.path.basename(class_sep_path),
            "pair_rule_csv": os.path.basename(pair_path),
            "pooled_recommendation_csv": os.path.basename(pooled_reco_path) if pooled_reco_path else None,
            "split_validation_csv": os.path.basename(split_validation_path) if split_validation_path else None,
            "raw_missing_train_csv": os.path.basename(raw_missing_train_path) if raw_missing_train_path else None,
            "raw_missing_test_csv": os.path.basename(raw_missing_test_path) if raw_missing_test_path else None,
            "raw_missing_pooled_csv": os.path.basename(raw_missing_pooled_path) if raw_missing_pooled_path else None,
            "pair_test_csv": os.path.basename(pair_test_path) if pair_test_path else None,
            "pair_pooled_csv": os.path.basename(pair_pooled_path) if pair_pooled_path else None,
        })

    if run_metadata:
        pd.DataFrame(run_metadata).to_csv(os.path.join(OUT_DIR, "run_metadata_summary.csv"), index=False, encoding="utf-8-sig")

    def concat_save(tables, filename):
        if tables:
            df = pd.concat(tables, axis=0, ignore_index=True)
            df.to_csv(os.path.join(OUT_DIR, filename), index=False, encoding="utf-8-sig")
            return df
        return pd.DataFrame()

    all_tp_df = concat_save(all_tp_rankings, "shap_rank__all_models__tp_like.csv")
    all_fp_df = concat_save(all_fp_rankings, "shap_rank__all_models__fp_like.csv")
    concat_save(all_group_distribution_tables, "group_distribution_summary__all_models.csv")
    all_stats_all_df = concat_save(all_stats_cmp_tables, "feature_stats_comparison__all_models.csv")
    all_model_reco_df = concat_save(all_model_feature_recommendation_tables, "monitor_feature_recommendation__all_models.csv")
    missingness_all_df = concat_save(all_missingness_tables, "missingness_feature_stats__all_models.csv")
    class_sep_all_df = concat_save(all_class_sep_tables, "class_distribution_separation__all_models.csv")
    pair_rule_all_df = concat_save(all_pair_rule_tables, "pair_rule_candidates__all_models.csv")

    all_test_tp_df = concat_save(all_test_tp_rankings, "shap_rank__test__all_models__tp_like.csv")
    all_test_fp_df = concat_save(all_test_fp_rankings, "shap_rank__test__all_models__fp_like.csv")
    concat_save(all_test_group_distribution_tables, "group_distribution_summary__test__all_models.csv")
    all_test_stats_all_df = concat_save(all_test_stats_cmp_tables, "feature_stats_comparison__test__all_models.csv")
    all_test_model_reco_df = concat_save(all_test_model_feature_recommendation_tables, "monitor_feature_recommendation__test__all_models.csv")
    test_class_sep_all_df = concat_save(all_test_class_sep_tables, "class_distribution_separation__test__all_models.csv")

    concat_save(all_pooled_group_distribution_tables, "group_distribution_summary__pooled__all_models.csv")
    all_pooled_stats_all_df = concat_save(all_pooled_stats_cmp_tables, "feature_stats_comparison__pooled__all_models.csv")
    concat_save(all_pooled_tp_rankings, "shap_rank__pooled__tp_like__all_models.csv")
    concat_save(all_pooled_fp_rankings, "shap_rank__pooled__fp_like__all_models.csv")
    all_pooled_model_reco_df = concat_save(all_pooled_model_feature_recommendation_tables, "pooled_monitor_feature_recommendation__all_models.csv")
    pooled_class_sep_all_df = concat_save(all_pooled_class_sep_tables, "class_distribution_separation__pooled__all_models.csv")

    split_validation_all_df = concat_save(all_split_validation_tables, "feature_split_stability_check__all_models.csv")
    split_stability_agg_df = aggregate_split_stability(all_split_validation_tables) if all_split_validation_tables else pd.DataFrame()
    write_df_or_empty(split_stability_agg_df, os.path.join(OUT_DIR, "feature_split_stability_check__aggregated.csv"))

    raw_missing_train_agg_df = aggregate_missingness_across_models(all_raw_missingness_train_tables) if all_raw_missingness_train_tables else pd.DataFrame()
    raw_missing_test_agg_df = aggregate_missingness_across_models(all_raw_missingness_test_tables) if all_raw_missingness_test_tables else pd.DataFrame()
    raw_missing_pooled_agg_df = aggregate_missingness_across_models(all_raw_missingness_pooled_tables) if all_raw_missingness_pooled_tables else pd.DataFrame()
    write_df_or_empty(concat_save(all_raw_missingness_train_tables, "raw_missingness_feature_stats__train_raw__all_models.csv"), os.path.join(OUT_DIR, "raw_missingness_feature_stats__train_raw__all_models.csv"))
    write_df_or_empty(concat_save(all_raw_missingness_test_tables, "raw_missingness_feature_stats__test_raw__all_models.csv"), os.path.join(OUT_DIR, "raw_missingness_feature_stats__test_raw__all_models.csv"))
    write_df_or_empty(concat_save(all_raw_missingness_pooled_tables, "raw_missingness_feature_stats__pooled_raw__all_models.csv"), os.path.join(OUT_DIR, "raw_missingness_feature_stats__pooled_raw__all_models.csv"))
    write_df_or_empty(raw_missing_train_agg_df, os.path.join(OUT_DIR, "raw_missingness_feature_stats__train_raw__aggregated.csv"))
    write_df_or_empty(raw_missing_test_agg_df, os.path.join(OUT_DIR, "raw_missingness_feature_stats__test_raw__aggregated.csv"))
    write_df_or_empty(raw_missing_pooled_agg_df, os.path.join(OUT_DIR, "raw_missingness_feature_stats__pooled_raw__aggregated.csv"))

    pair_rule_test_agg_df = aggregate_pair_rules_across_models(all_pair_rule_test_tables) if all_pair_rule_test_tables else pd.DataFrame()
    pair_rule_pooled_agg_df = aggregate_pair_rules_across_models(all_pair_rule_pooled_tables) if all_pair_rule_pooled_tables else pd.DataFrame()
    concat_save(all_pair_rule_test_tables, "pair_rule_candidates__test__all_models.csv")
    concat_save(all_pair_rule_pooled_tables, "pair_rule_candidates__pooled__all_models.csv")
    write_df_or_empty(pair_rule_test_agg_df, os.path.join(OUT_DIR, "pair_rule_candidates__test__aggregated.csv"))
    write_df_or_empty(pair_rule_pooled_agg_df, os.path.join(OUT_DIR, "pair_rule_candidates__pooled__aggregated.csv"))

    stability_df = aggregate_feature_stability(per_model_feature_tables=per_model_top_feature_tables_for_stability, top_n=TOP_N_PER_MODEL_FOR_STABILITY)
    write_df_or_empty(stability_df, os.path.join(OUT_DIR, "feature_stability_across_models.csv"))
    candidate_summary_df = build_candidate_feature_summary(stability_df=stability_df, model_level_tables=model_level_candidate_rows)
    write_df_or_empty(candidate_summary_df, os.path.join(OUT_DIR, "candidate_feature_summary__train_oof.csv"))

    test_stability_df = aggregate_feature_stability(per_model_feature_tables=per_model_test_top_feature_tables_for_stability, top_n=TOP_N_PER_MODEL_FOR_STABILITY)
    write_df_or_empty(test_stability_df, os.path.join(OUT_DIR, "feature_stability_across_models__test.csv"))
    test_candidate_summary_df = build_candidate_feature_summary(stability_df=test_stability_df, model_level_tables=test_model_level_candidate_rows) if test_model_level_candidate_rows else pd.DataFrame()
    write_df_or_empty(test_candidate_summary_df, os.path.join(OUT_DIR, "candidate_feature_summary__test.csv"))

    pooled_stability_df = aggregate_feature_stability(per_model_feature_tables=per_model_pooled_top_feature_tables_for_stability, top_n=TOP_N_PER_MODEL_FOR_STABILITY)
    write_df_or_empty(pooled_stability_df, os.path.join(OUT_DIR, "feature_stability_across_models__pooled.csv"))
    pooled_candidate_summary_df = build_candidate_feature_summary(stability_df=pooled_stability_df, model_level_tables=pooled_model_level_candidate_rows) if pooled_model_level_candidate_rows else pd.DataFrame()
    write_df_or_empty(pooled_candidate_summary_df, os.path.join(OUT_DIR, "pooled_candidate_feature_summary.csv"))

    missingness_agg_df = aggregate_missingness_across_models(all_missingness_tables) if all_missingness_tables else pd.DataFrame()
    class_sep_agg_df = aggregate_class_distribution_across_models(all_class_sep_tables) if all_class_sep_tables else pd.DataFrame()
    test_class_sep_agg_df = aggregate_class_distribution_across_models(all_test_class_sep_tables) if all_test_class_sep_tables else pd.DataFrame()
    pooled_class_sep_agg_df = aggregate_class_distribution_across_models(all_pooled_class_sep_tables) if all_pooled_class_sep_tables else pd.DataFrame()
    pair_rule_agg_df = aggregate_pair_rules_across_models(all_pair_rule_tables) if all_pair_rule_tables else pd.DataFrame()

    write_df_or_empty(missingness_agg_df, os.path.join(OUT_DIR, "missingness_feature_stats__aggregated.csv"))
    write_df_or_empty(class_sep_agg_df, os.path.join(OUT_DIR, "class_distribution_separation__aggregated.csv"))
    write_df_or_empty(test_class_sep_agg_df, os.path.join(OUT_DIR, "class_distribution_separation__test__aggregated.csv"))
    write_df_or_empty(pooled_class_sep_agg_df, os.path.join(OUT_DIR, "class_distribution_separation__pooled__aggregated.csv"))
    write_df_or_empty(pair_rule_agg_df, os.path.join(OUT_DIR, "pair_rule_candidates__aggregated.csv"))

    if "Base" in pooled_feature_space_cache:
        corr_base_df = pooled_feature_space_cache["Base"]
        base_y = pooled_label_cache.get("Base")
    elif "Base" in feature_train_cache:
        corr_base_df = feature_train_cache["Base"][0]
        base_y = label_cache.get("Base", (None, None))[0]
    elif pooled_feature_space_cache:
        first_key = next(iter(pooled_feature_space_cache.keys()))
        corr_base_df = pooled_feature_space_cache[first_key]
        base_y = pooled_label_cache.get(first_key)
    else:
        corr_base_df = pd.DataFrame()
        base_y = None

    final_candidate_source_df = test_candidate_summary_df if test_candidate_summary_df is not None and not test_candidate_summary_df.empty else candidate_summary_df
    final_candidate_source_df, redundancy_df = maybe_merge_redundancy(final_candidate_source_df, corr_base_df)
    write_df_or_empty(final_candidate_source_df, os.path.join(OUT_DIR, "candidate_feature_summary__final.csv"))

    primary_stats_df = all_test_stats_all_df if all_test_stats_all_df is not None and not all_test_stats_all_df.empty else all_stats_all_df
    primary_class_sep_agg_df = test_class_sep_agg_df if test_class_sep_agg_df is not None and not test_class_sep_agg_df.empty else class_sep_agg_df

    primary_anchor_df = build_primary_anchor_summary(
        candidate_summary_df=final_candidate_source_df,
        all_stats_cmp_df=primary_stats_df,
        class_sep_agg_df=primary_class_sep_agg_df,
    )
    write_df_or_empty(primary_anchor_df, os.path.join(OUT_DIR, "primary_monitor_anchor_summary.csv"))

    reco_consistency_df = aggregate_recommendation_consistency(all_test_model_feature_recommendation_tables) if all_test_model_feature_recommendation_tables else aggregate_recommendation_consistency(all_model_feature_recommendation_tables)
    write_df_or_empty(reco_consistency_df, os.path.join(OUT_DIR, "recommendation_consistency__final.csv"))

    validated_primary_anchor_df = build_validated_primary_anchor_summary(
        primary_anchor_df=primary_anchor_df,
        split_stability_agg_df=split_stability_agg_df,
        reco_consistency_df=reco_consistency_df,
    )
    write_df_or_empty(validated_primary_anchor_df, os.path.join(OUT_DIR, "primary_monitor_anchor_summary__validated.csv"))

    final_anchor_features = validated_primary_anchor_df.loc[
        validated_primary_anchor_df["final_anchor_recommendation"] == "primary_anchor", "feature"
    ].tolist() if validated_primary_anchor_df is not None and not validated_primary_anchor_df.empty else []

    corr_companion_df = build_priority_feature_corr_companions(
        X_df=corr_base_df,
        anchor_features=final_anchor_features,
        abs_corr_threshold=HIGH_CORR_COMPANION_THRESHOLD,
    )
    if corr_companion_df is not None and not corr_companion_df.empty and final_candidate_source_df is not None and not final_candidate_source_df.empty:
        companion_meta = final_candidate_source_df[["feature", "mean_tp_mean_positive_shap", "mean_tp_vs_tn_smd", "recommendation"]].rename(columns={
            "feature": "companion_feature",
            "mean_tp_mean_positive_shap": "companion_mean_tp_mean_positive_shap",
            "mean_tp_vs_tn_smd": "companion_mean_tp_vs_tn_smd",
            "recommendation": "companion_recommendation",
        })
        corr_companion_df = corr_companion_df.merge(companion_meta, on="companion_feature", how="left")
    write_df_or_empty(corr_companion_df, os.path.join(OUT_DIR, "priority_feature_corr_companions__final.csv"))

    baseline_stats_df = build_baseline_normal_stats(corr_base_df, base_y)
    write_df_or_empty(baseline_stats_df, os.path.join(OUT_DIR, "baseline_normal_stats_for_spc__final.csv"))

    missingness_final_df = raw_missing_test_agg_df if raw_missing_test_agg_df is not None and not raw_missing_test_agg_df.empty else (raw_missing_pooled_agg_df if raw_missing_pooled_agg_df is not None and not raw_missing_pooled_agg_df.empty else raw_missing_train_agg_df)
    pair_final_df = pair_rule_test_agg_df if pair_rule_test_agg_df is not None and not pair_rule_test_agg_df.empty else pair_rule_pooled_agg_df

    validated_for_pack = validated_primary_anchor_df.copy() if validated_primary_anchor_df is not None and not validated_primary_anchor_df.empty else pd.DataFrame()
    if validated_for_pack is not None and not validated_for_pack.empty:
        validated_for_pack["anchor_recommendation"] = validated_for_pack["final_anchor_recommendation"]
        validated_for_pack["overall_anchor_score"] = validated_for_pack["validated_anchor_score"]

    spc_prep_df = build_spc_monitoring_prep_pack(
        primary_anchor_df=validated_for_pack,
        baseline_stats_df=baseline_stats_df,
        missingness_agg_df=missingness_final_df,
        pair_agg_df=pair_final_df,
        corr_companion_df=corr_companion_df,
        redundancy_df=redundancy_df,
    )
    write_df_or_empty(spc_prep_df, os.path.join(OUT_DIR, "spc_monitoring_prep_pack__final.csv"))

    tp_like_source_rank_df = aggregate_cross_model_rank_table(prefer_non_empty(all_test_tp_df, all_tp_df))
    fp_like_source_rank_df = aggregate_cross_model_rank_table(prefer_non_empty(all_test_fp_df, all_fp_df))
    holdout_source_df = prefer_non_empty(test_candidate_summary_df, final_candidate_source_df, candidate_summary_df)
    split_source_df = split_stability_agg_df.copy() if split_stability_agg_df is not None else pd.DataFrame()

    missingness_source_df = prefer_non_empty(raw_missing_test_agg_df, missingness_final_df, raw_missing_train_agg_df, missingness_agg_df)
    interaction_source_df = prefer_non_empty(pair_rule_test_agg_df, pair_final_df, pair_rule_agg_df)
    distribution_source_df = prefer_non_empty(test_class_sep_agg_df, primary_class_sep_agg_df, class_sep_agg_df)

    main_signal_output_df, auxiliary_diagnostic_output_df = build_grouped_output_lists(
        tp_like_rank_df=tp_like_source_rank_df,
        fp_like_rank_df=fp_like_source_rank_df,
        holdout_candidate_df=holdout_source_df,
        split_stable_df=split_source_df,
        missingness_df=missingness_source_df,
        interaction_df=interaction_source_df,
        distribution_df=distribution_source_df,
        top_n=OUTPUT_TOP_N,
    )
    write_df_or_empty(main_signal_output_df, os.path.join(OUT_DIR, "spc_list__main_signal.csv"))
    write_df_or_empty(auxiliary_diagnostic_output_df, os.path.join(OUT_DIR, "spc_list__auxiliary_diagnostic.csv"))

    final_deployment_output_df = build_tiered_final_spc_list(spc_prep_df=spc_prep_df, top_n=OUTPUT_TOP_N)
    write_df_or_empty(final_deployment_output_df, os.path.join(OUT_DIR, "spc_list__final_deployment.csv"))

    final_monitor_candidates_df = final_deployment_output_df.copy() if final_deployment_output_df is not None and not final_deployment_output_df.empty else pd.DataFrame()
    write_df_or_empty(final_monitor_candidates_df, os.path.join(OUT_DIR, "final_monitor_candidates.csv"))

    manifest = {
        "DETAIL_CSV_PATH": DETAIL_CSV_PATH,
        "ARTIFACT_PATH": ARTIFACT_PATH,
        "TOP_K_MODELS": TOP_K_MODELS,
        "BACKGROUND_SIZE": BACKGROUND_SIZE,
        "POOLED_BACKGROUND_SIZE": POOLED_BACKGROUND_SIZE,
        "MAX_EXPLAIN_ROWS_PER_GROUP": MAX_EXPLAIN_ROWS_PER_GROUP,
        "MAX_EVALS_CAP": MAX_EVALS_CAP,
        "TOP_N_PER_MODEL_FOR_STABILITY": TOP_N_PER_MODEL_FOR_STABILITY,
        "CORR_THRESHOLD": CORR_THRESHOLD,
        "HIGH_CORR_COMPANION_THRESHOLD": HIGH_CORR_COMPANION_THRESHOLD,
        "OOF_N_SPLITS": OOF_N_SPLITS,
        "MIN_FP_EVIDENCE_ROWS": MIN_FP_EVIDENCE_ROWS,
        "PAIR_TOP_FEATURES": PAIR_TOP_FEATURES,
        "PAIR_MIN_SUPPORT": PAIR_MIN_SUPPORT,
        "PAIR_MIN_LIFT": PAIR_MIN_LIFT,
        "SPLIT_VALIDATION_TOP_FEATURES": SPLIT_VALIDATION_TOP_FEATURES,
        "SPLIT_VALIDATION_MIN_SCORE": SPLIT_VALIDATION_MIN_SCORE,
        "SPLIT_VALIDATION_MIN_PASS_RATE": SPLIT_VALIDATION_MIN_PASS_RATE,
        "PRIORITY_TP_FP_RATIO_MIN": PRIORITY_TP_FP_RATIO_MIN,
        "PRIORITY_TOP_N": PRIORITY_TOP_N,
        "NORMAL_HIGH_QUANTILE": NORMAL_HIGH_QUANTILE,
        "NORMAL_LOW_QUANTILE": NORMAL_LOW_QUANTILE,
        "TRAIN_CSV_MAP": TRAIN_CSV_MAP,
        "TEST_CSV_MAP": TEST_CSV_MAP,
        "LABEL_CSV_CANDIDATES": LABEL_CSV_CANDIDATES,
        "TEST_LABEL_CSV_CANDIDATES": TEST_LABEL_CSV_CANDIDATES,
        "RAW_TRAIN_CSV_PATH": RAW_TRAIN_CSV_PATH,
        "RAW_TEST_CSV_PATH": RAW_TEST_CSV_PATH,
        "OUTPUT_TOP_N": OUTPUT_TOP_N,
    }
    with open(os.path.join(OUT_DIR, "run_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 100)
    print("Saved outputs")
    print("=" * 100)
    print("- top_models_for_feature_attribution.csv")
    print("- train-side outputs now use OOF grouping when available")
    print("- test holdout SHAP / stats / recommendation outputs added")
    print("- pooled outputs retained as auxiliary only")
    print("- spc_list__main_signal.csv")
    print("- spc_list__auxiliary_diagnostic.csv")
    print("- spc_list__final_deployment.csv (Tier A first, then Tier B backfill to top 10)")
    print("- final_monitor_candidates.csv now mirrors spc_list__final_deployment.csv")
    print("- run_manifest.json")
    print("\nDone.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run SHAP/root-cause feature attribution for the SECOM project."
    )
    parser.add_argument(
        "--processed-dir",
        default="data/processed",
        help="Directory containing processed training CSV files from 02_preprocess.py.",
    )
    parser.add_argument(
        "--training-output-dir",
        default="outputs/training",
        help="Directory containing all_model_evaluation_detail.csv from 03_train_models.py.",
    )
    parser.add_argument(
        "--model-dir",
        default="models",
        help="Directory containing all_best_models_and_thresholds.joblib from 03_train_models.py.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/shap",
        help="Directory to save SHAP/root-cause outputs.",
    )
    parser.add_argument(
        "--top-k-models",
        type=int,
        default=10,
        help="Number of top models to explain. Default: 10.",
    )

    args = parser.parse_args()

    configure_project_paths(
        processed_dir=args.processed_dir,
        training_output_dir=args.training_output_dir,
        model_dir=args.model_dir,
        shap_output_dir=args.output_dir,
    )
    TOP_K_MODELS = args.top_k_models

    main()
