"""
03_train_models.py

Training script converted from the original training notebook/code.

This version keeps the original model/search/evaluation logic, but reads processed
datasets from data/processed/ and saves outputs to outputs/training/ and models/.

Run:
    python scripts/03_train_models.py

Optional:
    python scripts/03_train_models.py --processed-dir data/processed --output-dir outputs/training --model-dir models
"""

# =========================================================
# Imports
# =========================================================
import json
import os
import time
import joblib
import warnings
import subprocess
import argparse
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import (
    StratifiedKFold,
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    IsolationForest,
)
from sklearn.svm import SVC
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier


# =========================================================
# Warning control
# =========================================================
warnings.filterwarnings(
    "ignore",
    message=r".*'penalty' was deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*Falling back to prediction using DMatrix due to mismatched devices.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*X does not have valid feature names.*",
    category=UserWarning,
)



# =========================================================
# Script arguments
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train SECOM models from processed datasets.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"), help="Input directory from 02_preprocess.py")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/training"), help="Output directory for evaluation results")
    parser.add_argument("--model-dir", type=Path, default=Path("models"), help="Output directory for model artifacts")
    parser.add_argument("--pipeline-cache-dir", type=Path, default=Path("pipeline_cache"), help="Pipeline cache directory")
    return parser.parse_args()

ARGS = parse_args()
ARGS.output_dir.mkdir(parents=True, exist_ok=True)
ARGS.model_dir.mkdir(parents=True, exist_ok=True)
ARGS.pipeline_cache_dir.mkdir(parents=True, exist_ok=True)

# =========================================================
# Fast I/O / dtype config
# =========================================================
FORCE_FLOAT64_NUMERIC = True
PREFER_FAST_FILE_FORMATS = True
PIPELINE_CACHE_DIR = str(ARGS.pipeline_cache_dir)
PIPELINE_CACHE = joblib.Memory(location=PIPELINE_CACHE_DIR, verbose=0)
CPU_COUNT = os.cpu_count() or 1
SEARCH_N_JOBS = max(1, CPU_COUNT - 1)


def optimize_feature_dtypes(df):
    if not FORCE_FLOAT64_NUMERIC:
        return df
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        df = df.copy()
        df[numeric_cols] = df[numeric_cols].astype(np.float64)
    return df


def smart_read_dataframe(base_name):
    base_name = str(base_name)
    candidate_paths = []
    if PREFER_FAST_FILE_FORMATS:
        candidate_paths.extend([
            f"{base_name}.parquet",
            f"{base_name}.feather",
        ])
    candidate_paths.append(f"{base_name}.csv")

    for path in candidate_paths:
        if os.path.exists(path):
            if path.endswith(".parquet"):
                df = pd.read_parquet(path)
            elif path.endswith(".feather"):
                df = pd.read_feather(path)
            else:
                df = pd.read_csv(path, encoding="utf-8")
            return optimize_feature_dtypes(df)

    raise FileNotFoundError(f"No file found for base name: {base_name}")


def smart_read_series(base_name):
    base_name = str(base_name)
    candidate_paths = []
    if PREFER_FAST_FILE_FORMATS:
        candidate_paths.extend([
            f"{base_name}.parquet",
            f"{base_name}.feather",
        ])
    candidate_paths.append(f"{base_name}.csv")

    for path in candidate_paths:
        if os.path.exists(path):
            if path.endswith(".parquet"):
                obj = pd.read_parquet(path)
            elif path.endswith(".feather"):
                obj = pd.read_feather(path)
            else:
                obj = pd.read_csv(path, encoding="utf-8")

            if isinstance(obj, pd.DataFrame):
                series = obj.squeeze("columns")
            else:
                series = pd.Series(obj)

            if pd.api.types.is_numeric_dtype(series):
                series = series.astype(np.float64)
            return series

    raise FileNotFoundError(f"No file found for base name: {base_name}")


# =========================================================
# Read data
# =========================================================
X_train_base = smart_read_dataframe(ARGS.processed_dir / "X_train_base")
X_train_rferf = smart_read_dataframe(ARGS.processed_dir / "X_train_rferf")
X_train_rfexgb = smart_read_dataframe(ARGS.processed_dir / "X_train_rfexgb")
X_train_anova = smart_read_dataframe(ARGS.processed_dir / "X_train_anova")

X_test_base = smart_read_dataframe(ARGS.processed_dir / "X_test_base")
X_test_rferf = smart_read_dataframe(ARGS.processed_dir / "X_test_rferf")
X_test_rfexgb = smart_read_dataframe(ARGS.processed_dir / "X_test_rfexgb")
X_test_anova = smart_read_dataframe(ARGS.processed_dir / "X_test_anova")

y_train = smart_read_series(ARGS.processed_dir / "y_train")
y_test = smart_read_series(ARGS.processed_dir / "y_test")
y_train_base = y_train.copy()
y_test_base = y_test.copy()


# =========================================================
# Global config
# =========================================================
RANDOM_STATE = 666
SCORING = "average_precision"
POS_LABEL = 1
NEG_LABEL = 0
TYPE1_COST_WEIGHT = 1
TYPE2_COST_WEIGHT = 10

N_SPLITS = 5

BALANCE_METHODS = [None, "class_weight", "SMOTE", "ROSE"]
SAMPLER_BALANCE_METHODS = ["SMOTE", "ROSE"]
SAMPLING_STRATEGY_CANDIDATES = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
CLASS_WEIGHT_CANDIDATES = [{0: 1, 1: w} for w in [1, 2, 3, 5, 8, 10]]
XGB_SCALE_POS_WEIGHT_CANDIDATES = [1, 2, 3, 5, 8, 10]

N_ITER_SUPERVISED = 25
N_ITER_ANOMALY = 12
N_ITER_VOTING = 30

# progress / logging
SEARCH_VERBOSE = 0
SHOW_TIMESTAMP = False

FEATURE_SETS = {
    "Base": {
        "X_train": X_train_base,
        "X_test": X_test_base,
    },
    "RFERF": {
        "X_train": X_train_rferf,
        "X_test": X_test_rferf,
    },
    "RFEXGB": {
        "X_train": X_train_rfexgb,
        "X_test": X_test_rfexgb,
    },
    "ANOVA": {
        "X_train": X_train_anova,
        "X_test": X_test_anova,
    },
}

FLOW_CONFIG = {
    "Base": {
        "fixed_lr_models": [],
        "supervised_with_resampling": [],
        "supervised_no_resampling": [],
        "run_voting": False,
        "anomaly_models": [],
    },
    "RFERF": {
        "fixed_lr_models": ["LR"],
        "supervised_with_resampling": ["RF"],
        "supervised_no_resampling": ["BalancedRF"],
        "run_voting": False,
        "anomaly_models": [],
    },
    "RFEXGB": {
        "fixed_lr_models": ["LR"],
        "supervised_with_resampling": ["XGB", "HistGB"],
        "supervised_no_resampling": [],
        "run_voting": False,
        "anomaly_models": [],
    },
    "ANOVA": {
        "fixed_lr_models": ["LR"],
        "supervised_with_resampling": ["SVM"],
        "supervised_no_resampling": [],
        "run_voting": False,
        "anomaly_models": ["IsolationForest", "LOF"],
    },
}

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

BALANCE_TUNABLE_MODEL_POOL = ["LR", "RF", "XGB", "SVM", "HistGB"]
NO_RESAMPLING_MODEL_POOL = ["BalancedRF"]
VOTING_COMPONENT_MODELS = ["RF", "XGB", "HistGB", "BalancedRF"]
VOTING_COMPONENT_FEATURE_SETS = {
    "RF": "RFERF",
    "BalancedRF": "RFERF",
    "XGB": "RFEXGB",
    "HistGB": "RFEXGB",
}

FUSION_SCORE_MATRIX_CACHE = {}


# =========================================================
# Logging helpers
# =========================================================
def now_str():
    return ""


def format_seconds(sec):
    sec = float(sec)
    if sec < 60:
        return f"{sec:.1f}s"
    minutes = int(sec // 60)
    seconds = sec % 60
    if minutes < 60:
        return f"{minutes}m {seconds:.1f}s"
    hours = int(minutes // 60)
    minutes = minutes % 60
    return f"{hours}h {minutes}m {seconds:.1f}s"


def log(msg, line_char="-", width=100):
    print("\n" + line_char * width)
    print(str(msg))
    print(line_char * width)


# =========================================================
# Utils
# =========================================================
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


def detect_xgb_device():
    try:
        subprocess.run(
            ["nvidia-smi"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print("[Info] XGBoost device = CUDA")
        return {"tree_method": "hist", "device": "cuda"}
    except Exception:
        print("[Info] GPU not found. XGBoost device = CPU")
        return {"tree_method": "hist", "device": "cpu"}


XGB_DEVICE_KWARGS = detect_xgb_device()


def get_search_n_jobs(model_name):
    if model_name == "XGB" and XGB_DEVICE_KWARGS.get("device") == "cuda":
        return 1
    return SEARCH_N_JOBS


def build_unfitted_best_estimator(estimator, best_params):
    return clone(estimator).set_params(**best_params)


def get_contamination_candidates(y):
    y = to_1d_numpy(y)
    pos_rate = float(np.mean(y == POS_LABEL))
    base = np.clip(pos_rate, 0.01, 0.2)
    cands = sorted(
        set(
            float(np.clip(v, 0.005, 0.3))
            for v in [base / 2.0, base, base * 1.5]
        )
    )
    return cands


CONTAMINATION_CANDS = get_contamination_candidates(y_train)

CALIB_TRAIN_IDX, CALIB_VAL_IDX = train_test_split(
    np.arange(len(y_train)),
    test_size=0.2,
    stratify=to_1d_numpy(y_train),
    random_state=RANDOM_STATE,
)


def get_balance_method_label(balance_method):
    return "NoResample" if balance_method is None else str(balance_method)


def build_resampler(balance_method):
    if balance_method is None or balance_method == "class_weight":
        return "passthrough"

    if balance_method == "ROSE":
        return RandomOverSampler(
            sampling_strategy=1.0,
            shrinkage=0.15,
            random_state=RANDOM_STATE,
        )

    if balance_method == "SMOTE":
        return SMOTE(
            sampling_strategy=1.0,
            random_state=RANDOM_STATE,
            k_neighbors=5,
        )

    raise ValueError(f"Unknown balance method: {balance_method}")


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


def get_total_candidates(param_distributions):
    total = 1
    for values in param_distributions.values():
        if hasattr(values, "rvs"):
            return None
        total *= len(values)
    return total


def get_effective_n_iter(param_distributions, requested_n_iter):
    total = get_total_candidates(param_distributions)
    if total is None:
        return requested_n_iter
    return min(requested_n_iter, total)


def sample_random_candidates(candidates, n_iter, random_state):
    if n_iter >= len(candidates):
        return list(candidates)
    rng = np.random.RandomState(random_state)
    sampled_idx = rng.choice(len(candidates), size=n_iter, replace=False)
    return [candidates[i] for i in sampled_idx]


def safe_div(a, b):
    return float(a / b) if b != 0 else 0.0


# =========================================================
# Model builders
# =========================================================
def build_supervised_search_space(model_name, balance_method=None):
    if model_name in BALANCE_TUNABLE_MODEL_POOL and balance_method not in BALANCE_METHODS:
        raise ValueError(f"{model_name} requires balance_method in {BALANCE_METHODS}")

    if model_name in NO_RESAMPLING_MODEL_POOL and balance_method is not None:
        raise ValueError(f"{model_name} should not use extra balance search")

    if model_name == "LR":
        estimator = ImbPipeline([
            ("scaler", StandardScaler()),
            ("sampler", build_resampler(balance_method)),
            ("clf", LogisticRegression(
                C=1.0,
                penalty="l2",
                solver="liblinear",
                max_iter=3000,
                random_state=RANDOM_STATE,
                class_weight=None,
            )),
        ], memory=PIPELINE_CACHE)
        param_distributions = {
            "clf__C": [1.0],
            "clf__penalty": ["l2"],
            "clf__solver": ["liblinear"],
            "clf__max_iter": [3000],
        }
        if balance_method == "class_weight":
            param_distributions["clf__class_weight"] = CLASS_WEIGHT_CANDIDATES
        elif balance_method in SAMPLER_BALANCE_METHODS:
            param_distributions["sampler__sampling_strategy"] = SAMPLING_STRATEGY_CANDIDATES
        return estimator, param_distributions

    if model_name == "RF":
        estimator = ImbPipeline([
            ("sampler", build_resampler(balance_method)),
            ("clf", RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=1,
                class_weight=None,
            )),
        ], memory=PIPELINE_CACHE)
        param_distributions = {
            "clf__n_estimators": [400, 200, 600, 800],
            "clf__max_features": ["sqrt", 0.8, 0.5, None],
            "clf__max_depth": [None, 10, 20, 5, 15],
            "clf__min_samples_split": [2, 5, 10, 20],
            "clf__min_samples_leaf": [1, 2, 4, 6],
        }
        if balance_method == "class_weight":
            param_distributions["clf__class_weight"] = CLASS_WEIGHT_CANDIDATES
        elif balance_method in SAMPLER_BALANCE_METHODS:
            param_distributions["sampler__sampling_strategy"] = SAMPLING_STRATEGY_CANDIDATES
        return estimator, param_distributions

    if model_name == "XGB":
        estimator = ImbPipeline([
            ("sampler", build_resampler(balance_method)),
            ("clf", XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=1,
                verbosity=0,
                **XGB_DEVICE_KWARGS,
            )),
        ], memory=PIPELINE_CACHE)
        param_distributions = {
            "clf__n_estimators": [400, 200, 600, 800],
            "clf__learning_rate": [0.05, 0.03, 0.08, 0.1],
            "clf__max_depth": [4, 3, 5, 6],
            "clf__min_child_weight": [1, 2, 3, 5],
            "clf__subsample": [0.8, 0.7, 0.9, 1.0],
            "clf__colsample_bytree": [0.8, 0.7, 0.9, 1.0],
            "clf__reg_alpha": [0.0, 0.1, 0.3],
            "clf__reg_lambda": [1.0, 1.5, 2.0, 3.0],
        }
        if balance_method == "class_weight":
            param_distributions["clf__scale_pos_weight"] = XGB_SCALE_POS_WEIGHT_CANDIDATES
        elif balance_method in SAMPLER_BALANCE_METHODS:
            param_distributions["sampler__sampling_strategy"] = SAMPLING_STRATEGY_CANDIDATES
        return estimator, param_distributions

    if model_name == "SVM":
        estimator = ImbPipeline([
            ("scaler", StandardScaler()),
            ("sampler", build_resampler(balance_method)),
            ("clf", SVC(
                probability=True,
                random_state=RANDOM_STATE,
                class_weight=None,
            )),
        ], memory=PIPELINE_CACHE)
        param_distributions = {
            "clf__C": [1.0, 0.5, 2.0, 5.0, 10.0],
            "clf__kernel": ["rbf"],
            "clf__gamma": ["scale", 0.01, 0.1, 1.0],
        }
        if balance_method == "class_weight":
            param_distributions["clf__class_weight"] = CLASS_WEIGHT_CANDIDATES
        elif balance_method in SAMPLER_BALANCE_METHODS:
            param_distributions["sampler__sampling_strategy"] = SAMPLING_STRATEGY_CANDIDATES
        return estimator, param_distributions

    if model_name == "HistGB":
        estimator = ImbPipeline([
            ("sampler", build_resampler(balance_method)),
            ("clf", SampleWeightClassifier(
                base_estimator=HistGradientBoostingClassifier(
                    random_state=RANDOM_STATE,
                ),
                class_weight=None,
            )),
        ], memory=PIPELINE_CACHE)
        param_distributions = {
            "clf__base_estimator__max_iter": [100, 200, 300],
            "clf__base_estimator__learning_rate": [0.03, 0.05, 0.1],
            "clf__base_estimator__max_depth": [None, 3, 5, 10],
            "clf__base_estimator__min_samples_leaf": [20, 40, 60],
            "clf__base_estimator__l2_regularization": [0.0, 0.1, 1.0],
        }
        if balance_method == "class_weight":
            param_distributions["clf__class_weight"] = CLASS_WEIGHT_CANDIDATES
        elif balance_method in SAMPLER_BALANCE_METHODS:
            param_distributions["sampler__sampling_strategy"] = SAMPLING_STRATEGY_CANDIDATES
        return estimator, param_distributions

    if model_name == "BalancedRF":
        estimator = BalancedRandomForestClassifier(
            random_state=RANDOM_STATE,
            sampling_strategy="all",
            replacement=True,
            bootstrap=False,
            n_jobs=1,
        )
        param_distributions = {
            "n_estimators": [400, 200, 600, 800],
            "max_features": ["sqrt", 0.8, 0.5, None],
            "max_depth": [None, 10, 20, 5, 15],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4, 6],
        }
        return estimator, param_distributions

    raise ValueError(f"Unknown supervised model: {model_name}")


def get_voting_component_keys(balance_label):
    return {
        "RF": f"RFERF_RF_{balance_label}",
        "XGB": f"RFEXGB_XGB_{balance_label}",
        "HistGB": f"RFEXGB_HistGB_{balance_label}",
        "BalancedRF": "RFERF_BalancedRF_NoResample",
    }


def collect_score_fusion_components(all_artifacts, balance_label):
    component_keys = get_voting_component_keys(balance_label)
    missing = [artifact_key for artifact_key in component_keys.values() if artifact_key not in all_artifacts]
    if missing:
        raise ValueError(f"VotingEnsemble missing fitted component artifacts for {balance_label}: {missing}")

    component_artifacts = {}
    for model_name, artifact_key in component_keys.items():
        component_artifacts[model_name] = {
            **all_artifacts[artifact_key],
            "artifact_key": artifact_key,
        }
    return component_artifacts


def get_score_fusion_cache_key(component_artifacts, split):
    artifact_keys = tuple(
        component_artifacts[model_name]["artifact_key"]
        for model_name in VOTING_COMPONENT_MODELS
    )
    return split, artifact_keys


def build_score_matrix_from_fitted_components(component_artifacts, split="train"):
    cache_key = get_score_fusion_cache_key(component_artifacts, split)
    if cache_key in FUSION_SCORE_MATRIX_CACHE:
        return FUSION_SCORE_MATRIX_CACHE[cache_key]

    score_list = []

    for model_name in VOTING_COMPONENT_MODELS:
        feature_set_name = VOTING_COMPONENT_FEATURE_SETS[model_name]
        feature_data = FEATURE_SETS[feature_set_name]
        X_used = feature_data["X_train"] if split == "train" else feature_data["X_test"]

        fitted_model = component_artifacts[model_name]["final_model"]
        scores = get_supervised_scores(fitted_model, X_used)
        score_list.append(np.asarray(scores, dtype=float))

    score_matrix = np.column_stack(score_list)
    FUSION_SCORE_MATRIX_CACHE[cache_key] = score_matrix
    return score_matrix


# =========================================================
# Anomaly scorer / wrapper
# =========================================================
def anomaly_ap_scorer(estimator, X, y):
    y = np.asarray(y).ravel().astype(int)
    scores = np.asarray(estimator.decision_function(X)).ravel()
    return average_precision_score(y, scores)


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


def build_anomaly_search_space(model_name):
    if model_name == "IsolationForest":
        estimator = NormalOnlyAnomalyDetector(
            model_name="IsolationForest",
            random_state=RANDOM_STATE,
        )
        param_distributions = {
            "n_estimators": [300, 100, 500],
            "max_samples": ["auto", 0.7, 1.0],
            "contamination": CONTAMINATION_CANDS,
        }
        return estimator, param_distributions

    if model_name == "LOF":
        estimator = NormalOnlyAnomalyDetector(
            model_name="LOF",
            random_state=RANDOM_STATE,
        )
        param_distributions = {
            "n_neighbors": [20, 10, 35, 50],
            "contamination": CONTAMINATION_CANDS,
        }
        return estimator, param_distributions

    raise ValueError(f"Unknown anomaly model: {model_name}")


# =========================================================
# RandomizedSearchCV
# =========================================================
def run_random_search(
    name,
    estimator,
    param_distributions,
    X,
    y,
    cv,
    scoring,
    requested_n_iter=20,
    search_n_jobs=1,
):
    n_iter = get_effective_n_iter(param_distributions, requested_n_iter)
    start_time = time.time()

    log(f"[RandomizedSearchCV START] {name}", "=")
    print(f"Scoring = {scoring if isinstance(scoring, str) else getattr(scoring, '__name__', 'callable')}")
    print(f"n_iter  = {n_iter}")
    print(f"X shape  = {getattr(X, 'shape', 'N/A')}")
    print(f"y length = {len(y)}")
    print(f"search_n_jobs = {search_n_jobs}")

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        refit=False,
        random_state=RANDOM_STATE,
        n_jobs=search_n_jobs,
        pre_dispatch="2*n_jobs",
        verbose=SEARCH_VERBOSE,
        return_train_score=False,
        error_score="raise",
    )
    search.fit(X, y)

    best_estimator = build_unfitted_best_estimator(estimator, search.best_params_)
    elapsed = time.time() - start_time
    print(f"Best CV score = {search.best_score_:.6f}")
    print(f"Best params   = {search.best_params_}")
    print(f"Elapsed       = {format_seconds(elapsed)}")
    log(f"[RandomizedSearchCV END] {name}", "=")

    return {
        "name": name,
        "best_params": dict(search.best_params_),
        "best_score": float(search.best_score_),
        "best_estimator": best_estimator,
        "search_object": search,
    }


# =========================================================
# Voting weight search (same logic, avoid repeated retraining per weight)
# =========================================================
def get_positive_class_scores_from_proba_or_decision(estimator, X):
    if hasattr(estimator, "predict_proba"):
        return np.asarray(estimator.predict_proba(X)[:, 1], dtype=float)
    if hasattr(estimator, "decision_function"):
        return np.asarray(estimator.decision_function(X), dtype=float)
    raise ValueError("Estimator has neither predict_proba nor decision_function")


def build_oof_scores_for_estimator(estimator, X, y, cv):
    y_np = to_1d_numpy(y)
    oof_scores = np.zeros(len(y_np), dtype=float)

    for train_idx, valid_idx in cv.split(X, y_np):
        est = clone(estimator)
        X_tr = take_rows(X, train_idx)
        y_tr = y_np[train_idx]
        X_val = take_rows(X, valid_idx)

        est.fit(X_tr, y_tr)
        oof_scores[valid_idx] = get_positive_class_scores_from_proba_or_decision(est, X_val)

    return oof_scores


def combine_weighted_scores(score_matrix, weights):
    weights = np.asarray(weights, dtype=float)
    return np.average(score_matrix, axis=1, weights=weights)


def run_score_fusion_weight_search(
    name,
    component_artifacts,
    component_models,
    y,
    requested_n_iter=20,
):
    start_time = time.time()
    all_weight_candidates = list(product([1, 2, 3], repeat=len(component_models)))
    n_iter = min(requested_n_iter, len(all_weight_candidates))
    sampled_weight_candidates = sample_random_candidates(
        all_weight_candidates,
        n_iter=n_iter,
        random_state=RANDOM_STATE,
    )

    log(f"[ScoreFusionWeightSearch START] {name}", "=")
    print(f"n_iter  = {n_iter}")
    print(f"y length = {len(y)}")

    score_matrix = build_score_matrix_from_fitted_components(
        component_artifacts=component_artifacts,
        split="train",
    )
    print(f"score_matrix shape = {score_matrix.shape}")

    y_np = to_1d_numpy(y).astype(int)

    best_weights = None
    best_score = -np.inf
    for weights in sampled_weight_candidates:
        combined_scores = combine_weighted_scores(score_matrix, weights)
        score = average_precision_score(y_np, combined_scores)
        if score > best_score:
            best_score = float(score)
            best_weights = tuple(weights)

    elapsed = time.time() - start_time
    print(f"Best train AP = {best_score:.6f}")
    print(f"Best params   = {{'weights': {best_weights}}}")
    print(f"Elapsed       = {format_seconds(elapsed)}")
    log(f"[ScoreFusionWeightSearch END] {name}", "=")

    return {
        "name": name,
        "best_params": {"weights": best_weights},
        "best_score": float(best_score),
        "component_artifact_keys": {
            model_name: component_artifacts[model_name]["artifact_key"]
            for model_name in component_models
        },
        "search_object": {
            "method": "custom_score_fusion_weight_search",
            "n_iter": n_iter,
            "component_models": list(component_models),
        },
    }


# =========================================================
# Scoring helpers
# =========================================================
def get_supervised_scores(estimator, X):
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(X)
    raise ValueError("Estimator has neither predict_proba nor decision_function")


def get_anomaly_scores(estimator, X):
    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(X)
    if hasattr(estimator, "score_samples"):
        return estimator.score_samples(X)
    pred = estimator.predict(X)
    return np.asarray(pred, dtype=float)


def tune_threshold_by_f1(y_true, scores):
    y_true = to_1d_numpy(y_true)
    thresholds = np.linspace(float(np.min(scores)), float(np.max(scores)), 200)
    best_thr = thresholds[0]
    best_f1 = -1.0

    for thr in thresholds:
        y_pred = (scores >= thr).astype(int)
        cur_f1 = f1_score(y_true, y_pred, zero_division=0)
        if cur_f1 > best_f1:
            best_f1 = float(cur_f1)
            best_thr = float(thr)

    return best_thr, best_f1


def compute_metrics(y_true, scores, threshold):
    y_true = to_1d_numpy(y_true)
    y_pred = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[NEG_LABEL, POS_LABEL],
    ).ravel()

    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)

    try:
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
    except Exception:
        balanced_acc = (recall + specificity) / 2.0

    try:
        mcc = matthews_corrcoef(y_true, y_pred)
    except Exception:
        mcc = 0.0

    out = {
        "roc_auc": roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else np.nan,
        "average_precision": average_precision_score(y_true, scores) if len(np.unique(y_true)) > 1 else np.nan,
        "balanced_accuracy": float(balanced_acc),
        "recall": float(recall),
        "specificity": float(specificity),
        "mcc": float(mcc),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_negative": int(tn),
        "true_positive": int(tp),
        "type1_cost": float(fp * TYPE1_COST_WEIGHT),
        "type2_cost": float(fn * TYPE2_COST_WEIGHT),
        "overall_cost": float(fp * TYPE1_COST_WEIGHT + fn * TYPE2_COST_WEIGHT),
        "predicted_positive": int(np.sum(y_pred)),
        "threshold": float(threshold),
    }

    return out


# =========================================================
# Evaluation wrappers
# =========================================================
def evaluate_score_fusion_search_result(
    search_result,
    model_label,
    balance_method,
    feature_set,
    component_artifacts,
    y_train,
    y_test,
):
    eval_start = time.time()

    train_score_matrix = build_score_matrix_from_fitted_components(
        component_artifacts=component_artifacts,
        split="train",
    )
    test_score_matrix = build_score_matrix_from_fitted_components(
        component_artifacts=component_artifacts,
        split="test",
    )

    weights = search_result["best_params"]["weights"]
    val_scores = combine_weighted_scores(train_score_matrix[CALIB_VAL_IDX], weights)
    best_thr, best_val_f1 = tune_threshold_by_f1(
        take_rows(y_train, CALIB_VAL_IDX),
        val_scores,
    )

    test_scores = combine_weighted_scores(test_score_matrix, weights)
    metrics = compute_metrics(y_test, test_scores, best_thr)

    print(
        f"[Eval] {feature_set}_{model_label}_{balance_method} | "
        f"threshold={best_thr:.6f} | val_best_f1={best_val_f1:.6f} | "
        f"test_f1={metrics['f1_score']:.6f} | AP={metrics['average_precision']:.6f} | "
        f"elapsed={format_seconds(time.time() - eval_start)}"
    )

    row = {
        "feature_set": feature_set,
        "category": "supervised",
        "model": model_label,
        "balance_method": balance_method,
        "cv_ap": search_result["best_score"],
        "val_best_f1": best_val_f1,
        "best_params": json.dumps(search_result["best_params"], ensure_ascii=False),
        **metrics,
    }

    artifact = {
        "search_result": search_result,
        "final_model": {
            "type": "score_fusion_ensemble",
            "weights": weights,
            "component_models": list(VOTING_COMPONENT_MODELS),
            "component_artifact_keys": search_result["component_artifact_keys"],
        },
        "threshold": best_thr,
        "balance_method": balance_method,
        "feature_set": feature_set,
        "model": model_label,
    }
    return row, artifact


def evaluate_supervised_search_result(
    search_result,
    model_label,
    balance_method,
    feature_set,
    X_train,
    y_train,
    X_test,
    y_test,
):
    eval_start = time.time()

    X_tr = take_rows(X_train, CALIB_TRAIN_IDX)
    X_val = take_rows(X_train, CALIB_VAL_IDX)
    y_tr = take_rows(y_train, CALIB_TRAIN_IDX)
    y_val = take_rows(y_train, CALIB_VAL_IDX)

    calib_model = clone(search_result["best_estimator"])
    calib_model.fit(X_tr, y_tr)
    val_scores = get_supervised_scores(calib_model, X_val)
    best_thr, best_val_f1 = tune_threshold_by_f1(y_val, val_scores)

    final_model = clone(search_result["best_estimator"])
    final_model.fit(X_train, y_train)
    test_scores = get_supervised_scores(final_model, X_test)

    metrics = compute_metrics(y_test, test_scores, best_thr)

    print(
        f"[Eval] {feature_set}_{model_label}_{balance_method} | "
        f"threshold={best_thr:.6f} | val_best_f1={best_val_f1:.6f} | "
        f"test_f1={metrics['f1_score']:.6f} | AP={metrics['average_precision']:.6f} | "
        f"elapsed={format_seconds(time.time() - eval_start)}"
    )

    row = {
        "feature_set": feature_set,
        "category": "supervised",
        "model": model_label,
        "balance_method": balance_method,
        "cv_ap": search_result["best_score"],
        "val_best_f1": best_val_f1,
        "best_params": json.dumps(search_result["best_params"], ensure_ascii=False),
        **metrics,
    }

    artifact = {
        "search_result": search_result,
        "final_model": final_model,
        "threshold": best_thr,
        "balance_method": balance_method,
        "feature_set": feature_set,
        "model": model_label,
    }
    return row, artifact


def evaluate_anomaly_search_result(
    search_result,
    model_label,
    feature_set,
    X_train,
    y_train,
    X_test,
    y_test,
):
    eval_start = time.time()

    X_tr = take_rows(X_train, CALIB_TRAIN_IDX)
    X_val = take_rows(X_train, CALIB_VAL_IDX)
    y_tr = take_rows(y_train, CALIB_TRAIN_IDX)
    y_val = take_rows(y_train, CALIB_VAL_IDX)

    calib_model = clone(search_result["best_estimator"])
    calib_model.fit(X_tr, y_tr)
    val_scores = get_anomaly_scores(calib_model, X_val)
    best_thr, best_val_f1 = tune_threshold_by_f1(y_val, val_scores)

    final_model = clone(search_result["best_estimator"])
    final_model.fit(X_train, y_train)
    test_scores = get_anomaly_scores(final_model, X_test)

    metrics = compute_metrics(y_test, test_scores, best_thr)

    print(
        f"[Eval] {feature_set}_{model_label} | "
        f"threshold={best_thr:.6f} | val_best_f1={best_val_f1:.6f} | "
        f"test_f1={metrics['f1_score']:.6f} | AP={metrics['average_precision']:.6f} | "
        f"elapsed={format_seconds(time.time() - eval_start)}"
    )

    row = {
        "feature_set": feature_set,
        "category": "anomaly",
        "model": model_label,
        "balance_method": "N/A",
        "cv_ap": search_result["best_score"],
        "val_best_f1": best_val_f1,
        "best_params": json.dumps(search_result["best_params"], ensure_ascii=False),
        **metrics,
    }

    artifact = {
        "search_result": search_result,
        "final_model": final_model,
        "threshold": best_thr,
        "balance_method": "N/A",
        "feature_set": feature_set,
        "model": model_label,
    }
    return row, artifact


def run_tuned_balance_search_for_model(
    base_model_name,
    model_label,
    feature_set,
    X_train,
    y_train,
    X_test,
    y_test,
):
    rows = []
    artifacts = {}

    for balance_method in BALANCE_METHODS:
        balance_label = get_balance_method_label(balance_method)
        print(f"[Progress] feature_set={feature_set} | model={model_label} | balance={balance_label}")

        estimator, param_distributions = build_supervised_search_space(
            model_name=base_model_name,
            balance_method=balance_method,
        )

        search_result = run_random_search(
            name=f"{feature_set}_{model_label}_{balance_label}",
            estimator=estimator,
            param_distributions=param_distributions,
            X=X_train,
            y=y_train,
            cv=cv,
            scoring=SCORING,
            requested_n_iter=N_ITER_SUPERVISED,
            search_n_jobs=get_search_n_jobs(base_model_name),
        )

        row, artifact = evaluate_supervised_search_result(
            search_result=search_result,
            model_label=model_label,
            balance_method=balance_label,
            feature_set=feature_set,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
        rows.append(row)
        artifacts[f"{feature_set}_{model_label}_{balance_label}"] = artifact

    return rows, artifacts


# =========================================================
# Baseline model
# =========================================================
def run_baseline_lr():
    required_vars = ["X_train_base", "X_test_base", "y_train_base", "y_test_base"]
    missing_vars = [v for v in required_vars if v not in globals()]

    if missing_vars:
        log("[Baseline LR] Skip", "=")
        print(f"Missing variables: {missing_vars}")
        return [], {}

    return run_tuned_balance_search_for_model(
        base_model_name="LR",
        model_label="LR",
        feature_set="Base",
        X_train=X_train_base,
        y_train=y_train_base,
        X_test=X_test_base,
        y_test=y_test_base,
    )


# =========================================================
# Main
# =========================================================
overall_start = time.time()

all_rows = []
all_artifacts = {}

log("Training started", "=")
print(f"Feature sets: {list(FEATURE_SETS.keys())}")
print(f"Supervised n_iter: {N_ITER_SUPERVISED}")
print(f"Anomaly n_iter   : {N_ITER_ANOMALY}")
print(f"Voting n_iter    : {N_ITER_VOTING}")
print(f"SEARCH_N_JOBS    : {SEARCH_N_JOBS}")
print(f"PIPELINE_CACHE   : {PIPELINE_CACHE_DIR}")

baseline_rows, baseline_artifacts = run_baseline_lr()
all_rows.extend(baseline_rows)
all_artifacts.update(baseline_artifacts)

for feature_set_name, feature_data in FEATURE_SETS.items():
    feature_start = time.time()

    flow_cfg = FLOW_CONFIG[feature_set_name]
    X_train_used = feature_data["X_train"]
    X_test_used = feature_data["X_test"]

    log(f"Feature set = {feature_set_name}", "=")
    print(f"Train shape = {X_train_used.shape}")
    print(f"Test shape  = {X_test_used.shape}")

    for model_name in flow_cfg["fixed_lr_models"]:
        print(f"[Progress] feature_set={feature_set_name} | fixed_lr_model={model_name}")
        rows, artifacts = run_tuned_balance_search_for_model(
            base_model_name="LR",
            model_label=model_name,
            feature_set=feature_set_name,
            X_train=X_train_used,
            y_train=y_train,
            X_test=X_test_used,
            y_test=y_test,
        )
        all_rows.extend(rows)
        all_artifacts.update(artifacts)

    if flow_cfg["supervised_with_resampling"]:
        for balance_method in BALANCE_METHODS:
            balance_label = get_balance_method_label(balance_method)

            log(
                f"Feature set = {feature_set_name} | Balance method = {balance_label}",
                "#"
            )

            for model_name in flow_cfg["supervised_with_resampling"]:
                print(f"[Progress] feature_set={feature_set_name} | model={model_name} | balance={balance_label}")

                estimator, param_distributions = build_supervised_search_space(
                    model_name=model_name,
                    balance_method=balance_method,
                )

                search_result = run_random_search(
                    name=f"{feature_set_name}_{model_name}_{balance_label}",
                    estimator=estimator,
                    param_distributions=param_distributions,
                    X=X_train_used,
                    y=y_train,
                    cv=cv,
                    scoring=SCORING,
                    requested_n_iter=N_ITER_SUPERVISED,
                    search_n_jobs=get_search_n_jobs(model_name),
                )

                row, artifact = evaluate_supervised_search_result(
                    search_result=search_result,
                    model_label=model_name,
                    balance_method=balance_label,
                    feature_set=feature_set_name,
                    X_train=X_train_used,
                    y_train=y_train,
                    X_test=X_test_used,
                    y_test=y_test,
                )
                all_rows.append(row)
                all_artifacts[f"{feature_set_name}_{model_name}_{balance_label}"] = artifact

    if flow_cfg["supervised_no_resampling"]:
        log(
            f"Feature set = {feature_set_name} | Balance method = NoResample",
            "#"
        )

        for model_name in flow_cfg["supervised_no_resampling"]:
            print(f"[Progress] feature_set={feature_set_name} | model={model_name} | balance=NoResample")

            estimator, param_distributions = build_supervised_search_space(
                model_name=model_name,
                balance_method=None,
            )

            search_result = run_random_search(
                name=f"{feature_set_name}_{model_name}_NoResample",
                estimator=estimator,
                param_distributions=param_distributions,
                X=X_train_used,
                y=y_train,
                cv=cv,
                scoring=SCORING,
                requested_n_iter=N_ITER_SUPERVISED,
                search_n_jobs=get_search_n_jobs(model_name),
            )

            row, artifact = evaluate_supervised_search_result(
                search_result=search_result,
                model_label=model_name,
                balance_method="NoResample",
                feature_set=feature_set_name,
                X_train=X_train_used,
                y_train=y_train,
                X_test=X_test_used,
                y_test=y_test,
            )
            all_rows.append(row)
            all_artifacts[f"{feature_set_name}_{model_name}_NoResample"] = artifact

    # VotingEnsemble is handled after all component models finish training.

    for model_name in flow_cfg["anomaly_models"]:
        print(f"[Progress] feature_set={feature_set_name} | anomaly_model={model_name}")

        estimator, param_distributions = build_anomaly_search_space(model_name)

        search_result = run_random_search(
            name=f"{feature_set_name}_{model_name}",
            estimator=estimator,
            param_distributions=param_distributions,
            X=X_train_used,
            y=y_train,
            cv=cv,
            scoring=anomaly_ap_scorer,
            requested_n_iter=N_ITER_ANOMALY,
            search_n_jobs=get_search_n_jobs(model_name),
        )

        row, artifact = evaluate_anomaly_search_result(
            search_result=search_result,
            model_label=model_name,
            feature_set=feature_set_name,
            X_train=X_train_used,
            y_train=y_train,
            X_test=X_test_used,
            y_test=y_test,
        )
        all_rows.append(row)
        all_artifacts[f"{feature_set_name}_{model_name}"] = artifact

    print(f"[Feature set done] {feature_set_name} | elapsed={format_seconds(time.time() - feature_start)}")


# =========================================================
# Score-fusion VotingEnsemble
# =========================================================
log("Score-fusion VotingEnsemble", "=")

for balance_method in BALANCE_METHODS:
    balance_label = get_balance_method_label(balance_method)
    print(f"[Progress] feature_set=Fusion | model=VotingEnsemble | balance={balance_label}+NoResample")

    component_artifacts = collect_score_fusion_components(
        all_artifacts=all_artifacts,
        balance_label=balance_label,
    )

    search_result = run_score_fusion_weight_search(
        name=f"Fusion_VotingEnsemble_{balance_label}_plus_NoResample",
        component_artifacts=component_artifacts,
        component_models=VOTING_COMPONENT_MODELS,
        y=y_train,
        requested_n_iter=N_ITER_VOTING,
    )

    row, artifact = evaluate_score_fusion_search_result(
        search_result=search_result,
        model_label="VotingEnsemble",
        balance_method=f"{balance_label}+NoResample",
        feature_set="Fusion",
        component_artifacts=component_artifacts,
        y_train=y_train,
        y_test=y_test,
    )
    all_rows.append(row)
    all_artifacts[
        f"Fusion_VotingEnsemble_{balance_label}_plus_NoResample"
    ] = artifact


# =========================================================
# Summary table
# =========================================================
results_detail_df = pd.DataFrame(all_rows)

results_detail_df = results_detail_df.sort_values(
    by=["f1_score", "average_precision", "balanced_accuracy", "mcc"],
    ascending=[False, False, False, False],
).reset_index(drop=True)

summary_df = results_detail_df[
    [
        "feature_set",
        "category",
        "model",
        "balance_method",
        "cv_ap",
        "average_precision",
        "balanced_accuracy",
        "recall",
        "specificity",
        "mcc",
        "f1_score",
        "false_positive",
        "false_negative",
        "true_negative",
        "true_positive",
        "type1_cost",
        "type2_cost",
        "overall_cost",
    ]
].rename(columns={
    "feature_set": "Feature Set",
    "category": "Category",
    "model": "Model",
    "balance_method": "Balance Method",
    "cv_ap": "CV AP",
    "f1_score": "F1 Score",
    "recall": "Recall",
    "average_precision": "PR curve/AP",
    "balanced_accuracy": "Balanced Accuracy",
    "specificity": "Specificity",
    "mcc": "MCC",
    "false_positive": "Type I Error",
    "false_negative": "Type II Error",
    "true_negative": "True Negative (TN)",
    "true_positive": "True Positive (TP)",
    "type1_cost": "Type 1 Cost",
    "type2_cost": "Type 2 Cost",
    "overall_cost": "Overall Cost",
})

summary_df.insert(0, "Rank", range(1, len(summary_df) + 1))

log("All model evaluation summary", "=")
print(summary_df.to_string(index=False))


# =========================================================
# Save results
# =========================================================
results_detail_df.insert(0, "rank", range(1, len(results_detail_df) + 1))
summary_df.to_csv(ARGS.output_dir / "all_model_evaluation_summary.csv", index=False, encoding="utf-8-sig")
results_detail_df.to_csv(ARGS.output_dir / "all_model_evaluation_detail.csv", index=False, encoding="utf-8-sig")
joblib.dump(all_artifacts, ARGS.model_dir / "all_best_models_and_thresholds.joblib")

best_row = results_detail_df.iloc[0].to_dict()
with open(ARGS.output_dir / "best_model_summary.json", "w", encoding="utf-8") as f:
    json.dump(best_row, f, ensure_ascii=False, indent=2)

training_run_log = {
    "processed_dir": str(ARGS.processed_dir),
    "output_dir": str(ARGS.output_dir),
    "model_dir": str(ARGS.model_dir),
    "pipeline_cache_dir": str(ARGS.pipeline_cache_dir),
    "random_state": RANDOM_STATE,
    "n_splits": N_SPLITS,
    "n_iter_supervised": N_ITER_SUPERVISED,
    "n_iter_anomaly": N_ITER_ANOMALY,
    "n_iter_voting": N_ITER_VOTING,
    "search_n_jobs": SEARCH_N_JOBS,
    "total_elapsed": format_seconds(time.time() - overall_start),
}
with open(ARGS.output_dir / "training_run_log.json", "w", encoding="utf-8") as f:
    json.dump(training_run_log, f, ensure_ascii=False, indent=2)

log("Saved files", "=")
print(f"- {ARGS.output_dir / 'all_model_evaluation_summary.csv'}")
print(f"- {ARGS.output_dir / 'all_model_evaluation_detail.csv'}")
print(f"- {ARGS.model_dir / 'all_best_models_and_thresholds.joblib'}")
print(f"- {ARGS.output_dir / 'best_model_summary.json'}")
print(f"Total elapsed = {format_seconds(time.time() - overall_start)}")
