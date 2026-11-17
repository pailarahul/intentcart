"""IntentCart: reusable pieces of the analysis pipeline.

Everything that is *learned* from data (scaling, one-hot vocabularies, infrequent
category merging, model parameters, decision thresholds) is fitted on the
training split only, or chosen on the validation split. The test split is used
once, at the end, by `run_all.py`.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "online_shoppers_intention.csv"
DATA_SHA256 = "b3055ee355f59134d851d32641183cb4a8b45def7124d2f50442a042f358e0d9"
SEED = 42
TARGET = "Revenue"

# Page-count and duration features: non-negative and heavily right-skewed.
COUNT_DURATION = [
    "Administrative", "Administrative_Duration",
    "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration",
]
RATES = ["BounceRates", "ExitRates"]
PAGE_VALUES = ["PageValues"]
OTHER_NUMERIC = ["SpecialDay"]
BINARY = ["Weekend"]
# Integer codes with no numeric meaning (an OS id of 3 is not "more" than 2).
CATEGORICAL = ["Month", "VisitorType", "OperatingSystems", "Browser", "Region", "TrafficType"]

MONTH_ORDER = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Feature settings = what we assume is known when the prediction is made.
SETTINGS = {
    "A": {
        "name": "All features (end of session)",
        "short": "All features",
        "drop": [],
        "why": "Uses every column. PageValues is computed by Google Analytics from "
               "revenue of sessions that passed through the same pages, so it may "
               "partly encode the outcome. Treated as an optimistic upper bound.",
    },
    "B": {
        "name": "Without PageValues (main setting)",
        "short": "No PageValues",
        "drop": ["PageValues"],
        "why": "Drops the revenue-derived PageValues. Everything left describes "
               "what the visitor did (pages, time, rates, context). This is the "
               "setting our main claim rests on.",
    },
}


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_data(path: Path = DATA_PATH, check_hash: bool = True) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python src/download_data.py` first."
        )
    if check_hash and sha256(path) != DATA_SHA256:
        raise ValueError("Dataset checksum differs from the version used in this project.")
    df = pd.read_csv(path)
    # Booleans arrive as True/False; store target and Weekend as 0/1 ints.
    df[TARGET] = df[TARGET].astype(int)
    df["Weekend"] = df["Weekend"].astype(int)
    for c in ["OperatingSystems", "Browser", "Region", "TrafficType"]:
        df[c] = df[c].astype(str)
    return df


def quality_report(df: pd.DataFrame) -> dict:
    """Checks that do not learn anything, so they are safe on the full data."""
    dup_mask = df.duplicated(keep=False)
    num = df.select_dtypes("number")
    return {
        "n_rows": int(len(df)),
        "n_features": int(df.shape[1] - 1),
        "n_positive": int(df[TARGET].sum()),
        "positive_rate": float(df[TARGET].mean()),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows_extra_copies": int(df.duplicated().sum()),
        "rows_in_duplicate_groups": int(dup_mask.sum()),
        "duplicate_rows_all_nonpurchase": bool((df.loc[dup_mask, TARGET] == 0).all()),
        "duplicate_rows_all_single_page_bounce": bool(
            (df.loc[dup_mask, "ProductRelated_Duration"] == 0).all()
            and (df.loc[dup_mask, "BounceRates"] == 0.2).all()
        ),
        "negative_values": {c: int((num[c] < 0).sum()) for c in num.columns if (num[c] < 0).any()},
        "months_present": [m for m in MONTH_ORDER if m in set(df["Month"])],
        "months_missing": ["Jan", "Apr"],
    }


def make_split(df: pd.DataFrame, seed: int = SEED) -> pd.Series:
    """Stratified 70/15/15 split. Returns a Series of 'train'/'val'/'test'."""
    idx = np.arange(len(df))
    y = df[TARGET].values
    tr, rest = train_test_split(idx, test_size=0.30, stratify=y, random_state=seed)
    va, te = train_test_split(rest, test_size=0.50, stratify=y[rest], random_state=seed)
    split = pd.Series("train", index=df.index, name="split")
    split.iloc[va] = "val"
    split.iloc[te] = "test"
    return split


def features_for(setting: str) -> list[str]:
    drop = set(SETTINGS[setting]["drop"])
    cols = COUNT_DURATION + RATES + PAGE_VALUES + OTHER_NUMERIC + BINARY + CATEGORICAL
    return [c for c in cols if c not in drop]


# --------------------------------------------------------------------------
# Preprocessing (always inside a Pipeline, so it is fitted on train only)
# --------------------------------------------------------------------------
def _onehot() -> OneHotEncoder:
    # Categories seen < 20 times in TRAINING data are merged into one
    # "infrequent" column; categories unseen in training are ignored.
    return OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20,
                         sparse_output=False)


def make_preprocessor(features: list[str], for_linear: bool) -> ColumnTransformer:
    skewed = [c for c in COUNT_DURATION + PAGE_VALUES if c in features]
    plain = [c for c in RATES + OTHER_NUMERIC + BINARY if c in features]
    cats = [c for c in CATEGORICAL if c in features]
    if for_linear:
        # log1p tames the long tails (max ProductRelated_Duration ~ 64,000 s)
        # so a few extreme sessions do not dominate the linear model; then
        # standardise so the L2 penalty treats all features alike.
        skew_pipe = Pipeline([
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ])
        transformers = [("skewed", skew_pipe, skewed), ("plain", StandardScaler(), plain)]
    else:
        # Trees split on thresholds: monotone transforms and scaling change nothing.
        transformers = [("numeric", "passthrough", skewed + plain)]
    transformers.append(("cat", _onehot(), cats))
    return ColumnTransformer(transformers, verbose_feature_names_out=False)


# --------------------------------------------------------------------------
# Models and their (small, validation-tuned) hyper-parameter grids
# --------------------------------------------------------------------------
MODEL_INFO = {
    "majority": "Majority-class baseline",
    "stump": "One-rule baseline (depth-1 tree)",
    "logreg": "Logistic regression",
    "tree": "Decision tree",
    "forest": "Random forest",
    "gboost": "Gradient boosting",
}


def model_grid(name: str) -> list[dict]:
    if name == "majority":
        return [{}]
    if name == "stump":
        return [{}]
    if name == "logreg":
        return [{"C": c, "class_weight": w} for c in [0.01, 0.1, 1.0, 10.0]
                for w in [None, "balanced"]]
    if name == "tree":
        return [{"max_depth": d, "min_samples_leaf": l, "class_weight": w}
                for d in [3, 5, 8, None] for l in [5, 20, 50] for w in [None, "balanced"]]
    if name == "forest":
        return [{"max_depth": d, "min_samples_leaf": l, "max_features": f, "class_weight": w}
                for d in [8, None] for l in [1, 5] for f in ["sqrt", 0.5]
                for w in [None, "balanced_subsample"]]
    if name == "gboost":
        return [{"learning_rate": lr, "max_leaf_nodes": m, "l2_regularization": l2}
                for lr in [0.03, 0.1] for m in [15, 31] for l2 in [0.0, 1.0]]
    raise KeyError(name)


def build_model(name: str, params: dict, features: list[str]) -> Pipeline:
    if name == "majority":
        clf = DummyClassifier(strategy="prior")
    elif name == "stump":
        clf = DecisionTreeClassifier(max_depth=1, random_state=SEED)
    elif name == "logreg":
        clf = LogisticRegression(max_iter=5000, random_state=SEED, **params)
    elif name == "tree":
        clf = DecisionTreeClassifier(random_state=SEED, **params)
    elif name == "forest":
        clf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED, **params)
    elif name == "gboost":
        # Early stopping uses an internal slice of TRAINING data only.
        clf = HistGradientBoostingClassifier(max_iter=500, early_stopping=True,
                                             validation_fraction=0.15,
                                             random_state=SEED, **params)
    else:
        raise KeyError(name)
    return Pipeline([
        ("prep", make_preprocessor(features, for_linear=(name == "logreg"))),
        ("clf", clf),
    ])


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def f1_optimal_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Threshold that maximises F1 on the data given (we pass validation data)."""
    if np.unique(p).size == 1:
        return 0.5
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = np.where(prec[:-1] + rec[:-1] > 0,
                  2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12), 0)
    return float(thr[int(np.argmax(f1))])


def threshold_metrics(y: np.ndarray, p: np.ndarray, t: float) -> dict:
    yhat = (p >= t).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum())
    tn = int(((yhat == 0) & (y == 0)).sum())
    return {
        "threshold": float(t),
        "precision": float(precision_score(y, yhat, zero_division=0)),
        "recall": float(recall_score(y, yhat, zero_division=0)),
        "f1": float(f1_score(y, yhat, zero_division=0)),
        "accuracy": float(accuracy_score(y, yhat)),
        "balanced_accuracy": float(balanced_accuracy_score(y, yhat)),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def ranking_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)) if np.unique(p).size > 1 else 0.5,
        "brier": float(brier_score_loss(y, p)),
    }


def bootstrap_ci(y: np.ndarray, p: np.ndarray, t: float, n_boot: int = 1000,
                 seed: int = SEED) -> dict:
    """Percentile 95% CIs by resampling test sessions (model held fixed)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    stats = {"pr_auc": [], "f1": [], "precision": [], "recall": []}
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        yb, pb = y[i], p[i]
        if yb.sum() == 0:
            continue
        stats["pr_auc"].append(average_precision_score(yb, pb))
        yh = (pb >= t).astype(int)
        stats["f1"].append(f1_score(yb, yh, zero_division=0))
        stats["precision"].append(precision_score(yb, yh, zero_division=0))
        stats["recall"].append(recall_score(yb, yh, zero_division=0))
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            for k, v in stats.items()}


def paired_bootstrap_diff(y, p1, p2, n_boot: int = 1000, seed: int = SEED) -> dict:
    """PR-AUC(model 1) - PR-AUC(model 2) on the same resampled test sessions."""
    rng = np.random.default_rng(seed)
    n = len(y)
    d = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if y[i].sum() == 0:
            continue
        d.append(average_precision_score(y[i], p1[i]) - average_precision_score(y[i], p2[i]))
    d = np.array(d)
    return {"mean": float(d.mean()), "ci95": [float(np.percentile(d, 2.5)),
                                               float(np.percentile(d, 97.5))],
            "share_above_zero": float((d > 0).mean())}
