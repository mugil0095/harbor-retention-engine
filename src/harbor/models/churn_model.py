"""Churn prediction: baseline -> strong classical -> advanced, evaluated with
a time-aware split so no snapshot's features ever inform a prediction on an
earlier snapshot.

- Baseline: logistic regression using only static account attributes
  (plan, industry, size, region, tenure). No behavioral or NLP features.
  This establishes what you get "for free" before doing any feature work.
- Strong classical: logistic regression with the full engineered feature
  set (behavioral trends + ticket/NLP features).
- Advanced: histogram gradient boosting over the full feature set, with a
  small hyperparameter grid selected by rolling-origin cross-validation
  (not a random split) across the training snapshots.

The advanced model is only worth shipping if it beats the strong classical
model by a non-trivial margin on the held-out snapshot -- both are reported
so that claim is checkable, not asserted.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve as _sk_calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

STATIC_FEATURE_COLUMNS = ["plan_tier", "industry", "employee_size_band", "region", "tenure_months", "base_mrr"]
STATIC_CATEGORICAL = ["plan_tier", "industry", "employee_size_band", "region"]
STATIC_NUMERIC = ["tenure_months", "base_mrr"]

LABEL_COLUMN = "churn_within_horizon"


def _build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )


def rolling_origin_splits(snapshot_months: list[int], test_month: int) -> list[tuple[list[int], int]]:
    """Successive folds: train on all snapshots strictly before the fold's
    validation month, validate on that month. Excludes `test_month`, which
    is reserved as the final untouched holdout."""
    train_pool = sorted(m for m in snapshot_months if m != test_month)
    folds = []
    for i in range(1, len(train_pool)):
        folds.append((train_pool[:i], train_pool[i]))
    return folds


def train_baseline_model(train_df: pd.DataFrame, random_seed: int = 42) -> Pipeline:
    pre = _build_preprocessor(STATIC_NUMERIC, STATIC_CATEGORICAL)
    clf = LogisticRegression(max_iter=1000, random_state=random_seed)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(train_df[STATIC_FEATURE_COLUMNS], train_df[LABEL_COLUMN])
    return pipe


def train_classical_model(
    train_df: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str], random_seed: int = 42
) -> Pipeline:
    pre = _build_preprocessor(numeric_cols, categorical_cols)
    clf = LogisticRegression(max_iter=2000, random_state=random_seed, C=1.0)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(train_df[numeric_cols + categorical_cols], train_df[LABEL_COLUMN])
    return pipe


@dataclass
class GridSearchResult:
    best_params: dict
    best_score: float
    all_results: list[dict]


def tune_advanced_model(
    feature_store: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    snapshot_months: list[int],
    test_month: int,
    random_seed: int = 42,
) -> GridSearchResult:
    folds = rolling_origin_splits(snapshot_months, test_month)
    grid = {
        "max_iter": [150, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.03, 0.05, 0.1],
        "l2_regularization": [0.0, 1.0],
    }
    combos = [dict(zip(grid.keys(), values)) for values in product(*grid.values())]

    results = []
    for params in combos:
        # Pool predictions across folds before scoring, rather than averaging
        # a per-fold PR-AUC: at a ~5% base rate each fold has only a handful
        # of positives, so a single fold's PR-AUC is extremely noisy and can
        # dominate the average by chance. Pooling gives one stable estimate
        # over all held-out rows, which is standard practice for imbalanced
        # rolling-origin CV.
        pooled_y: list[np.ndarray] = []
        pooled_proba: list[np.ndarray] = []
        for train_months, val_month in folds:
            train_fold = feature_store[feature_store["snapshot_month"].isin(train_months)]
            val_fold = feature_store[feature_store["snapshot_month"] == val_month]
            if train_fold[LABEL_COLUMN].nunique() < 2 or val_fold[LABEL_COLUMN].nunique() < 2:
                continue
            pre = _build_preprocessor(numeric_cols, categorical_cols)
            clf = HistGradientBoostingClassifier(random_state=random_seed, **params)
            pipe = Pipeline([("pre", pre), ("clf", clf)])
            pipe.fit(train_fold[numeric_cols + categorical_cols], train_fold[LABEL_COLUMN])
            proba = pipe.predict_proba(val_fold[numeric_cols + categorical_cols])[:, 1]
            pooled_y.append(val_fold[LABEL_COLUMN].to_numpy())
            pooled_proba.append(proba)
        if pooled_y:
            y_all = np.concatenate(pooled_y)
            proba_all = np.concatenate(pooled_proba)
            results.append({"params": params, "mean_pr_auc": float(average_precision_score(y_all, proba_all))})

    results.sort(key=lambda r: r["mean_pr_auc"], reverse=True)
    best = results[0]
    return GridSearchResult(best_params=best["params"], best_score=best["mean_pr_auc"], all_results=results)


def train_advanced_model(
    train_df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    best_params: dict,
    random_seed: int = 42,
) -> Pipeline:
    pre = _build_preprocessor(numeric_cols, categorical_cols)
    clf = HistGradientBoostingClassifier(random_state=random_seed, **best_params)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(train_df[numeric_cols + categorical_cols], train_df[LABEL_COLUMN])
    return pipe


def _best_f1_over_thresholds(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, float]:
    """Best achievable F1 across all thresholds implied by the predicted
    scores, with the threshold that achieves it. A fixed 0.5 cutoff is
    meaningless at a ~5% base rate -- every model's predicted probabilities
    stay well under 0.5, so F1@0.5 is trivially 0 for baseline, classical,
    and advanced models alike and says nothing about relative quality."""
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1_scores = np.where(
        (precision + recall) > 0, 2 * precision * recall / (precision + recall + 1e-12), 0.0
    )
    best_idx = int(np.argmax(f1_scores[:-1])) if len(thresholds) > 0 else 0
    best_threshold = float(thresholds[best_idx]) if len(thresholds) > 0 else 0.5
    return float(f1_scores[best_idx]), best_threshold


def evaluate_model(
    pipe: Pipeline, test_df: pd.DataFrame, feature_cols: list[str], label_col: str = LABEL_COLUMN
) -> dict[str, float]:
    proba = pipe.predict_proba(test_df[feature_cols])[:, 1]
    y_true = test_df[label_col].to_numpy()

    decile_cut = np.quantile(proba, 0.9)
    top_decile_mask = proba >= decile_cut
    base_rate = y_true.mean()
    top_decile_rate = y_true[top_decile_mask].mean() if top_decile_mask.sum() > 0 else float("nan")
    best_f1, best_f1_threshold = _best_f1_over_thresholds(y_true, proba)

    return {
        "roc_auc": roc_auc_score(y_true, proba) if len(np.unique(y_true)) > 1 else float("nan"),
        "pr_auc": average_precision_score(y_true, proba),
        "brier_score": brier_score_loss(y_true, proba),
        "best_f1": best_f1,
        "best_f1_threshold": best_f1_threshold,
        "base_rate": float(base_rate),
        "top_decile_churn_rate": float(top_decile_rate),
        "top_decile_lift": float(top_decile_rate / base_rate) if base_rate > 0 else float("nan"),
        "n_samples": len(test_df),
    }


def segment_performance(
    pipe: Pipeline, test_df: pd.DataFrame, feature_cols: list[str], segment_col: str, label_col: str = LABEL_COLUMN
) -> pd.DataFrame:
    rows = []
    for segment_value, group in test_df.groupby(segment_col):
        if group[label_col].nunique() < 2:
            continue
        metrics = evaluate_model(pipe, group, feature_cols, label_col)
        metrics[segment_col] = segment_value
        rows.append(metrics)
    return pd.DataFrame(rows)


def top_feature_importances(
    pipe: Pipeline,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = LABEL_COLUMN,
    top_n: int = 15,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Permutation importance on the held-out set. Works identically for the
    logistic regression and gradient boosting pipelines, unlike coef_ /
    feature_importances_ (HistGradientBoostingClassifier exposes neither),
    and reflects actual predictive contribution rather than model-internal
    weight scale."""
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        pipe,
        test_df[feature_cols],
        test_df[label_col],
        scoring="average_precision",
        n_repeats=10,
        random_state=random_seed,
        n_jobs=-1,
    )
    df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return df.sort_values("importance_mean", ascending=False).head(top_n).reset_index(drop=True)


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    """Weighted average |predicted probability - observed frequency| across
    quantile bins of predicted probability. Quantile bins (not fixed-width
    bins) matter here specifically because predictions cluster in a narrow
    range at a ~5% base rate -- fixed-width bins would leave most of them
    empty and a handful massively overloaded."""
    bin_edges = np.quantile(proba, np.linspace(0, 1, n_bins + 1))
    bin_edges[0] -= 1e-9
    bin_ids = np.digitize(proba, bin_edges[1:-1], right=True)
    n = len(y_true)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(proba[mask].mean() - y_true[mask].mean())
    return float(ece)


def calibration_curve_points(
    y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10
) -> tuple[list[float], list[float]]:
    """Reliability-diagram points: (mean predicted probability, observed
    frequency) per quantile bin. Returns (predicted, observed) so the
    caller can plot predicted vs. observed and compare to the y=x diagonal
    a perfectly calibrated model would trace."""
    observed, predicted = _sk_calibration_curve(y_true, proba, n_bins=n_bins, strategy="quantile")
    return predicted.tolist(), observed.tolist()


def train_calibrated_advanced_model(
    train_df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    best_params: dict,
    random_seed: int = 42,
    cv: int = 3,
) -> CalibratedClassifierCV:
    """Isotonic-calibrated version of the advanced model, fit with internal
    cross-validation (not on an already-fitted model) so the calibration
    mapping isn't learned on the same rows it will be evaluated having seen.
    Compared against the uncalibrated model on Brier score and expected
    calibration error in the training pipeline -- kept only if it actually
    improves calibration, per the same "don't add a technique that doesn't
    pay off" standard applied to every other model in this project."""
    pre = _build_preprocessor(numeric_cols, categorical_cols)
    clf = HistGradientBoostingClassifier(random_state=random_seed, **best_params)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    calibrated = CalibratedClassifierCV(pipe, method="isotonic", cv=cv)
    calibrated.fit(train_df[numeric_cols + categorical_cols], train_df[LABEL_COLUMN])
    return calibrated
