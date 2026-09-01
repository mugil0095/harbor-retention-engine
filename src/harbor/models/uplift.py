"""Uplift / CATE estimation for the retention-offer experiment.

- **Baseline (S-learner)**: one model, treatment flag as an ordinary
  feature. Simple, but tends to under-fit heterogeneous effects because a
  regularized model can shrink the treatment coefficient toward zero when
  treatment isn't the dominant predictor of the outcome.
- **T-learner**: two separate models (treatment-arm, control-arm),
  CATE = predicted control-risk - predicted treatment-risk. Better at
  capturing heterogeneity at the cost of higher variance with small samples.
- **X-learner**: imputes each unit's individual treatment effect using the
  *other* arm's outcome model (a treated unit's effect is estimated against
  the control model's prediction for it, and vice versa), fits a regression
  on each arm's imputed effects, then blends the two via an estimated
  propensity score. Designed specifically to help when the two arms are
  unbalanced in size or composition -- each arm's effect estimate borrows
  statistical strength from the other arm's model, rather than relying
  purely on its own (possibly small) sample the way a T-learner does.

All three are evaluated the same two ways:
1. **Qini curve / AUUC** against the experiment's actual outcomes -- the
   standard uplift-modeling evaluation, usable on real data with no ground
   truth.
2. **Recovered vs. true CATE** -- only possible because the experiment data
   is synthetic with a known data-generating process. This is reported as a
   validation of methodology, not a claim that real-world uplift models can
   be checked this way.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge


@dataclass
class UpliftEvalResult:
    method: str
    auuc: float
    qini_curve: pd.DataFrame
    mean_predicted_cate: float
    correlation_with_true_effect: float | None
    mae_vs_true_effect: float | None


def _feature_matrix(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    return pd.get_dummies(df[feature_cols], drop_first=True)


class _ConstantClassifier:
    """Fallback for an arm whose observed outcomes are a single class (can
    happen with small experiment samples or rare outcomes) -- predicts that
    constant rate for every row rather than raising, since
    LogisticRegression cannot fit with only one class present."""

    def __init__(self, constant_value: int) -> None:
        self.constant_value = constant_value

    def predict_proba(self, X) -> np.ndarray:
        n = len(X)
        p1 = np.full(n, float(self.constant_value))
        return np.column_stack([1 - p1, p1])


BinaryClassifier = LogisticRegression | _ConstantClassifier


def _fit_binary_classifier(X: pd.DataFrame, y: pd.Series, random_seed: int) -> BinaryClassifier:
    if y.nunique() < 2:
        return _ConstantClassifier(constant_value=int(y.iloc[0]))
    model = LogisticRegression(max_iter=1000, random_state=random_seed)
    model.fit(X, y)
    return model


def fit_s_learner(
    df: pd.DataFrame, feature_cols: list[str], treatment_col: str, outcome_col: str, random_seed: int = 42
) -> tuple[BinaryClassifier, pd.DataFrame]:
    X = _feature_matrix(df, feature_cols)
    X = X.assign(_treatment=df[treatment_col].to_numpy())
    model = _fit_binary_classifier(X, df[outcome_col], random_seed)
    return model, X


def predict_cate_s_learner(model: BinaryClassifier, X: pd.DataFrame) -> np.ndarray:
    X_treat = X.copy()
    X_treat["_treatment"] = 1
    X_control = X.copy()
    X_control["_treatment"] = 0
    p_treat = model.predict_proba(X_treat)[:, 1]
    p_control = model.predict_proba(X_control)[:, 1]
    return p_control - p_treat  # positive = treatment reduces churn


def fit_t_learner(
    df: pd.DataFrame, feature_cols: list[str], treatment_col: str, outcome_col: str, random_seed: int = 42
) -> tuple[BinaryClassifier, BinaryClassifier, pd.DataFrame]:
    X_full = _feature_matrix(df, feature_cols)
    treated_mask = df[treatment_col] == 1

    model_treat = _fit_binary_classifier(X_full[treated_mask], df.loc[treated_mask, outcome_col], random_seed)
    model_control = _fit_binary_classifier(X_full[~treated_mask], df.loc[~treated_mask, outcome_col], random_seed)

    return model_treat, model_control, X_full


def predict_cate_t_learner(
    model_treat: BinaryClassifier, model_control: BinaryClassifier, X: pd.DataFrame
) -> np.ndarray:
    p_treat = model_treat.predict_proba(X)[:, 1]
    p_control = model_control.predict_proba(X)[:, 1]
    return p_control - p_treat


def fit_x_learner(
    df: pd.DataFrame, feature_cols: list[str], treatment_col: str, outcome_col: str, random_seed: int = 42
) -> tuple[Ridge, Ridge, LogisticRegression, pd.DataFrame]:
    """X-learner: stage 1 reuses the T-learner's per-arm outcome models;
    stage 2 imputes each unit's individual treatment effect using the
    *other* arm's model, fits a regression on each arm's imputed effects,
    and stage 3 blends the two regressions via an estimated propensity
    score. Imputed effects (`d1`, `d0`) are in raw Y(1)-Y(0) units, i.e.
    negative = treatment reduces churn = good; the final prediction is
    sign-flipped before returning so positive = good, matching the S-/T-
    learner convention used everywhere else in this module.
    """
    model_treat, model_control, X_full = fit_t_learner(df, feature_cols, treatment_col, outcome_col, random_seed)
    treated_mask = (df[treatment_col] == 1).to_numpy()
    y = df[outcome_col].to_numpy().astype(float)

    mu0_on_treated = model_control.predict_proba(X_full[treated_mask])[:, 1]
    imputed_effect_treated = y[treated_mask] - mu0_on_treated

    mu1_on_control = model_treat.predict_proba(X_full[~treated_mask])[:, 1]
    imputed_effect_control = mu1_on_control - y[~treated_mask]

    tau1_model = Ridge(random_state=random_seed).fit(X_full[treated_mask], imputed_effect_treated)
    tau0_model = Ridge(random_state=random_seed).fit(X_full[~treated_mask], imputed_effect_control)

    propensity_model = LogisticRegression(max_iter=1000, random_state=random_seed)
    propensity_model.fit(X_full, df[treatment_col])

    return tau1_model, tau0_model, propensity_model, X_full


def predict_cate_x_learner(
    tau1_model: Ridge, tau0_model: Ridge, propensity_model: LogisticRegression, X: pd.DataFrame
) -> np.ndarray:
    propensity = propensity_model.predict_proba(X)[:, 1]  # P(treatment=1 | X)
    tau1_pred = tau1_model.predict(X)
    tau0_pred = tau0_model.predict(X)
    blended = propensity * tau0_pred + (1 - propensity) * tau1_pred  # raw units, negative = good
    return -blended  # positive = treatment reduces churn


def qini_curve(predicted_cate: np.ndarray, treatment: np.ndarray, outcome: np.ndarray) -> pd.DataFrame:
    """Standard Qini curve: rank accounts by predicted uplift (descending),
    and at each fraction of the population, compare the cumulative
    (observed treatment benefit) against what random targeting would give.
    Outcome here is churn (1 = bad), so benefit = avoided churns among
    treated minus what would be expected under random assignment.
    """
    order = np.argsort(-predicted_cate)
    treatment = treatment[order]
    outcome = outcome[order]
    n = len(outcome)

    n_t_cum = np.cumsum(treatment)
    n_c_cum = np.cumsum(1 - treatment)
    y_t_cum = np.cumsum(outcome * treatment)
    y_c_cum = np.cumsum(outcome * (1 - treatment))

    with np.errstate(divide="ignore", invalid="ignore"):
        qini = y_t_cum - y_c_cum * (n_t_cum / np.where(n_c_cum == 0, np.nan, n_c_cum))
    qini = np.nan_to_num(qini, nan=0.0)
    # qini here counts churns attributable to treatment; flip sign so
    # "higher is better" (treatment reducing churn is a positive outcome)
    qini = -qini

    fraction = np.arange(1, n + 1) / n
    return pd.DataFrame({"fraction": fraction, "qini": qini})


def compute_auuc(qini_df: pd.DataFrame) -> float:
    trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy >=2.0 renamed trapz -> trapezoid
    return float(trapezoid(qini_df["qini"], qini_df["fraction"]))


def evaluate_uplift_method(
    method_name: str,
    predicted_cate: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    true_effect: np.ndarray | None = None,
) -> UpliftEvalResult:
    qini_df = qini_curve(predicted_cate, treatment, outcome)
    auuc = compute_auuc(qini_df)

    corr, mae = None, None
    if true_effect is not None:
        corr = float(np.corrcoef(predicted_cate, -true_effect)[0, 1])
        mae = float(np.mean(np.abs(predicted_cate - (-true_effect))))

    return UpliftEvalResult(
        method=method_name,
        auuc=auuc,
        qini_curve=qini_df,
        mean_predicted_cate=float(np.mean(predicted_cate)),
        correlation_with_true_effect=corr,
        mae_vs_true_effect=mae,
    )
