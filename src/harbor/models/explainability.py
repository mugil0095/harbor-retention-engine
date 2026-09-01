"""SHAP-based explainability for the tree-based advanced churn model.

Kept in its own module because `shap` is a heavier dependency than the
rest of this project needs by default -- only the training pipeline
(which wants both SHAP and permutation importance to cross-check each
other) imports it.

Permutation importance (in `churn_model.py`) answers "how much does the
model's score degrade if I scramble this feature." SHAP answers "how much
does each feature push each individual prediction away from the average."
They can disagree, especially under correlated features, which is exactly
why both are computed and compared rather than trusting either alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline


def _normalize_shap_output(shap_values) -> np.ndarray:
    """Collapses whatever shape `TreeExplainer.shap_values()` returns into a
    single (n_samples, n_features) array of positive-class contributions.

    This exists as its own function -- not inlined -- because the shape
    genuinely varies across shap versions and isn't something this project
    can fully verify: older shap returns a list of one array per class for
    a classifier; some versions return a single 3D (n_samples, n_features,
    n_classes) array for binary classification; the pinned version here
    (>=0.45) returns a plain 2D array directly, so branches below are real
    compatibility handling, not speculative -- but only the 2D case is
    actually exercised end-to-end by this project's own tests, since that's
    the only shape the pinned shap+sklearn combination ever produces. The
    other branches are unit-tested directly against synthetic inputs
    instead, since a real SHAP call can't be made to produce them here.
    """
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # binary classification: positive-class contributions
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]
    return shap_values


def _raw_feature_name(encoded_name: str, categorical_cols: list[str]) -> str:
    """Maps a ColumnTransformer-encoded name back to the raw column it came
    from, e.g. "num__tenure_months" -> "tenure_months" and
    "cat__plan_tier_growth" -> "plan_tier" (one-hot columns collapse back to
    their source categorical column so they're comparable to permutation
    importance, which is computed on raw, un-encoded columns)."""
    name = encoded_name.removeprefix("num__").removeprefix("cat__")
    for col in categorical_cols:
        if name.startswith(col + "_"):
            return col
    return name


def compute_shap_importance(
    pipe: Pipeline,
    df: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str] | None = None,
    max_samples: int = 500,
    top_n: int = 15,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Mean absolute SHAP value per raw feature, for a Pipeline whose final
    step is a tree ensemble (TreeExplainer requires direct access to the
    tree structure, so this does not work through an arbitrary estimator --
    e.g. a CalibratedClassifierCV wrapper -- only through the raw tree
    pipeline). One-hot-encoded categorical columns are aggregated back to
    their source column (summing |SHAP|) so the ranking is directly
    comparable to permutation importance, which operates on raw columns.
    """
    categorical_cols = categorical_cols or []
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]

    sample = df.sample(n=min(max_samples, len(df)), random_state=random_seed)
    X_transformed = pre.transform(sample[feature_cols])
    if hasattr(X_transformed, "toarray"):
        # Defensive, not exercised by this project's own preprocessor: the
        # StandardScaler + OneHotEncoder ColumnTransformer used throughout
        # this codebase produces a dense ndarray in practice (verified
        # directly, not assumed), but a differently configured
        # ColumnTransformer (e.g. sparse_output=True with enough categorical
        # columns) can return a scipy sparse matrix, which shap.TreeExplainer
        # cannot consume directly.
        X_transformed = X_transformed.toarray()
    encoded_names = list(pre.get_feature_names_out())

    explainer = shap.TreeExplainer(clf)
    shap_values = _normalize_shap_output(explainer.shap_values(X_transformed))

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    raw_names = [_raw_feature_name(n, categorical_cols) for n in encoded_names]
    result = (
        pd.DataFrame({"feature": raw_names, "mean_abs_shap": mean_abs_shap})
        .groupby("feature", as_index=False)["mean_abs_shap"]
        .sum()
    )
    return result.sort_values("mean_abs_shap", ascending=False).head(top_n).reset_index(drop=True)
