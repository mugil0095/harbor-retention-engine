"""Behavioral customer segmentation.

Three genuinely different clustering families are compared, not just
K-Means, because they answer different questions:

- **K-Means**: assumes roughly spherical, similarly-sized segments. Fast,
  interpretable centroids, needs `k` chosen in advance (done here via
  silhouette score, not guessed).
- **Agglomerative (hierarchical)**: no assumption of cluster shape/size,
  gives a merge history (useful for deciding how many segments are
  "real"), but is O(n^2) memory -- fine at the account counts here, not at
  millions of rows.
- **HDBSCAN (density-based)**: does not force every account into a
  segment -- accounts that don't belong to any dense group are labeled
  noise (-1), which K-Means and hierarchical clustering cannot express and
  which is often the most business-relevant finding ("this cluster of
  accounts doesn't look like anyone else").

All three are run on the same standardized behavioral feature set so their
segment counts and quality (silhouette score) are directly comparable.
Segment *stability* (do these clusters replicate on a different sample of
the same population?) is checked separately via bootstrap ARI, since a
clustering can look clean on one sample and be an artifact of noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

# A compact RFM-style behavioral feature set: monetary (current_mrr),
# recency/relationship length (tenure_months), frequency/engagement
# (trailing login and feature-adoption averages), trend (login_trend_3v6),
# and support burden (ticket_count, sentiment). Deliberately excludes the
# label and excludes categorical demographic columns -- this segments
# accounts by *behavior*, not by the plan they happen to be on.
SEGMENTATION_FEATURE_COLUMNS = [
    "current_mrr",
    "tenure_months",
    "avg_login_trailing_3m",
    "avg_login_trailing_12m",
    "login_trend_3v6",
    "avg_feature_adoption_trailing_3m",
    "ticket_count_trailing_3m",
    "avg_ticket_sentiment_trailing_3m",
]


def prepare_features(df: pd.DataFrame, feature_cols: list[str] = SEGMENTATION_FEATURE_COLUMNS) -> tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_cols].to_numpy())
    return X, scaler


def choose_k_via_silhouette(
    X: np.ndarray, k_range: range = range(2, 9), random_seed: int = 42
) -> tuple[int, dict[int, float]]:
    scores = {}
    for k in k_range:
        labels = KMeans(n_clusters=k, random_state=random_seed, n_init=10).fit_predict(X)
        scores[k] = float(silhouette_score(X, labels))
    best_k = max(scores, key=scores.get)
    return best_k, scores


def fit_kmeans(X: np.ndarray, k: int, random_seed: int = 42) -> tuple[np.ndarray, KMeans]:
    model = KMeans(n_clusters=k, random_state=random_seed, n_init=10)
    labels = model.fit_predict(X)
    return labels, model


def fit_hierarchical(X: np.ndarray, k: int) -> tuple[np.ndarray, AgglomerativeClustering]:
    model = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels = model.fit_predict(X)
    return labels, model


def fit_density_based(X: np.ndarray, min_cluster_size: int = 30) -> tuple[np.ndarray, HDBSCAN]:
    model = HDBSCAN(min_cluster_size=min_cluster_size, copy=True)
    labels = model.fit_predict(X)  # -1 = noise, not assigned to any dense cluster
    return labels, model


def choose_min_cluster_size_via_silhouette(
    X: np.ndarray, candidate_fractions: tuple[float, ...] = (0.01, 0.02, 0.05, 0.1, 0.15, 0.2)
) -> tuple[int, dict[int, dict]]:
    """Sweeps `min_cluster_size` the same way `choose_k_via_silhouette`
    sweeps k for K-Means, rather than trying one arbitrary value and
    accepting whatever HDBSCAN returns. Concluding "no dense clusters
    exist" off a single untested threshold isn't a real finding -- it might
    just mean that one guess was wrong. Each candidate is scored on
    silhouette *excluding noise*, but a candidate that calls almost
    everything noise is penalized (a threshold so strict it finds a
    "perfect" silhouette on 2% of the data isn't a useful segmentation),
    and candidates that produce fewer than 2 real clusters are dropped.
    """
    n = X.shape[0]
    results: dict[int, dict] = {}
    for frac in candidate_fractions:
        min_cluster_size = max(5, round(n * frac))
        labels, _ = fit_density_based(X, min_cluster_size=min_cluster_size)
        noise_fraction = float((labels == -1).mean())
        silhouette = silhouette_excluding_noise(X, labels)
        if silhouette is None or noise_fraction > 0.8:
            continue
        results[min_cluster_size] = {
            "silhouette": silhouette,
            "noise_fraction": noise_fraction,
            "n_clusters": len(set(labels)) - (1 if -1 in labels else 0),
        }
    if not results:
        # Every candidate either found <2 clusters or was mostly noise --
        # a real, reportable finding (see module docstring), not a bug.
        return max(5, round(n * candidate_fractions[0])), {}
    best = max(results, key=lambda k: results[k]["silhouette"])
    return best, results


def reduce_to_2d_pca(X: np.ndarray, random_seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    pca = PCA(n_components=2, random_state=random_seed)
    coords = pca.fit_transform(X)
    return coords, pca.explained_variance_ratio_


def silhouette_excluding_noise(X: np.ndarray, labels: np.ndarray) -> float | None:
    """HDBSCAN's noise points (-1) aren't a cluster, so silhouette must
    exclude them; returns None if fewer than 2 real clusters remain."""
    mask = labels != -1
    if len(set(labels[mask])) < 2:
        return None
    return float(silhouette_score(X[mask], labels[mask]))


def bootstrap_stability(
    X: np.ndarray, cluster_fn, n_bootstrap: int = 15, subsample_frac: float = 0.8, random_seed: int = 42
) -> float:
    """Stability check: cluster two independent 80% subsamples, then measure
    agreement (Adjusted Rand Index) on the rows they have in common. A
    clustering that isn't an artifact of sampling noise should agree with
    itself across resamples; ARI near 0 means the "segments" don't
    replicate and shouldn't be presented as real customer segments.
    """
    rng = np.random.default_rng(random_seed)
    n = X.shape[0]
    ari_scores = []
    for _ in range(n_bootstrap):
        idx_a = rng.choice(n, size=int(n * subsample_frac), replace=False)
        idx_b = rng.choice(n, size=int(n * subsample_frac), replace=False)
        common = np.intersect1d(idx_a, idx_b)
        if len(common) < 10:
            continue
        labels_a_full = cluster_fn(X[idx_a])
        labels_b_full = cluster_fn(X[idx_b])
        pos_in_a = {row_id: i for i, row_id in enumerate(idx_a)}
        pos_in_b = {row_id: i for i, row_id in enumerate(idx_b)}
        labels_a_common = [labels_a_full[pos_in_a[i]] for i in common]
        labels_b_common = [labels_b_full[pos_in_b[i]] for i in common]
        ari_scores.append(adjusted_rand_score(labels_a_common, labels_b_common))
    return float(np.mean(ari_scores)) if ari_scores else float("nan")


@dataclass
class SegmentProfile:
    label: int
    size: int
    share: float
    feature_means: dict[str, float]
    avg_mrr: float
    total_mrr: float
    churn_rate: float | None


def profile_segments(
    df: pd.DataFrame,
    labels: np.ndarray,
    feature_cols: list[str] = SEGMENTATION_FEATURE_COLUMNS,
    mrr_col: str = "current_mrr",
    churn_col: str | None = "churn_within_horizon",
) -> list[SegmentProfile]:
    """Interpretability (what does each segment look like) and business
    value (how much revenue, how much churn risk) in one pass -- a
    segmentation without both is just a clustering exercise, not something
    a retention team can act on."""
    profiles = []
    n_total = len(df)
    for label in sorted(set(labels)):
        mask = labels == label
        subset = df[mask]
        profiles.append(
            SegmentProfile(
                label=int(label),
                size=int(mask.sum()),
                share=float(mask.sum() / n_total),
                feature_means={c: float(subset[c].mean()) for c in feature_cols},
                avg_mrr=float(subset[mrr_col].mean()),
                total_mrr=float(subset[mrr_col].sum()),
                churn_rate=float(subset[churn_col].mean()) if churn_col and churn_col in subset.columns else None,
            )
        )
    return profiles
