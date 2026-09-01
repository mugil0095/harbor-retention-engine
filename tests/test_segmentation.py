"""Segmentation/clustering: correctness of each method plus the comparison
and stability logic used to decide which segmentation to trust."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_blobs

from harbor.features.build_features import build_feature_store
from harbor.nlp.ticket_analysis import compare_nlp_approaches
from harbor.segmentation.clustering import (
    SEGMENTATION_FEATURE_COLUMNS,
    bootstrap_stability,
    choose_k_via_silhouette,
    choose_min_cluster_size_via_silhouette,
    fit_density_based,
    fit_hierarchical,
    fit_kmeans,
    prepare_features,
    profile_segments,
    reduce_to_2d_pca,
    silhouette_excluding_noise,
)


@pytest.fixture(scope="module")
def feature_store(synthetic_tables, test_config):
    _, clf = compare_nlp_approaches(synthetic_tables["support_tickets"], random_seed=123)
    data_cfg = test_config.section("data")
    return build_feature_store(
        synthetic_tables["accounts"], synthetic_tables["monthly_usage"], synthetic_tables["support_tickets"],
        clf, data_cfg["snapshot_months"], data_cfg["churn_horizon_months"],
    )


@pytest.fixture(scope="module")
def latest_snapshot(feature_store, test_config):
    test_month = test_config.section("data")["test_snapshot_month"]
    return feature_store[feature_store["snapshot_month"] == test_month].reset_index(drop=True)


def _well_separated_blobs(n_per_cluster=100, n_clusters=3, seed=0):
    X, true_labels = make_blobs(
        n_samples=n_per_cluster * n_clusters, centers=n_clusters, cluster_std=0.5, random_state=seed
    )
    return X, true_labels


def test_prepare_features_standardizes_to_zero_mean_unit_variance(latest_snapshot):
    if len(latest_snapshot) < 10:
        pytest.skip("too few active accounts in tiny test fixture")
    X, _scaler = prepare_features(latest_snapshot)
    assert X.shape == (len(latest_snapshot), len(SEGMENTATION_FEATURE_COLUMNS))
    np.testing.assert_allclose(X.mean(axis=0), 0.0, atol=1e-8)
    np.testing.assert_allclose(X.std(axis=0), 1.0, atol=1e-6)


def test_choose_k_via_silhouette_recovers_known_cluster_count():
    X, _ = _well_separated_blobs(n_clusters=4)
    best_k, scores = choose_k_via_silhouette(X, k_range=range(2, 7), random_seed=0)
    assert best_k == 4
    assert all(-1.0 <= v <= 1.0 for v in scores.values())


def test_kmeans_recovers_well_separated_clusters():
    X, true_labels = _well_separated_blobs(n_clusters=3)
    labels, _model = fit_kmeans(X, k=3, random_seed=0)
    assert len(set(labels)) == 3
    assert len(labels) == len(X)
    # k-means should near-perfectly agree with the true generating clusters
    from sklearn.metrics import adjusted_rand_score

    assert adjusted_rand_score(true_labels, labels) > 0.9


def test_hierarchical_recovers_well_separated_clusters():
    X, true_labels = _well_separated_blobs(n_clusters=3)
    labels, _model = fit_hierarchical(X, k=3)
    assert len(set(labels)) == 3
    from sklearn.metrics import adjusted_rand_score

    assert adjusted_rand_score(true_labels, labels) > 0.9


def test_density_based_labels_noise_for_scattered_points():
    rng = np.random.default_rng(0)
    dense_blob, _ = _well_separated_blobs(n_per_cluster=100, n_clusters=1, seed=0)
    scattered_noise = rng.uniform(-15, 15, size=(20, 2))
    X = np.vstack([dense_blob, scattered_noise])
    labels, _ = fit_density_based(X, min_cluster_size=20)
    assert -1 in labels  # some scattered points should be flagged as noise
    assert len(labels) == len(X)


def test_silhouette_excluding_noise_handles_all_noise():
    labels = np.full(50, -1)
    X = np.random.default_rng(0).normal(size=(50, 2))
    assert silhouette_excluding_noise(X, labels) is None


def test_silhouette_excluding_noise_computes_for_valid_clusters():
    X, true_labels = _well_separated_blobs(n_clusters=3)
    score = silhouette_excluding_noise(X, true_labels)
    assert score is not None
    assert score > 0.5  # well-separated blobs should have high silhouette


def test_reduce_to_2d_pca_shape_and_variance():
    X, _ = _well_separated_blobs(n_clusters=3)
    coords, explained_variance = reduce_to_2d_pca(X, random_seed=0)
    assert coords.shape == (len(X), 2)
    assert len(explained_variance) == 2
    assert 0.0 <= explained_variance.sum() <= 1.0 + 1e-9


def test_bootstrap_stability_is_high_for_well_separated_clusters():
    X, _ = _well_separated_blobs(n_per_cluster=200, n_clusters=3, seed=1)
    stability = bootstrap_stability(
        X, cluster_fn=lambda Xs: fit_kmeans(Xs, k=3, random_seed=1)[0], n_bootstrap=10, random_seed=1
    )
    assert stability > 0.7  # clean, separated clusters should replicate across resamples


def test_stability_alone_cannot_detect_meaningless_clusters_silhouette_can():
    """K-Means forced onto pure, unimodal noise still produces a highly
    *reproducible* partition (it consistently slices the same blob into the
    same angular thirds), so bootstrap stability alone is NOT a reliable
    "are these real clusters" check -- it can be high even with zero
    underlying structure. Silhouette score is what actually distinguishes
    real, well-separated clusters from an artificial partition: it's near
    the conventional "weak structure" range for the noise case and clearly
    high for genuine clusters. This is exactly why the segmentation
    pipeline reports both, not stability alone."""
    rng = np.random.default_rng(2)
    noise = rng.normal(size=(300, 2))
    noise_labels, _ = fit_kmeans(noise, k=3, random_seed=2)
    noise_stability = bootstrap_stability(
        noise, cluster_fn=lambda Xs: fit_kmeans(Xs, k=3, random_seed=2)[0], n_bootstrap=10, random_seed=2
    )
    noise_silhouette = silhouette_excluding_noise(noise, noise_labels)

    real_blobs, _ = _well_separated_blobs(n_per_cluster=100, n_clusters=3, seed=1)
    real_silhouette = silhouette_excluding_noise(real_blobs, fit_kmeans(real_blobs, k=3, random_seed=1)[0])

    assert noise_stability > 0.7  # reproducible, but NOT evidence of real structure
    assert noise_silhouette < 0.5  # silhouette correctly flags it as weak/artificial
    assert real_silhouette > noise_silhouette  # real clusters are unambiguously better separated


def test_profile_segments_covers_every_account_exactly_once(latest_snapshot):
    if len(latest_snapshot) < 10:
        pytest.skip("too few active accounts in tiny test fixture")
    X, _ = prepare_features(latest_snapshot)
    labels, _ = fit_kmeans(X, k=3, random_seed=0)
    profiles = profile_segments(latest_snapshot, labels)
    assert sum(p.size for p in profiles) == len(latest_snapshot)
    assert abs(sum(p.share for p in profiles) - 1.0) < 1e-9


def test_profile_segments_churn_rate_is_valid_probability(latest_snapshot):
    if len(latest_snapshot) < 10:
        pytest.skip("too few active accounts in tiny test fixture")
    X, _ = prepare_features(latest_snapshot)
    labels, _ = fit_kmeans(X, k=3, random_seed=0)
    profiles = profile_segments(latest_snapshot, labels)
    for p in profiles:
        if p.churn_rate is not None:
            assert 0.0 <= p.churn_rate <= 1.0


def test_choose_min_cluster_size_via_silhouette_finds_real_clusters_when_present():
    """HDBSCAN should get the same fair chance K-Means gets: swept across
    thresholds, not judged on one guess."""
    X, _ = _well_separated_blobs(n_per_cluster=150, n_clusters=3, seed=2)
    best_min_cluster_size, sweep = choose_min_cluster_size_via_silhouette(X)
    assert len(sweep) > 0  # at least one threshold should find real structure
    assert sweep[best_min_cluster_size]["silhouette"] > 0.5
    assert sweep[best_min_cluster_size]["n_clusters"] >= 2


def test_min_cluster_size_sweep_scores_noise_far_below_real_clusters():
    """A single unimodal blob has no real sub-cluster structure, but at a
    very permissive min_cluster_size, HDBSCAN can still carve spurious
    "clusters" out of local density fluctuations in pure noise -- a known
    HDBSCAN failure mode, not a bug in the sweep. The comparative signal is
    what's trustworthy: noise-driven candidates score far below genuine
    clusters on silhouette, the same "compare, don't assert absolutes"
    pattern used for K-Means stability elsewhere in this test file."""
    rng = np.random.default_rng(4)
    noise = rng.normal(size=(600, 2))
    _best_noise, noise_sweep = choose_min_cluster_size_via_silhouette(noise)
    best_noise_silhouette = max((v["silhouette"] for v in noise_sweep.values()), default=0.0)

    real_blobs, _ = _well_separated_blobs(n_per_cluster=150, n_clusters=3, seed=2)
    _best_real, real_sweep = choose_min_cluster_size_via_silhouette(real_blobs)
    best_real_silhouette = max(v["silhouette"] for v in real_sweep.values())

    assert best_real_silhouette > best_noise_silhouette
    assert best_noise_silhouette < 0.4  # at best, weak/ambiguous structure -- not a confident finding


def test_choose_min_cluster_size_via_silhouette_always_returns_a_usable_fallback():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(600, 2))
    best_min_cluster_size, _sweep = choose_min_cluster_size_via_silhouette(X)
    assert isinstance(best_min_cluster_size, int)
    assert best_min_cluster_size > 0
