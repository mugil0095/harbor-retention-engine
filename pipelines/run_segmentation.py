#!/usr/bin/env python
"""Behavioral customer segmentation on the latest feature-store snapshot:
K-Means vs. hierarchical vs. HDBSCAN (density-based), with silhouette-based
model comparison, bootstrap stability, PCA projection for visualization,
and per-segment interpretability + business-value profiling.

Requires `pipelines/run_training_pipeline.py` to have been run first (reads
its feature_store.parquet output).

Usage:
    python pipelines/run_segmentation.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from sklearn.metrics import silhouette_score

from harbor.config import load_config
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
from harbor.utils.artifacts import require_files_exist
from harbor.utils.logging_config import configure_logging

logger = logging.getLogger(__name__)


def main() -> None:
    configure_logging()
    cfg = load_config()
    data_cfg = cfg.section("data")
    processed_dir = cfg.path(key="processed_dir")
    models_dir = cfg.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)

    feature_store_path = processed_dir / data_cfg["feature_store_file"]
    require_files_exist([feature_store_path], context="Segmentation")

    feature_store = pd.read_parquet(feature_store_path)
    snapshot_month = data_cfg["test_snapshot_month"]
    df = feature_store[feature_store["snapshot_month"] == snapshot_month].reset_index(drop=True)
    logger.info("Segmenting %d accounts from snapshot_month=%d", len(df), snapshot_month)

    X, _scaler = prepare_features(df)

    logger.info("Choosing k for K-Means via silhouette score...")
    best_k, k_scores = choose_k_via_silhouette(X, k_range=range(2, 9), random_seed=cfg.random_seed)
    logger.info("Best k=%d (silhouette=%.3f)", best_k, k_scores[best_k])

    kmeans_labels, _ = fit_kmeans(X, best_k, random_seed=cfg.random_seed)
    hier_labels, _ = fit_hierarchical(X, best_k)

    logger.info("Sweeping HDBSCAN min_cluster_size via silhouette score (not a single guess)...")
    best_min_cluster_size, density_sweep = choose_min_cluster_size_via_silhouette(X)
    density_labels, _ = fit_density_based(X, min_cluster_size=best_min_cluster_size)
    logger.info(
        "HDBSCAN sweep tried %d candidate thresholds, %d produced >=2 usable clusters; best min_cluster_size=%d",
        6, len(density_sweep), best_min_cluster_size,
    )

    kmeans_silhouette = float(silhouette_score(X, kmeans_labels))
    hier_silhouette = float(silhouette_score(X, hier_labels))
    density_silhouette = silhouette_excluding_noise(X, density_labels)
    density_noise_fraction = float((density_labels == -1).mean())

    logger.info(
        "Method comparison -- KMeans silhouette=%.3f, Hierarchical silhouette=%.3f, "
        "HDBSCAN silhouette=%s (noise fraction=%.2f)",
        kmeans_silhouette, hier_silhouette, density_silhouette, density_noise_fraction,
    )

    logger.info("Checking K-Means stability via bootstrap ARI...")
    stability_ari = bootstrap_stability(
        X,
        cluster_fn=lambda Xs: fit_kmeans(Xs, best_k, random_seed=cfg.random_seed)[0],
        n_bootstrap=15,
        random_seed=cfg.random_seed,
    )
    logger.info("K-Means stability (mean bootstrap ARI): %.3f", stability_ari)

    coords, explained_variance = reduce_to_2d_pca(X, random_seed=cfg.random_seed)

    profiles = profile_segments(df, kmeans_labels)
    for p in profiles:
        logger.info(
            "Segment %d: n=%d (%.1f%%), avg_mrr=$%.0f, churn_rate=%.1f%%",
            p.label, p.size, p.share * 100, p.avg_mrr, (p.churn_rate or 0) * 100,
        )

    projection = pd.DataFrame(
        {
            "account_id": df["account_id"],
            "pca_x": coords[:, 0],
            "pca_y": coords[:, 1],
            "kmeans_label": kmeans_labels,
            "hierarchical_label": hier_labels,
            "density_label": density_labels,
            "current_mrr": df["current_mrr"],
            "churn_within_horizon": df["churn_within_horizon"],
            "plan_tier": df["plan_tier"],
        }
    )
    projection.to_csv(processed_dir / "segmentation_projection.csv", index=False)

    summary = {
        "snapshot_month": snapshot_month,
        "n_accounts": len(df),
        "feature_columns": SEGMENTATION_FEATURE_COLUMNS,
        "kmeans_k_search": {str(k): v for k, v in k_scores.items()},
        "best_k": best_k,
        "method_comparison": {
            "kmeans_silhouette": kmeans_silhouette,
            "hierarchical_silhouette": hier_silhouette,
            "hdbscan_silhouette_excl_noise": density_silhouette,
            "hdbscan_noise_fraction": density_noise_fraction,
            "hdbscan_min_cluster_size_used": best_min_cluster_size,
            "hdbscan_min_cluster_size_sweep": {str(k): v for k, v in density_sweep.items()},
        },
        "kmeans_stability_bootstrap_ari": stability_ari,
        "pca_explained_variance_ratio": explained_variance.tolist(),
        "segment_profiles": [
            {
                "label": p.label,
                "size": p.size,
                "share": p.share,
                "avg_mrr": p.avg_mrr,
                "total_mrr": p.total_mrr,
                "churn_rate": p.churn_rate,
                "feature_means": p.feature_means,
            }
            for p in profiles
        ],
    }
    with open(models_dir / "segmentation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Wrote segmentation_summary.json and segmentation_projection.csv")


if __name__ == "__main__":
    main()
