"""Describe how Premier League teams differ across completed seasons."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

PROFILE_COLUMNS = ["home_attack", "away_attack", "home_defense", "away_defense"]


def team_season_profiles(matches: pd.DataFrame) -> pd.DataFrame:
    """Return completed-season scoring and conceding rates in goals per match."""
    home = matches.groupby(["season", "home"])[["home_goals", "away_goals"]].mean()
    away = matches.groupby(["season", "away"])[["away_goals", "home_goals"]].mean()
    home.index.names = away.index.names = ["season", "team"]
    home.columns = ["home_attack", "home_defense"]
    away.columns = ["away_attack", "away_defense"]
    return home.join(away).loc[:, PROFILE_COLUMNS]


def profile_experiment(
    matches: pd.DataFrame, seed: int = 438
) -> tuple[pd.DataFrame, dict]:
    """Audit geometry and seed sensitivity of retrospective scoring profiles."""
    profiles = team_season_profiles(matches)
    standardized = StandardScaler().fit_transform(profiles)
    pca = PCA(n_components=2)
    coordinates = pca.fit_transform(standardized)
    results = []
    for clusters in (2, 3, 4, 5):
        labels = KMeans(n_clusters=clusters, n_init=10, random_state=seed).fit_predict(
            standardized
        )
        agreement = [
            adjusted_rand_score(
                labels,
                KMeans(
                    n_clusters=clusters, n_init=10, random_state=seed + offset
                ).fit_predict(standardized),
            )
            for offset in range(1, 6)
        ]
        results.append(
            {
                "clusters": clusters,
                "silhouette": silhouette_score(standardized, labels),
                "minimum_seed_agreement": min(agreement),
                "assignments": labels,
            }
        )
    preferred = max(results, key=lambda candidate: candidate["silhouette"])
    labels = preferred["assignments"].copy()
    for result in results:
        result.pop("assignments")
    report = {
        "rows": len(profiles),
        "explained_variance": pca.explained_variance_ratio_.tolist(),
        "reconstruction_rmse": float(
            np.mean((standardized - pca.inverse_transform(coordinates)) ** 2) ** 0.5
        ),
        "pca_loadings": dict(
            zip(PROFILE_COLUMNS, pca.components_.T.tolist(), strict=True)
        ),
        "candidates": results,
        "chosen_k": preferred["clusters"],
        "cluster_sizes": np.bincount(labels).tolist(),
    }
    return profiles.assign(
        pc1=coordinates[:, 0], pc2=coordinates[:, 1], cluster=labels
    ), report
