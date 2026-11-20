"""Build a versioned historical demonstration using the template bundle format."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import sklearn
from sklearn.base import BaseEstimator
from sklearn.metrics import log_loss

from dsm_project.artifacts.bundle import save_bundle
from dsm_project.config import ProjectConfig
from dsm_project.contracts import ProjectDataset
from dsm_project.data import football_features, openfootball
from dsm_project.data.football_features import FEATURES, OUTCOMES
from dsm_project.experiments import football
from dsm_project.experiments.supervised import ExperimentResult


class FootballReference(BaseEstimator):
    """Keep outcome probabilities and signed goal difference in one artifact."""

    def fit(self, features, outcomes, goal_differences) -> FootballReference:
        """Fit the two reference heads on the same chronological partition."""
        ridge, logistic = football.demonstration_models()
        self.regressor_ = ridge.fit(features, goal_differences)
        self.classifier_ = logistic.fit(features, outcomes)
        return self

    def predict(self, features):
        """Return most probable outcome labels."""
        return self.classifier_.predict(features)

    def predict_proba(self, features):
        """Return three probability columns in alphabetic class order."""
        return self.classifier_.predict_proba(features)

    def predict_difference(self, features):
        """Return predicted home goals minus away goals."""
        return self.regressor_.predict(features)


def archive_hashes(config: ProjectConfig) -> dict[str, str]:
    """Identify the exact bytes bound to a demonstration and its history."""
    manifest = openfootball.data_directory(config) / "manifest.json"
    files = json.loads(manifest.read_text(encoding="utf-8"))["files"]
    return {season: item["sha256"] for season, item in files.items()}


def text_fingerprint(path: str | Path) -> str:
    """Hash source or config text consistently across checkout line endings."""
    return hashlib.sha256(Path(path).read_text(encoding="utf-8").encode()).hexdigest()


def train_demo(
    dataset: ProjectDataset, config: ProjectConfig, directory: str | Path
) -> Path:
    """Fit on the first fold to permit prediction throughout both validation seasons."""
    season, train, validation = next(
        football.validation_folds(dataset.observations, config)
    )
    train_x, validation_x = train[list(FEATURES)], validation[list(FEATURES)]
    estimator = FootballReference().fit(
        train_x, train["outcome"], train["goal_difference"]
    )
    actual = validation["outcome"]
    result = ExperimentResult(
        "reference:logistic_c_0_1+ridge_10",
        "historical_matchup",
        "validation_log_loss",
        log_loss(actual, estimator.predict_proba(validation_x), labels=OUTCOMES),
        len(train),
        len(validation),
        FEATURES,
        estimator,
        validation_x,
        actual.to_numpy(),
        estimator.predict(validation_x),
    )
    modules = (football, football_features, openfootball)
    part_hashes = [text_fingerprint(module.__file__) for module in modules]
    part_hashes.append(text_fingerprint(__file__))
    metadata = {
        "product": "football",
        "provenance": "scikit-learn reference demonstration",
        "train_through": train["date"].max().isoformat(),
        "first_validation_season": season,
        "validation_seasons": config.split["validation_seasons"],
        "final_test_season": config.split["final_test_season"],
        "data_sha256": archive_hashes(config),
        "source_sha256": hashlib.sha256("".join(part_hashes).encode()).hexdigest(),
        "feature_code_sha256": text_fingerprint(football_features.__file__),
        "configuration_sha256": text_fingerprint(config.path),
        "sklearn_version": sklearn.__version__,
        "outcome_order": OUTCOMES,
        "limitations": (
            "Historical Premier League demonstration. Draws can receive meaningful "
            "probability even when the draw is never the most likely class. "
            "Goal difference is estimated separately. For analyst exploration only."
        ),
    }
    fingerprint = hashlib.sha256(
        json.dumps(metadata, sort_keys=True).encode()
    ).hexdigest()
    metadata["artifact_version"] = f"football-reference-v1-{fingerprint[:12]}"
    return save_bundle(result, dataset, config, directory, extra_metadata=metadata)
