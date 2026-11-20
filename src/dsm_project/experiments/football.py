"""Expanding-season validation for the Premier League reference models."""

from __future__ import annotations

from collections.abc import Iterator
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    log_loss,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from dsm_project.config import ProjectConfig
from dsm_project.contracts import TaskKind
from dsm_project.data.football_features import FEATURES, OUTCOMES
from dsm_project.data.openfootball import development_seasons
from dsm_project.experiments.supervised import make_reference_candidate


def demonstration_models() -> tuple[BaseEstimator, BaseEstimator]:
    """Create the same ridge and logistic pipelines evaluated in validation."""
    return (
        make_pipeline(StandardScaler(), Ridge(alpha=10)),
        make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000)),
    )


def validation_folds(
    observations: pd.DataFrame, config: ProjectConfig
) -> Iterator[tuple[str, pd.DataFrame, pd.DataFrame]]:
    """Yield each validation season and its strictly earlier training rows."""
    seasons = development_seasons(config)
    validation = tuple(config.split["validation_seasons"])
    warmup = config.split["warmup_season"]
    if (
        len(seasons) < 4
        or warmup != seasons[0]
        or not validation
        or validation != seasons[-len(validation) :]
        or validation[0] <= seasons[2]
        or set(observations["season"]) != set(seasons)
    ):
        raise ValueError("Invalid warmup or validation boundaries")
    for season in validation:
        training = observations[
            (observations["season"] > warmup) & (observations["season"] < season)
        ]
        holdout = observations[observations["season"] == season]
        if training["date"].max() >= holdout["date"].min():
            raise ValueError("Training must end before validation begins")
        yield season, training, holdout


def _block_interval(dates: pd.Series, gain: np.ndarray, seed: int) -> list[float]:
    """Resample paired 28-day calendar blocks within one validation season."""
    blocks = dates.map(lambda day: (day - dates.min()).days // 28)
    grouped = pd.DataFrame({"block": blocks.to_numpy(), "gain": gain}).groupby("block")
    statistics = grouped["gain"].agg(["sum", "count"]).to_numpy()
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(statistics), (1000, len(statistics)))
    totals = statistics[draws].sum(axis=1)
    return np.quantile(totals[:, 0] / totals[:, 1], [0.025, 0.975]).tolist()


def _classification_metrics(
    holdout: pd.DataFrame, probability: np.ndarray, baseline: np.ndarray, seed: int
) -> dict:
    actual = holdout["outcome"]
    predicted = np.asarray(OUTCOMES)[probability.argmax(axis=1)]
    target = actual.to_numpy()[:, None] == np.asarray(OUTCOMES)
    row_loss = -(target * np.log(probability.clip(1e-15))).sum(axis=1)
    prior_loss = -(target * np.log(baseline.clip(1e-15))).sum(axis=1)
    confidence = probability.max(axis=1)
    calibration = []
    for low, high in ((0, 0.4), (0.4, 0.55), (0.55, 0.7), (0.7, 1.01)):
        mask = (confidence >= low) & (confidence < high)
        if mask.any():
            calibration.append(
                {
                    "rows": int(mask.sum()),
                    "confidence": float(confidence[mask].mean()),
                    "accuracy": float(
                        (predicted[mask] == actual.to_numpy()[mask]).mean()
                    ),
                }
            )
    draw_probability = probability[:, OUTCOMES.index("draw")]
    observed_draw = actual.to_numpy() == "draw"
    draw_calibration = []
    for low, high in ((0, 0.15), (0.15, 0.25), (0.25, 0.35), (0.35, 1.01)):
        mask = (draw_probability >= low) & (draw_probability < high)
        if mask.any():
            draw_calibration.append(
                {
                    "rows": int(mask.sum()),
                    "probability": float(draw_probability[mask].mean()),
                    "observed_rate": float(observed_draw[mask].mean()),
                }
            )
    sparse = holdout[["home_recent_games", "away_recent_games"]].min(axis=1) < 5
    ratings = (holdout["home_elo"] - holdout["away_elo"]).abs()
    slices = {}
    for name, mask in {
        "sparse_history": sparse,
        "full_history": ~sparse,
        "elo_gap_under_100": ratings < 100,
        "elo_gap_at_least_100": ratings >= 100,
    }.items():
        if mask.any():
            slices[name] = {"rows": int(mask.sum()), "log_loss": row_loss[mask].mean()}
    return {
        "log_loss": log_loss(actual, probability, labels=OUTCOMES),
        "accuracy": accuracy_score(actual, predicted),
        "balanced_accuracy": balanced_accuracy_score(actual, predicted),
        "brier": float(np.square(probability - target).sum(axis=1).mean()),
        "class_order": OUTCOMES,
        "confusion": confusion_matrix(actual, predicted, labels=OUTCOMES).tolist(),
        "calibration": calibration,
        "draw_calibration": draw_calibration,
        "slices": slices,
        "gain_95_percent_block_interval": _block_interval(
            holdout["date"], prior_loss - row_loss, seed
        ),
    }


def _calibrated_margin_probabilities(
    train: pd.DataFrame, holdout_x: pd.DataFrame
) -> np.ndarray:
    """Fit a linear SVM and calibrate its scores on the latest training season."""
    latest = train["season"].max()
    fit = train[train["season"] < latest]
    calibrate = train[train["season"] == latest]
    if fit.empty:
        raise ValueError(
            "SVM calibration needs separate training and calibration seasons"
        )
    margin = make_pipeline(StandardScaler(), SVC(kernel="linear", C=0.5))
    margin.fit(fit[list(FEATURES)], fit["outcome"])
    calibration = LogisticRegression(max_iter=1000).fit(
        margin.decision_function(calibrate[list(FEATURES)]), calibrate["outcome"]
    )
    if tuple(calibration.classes_) != OUTCOMES:
        raise ValueError("Unexpected SVM calibration label order")
    return calibration.predict_proba(margin.decision_function(holdout_x))


def run_validation(observations: pd.DataFrame, config: ProjectConfig) -> dict:
    """Compare fixed, bounded reference candidates on expanding-season folds."""
    seed = int(config.split["random_state"])
    reports = {}
    for season, train, holdout in validation_folds(observations, config):
        start = perf_counter()
        train_x, holdout_x = train[list(FEATURES)], holdout[list(FEATURES)]
        scale = train_x.std(ddof=0).replace(0, np.nan)
        shifts = ((holdout_x.mean() - train_x.mean()) / scale).dropna()
        report = {
            "train_rows": len(train),
            "validation_rows": len(holdout),
            "train_through": train["date"].max().isoformat(),
            "validation_start": holdout["date"].min().isoformat(),
            "validation_end": holdout["date"].max().isoformat(),
            "outcome_counts": holdout["outcome"].value_counts().to_dict(),
            "feature_mean_shift_in_training_sd": shifts.to_dict(),
            "constant_training_features": scale[scale.isna()].index.tolist(),
            "regression": {},
            "classification": {},
        }
        ridge, logistic = demonstration_models()
        regressors = {
            "training_mean": DummyRegressor(strategy="mean"),
            "ridge_10": ridge,
            **{
                name: make_reference_candidate(name, TaskKind.REGRESSION, seed)
                for name in (
                    "linear_regression",
                    "neighbors",
                    "decision_tree",
                    "gradient_boosting",
                )
            },
        }
        actual = holdout["goal_difference"]
        baseline_error = np.abs(actual - train["goal_difference"].mean())
        for name, estimator in regressors.items():
            estimator.fit(train_x, train["goal_difference"])
            prediction = estimator.predict(holdout_x)
            residuals = actual.to_numpy() - prediction
            report["regression"][name] = {
                "mae": mean_absolute_error(actual, prediction),
                "rmse": root_mean_squared_error(actual, prediction),
                "r2": r2_score(actual, prediction),
                "residual_p10_p50_p90": np.quantile(
                    residuals, [0.1, 0.5, 0.9]
                ).tolist(),
                "gain_95_percent_block_interval": _block_interval(
                    holdout["date"], baseline_error.to_numpy() - abs(residuals), seed
                ),
            }
        prior = DummyClassifier(strategy="prior").fit(train_x, train["outcome"])
        baseline = prior.predict_proba(holdout_x)
        classifiers = {
            "class_frequency": prior,
            "logistic_c_0_1": logistic,
            "small_neural_network": make_pipeline(
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=(12,),
                    alpha=5.0,
                    max_iter=500,
                    random_state=seed,
                    solver="lbfgs",
                ),
            ),
            **{
                name: make_reference_candidate(name, TaskKind.CLASSIFICATION, seed)
                for name in (
                    "logistic_regression",
                    "neighbors",
                    "decision_tree",
                    "random_forest",
                    "gradient_boosting",
                )
            },
        }
        for name, estimator in classifiers.items():
            if name != "class_frequency":
                estimator.fit(train_x, train["outcome"])
            if tuple(estimator.classes_) != OUTCOMES:
                raise ValueError("Unexpected classification label order")
            report["classification"][name] = _classification_metrics(
                holdout, estimator.predict_proba(holdout_x), baseline, seed
            )
            if name == "small_neural_network":
                network = estimator[-1]
                report["classification"][name]["optimizer_iterations"] = network.n_iter_
                report["classification"][name]["optimizer_hit_limit"] = (
                    network.n_iter_ >= network.max_iter
                )
        svm_probability = _calibrated_margin_probabilities(train, holdout_x)
        report["classification"]["chronologically_calibrated_linear_svm"] = (
            _classification_metrics(holdout, svm_probability, baseline, seed)
        )
        report["elapsed_seconds"] = round(perf_counter() - start, 3)
        reports[season] = report
    return {"provenance": "scikit-learn reference", "folds": reports}
