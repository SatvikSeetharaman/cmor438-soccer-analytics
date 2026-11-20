"""Offline checks for the historical information and ingestion boundaries."""

import json
from datetime import date, timedelta
from hashlib import sha256
from itertools import permutations

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pandas.testing import assert_frame_equal

from dsm_project.artifacts.football import train_demo
from dsm_project.config import load_config
from dsm_project.data import load_project_dataset
from dsm_project.data.football_features import (
    FEATURES,
    build_features,
    matchup_features,
    replay,
)
from dsm_project.data.openfootball import parse_season, source_url
from dsm_project.experiments.football import validation_folds
from dsm_project.experiments.team_profiles import team_season_profiles
from dsm_project.service import create_app


@pytest.fixture
def matches() -> pd.DataFrame:
    """Construct a small offline fixture with two games sharing a date."""
    return pd.DataFrame(
        [
            ("1", "2018-19", date(2018, 9, 1), "A", "B", 2, 0),
            ("2", "2018-19", date(2018, 9, 1), "C", "D", 3, 1),
            ("3", "2018-19", date(2018, 9, 2), "A", "C", 0, 1),
            ("4", "2018-19", date(2018, 9, 3), "B", "D", 2, 2),
        ],
        columns=[
            "match_id",
            "season",
            "date",
            "home",
            "away",
            "home_goals",
            "away_goals",
        ],
    )


def test_features_invariant_to_current_same_day_and_future_scores(matches) -> None:
    modified = matches.copy()
    modified.loc[[0, 1, 3], "home_goals"] = 20
    original = build_features(matches).set_index("match_id")
    altered = build_features(modified.sample(frac=1, random_state=438)).set_index(
        "match_id"
    )
    assert_frame_equal(
        original.loc[["1", "2"], list(FEATURES)],
        altered.loc[["1", "2"], list(FEATURES)],
    )
    assert original.loc["1", "home_elo"] == 1500
    assert original.loc["3", "home_goals_for"] == 2
    assert altered.loc["3", "home_goals_for"] == 20
    assert original.loc["3", "home_elo"] == pytest.approx(
        1500 + 20 * (1 - 1 / (1 + 10 ** (-60 / 400)))
    )
    only_future = matches.copy()
    only_future.loc[3, "away_goals"] = 23
    assert_frame_equal(
        original.loc["1":"3", list(FEATURES)],
        build_features(only_future).set_index("match_id").loc["1":"3", list(FEATURES)],
    )


def test_replay_matches_training_feature_order_and_cutoff(matches) -> None:
    _, histories = replay(matches, before=date(2018, 9, 2))
    snapshot = matchup_features(histories, "A", "C")
    assert tuple(snapshot) == FEATURES
    assert snapshot == build_features(matches).loc[2, list(FEATURES)].to_dict()
    assert histories["A"].profile()["points"] == 3
    assert histories["B"].profile()["points"] == 0


def test_same_team_twice_on_date_is_rejected(matches) -> None:
    matches.loc[1, "home"] = "A"
    with pytest.raises(ValueError, match="within one date"):
        build_features(matches)


def test_team_season_profile_preserves_venue_and_goal_direction(matches) -> None:
    complete = matches.loc[[0, 2]].copy()
    complete.loc[2, ["home", "away"]] = ["B", "A"]
    profile = team_season_profiles(complete).loc[("2018-19", "A")]
    assert profile.to_dict() == {
        "home_attack": 2.0,
        "away_attack": 1.0,
        "home_defense": 0.0,
        "away_defense": 0.0,
    }


def synthetic_season(year: int) -> list[dict]:
    """Create a complete season with deterministic outcomes and no real clubs."""
    teams = [f"Team {index}" for index in range(20)]
    matches = []
    for home, away in permutations(range(20), 2):
        # On any date an opponent's index is uniquely determined by a team's index.
        slot = (home + away) % 20 + (20 if home > away else 0)
        matches.append(
            {
                "team1": teams[home],
                "team2": teams[away],
                "date": str(date(year, 8, 1) + timedelta(days=slot)),
                "score": {"ft": [(home + away + year) % 3, away % 3]},
            }
        )
    return matches


def test_parser_requires_complete_round_robin_and_full_time_scores() -> None:
    matches = synthetic_season(2018)
    content = json.dumps({"matches": matches}).encode()
    assert len(parse_season(content, "2018-19")) == 380
    matches[0]["score"]["ft"] = [1, -1]
    with pytest.raises(ValueError, match="Invalid fixture"):
        parse_season(json.dumps({"matches": matches}).encode(), "2018-19")
    with pytest.raises(ValueError, match="round-robin"):
        parse_season(json.dumps({"matches": matches[1:]}).encode(), "2018-19")


def test_final_season_is_excluded_and_folds_are_strictly_ordered() -> None:
    config = load_config("configs/football.toml")
    with pytest.raises(ValueError, match="development archive"):
        source_url(config, "2024-25")
    observations = pd.DataFrame(
        [(season, date(int(season[:4]), 8, 1)) for season in config.dataset["seasons"]],
        columns=["season", "date"],
    )
    folds = list(validation_folds(observations, config))
    assert [len(train) for _, train, _ in folds] == [3, 4]
    for _, train, validation in folds:
        assert train.date.max() < validation.date.min()


def test_fit_serialize_and_serve_offline(tmp_path, caplog) -> None:
    source = load_config("configs/football.toml")
    config_path = tmp_path / "configs" / "football.toml"
    config_path.parent.mkdir()
    config_path.write_bytes(source.path.read_bytes())
    config = load_config(config_path)
    archive = tmp_path / config.dataset["directory"]
    archive.mkdir(parents=True)
    files = {}
    for season in config.dataset["seasons"]:
        content = json.dumps({"matches": synthetic_season(int(season[:4]))}).encode()
        (archive / f"{season}.json").write_bytes(content)
        files[season] = {
            "url": source_url(config, season),
            "sha256": sha256(content).hexdigest(),
        }
    (archive / "manifest.json").write_text(json.dumps({"files": files}))
    dataset = load_project_dataset(config)
    artifact = train_demo(dataset, config, tmp_path / "artifact")
    with TestClient(create_app(artifact)) as client:
        payload = {"home": "Team 1", "away": "Team 2", "date": "2022-08-02"}
        result = client.post("/predict", json=payload)
        assert result.status_code == 200
        prediction = result.json()
        assert sum(prediction["probabilities"].values()) == pytest.approx(1)
        assert prediction["request_id"] == result.headers["X-Request-ID"]
        assert len(prediction["home_profile"]) == 5
        assert (
            client.post("/predict", json={**payload, "date": "2024-08-02"}).status_code
            == 422
        )
        assert (
            client.post("/predict", json={**payload, "away": "Team 1"}).status_code
            == 422
        )
        assert (
            sum(
                getattr(record, "validation_failure", False)
                for record in caplog.records
            )
            == 2
        )
    with (archive / "2022-23.json").open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="checksum"):
        create_app(artifact)
