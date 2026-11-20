"""Pinned public Premier League data and the football dataset adapter."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from pathlib import Path

import pandas as pd

from dsm_project.config import ProjectConfig
from dsm_project.contracts import (
    DatasetProvenance,
    ProjectDataset,
    TaskDefinition,
    TaskKind,
)
from dsm_project.data.football_features import FEATURES, build_features

ALIASES = {
    "Brighton & Hove Albion FC": "Brighton & Hove Albion",
    "Wolverhampton Wanderers FC": "Wolverhampton Wanderers",
    "Wolverhampton": "Wolverhampton Wanderers",
    "Spurs": "Tottenham Hotspur",
    "Bournemouth": "AFC Bournemouth",
    "Brighton": "Brighton & Hove Albion",
}


def development_seasons(config: ProjectConfig) -> tuple[str, ...]:
    """Check that the configured data cannot include the final season."""
    seasons = tuple(config.dataset["seasons"])
    final = config.split["final_test_season"]
    if (
        not seasons
        or seasons != tuple(sorted(set(seasons)))
        or any(not re.fullmatch(r"\d{4}-\d{2}", season) for season in (*seasons, final))
        or seasons[-1] >= final
    ):
        raise ValueError("Development seasons must be unique, ordered and pre-holdout")
    return seasons


def source_url(config: ProjectConfig, season: str) -> str:
    """Build an immutable GitHub raw URL for an approved development season."""
    revision = config.dataset["revision"]
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("An immutable source commit is required")
    if season not in development_seasons(config):
        raise ValueError("Season lies outside the development archive")
    return (
        "https://raw.githubusercontent.com/openfootball/football.json/"
        f"{revision}/{season}/en.1.json"
    )


def parse_season(content: bytes, season: str) -> pd.DataFrame:
    """Normalize a complete season and reject missing or inconsistent fixtures."""
    rows = []
    for fixture in json.loads(content)["matches"]:
        day = date.fromisoformat(fixture["date"])
        if not date(int(season[:4]), 7, 1) <= day < date(int(season[:4]) + 1, 8, 1):
            raise ValueError(f"Unexpected fixture date in {season}: {day}")
        teams = [fixture[key].strip().removesuffix(" FC") for key in ("team1", "team2")]
        home, away = (ALIASES.get(team, team) for team in teams)
        score = fixture.get("score", {}).get("ft", [])
        if (
            not home
            or not away
            or home == away
            or len(score) != 2
            or any(type(goals) is not int or goals < 0 for goals in score)
        ):
            raise ValueError(f"Invalid fixture in {season}: {day}, {home}, {away}")
        rows.append((f"{season}|{home}|{away}", season, day, home, away, *score))
    frame = pd.DataFrame(
        rows,
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
    if (
        len(frame) != 380
        or frame["match_id"].nunique() != 380
        or frame["home"].nunique() != 20
        or set(frame["home"]) != set(frame["away"])
    ):
        raise ValueError(f"{season} is missing a complete double round-robin")
    appearances = pd.concat(
        [
            frame[["date", "home"]].rename(columns={"home": "team"}),
            frame[["date", "away"]].rename(columns={"away": "team"}),
        ]
    )
    if appearances.duplicated().any():
        raise ValueError(f"A club appears twice on one date in {season}")
    return frame.sort_values(["date", "match_id"], ignore_index=True)


def data_directory(config: ProjectConfig) -> Path:
    """Locate raw files relative to the repository root."""
    return config.root / config.dataset["directory"]


def load_matches(config: ProjectConfig) -> pd.DataFrame:
    """Load only validated files whose bytes agree with the retrieval manifest."""
    directory = data_directory(config)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    frames = []
    for season in development_seasons(config):
        content = (directory / f"{season}.json").read_bytes()
        record = manifest["files"][season]
        if (
            record["url"] != source_url(config, season)
            or record["sha256"] != hashlib.sha256(content).hexdigest()
        ):
            raise ValueError(f"Unexpected source or checksum for {season}")
        frames.append(parse_season(content, season))
    for previous, following in zip(frames, frames[1:], strict=False):
        if previous["date"].max() >= following["date"].min():
            raise ValueError("Development seasons must not overlap")
    return pd.concat(frames, ignore_index=True)


class OpenFootballAdapter:
    """Build the template dataset contract from development fixtures."""

    def load(self, config: ProjectConfig) -> ProjectDataset:
        """Load scores and compute strictly prior-date features."""
        metadata = config.dataset
        observations = build_features(load_matches(config))
        tasks = {
            name: TaskDefinition(
                spec["name"],
                TaskKind(name),
                FEATURES,
                spec.get("target"),
                spec["metric"],
            )
            for name, spec in config.tasks.items()
        }
        dataset = ProjectDataset(
            observations,
            metadata["identifier"],
            tasks,
            DatasetProvenance(
                metadata["name"],
                f"{metadata['source']}/tree/{metadata['revision']}",
                metadata["license"],
                metadata["observation_unit"],
            ),
        )
        dataset.validate()
        return dataset
