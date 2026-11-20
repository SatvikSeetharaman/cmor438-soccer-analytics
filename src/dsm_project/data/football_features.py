"""Create identical point-in-time features for training and serving."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

PROFILE_FIELDS = ("elo", "recent_games", "goals_for", "goals_against", "points")
FEATURES = tuple(
    f"{side}_{field}" for side in ("home", "away") for field in PROFILE_FIELDS
)
OUTCOMES = ("away", "draw", "home")


@dataclass
class TeamHistory:
    """Maintain pre-match Elo and up to five preceding league results."""

    elo: float = 1500.0
    recent: deque[tuple[int, int, int]] = field(default_factory=lambda: deque(maxlen=5))

    def profile(self) -> dict[str, float]:
        """Return match counts and per-game averages, with neutral cold starts."""
        count = len(self.recent)
        goals_for, goals_against, points = (
            [sum(values) / count for values in zip(*self.recent, strict=True)]
            if count
            else (1.35, 1.35, 1.35)
        )
        return dict(
            zip(
                PROFILE_FIELDS,
                (self.elo, count, goals_for, goals_against, points),
                strict=True,
            )
        )


def matchup_features(
    histories: dict[str, TeamHistory], home: str, away: str
) -> dict[str, float]:
    """Snapshot the two teams in the exact order promised by FEATURES."""
    return {
        f"{side}_{key}": value
        for side, team in (("home", home), ("away", away))
        for key, value in histories.get(team, TeamHistory()).profile().items()
    }


def replay(
    matches: pd.DataFrame, *, before: date | None = None
) -> tuple[list[dict[str, object]], dict[str, TeamHistory]]:
    """Snapshot all matches on a date before applying that date's results.

    Elo updates are based on ratings at the start of the date. If a club
    appears twice on a date its recent form ordering is ambiguous, so reject it.
    """
    histories: dict[str, TeamHistory] = defaultdict(TeamHistory)
    rows: list[dict[str, object]] = []
    if before is not None:
        matches = matches[matches["date"] < before]
    for _, daily in matches.groupby("date", sort=True):
        updates = []
        teams = pd.concat([daily["home"], daily["away"]])
        if teams.duplicated().any():
            raise ValueError("Team results cannot be ordered within one date")
        for match in daily.itertuples(index=False):
            features = matchup_features(histories, match.home, match.away)
            difference = match.home_goals - match.away_goals
            home_score = 1.0 if difference > 0 else 0.5 if difference == 0 else 0.0
            expected = 1 / (
                1 + 10 ** ((features["away_elo"] - features["home_elo"] - 60) / 400)
            )
            delta = 20 * (home_score - expected)
            if before is None:
                rows.append(
                    {
                        **match._asdict(),
                        **features,
                        "goal_difference": difference,
                        "outcome": OUTCOMES[1 + (difference > 0) - (difference < 0)],
                    }
                )
            updates.append((match, delta, home_score))
        for match, delta, score in updates:
            home, away = histories[match.home], histories[match.away]
            home.elo += delta
            away.elo -= delta
            home.recent.append(
                (match.home_goals, match.away_goals, {0: 0, 0.5: 1, 1: 3}[score])
            )
            away.recent.append(
                (match.away_goals, match.home_goals, {0: 3, 0.5: 1, 1: 0}[score])
            )
    return rows, dict(histories)


def build_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Append supervised targets and strictly lagged features to the fixtures."""
    rows, _ = replay(matches)
    return pd.DataFrame(rows)
