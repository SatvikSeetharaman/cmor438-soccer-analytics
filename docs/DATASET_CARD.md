# Dataset card

## Source and scope

[openfootball football.json](https://github.com/openfootball/football.json) is a
community-maintained public-domain archive. The project uses `en.1.json` for
six complete Premier League seasons, 2018–19 through 2023–24, from
[commit e674442](https://github.com/openfootball/football.json/tree/e6744429ee395bc86f247348c6184bb08d4eb361).
See the upstream [public-domain dedication](https://github.com/openfootball/football.json/blob/e6744429ee395bc86f247348c6184bb08d4eb361/LICENSE.md).
The downloader accessed 694,168 bytes on November 20, 2026. The
[retrieval manifest](evidence/football-source-manifest.json) records each URL and checksum.

[StatsBomb Open Data](https://github.com/hudl/open-data) is designed for research
and general football analysis. Publication asks for StatsBomb attribution and
its logo. Its agreement restricts data redistribution and commercial use.
Its public catalog lists Premier League 2003–04 and 2015–16, which
does not give the contiguous recent seasons desired here. Events and 360 data
would also increase work substantially. Use the
[competition catalog](https://github.com/hudl/open-data/blob/master/data/competitions.json)
and [license](https://github.com/hudl/open-data/blob/master/LICENSE.pdf) for any
future event extension. No StatsBomb data enters these experiments.

## Observation and quality

One row is a completed men's Premier League fixture. Each season contributed
380 unique directed home-away fixtures, 20 clubs, and zero missing values in
retained columns. Across seasons there are 28 normalized club names and 2,280
rows. The adapter refuses malformed, incomplete or duplicate seasons. No rows
were removed from the six files. GPS, events, lineups, injuries, expected goals,
betting data, kickoff time, and other competitions are outside this archive.

| Fields | Meaning and availability |
| --- | --- |
| Raw `date`, `team1`, `team2` | Calendar date and club names, normalized into date, home and away. These identify a prediction scenario |
| Raw `score.ft` | Full-time goals, used only as targets or to update future history |
| Other raw fields | Round names, kickoff time, half-time scores and any ancillary fields are ignored |
| `match_id`, `season` | Season and normalized directed pairing, with human-readable names used as IDs |
| `home_elo`, `away_elo` | Prior-date rating, initial value 1500 |
| `*_recent_games` | Count from zero to five earlier top-flight results |
| `*_goals_for`, `*_goals_against`, `*_points` | Per-match form over that window. With zero games, fixed neutral 1.35 values and a count of zero |
| `goal_difference`, `outcome` | Home goals minus away goals, and away/draw/home based on its sign |
| Team profiles | Completed-season venue-specific rates, four values in goals per match for retrospective PCA and clustering |

Same-date fixtures are all predicted before that date updates the feature
state. A club with no history has a cold start. Returning clubs retain
their previous Premier League appearances and ratings. Clubs repeat across
time, matches share opponents, and seasons affected by COVID-era scheduling
may differ from later seasons. One competition offers no direct evidence for
women's football, lower tiers, knockout play, or other leagues. Archive
revision and name normalization can affect identity and measurements.

## Partitions and access

2018–19 initializes history. The first fold fits 2019–20 through 2021–22 and
validates 2022–23. The second expands through 2022–23 and validates 2023–24.
Each validation date can use previous dates' scores under a sequential
historical replay. The 2024–25 final season was never downloaded or scored in
this sprint. Raw files are ignored under `data/local/openfootball`. CI uses
constructed fixtures with no credentials or research claims.
