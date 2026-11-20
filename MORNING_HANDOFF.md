# Morning handoff

The initial Football Intelligence Engine sprint is ready for review.
The local demo compares historical Premier League matchups and shows
three-way outcome probabilities, predicted home-minus-away goals, and
both teams' prior form. It serves a scikit-learn reference bundle.
The assignment still needs student-owned algorithm code and final review.

## GitHub

Work is on `project/football-foundation` in the existing
[private repository](https://github.com/SatvikSeetharaman/cmor438-soccer-analytics).
The data boundary is commit `10d36b1` and the experiments are commit
`9e57e01`. The local product is commit `f2b30f0`.
Review and current GitHub Actions status are on
[draft PR #11](https://github.com/SatvikSeetharaman/cmor438-soccer-analytics/pull/11).

- [#7 Foundation](https://github.com/SatvikSeetharaman/cmor438-soccer-analytics/issues/7)
- [#8 Course scope and student implementations](https://github.com/SatvikSeetharaman/cmor438-soccer-analytics/issues/8)
- [#9 Final evaluation and calibration](https://github.com/SatvikSeetharaman/cmor438-soccer-analytics/issues/9)
- [#10 Monitoring, promotion and release](https://github.com/SatvikSeetharaman/cmor438-soccer-analytics/issues/10)

## Open these first

1. [Project overview](docs/PROJECT_OVERVIEW.md) for the flow from raw data to API.
2. [Validation results](docs/VALIDATION_RESULTS.md) for measured comparisons,
   uncertainty, and both figures.
3. [Professor meeting prep](docs/PROFESSOR_MEETING_PREP.md) for the scope,
   authorship, deadline, deployment and oral-defense questions.
4. [Course requirements](docs/COURSE_REQUIREMENTS.md) for the evidence checklist.

## Run and reproduce

From the repository root, use the commands below. The archive is under
`data/local/openfootball` and the build creates the ignored
`artifacts/football-reference` directory.

```sh
uv sync --locked
uv run python scripts/setup_project.py
uv run python scripts/download_football.py
uv run python scripts/run_football.py
uv run python scripts/verify_football_notebooks.py
uv run python scripts/smoke_football.py
uv run dsm-project serve --artifact artifacts/football-reference
```

Open `http://127.0.0.1:8000/` for the UI or `http://127.0.0.1:8000/docs`
for interactive API documentation. The server is launched on demand.
The full local experiment report is `data/local/football-validation.json`.
The [dataset card](docs/DATASET_CARD.md) records provenance and retrieval
checksums. Use `--figures data/local/figures` with the notebook verifier to
export the two main figures.

## What the evidence says

All 2,280 fixtures from six public, complete Premier League seasons passed
ingestion. The two expanding folds each evaluated 380 games. Logistic log
loss improved from 1.057 to 0.996 in 2022–23 and from 1.054 to 0.928 in
2023–24. Ridge MAE fell from 1.464 and 1.603 to 1.367 goals in each fold.
Logistic never ranked a draw first. PCA's first two components explained
about 82% of profile variance, with only moderate clustering separation.
These are real validation and descriptive results. The synthetic fixtures
are confined to tests. The 2024–25 final season is untouched.

## Verification and course status

- The original locked template, setup, `dsm-project verify`, and 63 tests
  passed. The expanded local pytest suite passed 70 tests. An earlier
  restricted run hit Windows temp-directory permissions. A retry with a
  writable test directory passed without modifying the assertions.
- The two football notebooks ran from clean kernels. Ruff, the package build
  and the template smoke path passed locally. The wheel includes the football
  UI. The source distribution excludes the downloaded data. Local Docker
  verification was unavailable because Docker was absent from the shell.
- Real-data API and Chrome smoke tests exercised form selection and validation.
  Invalid requests logged 422 with request IDs. Reversed feature order
  triggered a startup rejection and the original bundle recovered.
- The current evidence covers lawful data, point-in-time features, reusable
  task code, preliminary experiments, a versioned reference artifact, local
  product behavior, and an incident record. Student algorithms and their
  reviews, a locked final-test protocol, student-model promotion, and
  deployment with delayed quality monitoring remain open.

The [AI assistance log](docs/AI_ASSISTANCE_LOG.md) describes the generated
code, analysis and documentation. The [learning guide](docs/STUDENT_LEARNING_GUIDE.md)
walks through Elo, regression and softmax and lists verification exercises.

## Three next tasks

1. Confirm course scope, AI authorship policy and individual or team review
   expectations with Professor Dávila.
2. Reproduce one lagged feature row by hand, then independently implement
   and verify the first approved student estimator.
3. Audit draw calibration, agree on model acceptance and uncertainty rules,
   and freeze the selection protocol before opening 2024–25.
