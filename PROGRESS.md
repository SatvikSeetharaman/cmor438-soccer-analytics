# Development progress

- Found the existing course template on `main` with a clean working tree and
  the intended GitHub origin. The restricted shell could not verify GitHub
  CLI auth. The authorized connection verified the repository and login.
- Locked setup and original template verification passed. The untouched suite
  reported 63 passing tests, including the example and guided notebooks.
- GitHub confirmed the existing origin is private. Work is on
  `project/football-foundation`.
- Downloaded and validated six pinned seasons, 2,280 fixtures. Saved the SHA-256
  manifest locally. The download excludes 2024–25.
- Ran expanding-season reference comparisons. Ridge improved on the
  mean baseline in both folds. Regularized logistic improved log loss in both
  folds while achieving zero draw recall at argmax.
- Data and feature boundary committed as `10d36b1`. The public source card
  and manifest identify the raw bytes. Offline tests verify cold starts,
  chronology, validation isolation and rejection of malformed fixtures.
- Reference experiments and notebooks committed as `9e57e01`.
  [Validation evidence](docs/VALIDATION_RESULTS.md) includes paired block
  intervals, draw diagnostics, drift checks, and PCA/K-means profiles.
  Both football case studies passed from clean kernels.
- Verified the local browser journey, real-data API smoke and controlled
  feature-order incident. The final local artifact is
  `football-reference-v1-f5a6e664a223`. The 70-test suite and Ruff passed.
- [Morning handoff](MORNING_HANDOFF.md) links the GitHub work and reproduction
  instructions. The next work is independent student implementation and
  reviewed model selection. Keep 2024–25 sealed until the protocol is frozen.
