# Product contract

## User and decision

A soccer analyst can compare historical Premier League pairings and understand
how team strength and recent form relate to outcomes. The demo supports
historical dates from the first to last match in 2022–23 and 2023–24 and
distinct clubs in that season. The pairing may be hypothetical. A model
trained through May 2022 is used throughout the interface.

## Tasks and course methods

See the [dataset card](DATASET_CARD.md) for availability and provenance and the
[learning guide](STUDENT_LEARNING_GUIDE.md) for derivations and checks.
The following statuses refer to trusted-library reference experiments.

| Family | Project task | Status and justification |
| --- | --- | --- |
| Historical baselines | Signed goal difference, H/D/A | Training mean and empirical class frequencies |
| Linear regression, ridge and gradient descent | Signed goal difference | OLS and ridge compared. Gradient descent is an applicable student implementation route |
| Logistic regression | H/D/A probabilities | Two penalty strengths compared to the historical baseline |
| Nearest neighbors | Both supervised targets | Tests whether local similarity adds value. Small neighborhoods can produce brittle zero probabilities |
| Linear SVM and kernels | H/D/A classification | Linear margins fitted on earlier seasons, calibrated on the latest training season. Nonlinear kernels are deferred |
| Trees | Both supervised targets | Shallower trees test cutoffs and interactions |
| Ensembles | Both supervised targets | Forest classification and gradient boosting report retained. Test variance and nonlinearity |
| Neural networks | H/D/A classification | Single 12-unit hidden layer with regularization and a capped iteration budget |
| K-means | Retrospective team profiles | Tests whether venue-adjusted scoring summaries form separated groups |
| PCA | Retrospective team profiles | Tests how much scoring variation can be summarized in two dimensions |

Event-level expected goals and cross-league predictions are deferred on
data coverage grounds. No course algorithm family is claimed as exempt.

## Evaluation

Use expanding folds with MAE for signed goal difference, log loss for
probabilities, and secondary RMSE, R-squared, Brier, balanced accuracy,
confusion, confidence and subgroup diagnostics. Compare paired errors by
28-day calendar blocks. Rank preliminary candidates by the mean validation
metric, keeping all candidates. For course promotion, freeze the scope,
metrics, dependence checks, and acceptance thresholds with genuine review
before touching 2024–25. Validate classwise reliability and sparse clubs
before drawing conclusions about live use.

## Interface and responsible use

The local web app shows a signed goal-difference estimate, home/draw/away
probabilities, both teams' earlier form, and model identity. It serves
`GET /health`, `GET /model`, `GET /teams`, `POST /predict`, and OpenAPI.
See [operations](OPERATIONS.md) for invocation and failure response. Treat
outputs as exploratory estimates for analyst judgment. The product scope
excludes automated betting decisions, live predictions, and guaranteed results.
