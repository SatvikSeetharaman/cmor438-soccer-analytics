# Student learning guide

## Reconstruct one row on paper

Pick a day with two fixtures. Make a table of each club's last five completed
matches as of the start of the day. For a club with results 2–0 and 1–1,
the form is 1.5 goals scored, 0.5 goals conceded and 2 points per match.
Through the match of interest, those summaries stay fixed for that date. After
the date, add results. In `data/football_features.py`, `replay` separates
reading the history from writing results.

Each club starts at Elo 1500. With a 60-point home boost and two equal ratings,
the expected home score is 1 / (1 + 10^(-60/400)), about 0.5855. A home
win adds about 8.29 Elo to the home club and subtracts the same from the away
club. A draw counts as 0.5 for the update. Elo is a rating feature and the
expected Elo score is different from a three-way win probability.

## Derive and verify the estimator you own

For a design matrix X and signed goal differences y, write predictions Xw+b
and define e = Xw+b−y. With half mean squared loss, the weight gradient is
Xᵀe/n and the intercept gradient is mean(e). Add λw for a λ‖w‖²/2 penalty.
A gradient step subtracts the learning rate times these derivatives. Try
x=[0,1,2] and y=[1,3,5]. The solution is slope 2 and intercept 1.
Check finite differences at a point away from the optimum. The template's
`BatchLinearRegressor` and scikit-learn least squares provide two independent
comparison paths. Match penalty conventions and intercept treatment before
comparing library coefficients.

For multinomial logistic, the model gives each row three affine scores.
Subtract the maximum score before exponentiating and normalize by the sum.
For scores [0,0,0] the probabilities are all 1/3, so a realized class has log
loss log(3). For a one-hot target, the log-loss derivative with respect to
each score is its predicted probability minus its target indicator. Check
this gradient numerically. Check probability sum and class ordering after a
row permutation. An intercept-only fit should recover training class shares
when every outcome is present.

## Connect each family to a verification task

| Family | Objective or prediction rule | First check and caveat |
| --- | --- | --- |
| Linear and ridge | Squared loss, with optional squared-weight penalty | Analytic line, gradient sign and convergence. Scale training data before penalizing |
| Logistic | Cross-entropy plus L2 penalty | Uniform softmax, finite differences and extreme scores. Compare probabilities after matching label order |
| Neighbors | Standardized Euclidean distance, local target mean or vote | Three points with one clear nearest neighbor. Set deterministic tie rules and inspect zero class probabilities |
| SVM | Hinge loss plus squared-weight penalty, decision by margin | Separate two 1D classes and check hinge subgradient. Tune C and calibrate on later data if needed |
| Trees | Split by weighted impurity reduction, follow learned thresholds | Two pure child nodes should yield maximum gain on a balanced binary node. Match tie and minimum-leaf rules |
| Forest and boosting | Average bootstrap trees, or add learners that reduce stage loss | One tree or boosting stage should match its base learner when configured identically. Bound depth, stages and learning rate |
| Neural networks | Layered affine maps and nonlinearities with cross-entropy | Compare a tiny backpropagation gradient to centered finite differences. Control initialization and iteration budget |
| K-means | Assign to nearest centroid and replace centroids with assigned means | Two compact point pairs with centers known by hand. Compare partitions across seeds up to label permutation |
| PCA | Center or standardize, then project onto top covariance directions | Collinear data should reconstruct with one component. Compare subspaces and explained variance because signs are arbitrary |

Follow the [student implementation guide](../STUDENT_IMPLEMENTATION_GUIDE.md).
Choose an interface with `fit` and `predict` or `transform`, then test invalid
shape, invalid parameters, the unfitted state, and seeded reproducibility.
Adapt the registered course exemplar interface and document the training
fold. Add new student registrations only when authorship and review are clear.

## Read the evidence

MAE is an average absolute miss measured in goals. Log loss averages minus the
log of probability assigned to the observed class. Assigning zero to an event
that happens is heavily penalized. Brier is mean squared error summed across
the three probabilities. Balanced accuracy averages per-class recall.
A model can improve log loss while producing zero draw decisions because
argmax uses only the top probability. Use 28-day block bootstrap results to
discuss variation without treating games as independent. Teams recur across
blocks, so try longer blocks or an alternative season-level design.

Read `configs/football.toml`, the adapter, `football_features.py`,
`experiments/football.py`, and `service/football.py` in order. For an oral
defense, explain what a cold start means, why the standardizer fits on earlier
data, why signed goal difference is a different target from win probability,
and how a request ID connects a result to operations.
