# Professor meeting prep

Football Intelligence Engine uses previous Premier League results to compare
two clubs, predict signed goal difference and estimate three-way probabilities.
Six complete openfootball seasons provide daily dates, scores and consistent
club identities. The small, public-domain archive allows reproducible
match-level experiments. We selected it over the discontinuous Premier League
coverage in StatsBomb Open Data.

In the 2022–23 and 2023–24 chronological validation folds, the regularized
logistic reference achieved log loss 0.996 and 0.928 against class-frequency
baselines of 1.057 and 1.054. Ridge reduced goal-difference MAE to 1.367 in both
seasons. Logistic never predicted draw as its top class. Show the
[reliability figure](evidence/reliability.png) and [profile PCA](evidence/team-profiles.png).
2024–25 is sealed.

The local UI accepts historical dates and teams, returns probabilities, goal
difference and form, and identifies the reference artifact. The test suite
checks date leakage, and an incident exercise rejected reversed feature order.
This preliminary code and analysis are AI-assisted. Student implementations,
review, final-test protocol, and deployment remain open.

## Discuss with Professor Dávila

1. Is the single-league scope sufficient, and would you prioritize draw-specific
   calibration or cross-competition replication next?
2. Can this be an individual project? How should required peer review and
   contribution evidence work in that case?
3. Which AI-assisted activities are permitted, how should we disclose them, and
   what must be written independently? Can attributed adaptation of course
   implementations count under any conditions?
4. Which combinations of supervised, ensemble, neural and unsupervised methods
   meet the coverage standard here? Is a calibrated linear SVM meaningful for
   these probability targets, and what constitutes an approved exclusion?
5. What is the final deadline and oral-defense format? What operational and
   deployment evidence should the final release include?
6. Does the proposed expanding-window design and sealed 2024–25 season match
   your expectations? Should we test uncertainty with a different time block?
