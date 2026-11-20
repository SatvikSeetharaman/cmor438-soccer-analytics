# Model card

The local artifact contains scikit-learn multinomial logistic regression
(C=0.1) and ridge regression (α=10). Both use standardization fit on the
training partition. The reference demonstration uses 1,140 training matches,
from 2019–20 through 2021–22. The version, exact feature and class order,
configuration fingerprint, package version and six archive checksums are
available through `GET /model`. Rebuild with `scripts/run_football.py`.

The demo accepts dates in 2022–23 and 2023–24. Pre-match feature values come
from earlier dates. The model choice uses validation evidence in hindsight.
The expanding run fits a newer model for the second fold. Its 2023–24 scores
are comparison evidence and describe a different fit from the local
historical bundle. The choice favors strong simple reference candidates before
student implementations are reviewed. The separate heads can imply slightly
different intuitions about goals and outcomes.

See the [validation report](VALIDATION_RESULTS.md) for retained models, fold
sizes, metrics, uncertainty and diagnostics. Positional class probabilities
use the order away, draw, home. Draw was never the argmax for the selected
reference in either fold. Confidence reliability changes between years and
sparse club evidence is small. Historical profiles omit injuries, sub-league
form and calendar-time decay. The model is exploratory and provides no live
decision guarantee. No final-test opening or student-model promotion took
place. See the [operations guide](OPERATIONS.md) for monitoring and rollback.
