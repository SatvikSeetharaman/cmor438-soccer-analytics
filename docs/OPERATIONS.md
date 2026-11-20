# Operations and incident response

## Local service

From the repository root, run the downloader and training commands in the
[README](../README.md). Start with the CLI's default loopback binding.

```sh
uv run dsm-project serve --artifact artifacts/football-reference
uv run python scripts/smoke_football.py
```

The artifact is a locally rebuilt Joblib bundle. Use a bundle produced by
this project in the locked environment, as loading Joblib can execute code.
Startup checks the configured archive hashes, model feature names and class
order. Keep known-good candidate directories separate and archive each as a
unit with its config and data manifest. The artifact reads its config from the
path used during training. Rebuild after relocation if a custom absolute
path was supplied.

## Observability and response

Each prediction emits JSON with the model name, artifact version, request ID,
elapsed milliseconds, HTTP status, and a validation-failure flag. The response
includes the same request ID as a header. Request payloads are excluded from
these logs. The historical report tracks training-scaled feature mean shifts
and observed outcomes by season. It can be rerun when a reviewed archive is
available. The local app accepts hypothetical fixtures, so an automated
outcome join requires a separate actual-fixture identity workflow.

For local investigation, start with sustained 422s, any 5xx, or prediction
latency above one second across a batch. Look up the artifact version and
request ID, check `/model` and `/health`, and inspect the date/team schema or
the model, configuration, and raw data compatibility. These are manual
investigation triggers based on a local smoke exercise. Alerting, dashboard
and service-level targets require a real deployment and load measurements.

## Exercised incident and recovery

The local smoke test sent a valid Arsenal–Liverpool request on August 5,
2022, then an unknown club, an out-of-range date and an incomplete payload.
The valid request returned a complete response. The invalid requests returned
422 and emitted a JSON record with a request ID and validation-failure flag.
Next, the exercise reversed feature names in a disposable copy of the bundle.
Startup rejected the copy with a compatibility error. A restart of the
original bundle passed the health check. A separate headless Chrome journey
rendered the probabilities and profile rows and verified identical-club
validation. See the [smoke record](evidence/football-smoke.json).

When a candidate fails, stop it, restart with the previous compatible
artifact directory using the same command, verify its identity in `/model`,
then rerun health and a known valid matchup. When no known-good bundle is
available, rebuild from the recorded source/config/manifest and rerun smoke
checks. A deployed rollback, automatic drift alert and persistent
prediction-to-outcome join remain follow-up work.
