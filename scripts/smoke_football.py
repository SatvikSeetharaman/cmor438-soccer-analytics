"""Exercise prediction, invalid input, and recovery on the real local archive."""

import argparse
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

from dsm_project.monitoring import configure_logging
from dsm_project.service import create_app


def main() -> None:
    """Run the historical matchup path and a reversible artifact incident."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact", type=Path, default=Path("artifacts/football-reference")
    )
    artifact = parser.parse_args().artifact
    configure_logging()
    client = TestClient(create_app(artifact))
    assert client.get("/health").json()["status"] == "ok"
    assert "Football Intelligence Engine" in client.get("/").text
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200
    season = client.get("/teams").json()["2022-23"]
    payload = {"home": "Arsenal", "away": "Liverpool", "date": season["first"]}
    response = client.post("/predict", json=payload)
    response.raise_for_status()
    result = response.json()
    assert abs(sum(result["probabilities"].values()) - 1) < 1e-12
    assert result["request_id"] == response.headers["X-Request-ID"]
    assert result["artifact_version"] == client.get("/model").json()["artifact_version"]
    assert result["home_profile"]["recent_games"] == 5
    failure = client.post("/predict", json={**payload, "home": "Invalid club"})
    assert failure.status_code == 422 and failure.headers["X-Request-ID"]
    assert (
        client.post("/predict", json={**payload, "date": "2024-10-01"}).status_code
        == 422
    )
    assert client.post("/predict", json={"home": "Arsenal"}).status_code == 422
    with TemporaryDirectory(dir=artifact.parent) as temporary:
        copy = Path(temporary) / "mismatch"
        shutil.copytree(artifact, copy)
        metadata_path = copy / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["feature_names"].reverse()
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        try:
            create_app(copy)
        except ValueError as error:
            assert "mismatch" in str(error)
        else:
            raise AssertionError("Reversed feature order unexpectedly accepted")
        assert TestClient(create_app(artifact)).get("/health").status_code == 200
    print(
        json.dumps(
            {
                "prediction": result,
                "invalid_team_status": failure.status_code,
                "reversed_feature_order": "startup rejected",
                "original_artifact": "healthy",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
