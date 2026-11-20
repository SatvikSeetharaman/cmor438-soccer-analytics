"""Fetch only the pinned development fixtures and write a local checksum manifest."""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from urllib.request import urlopen

from dsm_project.config import load_config
from dsm_project.data.openfootball import (
    data_directory,
    development_seasons,
    parse_season,
    source_url,
)


def main() -> None:
    """Fetch and validate all files before replacing the local archive."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/football.toml")
    config = load_config(parser.parse_args().config)
    root = data_directory(config)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"retrieved_at": datetime.now(UTC).isoformat(), "files": {}}
    files = {}
    for season in development_seasons(config):
        url = source_url(config, season)
        with urlopen(url, timeout=20) as response:
            content = response.read(1_000_001)
        if len(content) > 1_000_000:
            raise ValueError(f"Fixture file exceeds the 1 MB budget: {season}")
        rows = parse_season(content, season)
        files[season] = content
        manifest["files"][season] = {
            "url": url,
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
            "matches": len(rows),
        }
    for season, content in files.items():
        (root / f"{season}.json").write_bytes(content)
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Verified {len(files)} seasons and wrote {root / 'manifest.json'}")


if __name__ == "__main__":
    main()
