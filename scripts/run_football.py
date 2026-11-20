"""Run reference validation and profiling, then build a historical demo artifact."""

import argparse
import json
from pathlib import Path

from dsm_project.artifacts.football import train_demo
from dsm_project.config import load_config
from dsm_project.data import load_project_dataset
from dsm_project.experiments.football import run_validation
from dsm_project.experiments.team_profiles import profile_experiment


def main() -> None:
    """Write metric evidence and train the provisional local artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/football.toml")
    parser.add_argument(
        "--report", type=Path, default=Path("data/local/football-validation.json")
    )
    parser.add_argument(
        "--artifact", type=Path, default=Path("artifacts/football-reference")
    )
    args = parser.parse_args()
    config = load_config(args.config)
    dataset = load_project_dataset(config)
    report = run_validation(dataset.observations, config)
    _, report["team_profiling"] = profile_experiment(dataset.observations)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Validation evidence: {args.report}")
    print(f"Reference artifact: {train_demo(dataset, config, args.artifact)}")


if __name__ == "__main__":
    main()
