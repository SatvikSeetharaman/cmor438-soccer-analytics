"""Execute the football case studies in clean kernels."""

import argparse
from base64 import b64decode
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = {
    "01-does-team-history-help.ipynb": "reliability.png",
    "02-what-do-scoring-profiles-capture.ipynb": "team-profiles.png",
}


def main() -> None:
    """Validate notebook execution and optionally export the main figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", type=Path)
    figures = parser.parse_args().figures
    if figures:
        figures.mkdir(parents=True, exist_ok=True)
    for filename, figure in NOTEBOOKS.items():
        notebook = nbformat.read(ROOT / "notebooks" / filename, as_version=4)
        NotebookClient(
            notebook,
            kernel_name="dsm-project",
            timeout=120,
            resources={"metadata": {"path": str(ROOT)}},
        ).execute()
        if figures:
            images = [
                result.data["image/png"]
                for cell in notebook.cells
                for result in cell.get("outputs", [])
                if "image/png" in result.get("data", {})
            ]
            (figures / figure).write_bytes(b64decode(images[-1]))
        print(f"Clean kernel passed: {filename}")


if __name__ == "__main__":
    main()
