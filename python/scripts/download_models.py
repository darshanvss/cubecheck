"""Download the published models from Hugging Face into runs/, where the other scripts expect them.

    uv run --extra models python scripts/download_models.py

Gives runs/corners/c2/{finder,refiner}.pt and runs/reader/r2/best.pt.
"""
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = "darshanvss/cubecheck-3x3"
REVISION = "v0.1.0"  # a release tag, so a checkout gets the weights it was tested with


def main() -> None:
    runs = Path(__file__).resolve().parents[2] / "runs"
    snapshot_download(REPO, revision=REVISION, local_dir=runs, allow_patterns=["corners/*", "reader/*"])
    print(f"models in {runs}")


if __name__ == "__main__":
    main()
