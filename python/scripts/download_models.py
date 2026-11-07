"""Download the published models from Hugging Face into runs/, where the other scripts expect them.

    uv run --extra models python scripts/download_models.py

Gives runs/corners/c7/{finder,refiner}.pt and runs/reader/r4/best.pt (and the earlier models, c2 and r2).
"""
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = "darshanvss/cubecheck"
REVISION = "v0.2.0"  # a release tag, so a checkout gets the weights it was tested with


def main() -> None:
    runs = Path(__file__).resolve().parents[2] / "runs"
    snapshot_download(REPO, revision=REVISION, local_dir=runs, allow_patterns=["corners/*", "reader/*"])
    print(f"models in {runs}")


if __name__ == "__main__":
    main()
