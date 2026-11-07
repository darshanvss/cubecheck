"""Download the CubeCheck renders from Hugging Face and unpack them into data/raw/, where training expects them.

    uv run --extra models python scripts/download_dataset.py                     # everything, about 19 GB
    uv run --extra models python scripts/download_dataset.py 444 777             # whole cube sizes
    uv run --extra models python scripts/download_dataset.py 333/test-full-v1    # single sets

Gives data/raw/<size>/<set>/{images,labels}/ and the set's manifest.json, as docs/training.md uses them
(<size> is the WCA event id, 222 ... 777, or none). The dataset is darshanvss/cubecheck at a release tag;
its shards are WebDataset TAR files (data/<size>/<set>-NNNN.tar) that `datasets.load_dataset` can also read
directly, without unpacking.
"""
import argparse
import shutil
import tarfile
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = "darshanvss/cubecheck"
REVISION = "v1.0"  # a release tag: the renders c7 and r4 were trained on


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="*", help="sizes (444) or sets (444/test-mixed-v2); none given: everything")
    ap.add_argument("--keep-shards", action="store_true", help="keep the downloaded TAR files")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2] / "data"
    cache = root / "hf-shards"
    patterns = [f"data/{s}/*.tar" if "/" not in s else f"data/{s}-*.tar" for s in args.sets] or ["data/*/*.tar"]
    patterns += [f"manifests/{s}/*.json" if "/" not in s else f"manifests/{s}.json" for s in args.sets] or ["manifests/*/*.json"]
    snapshot_download(REPO, repo_type="dataset", revision=REVISION, local_dir=cache, allow_patterns=patterns + ["sources.json"])

    for shard in sorted((cache / "data").glob("*/*.tar")):
        size, name = shard.parent.name, shard.stem.rsplit("-", 1)[0]  # 444/test-mixed-v2-0000.tar
        out = root / "raw" / size / name
        (out / "images").mkdir(parents=True, exist_ok=True)
        (out / "labels").mkdir(parents=True, exist_ok=True)
        with tarfile.open(shard) as tar:
            for member in tar:
                if not member.isfile() or "/" in member.name:
                    continue  # the shards hold flat syn_NNNNNNN.jpg / .json pairs only
                folder = "images" if member.name.endswith(".jpg") else "labels"
                with tar.extractfile(member) as src, open(out / folder / member.name, "wb") as dst:
                    shutil.copyfileobj(src, dst)
        manifest = cache / "manifests" / size / f"{name}.json"
        if manifest.exists():
            shutil.copy(manifest, out / "manifest.json")
        print(f"{size}/{name}: {shard.name}")
    if not args.keep_shards:
        shutil.rmtree(cache / "data", ignore_errors=True)
    print(f"renders in {root / 'raw'}")


if __name__ == "__main__":
    main()
