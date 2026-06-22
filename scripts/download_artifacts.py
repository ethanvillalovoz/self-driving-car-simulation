"""Download release artifacts needed for training and simulator inference."""

from __future__ import annotations

import argparse
import os
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path


DEFAULT_REPO = "ethanvillalovoz/self-driving-car-simulation"
DEFAULT_TAG = "v1.0.0"


@dataclass(frozen=True)
class Artifact:
    name: str
    archive: str
    expected_paths: tuple[str, ...]


ARTIFACTS = {
    "model": Artifact(
        name="trained model",
        archive="self-driving-car-model-v1.tar.gz",
        expected_paths=("model/model.h5",),
    ),
    "data": Artifact(
        name="training data",
        archive="self-driving-car-data-v1.tar.gz",
        expected_paths=("data/driving_log.csv", "data/IMG"),
    ),
    "simulator": Artifact(
        name="Linux simulator",
        archive="self-driving-car-simulator-linux-v1.tar.gz",
        expected_paths=("simulator-linux",),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the model, data, or simulator from GitHub Releases."
    )
    parser.add_argument("--repo", default=DEFAULT_REPO, help="GitHub repo owner/name.")
    parser.add_argument("--tag", default=DEFAULT_TAG, help="Release tag to download.")
    parser.add_argument("--model", action="store_true", help="Download model/model.h5.")
    parser.add_argument("--data", action="store_true", help="Download training data.")
    parser.add_argument(
        "--simulator", action="store_true", help="Download the Linux simulator."
    )
    parser.add_argument("--all", action="store_true", help="Download all artifacts.")
    parser.add_argument(
        "--force", action="store_true", help="Download and extract even if files exist."
    )
    return parser.parse_args()


def selected_artifacts(args: argparse.Namespace) -> list[str]:
    selected = [
        key
        for key in ("model", "data", "simulator")
        if args.all or getattr(args, key)
    ]
    return selected or ["model"]


def artifact_exists(artifact: Artifact) -> bool:
    return all(Path(path).exists() for path in artifact.expected_paths)


def release_url(repo: str, tag: str, archive: str) -> str:
    return f"https://github.com/{repo}/releases/download/{tag}/{archive}"


def download(url: str, destination: Path) -> None:
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response:
        with destination.open("wb") as file:
            file.write(response.read())


def safe_extract(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            member_path = (destination / member.name).resolve()
            if os.path.commonpath([root, member_path]) != str(root):
                raise RuntimeError(f"Refusing to extract unsafe path: {member.name}")
        tar.extractall(destination)


def main() -> None:
    args = parse_args()
    for key in selected_artifacts(args):
        artifact = ARTIFACTS[key]
        if artifact_exists(artifact) and not args.force:
            print(f"Skipping {artifact.name}; expected files already exist.")
            continue

        url = release_url(args.repo, args.tag, artifact.archive)
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / artifact.archive
            download(url, archive_path)
            safe_extract(archive_path, Path.cwd())
        print(f"Restored {artifact.name}.")


if __name__ == "__main__":
    main()
