"""Verified downloads for the model, data, and simulator release artifacts."""

from __future__ import annotations

import argparse
import hashlib
import shutil
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
    sha256: str
    expected_paths: tuple[str, ...]


ARTIFACTS = {
    "model": Artifact(
        name="trained model",
        archive="self-driving-car-model-v1.tar.gz",
        sha256="a4a551864792d24c78af9b31efcea1173207787bd72c939325a4c3795d0fc483",
        expected_paths=("model/model.h5",),
    ),
    "data": Artifact(
        name="training data",
        archive="self-driving-car-data-v1.tar.gz",
        sha256="5ebb75a78a11ca35f05b285674240e5f42f534c9094b183f80fa58a38162f845",
        expected_paths=("data/driving_log.csv", "data/IMG"),
    ),
    "simulator": Artifact(
        name="Linux simulator",
        archive="self-driving-car-simulator-linux-v1.tar.gz",
        sha256="d8e5d79e7757e8cea4e26cfd6287313cdbdd108a3a972d6e88f0723e5cd2baa8",
        expected_paths=("simulator-linux",),
    ),
}


def release_url(repo: str, tag: str, archive: str) -> str:
    return f"https://github.com/{repo}/releases/download/{tag}/{archive}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(
            f"Checksum mismatch for {path.name}: expected {expected}, got {actual}"
        )


def download(url: str, destination: Path) -> None:
    print(f"Downloading {url}")
    with urllib.request.urlopen(url, timeout=60) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)


def _safe_target(destination: Path, member_name: str) -> Path:
    root = destination.resolve()
    target = (destination / member_name).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"Refusing to extract unsafe path: {member_name}")
    return target


def safe_extract(archive: Path, destination: Path) -> None:
    """Extract regular files and directories without trusting tar links or devices."""
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            target = _safe_target(destination, member.name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError(f"Refusing to extract non-regular file: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                raise ValueError(f"Could not read archive member: {member.name}")
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def artifact_exists(artifact: Artifact, root: Path) -> bool:
    return all((root / path).exists() for path in artifact.expected_paths)


def restore_artifact(
    artifact: Artifact,
    *,
    repo: str,
    tag: str,
    destination: Path,
    force: bool = False,
) -> None:
    if artifact_exists(artifact, destination) and not force:
        print(f"Skipping {artifact.name}; expected files already exist.")
        return
    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = Path(tmpdir) / artifact.archive
        download(release_url(repo, tag, artifact.archive), archive_path)
        verify_sha256(archive_path, artifact.sha256)
        safe_extract(archive_path, destination)
    if not artifact_exists(artifact, destination):
        raise RuntimeError(f"Archive did not contain the expected {artifact.name} paths")
    print(f"Restored {artifact.name}.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Restore verified GitHub release artifacts.")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="GitHub repo owner/name")
    parser.add_argument("--tag", default=DEFAULT_TAG, help="Release tag")
    parser.add_argument("--destination", type=Path, default=Path.cwd())
    parser.add_argument("--model", action="store_true")
    parser.add_argument("--data", action="store_true")
    parser.add_argument("--simulator", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    selected = [
        key for key in ("model", "data", "simulator") if args.all or getattr(args, key)
    ] or ["model"]
    for key in selected:
        restore_artifact(
            ARTIFACTS[key],
            repo=args.repo,
            tag=args.tag,
            destination=args.destination,
            force=args.force,
        )


if __name__ == "__main__":
    main()
