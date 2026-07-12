import io
import tarfile
from pathlib import Path

import pytest

from behavioral_cloning.artifacts import safe_extract, sha256, verify_sha256


def write_archive(path: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for name, content in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))


def test_safe_extract_restores_regular_files(tmp_path):
    archive = tmp_path / "artifact.tar.gz"
    write_archive(archive, {"model/model.h5": b"model"})

    destination = tmp_path / "restored"
    safe_extract(archive, destination)

    assert (destination / "model/model.h5").read_bytes() == b"model"


def test_safe_extract_rejects_path_traversal(tmp_path):
    archive = tmp_path / "unsafe.tar.gz"
    write_archive(archive, {"../outside.txt": b"nope"})

    with pytest.raises(ValueError, match="unsafe path"):
        safe_extract(archive, tmp_path / "restored")


def test_safe_extract_rejects_symbolic_links(tmp_path):
    archive = tmp_path / "links.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        link = tarfile.TarInfo("model/latest.h5")
        link.type = tarfile.SYMTYPE
        link.linkname = "../../outside.h5"
        tar.addfile(link)

    with pytest.raises(ValueError, match="non-regular"):
        safe_extract(archive, tmp_path / "restored")


def test_checksum_verification(tmp_path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"verified")
    digest = sha256(artifact)

    verify_sha256(artifact, digest)
    with pytest.raises(ValueError, match="Checksum mismatch"):
        verify_sha256(artifact, "0" * 64)
