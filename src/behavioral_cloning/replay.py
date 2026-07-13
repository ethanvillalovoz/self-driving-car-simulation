"""Camera ordering contracts for offline behavioral-cloning replays."""

from __future__ import annotations

from pathlib import Path

from .training import DrivingRecord

CAMERA_LABELS = ("LEFT", "CENTER", "RIGHT")


def replay_camera_paths(record: DrivingRecord) -> tuple[Path, Path, Path]:
    """Return camera paths in the left-to-right order used by the replay."""
    return record.left, record.center, record.right


def replay_model_camera_path(record: DrivingRecord) -> Path:
    """Return the center camera associated with the recorded steering target."""
    return record.center
