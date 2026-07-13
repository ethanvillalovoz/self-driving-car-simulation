from pathlib import Path

from behavioral_cloning.replay import (
    CAMERA_LABELS,
    replay_camera_paths,
    replay_model_camera_path,
)
from behavioral_cloning.training import DrivingRecord


def test_replay_orders_cameras_and_predicts_from_center_frame():
    record = DrivingRecord(
        center=Path("center.jpg"),
        left=Path("left.jpg"),
        right=Path("right.jpg"),
        steering=0.2,
    )

    assert CAMERA_LABELS == ("LEFT", "CENTER", "RIGHT")
    assert replay_camera_paths(record) == (
        Path("left.jpg"),
        Path("center.jpg"),
        Path("right.jpg"),
    )
    assert replay_model_camera_path(record) == Path("center.jpg")
