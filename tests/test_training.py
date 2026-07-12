import csv
from pathlib import Path

import pytest

from behavioral_cloning.training import (
    CameraSample,
    DrivingRecord,
    balance_records,
    expand_camera_samples,
    read_driving_log,
    split_samples,
)


def record(index: int, steering: float) -> DrivingRecord:
    return DrivingRecord(
        center=Path(f"center-{index}.jpg"),
        left=Path(f"left-{index}.jpg"),
        right=Path(f"right-{index}.jpg"),
        steering=steering,
    )


def test_read_driving_log_normalizes_mixed_platform_paths(tmp_path):
    log = tmp_path / "driving_log.csv"
    with log.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["C:\\capture\\center.jpg", "/capture/left.jpg", "right.jpg", "0.2"])

    records = read_driving_log(log, tmp_path / "IMG")

    assert records[0].center.name == "center.jpg"
    assert records[0].left.name == "left.jpg"
    assert records[0].steering == 0.2


def test_balance_records_is_deterministic_and_caps_bins():
    records = [record(index, 0.0) for index in range(20)] + [record(20, 1.0)]

    first = balance_records(records, bins=2, max_per_bin=3, seed=7)
    second = balance_records(records, bins=2, max_per_bin=3, seed=7)

    assert first == second
    assert len(first) == 4


def test_expand_camera_samples_applies_side_camera_correction():
    samples = expand_camera_samples([record(0, 0.25)], correction=0.15)

    assert [sample.steering for sample in samples] == pytest.approx([0.25, 0.4, 0.1])


def test_split_samples_is_deterministic_and_disjoint():
    samples = [CameraSample(Path(f"{index}.jpg"), float(index)) for index in range(10)]

    training, validation = split_samples(samples, validation_fraction=0.2, seed=42)
    repeated = split_samples(samples, validation_fraction=0.2, seed=42)

    assert (training, validation) == repeated
    assert len(training) == 8
    assert len(validation) == 2
    assert set(training).isdisjoint(validation)
