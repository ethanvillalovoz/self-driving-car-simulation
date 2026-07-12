"""Deterministic data preparation and training for behavioral cloning."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from .preprocessing import image_preprocess


@dataclass(frozen=True)
class DrivingRecord:
    center: Path
    left: Path
    right: Path
    steering: float


@dataclass(frozen=True)
class CameraSample:
    image: Path
    steering: float


@dataclass(frozen=True)
class TrainingConfig:
    data_dir: Path = Path("data")
    output_model: Path = Path("model/model.keras")
    seed: int = 42
    max_samples_per_bin: int = 400
    steering_correction: float = 0.15
    validation_fraction: float = 0.2
    batch_size: int = 100
    epochs: int = 10

    def __post_init__(self) -> None:
        if self.max_samples_per_bin <= 0 or self.batch_size <= 0 or self.epochs <= 0:
            raise ValueError("Sample cap, batch size, and epochs must be positive")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("Validation fraction must be between 0 and 1")


def _camera_path(raw_path: str, image_dir: Path) -> Path:
    filename = raw_path.strip().replace("\\", "/").rsplit("/", 1)[-1]
    if not filename:
        raise ValueError("Camera path is empty")
    return image_dir / filename


def read_driving_log(csv_path: Path, image_dir: Path) -> list[DrivingRecord]:
    records: list[DrivingRecord] = []
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        for line_number, row in enumerate(csv.reader(stream), start=1):
            if not row or all(not value.strip() for value in row):
                continue
            if len(row) < 4:
                raise ValueError(f"Line {line_number} has fewer than four columns")
            try:
                steering = float(row[3])
            except ValueError as exc:
                if line_number == 1 and row[3].strip().lower() == "steering":
                    continue
                raise ValueError(f"Line {line_number} has an invalid steering value") from exc
            records.append(
                DrivingRecord(
                    center=_camera_path(row[0], image_dir),
                    left=_camera_path(row[1], image_dir),
                    right=_camera_path(row[2], image_dir),
                    steering=steering,
                )
            )
    if not records:
        raise ValueError("Driving log contains no usable records")
    return records


def balance_records(
    records: Sequence[DrivingRecord],
    *,
    bins: int = 25,
    max_per_bin: int = 400,
    seed: int = 42,
) -> list[DrivingRecord]:
    if bins <= 0 or max_per_bin <= 0:
        raise ValueError("Bins and per-bin cap must be positive")
    steering = np.asarray([record.steering for record in records], dtype=float)
    if steering.size == 0:
        raise ValueError("Cannot balance an empty record set")
    _, edges = np.histogram(steering, bins=bins)
    assignments = np.clip(np.digitize(steering, edges[1:-1]), 0, bins - 1)
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for index in range(bins):
        candidates = np.flatnonzero(assignments == index)
        rng.shuffle(candidates)
        selected.extend(candidates[:max_per_bin].tolist())
    return [records[index] for index in sorted(selected)]


def expand_camera_samples(
    records: Sequence[DrivingRecord],
    *,
    correction: float = 0.15,
) -> list[CameraSample]:
    samples: list[CameraSample] = []
    for record in records:
        samples.extend(
            [
                CameraSample(record.center, record.steering),
                CameraSample(record.left, record.steering + correction),
                CameraSample(record.right, record.steering - correction),
            ]
        )
    return samples


def split_samples(
    samples: Sequence[CameraSample],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[CameraSample], list[CameraSample]]:
    if len(samples) < 2:
        raise ValueError("At least two samples are required")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(samples))
    validation_count = max(1, round(len(samples) * validation_fraction))
    validation_indices = set(indices[:validation_count].tolist())
    training = [
        sample for index, sample in enumerate(samples) if index not in validation_indices
    ]
    validation = [sample for index, sample in enumerate(samples) if index in validation_indices]
    return training, validation


def load_rgb(path: Path) -> NDArray[np.uint8]:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read camera frame: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def augment(
    image: NDArray[np.uint8],
    steering: float,
    rng: np.random.Generator,
) -> tuple[NDArray[np.uint8], float]:
    height, width = image.shape[:2]
    translation_x = float(rng.uniform(-0.1, 0.1) * width)
    translation_y = float(rng.uniform(-0.05, 0.05) * height)
    transform = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
    shifted = cv2.warpAffine(image, transform, (width, height), borderMode=cv2.BORDER_REFLECT)
    hsv = cv2.cvtColor(shifted, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 2] *= float(rng.uniform(0.45, 1.15))
    brightened = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
    if rng.random() < 0.5:
        return cv2.flip(brightened, 1), -steering
    return brightened, steering


def batch_generator(
    samples: Sequence[CameraSample],
    *,
    batch_size: int,
    training: bool,
    seed: int,
) -> Iterator[tuple[NDArray[np.float32], NDArray[np.float32]]]:
    if not samples:
        raise ValueError("Cannot generate batches from an empty sample set")
    rng = np.random.default_rng(seed)
    while True:
        batch_images = []
        batch_steering = []
        for index in rng.integers(0, len(samples), size=batch_size):
            sample = samples[int(index)]
            image = load_rgb(sample.image)
            steering = sample.steering
            if training:
                image, steering = augment(image, steering, rng)
            batch_images.append(image_preprocess(image))
            batch_steering.append(steering)
        yield (
            np.asarray(batch_images, dtype=np.float32),
            np.asarray(batch_steering, dtype=np.float32),
        )


def build_nvidia_model():
    from tensorflow import keras

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(66, 200, 3)),
            keras.layers.Conv2D(24, 5, strides=2, activation="elu"),
            keras.layers.Conv2D(36, 5, strides=2, activation="elu"),
            keras.layers.Conv2D(48, 5, strides=2, activation="elu"),
            keras.layers.Conv2D(64, 3, activation="elu"),
            keras.layers.Conv2D(64, 3, activation="elu"),
            keras.layers.Dropout(0.5),
            keras.layers.Flatten(),
            keras.layers.Dense(100, activation="elu"),
            keras.layers.Dense(50, activation="elu"),
            keras.layers.Dense(10, activation="elu"),
            keras.layers.Dense(1),
        ],
        name="nvidia_behavioral_cloning",
    )
    model.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=1e-3))
    return model


def train(config: TrainingConfig) -> dict[str, list[float]]:
    records = read_driving_log(config.data_dir / "driving_log.csv", config.data_dir / "IMG")
    balanced = balance_records(
        records,
        max_per_bin=config.max_samples_per_bin,
        seed=config.seed,
    )
    samples = expand_camera_samples(balanced, correction=config.steering_correction)
    training, validation = split_samples(
        samples,
        validation_fraction=config.validation_fraction,
        seed=config.seed,
    )
    model = build_nvidia_model()
    history = model.fit(
        batch_generator(
            training,
            batch_size=config.batch_size,
            training=True,
            seed=config.seed,
        ),
        steps_per_epoch=max(1, len(training) // config.batch_size),
        epochs=config.epochs,
        validation_data=batch_generator(
            validation,
            batch_size=config.batch_size,
            training=False,
            seed=config.seed + 1,
        ),
        validation_steps=max(1, len(validation) // config.batch_size),
    )
    config.output_model.parent.mkdir(parents=True, exist_ok=True)
    model.save(config.output_model)
    history_path = config.output_model.with_suffix(".history.json")
    serialized_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }
    history_path.write_text(
        json.dumps(
            {
                "config": serialized_config,
                "history": history.history,
            },
            indent=2,
        )
        + "\n"
    )
    return history.history
