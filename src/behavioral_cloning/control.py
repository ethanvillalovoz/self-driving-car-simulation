"""Validated steering and throttle prediction."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from .preprocessing import decode_telemetry_image, image_preprocess


class PredictionModel(Protocol):
    def predict(self, batch: NDArray[np.float32], **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class ControlConfig:
    speed_limit: float = 10.0
    steering_limit: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.speed_limit) or self.speed_limit <= 0:
            raise ValueError("Speed limit must be a finite positive number")
        if not math.isfinite(self.steering_limit) or self.steering_limit <= 0:
            raise ValueError("Steering limit must be a finite positive number")


@dataclass(frozen=True)
class ControlCommand:
    steering_angle: float
    throttle: float


def calculate_throttle(speed: float | str, speed_limit: float = 10.0) -> float:
    """Return the original proportional controller, clamped to a safe range."""
    try:
        numeric_speed = float(speed)
        numeric_limit = float(speed_limit)
    except (TypeError, ValueError) as exc:
        raise ValueError("Speed and speed limit must be numeric") from exc
    if not math.isfinite(numeric_speed):
        raise ValueError("Speed must be finite")
    if not math.isfinite(numeric_limit) or numeric_limit <= 0:
        raise ValueError("Speed limit must be a finite positive number")
    return float(np.clip(1.0 - numeric_speed / numeric_limit, 0.0, 1.0))


def _predict_scalar(model: PredictionModel, batch: NDArray[np.float32]) -> float:
    try:
        prediction = model.predict(batch, verbose=0)
    except TypeError:
        prediction = model.predict(batch)
    values = np.asarray(prediction, dtype=float)
    if values.size != 1:
        raise ValueError("Steering model must return exactly one value per frame")
    steering = float(values.reshape(-1)[0])
    if not math.isfinite(steering):
        raise ValueError("Steering model returned a non-finite value")
    return steering


def predict_control(
    telemetry: Mapping[str, Any],
    model: PredictionModel,
    config: ControlConfig | None = None,
) -> ControlCommand:
    """Validate one telemetry payload and produce a bounded control command."""
    active_config = config or ControlConfig()
    if "speed" not in telemetry or "image" not in telemetry:
        raise ValueError("Telemetry must include speed and image fields")

    image = decode_telemetry_image(telemetry["image"])
    batch = np.expand_dims(image_preprocess(image), axis=0)
    raw_steering = _predict_scalar(model, batch)
    steering = float(
        np.clip(raw_steering, -active_config.steering_limit, active_config.steering_limit)
    )
    throttle = calculate_throttle(telemetry["speed"], active_config.speed_limit)
    return ControlCommand(steering_angle=steering, throttle=throttle)
