"""Compatibility entry point for the simulator control server."""

from behavioral_cloning.cli import main
from behavioral_cloning.control import (
    ControlCommand,
    ControlConfig,
    calculate_throttle,
    predict_control,
)
from behavioral_cloning.preprocessing import decode_telemetry_image, image_preprocess

__all__ = [
    "ControlCommand",
    "ControlConfig",
    "calculate_throttle",
    "decode_telemetry_image",
    "image_preprocess",
    "predict_control",
]


if __name__ == "__main__":
    main()
