import base64
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from behavioral_cloning.control import (
    ControlConfig,
    calculate_throttle,
    predict_control,
)
from behavioral_cloning.preprocessing import decode_telemetry_image, image_preprocess


def encoded_test_image(size=(320, 160)):
    image = Image.new("RGB", size, color=(128, 64, 32))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class FakeModel:
    def __init__(self, prediction=0.25):
        self.prediction = prediction

    def predict(self, batch, verbose=0):
        assert batch.shape == (1, 66, 200, 3)
        assert verbose == 0
        return np.array([[self.prediction]])


def test_image_preprocess_returns_normalized_model_input():
    image = np.full((160, 320, 3), 128, dtype=np.uint8)

    processed = image_preprocess(image)

    assert processed.shape == (66, 200, 3)
    assert processed.dtype == np.float32
    assert processed.min() >= 0.0
    assert processed.max() <= 1.0


def test_decode_telemetry_image_returns_rgb_array():
    decoded = decode_telemetry_image(encoded_test_image())

    assert decoded.shape == (160, 320, 3)
    assert decoded.dtype == np.uint8


@pytest.mark.parametrize(
    "encoded",
    ["", "not base64", base64.b64encode(b"not an image").decode()],
)
def test_decode_telemetry_image_rejects_invalid_payloads(encoded):
    with pytest.raises(ValueError, match=r"image|base64"):
        decode_telemetry_image(encoded)


def test_calculate_throttle_clamps_original_controller():
    assert calculate_throttle(0) == 1.0
    assert calculate_throttle(5) == 0.5
    assert calculate_throttle(10) == 0.0
    assert calculate_throttle(15) == 0.0
    assert calculate_throttle(-3) == 1.0


def test_predict_control_returns_bounded_command():
    command = predict_control(
        {"speed": "5", "image": encoded_test_image()},
        FakeModel(prediction=3.5),
        ControlConfig(steering_limit=1.0),
    )

    assert command.steering_angle == 1.0
    assert command.throttle == 0.5


@pytest.mark.parametrize(
    "telemetry",
    [
        {"image": encoded_test_image()},
        {"speed": "5"},
        {"speed": "fast", "image": encoded_test_image()},
    ],
)
def test_predict_control_rejects_invalid_telemetry(telemetry):
    with pytest.raises(ValueError):
        predict_control(telemetry, FakeModel())


def test_predict_control_rejects_non_finite_model_output():
    with pytest.raises(ValueError, match="non-finite"):
        predict_control(
            {"speed": "5", "image": encoded_test_image()},
            FakeModel(prediction=np.nan),
        )
