import base64
from io import BytesIO

import numpy as np
from PIL import Image

from drive import (
    calculate_throttle,
    decode_telemetry_image,
    image_preprocess,
    predict_control,
)


def _encoded_test_image():
    image = Image.new("RGB", (320, 160), color=(128, 64, 32))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_image_preprocess_returns_model_input_shape():
    image = np.full((160, 320, 3), 128, dtype=np.uint8)

    processed = image_preprocess(image)

    assert processed.shape == (66, 200, 3)
    assert processed.min() >= 0.0
    assert processed.max() <= 1.0


def test_decode_telemetry_image_returns_rgb_array():
    decoded = decode_telemetry_image(_encoded_test_image())

    assert decoded.shape == (160, 320, 3)
    assert decoded.dtype == np.uint8


def test_calculate_throttle_matches_original_controller():
    assert calculate_throttle(0) == 1.0
    assert calculate_throttle(5) == 0.5
    assert calculate_throttle(10) == 0.0


def test_predict_control_uses_preprocessed_image_batch():
    class FakeModel:
        def predict(self, batch):
            assert batch.shape == (1, 66, 200, 3)
            return np.array([[0.25]])

    steering, throttle = predict_control(
        {"speed": "5", "image": _encoded_test_image()},
        FakeModel(),
    )

    assert steering == 0.25
    assert throttle == 0.5
