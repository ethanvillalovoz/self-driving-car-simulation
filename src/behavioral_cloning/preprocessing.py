"""Image decoding and preprocessing shared by training and inference."""

from __future__ import annotations

import base64
import binascii
from io import BytesIO

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError

MAX_ENCODED_FRAME_BYTES = 8 * 1024 * 1024
MAX_FRAME_PIXELS = 3840 * 2160
CROP_TOP = 60
CROP_BOTTOM = 135
MODEL_WIDTH = 200
MODEL_HEIGHT = 66


def decode_telemetry_image(
    encoded_image: str,
    *,
    max_bytes: int = MAX_ENCODED_FRAME_BYTES,
) -> NDArray[np.uint8]:
    """Decode a bounded base64 simulator frame as an RGB array."""
    if not isinstance(encoded_image, str) or not encoded_image:
        raise ValueError("Telemetry image must be a non-empty base64 string")
    try:
        payload = base64.b64decode(encoded_image, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Telemetry image is not valid base64") from exc
    if not payload or len(payload) > max_bytes:
        raise ValueError(f"Telemetry image must be between 1 and {max_bytes} bytes")

    try:
        with Image.open(BytesIO(payload)) as image:
            width, height = image.size
            if width * height > MAX_FRAME_PIXELS:
                raise ValueError("Telemetry image dimensions exceed the safety limit")
            rgb = image.convert("RGB")
            rgb.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Telemetry image is not a supported image") from exc
    return np.asarray(rgb, dtype=np.uint8)


def _as_uint8_rgb(image: NDArray) -> NDArray[np.uint8]:
    values = np.asarray(image)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("Frame must have shape (height, width, 3)")
    if values.shape[0] < CROP_BOTTOM or values.shape[1] < 2:
        raise ValueError(f"Frame must be at least {CROP_BOTTOM} pixels tall")
    if not np.issubdtype(values.dtype, np.number) or not np.all(np.isfinite(values)):
        raise ValueError("Frame pixels must be finite numeric values")
    if np.issubdtype(values.dtype, np.floating) and values.max(initial=0) <= 1.0:
        values = values * 255.0
    return np.clip(values, 0, 255).astype(np.uint8)


def image_preprocess(image: NDArray) -> NDArray[np.float32]:
    """Transform an RGB frame into the NVIDIA-style CNN input tensor."""
    rgb = _as_uint8_rgb(image)
    cropped = rgb[CROP_TOP:CROP_BOTTOM, :, :]
    yuv = cv2.cvtColor(cropped, cv2.COLOR_RGB2YUV)
    blurred = cv2.GaussianBlur(yuv, (3, 3), 0)
    resized = cv2.resize(blurred, (MODEL_WIDTH, MODEL_HEIGHT))
    return resized.astype(np.float32) / 255.0
