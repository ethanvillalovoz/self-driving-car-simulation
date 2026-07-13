"""Render an honest offline replay from restored driving data and a trained model."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from tensorflow import keras

from behavioral_cloning.preprocessing import image_preprocess
from behavioral_cloning.training import load_rgb

WIDTH = 1200
HEIGHT = 676
CAMERA_WIDTH = 368
CAMERA_HEIGHT = 184
CAMERA_TOP = 78
CAMERA_GAP = 16
BACKGROUND = (18, 15, 12)
PANEL = (25, 23, 20)
LINE = (62, 57, 51)
TEXT = (235, 232, 226)
MUTED = (151, 146, 137)
RECORDED = (76, 184, 232)
PREDICTED = (113, 213, 143)


def draw_text(
    canvas: np.ndarray,
    text: str,
    position: tuple[int, int],
    *,
    color: tuple[int, int, int] = TEXT,
    scale: float = 0.48,
    weight: int = 1,
) -> None:
    cv2.putText(
        canvas,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        weight,
        cv2.LINE_AA,
    )


def fit_camera(image: np.ndarray) -> np.ndarray:
    return cv2.resize(image, (CAMERA_WIDTH, CAMERA_HEIGHT), interpolation=cv2.INTER_CUBIC)


def plot_trace(
    canvas: np.ndarray,
    values: list[float],
    color: tuple[int, int, int],
    origin: tuple[int, int],
    size: tuple[int, int],
) -> None:
    if len(values) < 2:
        return
    x0, y0 = origin
    width, height = size
    points = []
    for index, value in enumerate(values):
        x = x0 + round(index / max(1, len(values) - 1) * width)
        y = y0 + round((1.0 - (np.clip(value, -1.0, 1.0) + 1.0) / 2.0) * height)
        points.append((x, y))
    cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], False, color, 2, cv2.LINE_AA)


def compose_frame(
    camera_frames: tuple[np.ndarray, np.ndarray, np.ndarray],
    model_input: np.ndarray,
    recorded_trace: list[float],
    predicted_trace: list[float],
    frame_number: int,
    total_frames: int,
) -> np.ndarray:
    canvas = np.full((HEIGHT, WIDTH, 3), BACKGROUND, dtype=np.uint8)
    draw_text(canvas, "BEHAVIORAL CLONING", (28, 32), scale=0.58, weight=2)
    draw_text(canvas, "OFFLINE REPLAY", (28, 54), color=MUTED, scale=0.39)
    draw_text(
        canvas,
        f"FRAME {frame_number:03d} / {total_frames:03d}",
        (1023, 38),
        color=MUTED,
        scale=0.38,
    )
    cv2.line(canvas, (28, 65), (1172, 65), LINE, 1)

    for index, (label, image) in enumerate(
        zip(("LEFT", "CENTER", "RIGHT"), camera_frames, strict=True)
    ):
        x = 28 + index * (CAMERA_WIDTH + CAMERA_GAP)
        canvas[CAMERA_TOP : CAMERA_TOP + CAMERA_HEIGHT, x : x + CAMERA_WIDTH] = fit_camera(
            image
        )
        draw_text(canvas, label, (x, CAMERA_TOP - 9), color=MUTED, scale=0.34)

    trace_x, trace_y = 28, 338
    trace_width, trace_height = 758, 242
    draw_text(canvas, "STEERING TRACE", (trace_x, trace_y - 13), color=MUTED, scale=0.36)
    canvas[trace_y : trace_y + trace_height, trace_x : trace_x + trace_width] = PANEL
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = trace_y + round(fraction * trace_height)
        cv2.line(canvas, (trace_x, y), (trace_x + trace_width, y), LINE, 1)
    cv2.line(
        canvas,
        (trace_x, trace_y + trace_height // 2),
        (trace_x + trace_width, trace_y + trace_height // 2),
        (94, 88, 80),
        1,
    )
    plot_trace(
        canvas, recorded_trace, RECORDED, (trace_x, trace_y), (trace_width, trace_height)
    )
    plot_trace(
        canvas, predicted_trace, PREDICTED, (trace_x, trace_y), (trace_width, trace_height)
    )
    draw_text(canvas, "RECORDED", (trace_x, 608), color=RECORDED, scale=0.34, weight=2)
    draw_text(canvas, "MODEL", (trace_x + 100, 608), color=PREDICTED, scale=0.34, weight=2)
    draw_text(canvas, "+1.0", (trace_x + trace_width + 7, trace_y + 7), color=MUTED, scale=0.31)
    draw_text(
        canvas,
        " 0.0",
        (trace_x + trace_width + 7, trace_y + trace_height // 2 + 4),
        color=MUTED,
        scale=0.31,
    )
    draw_text(
        canvas,
        "-1.0",
        (trace_x + trace_width + 7, trace_y + trace_height),
        color=MUTED,
        scale=0.31,
    )

    detail_x = 846
    draw_text(canvas, "MODEL INPUT", (detail_x, trace_y - 13), color=MUTED, scale=0.36)
    preview = cv2.cvtColor(
        np.clip(model_input * 255.0, 0, 255).astype(np.uint8),
        cv2.COLOR_YUV2BGR,
    )
    preview = cv2.resize(preview, (326, 108), interpolation=cv2.INTER_NEAREST)
    canvas[trace_y : trace_y + 108, detail_x : detail_x + 326] = preview
    cv2.rectangle(canvas, (detail_x, trace_y), (detail_x + 326, trace_y + 108), LINE, 1)

    latest_recorded = recorded_trace[-1]
    latest_predicted = predicted_trace[-1]
    draw_text(canvas, "RECORDED STEERING", (detail_x, 481), color=MUTED, scale=0.34)
    draw_text(
        canvas, f"{latest_recorded:+.4f}", (detail_x, 511), color=RECORDED, scale=0.74, weight=2
    )
    draw_text(canvas, "MODEL STEERING", (detail_x, 548), color=MUTED, scale=0.34)
    draw_text(
        canvas,
        f"{latest_predicted:+.4f}",
        (detail_x, 578),
        color=PREDICTED,
        scale=0.74,
        weight=2,
    )
    draw_text(
        canvas,
        "66 x 200 YUV input / NVIDIA-style CNN",
        (detail_x, 612),
        color=MUTED,
        scale=0.32,
    )
    draw_text(
        canvas,
        "Replay uses recorded simulator frames; it is not a closed-loop driving benchmark.",
        (28, 652),
        color=MUTED,
        scale=0.34,
    )
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--poster", type=Path)
    parser.add_argument("--start", type=int, default=6414)
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()

    if args.start < 0 or args.frames <= 1 or args.fps <= 0:
        raise ValueError("Start must be non-negative and frames/fps must be positive")
    csv_path = args.data_dir / "driving_log.csv"
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.reader(stream))
    selected = rows[args.start : args.start + args.frames]
    if len(selected) != args.frames:
        raise ValueError("Requested replay window exceeds the available driving log")

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to encode the replay")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.poster:
        args.poster.parent.mkdir(parents=True, exist_ok=True)

    model = keras.models.load_model(args.model, compile=False)
    command = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-r",
        str(args.fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(args.output),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    recorded_trace: deque[float] = deque(maxlen=90)
    predicted_trace: deque[float] = deque(maxlen=90)
    poster_frame = len(selected) // 2

    try:
        for index, row in enumerate(selected, start=1):
            camera_paths = [
                args.data_dir / "IMG" / Path(row[column].strip()).name for column in range(3)
            ]
            rgb_frames = tuple(load_rgb(path) for path in camera_paths)
            bgr_frames = tuple(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in rgb_frames)
            model_input = image_preprocess(rgb_frames[1])
            prediction = float(
                np.asarray(
                    model.predict(np.expand_dims(model_input, axis=0), verbose=0)
                ).reshape(-1)[0]
            )
            recorded_trace.append(float(row[3]))
            predicted_trace.append(prediction)
            frame = compose_frame(
                bgr_frames,
                model_input,
                list(recorded_trace),
                list(predicted_trace),
                index,
                len(selected),
            )
            if encoder.stdin is None:
                raise RuntimeError("ffmpeg input stream is unavailable")
            encoder.stdin.write(frame.tobytes())
            if args.poster and index == poster_frame:
                cv2.imwrite(str(args.poster), frame, [cv2.IMWRITE_WEBP_QUALITY, 94])
    finally:
        if encoder.stdin:
            encoder.stdin.close()
        return_code = encoder.wait()

    if return_code != 0:
        raise RuntimeError(f"ffmpeg exited with status {return_code}")
    print(f"Wrote {args.output} ({len(selected)} frames at {args.fps} fps)")
    if args.poster:
        print(f"Wrote {args.poster}")


if __name__ == "__main__":
    main()
