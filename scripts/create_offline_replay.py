"""Render an honest offline replay from restored driving data and a trained model."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from tensorflow import keras

from behavioral_cloning.preprocessing import image_preprocess
from behavioral_cloning.replay import (
    CAMERA_LABELS,
    replay_camera_paths,
    replay_model_camera_path,
)
from behavioral_cloning.training import load_rgb, read_driving_log

WIDTH = 1200
HEIGHT = 675
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
    render_scale: float = 1.0,
) -> None:
    cv2.putText(
        canvas,
        text,
        tuple(round(value * render_scale) for value in position),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale * render_scale,
        color,
        max(1, round(weight * render_scale)),
        cv2.LINE_AA,
    )


def fit_camera(image: np.ndarray, render_scale: float) -> np.ndarray:
    size = (
        round(CAMERA_WIDTH * render_scale),
        round(CAMERA_HEIGHT * render_scale),
    )
    return cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)


def plot_trace(
    canvas: np.ndarray,
    values: list[float],
    color: tuple[int, int, int],
    origin: tuple[int, int],
    size: tuple[int, int],
    render_scale: float,
) -> None:
    if len(values) < 2:
        return
    x0, y0 = (round(value * render_scale) for value in origin)
    width, height = (round(value * render_scale) for value in size)
    points = []
    for index, value in enumerate(values):
        x = x0 + round(index / max(1, len(values) - 1) * width)
        y = y0 + round((1.0 - (np.clip(value, -1.0, 1.0) + 1.0) / 2.0) * height)
        points.append((x, y))
    cv2.polylines(
        canvas,
        [np.asarray(points, dtype=np.int32)],
        False,
        color,
        max(2, round(2 * render_scale)),
        cv2.LINE_AA,
    )


def compose_frame(
    camera_frames: tuple[np.ndarray, np.ndarray, np.ndarray],
    model_input: np.ndarray,
    recorded_trace: list[float],
    predicted_trace: list[float],
    frame_number: int,
    total_frames: int,
    render_scale: float,
) -> np.ndarray:
    width = round(WIDTH * render_scale)
    height = round(HEIGHT * render_scale)
    canvas = np.full((height, width, 3), BACKGROUND, dtype=np.uint8)
    draw_text(
        canvas,
        "BEHAVIORAL CLONING",
        (28, 32),
        scale=0.58,
        weight=2,
        render_scale=render_scale,
    )
    draw_text(
        canvas,
        "OFFLINE REPLAY",
        (28, 54),
        color=MUTED,
        scale=0.39,
        render_scale=render_scale,
    )
    draw_text(
        canvas,
        f"FRAME {frame_number:03d} / {total_frames:03d}",
        (1023, 38),
        color=MUTED,
        scale=0.38,
        render_scale=render_scale,
    )
    cv2.line(
        canvas,
        (round(28 * render_scale), round(65 * render_scale)),
        (round(1172 * render_scale), round(65 * render_scale)),
        LINE,
        max(1, round(render_scale)),
    )

    for index, (label, image) in enumerate(zip(CAMERA_LABELS, camera_frames, strict=True)):
        x = 28 + index * (CAMERA_WIDTH + CAMERA_GAP)
        scaled_x = round(x * render_scale)
        scaled_top = round(CAMERA_TOP * render_scale)
        camera = fit_camera(image, render_scale)
        canvas[
            scaled_top : scaled_top + camera.shape[0],
            scaled_x : scaled_x + camera.shape[1],
        ] = camera
        draw_text(
            canvas,
            label,
            (x, CAMERA_TOP - 9),
            color=MUTED,
            scale=0.34,
            render_scale=render_scale,
        )

    trace_x, trace_y = 28, 338
    trace_width, trace_height = 758, 242
    draw_text(
        canvas,
        "STEERING TRACE",
        (trace_x, trace_y - 13),
        color=MUTED,
        scale=0.36,
        render_scale=render_scale,
    )
    trace_left = round(trace_x * render_scale)
    trace_top = round(trace_y * render_scale)
    scaled_trace_width = round(trace_width * render_scale)
    scaled_trace_height = round(trace_height * render_scale)
    canvas[
        trace_top : trace_top + scaled_trace_height,
        trace_left : trace_left + scaled_trace_width,
    ] = PANEL
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = trace_top + round(fraction * scaled_trace_height)
        cv2.line(
            canvas,
            (trace_left, y),
            (trace_left + scaled_trace_width, y),
            LINE,
            max(1, round(render_scale)),
        )
    cv2.line(
        canvas,
        (trace_left, trace_top + scaled_trace_height // 2),
        (trace_left + scaled_trace_width, trace_top + scaled_trace_height // 2),
        (94, 88, 80),
        max(1, round(render_scale)),
    )
    plot_trace(
        canvas,
        recorded_trace,
        RECORDED,
        (trace_x, trace_y),
        (trace_width, trace_height),
        render_scale,
    )
    plot_trace(
        canvas,
        predicted_trace,
        PREDICTED,
        (trace_x, trace_y),
        (trace_width, trace_height),
        render_scale,
    )
    draw_text(
        canvas,
        "RECORDED",
        (trace_x, 608),
        color=RECORDED,
        scale=0.34,
        weight=2,
        render_scale=render_scale,
    )
    draw_text(
        canvas,
        "MODEL",
        (trace_x + 100, 608),
        color=PREDICTED,
        scale=0.34,
        weight=2,
        render_scale=render_scale,
    )
    draw_text(
        canvas,
        "+1.0",
        (trace_x + trace_width + 7, trace_y + 7),
        color=MUTED,
        scale=0.31,
        render_scale=render_scale,
    )
    draw_text(
        canvas,
        " 0.0",
        (trace_x + trace_width + 7, trace_y + trace_height // 2 + 4),
        color=MUTED,
        scale=0.31,
        render_scale=render_scale,
    )
    draw_text(
        canvas,
        "-1.0",
        (trace_x + trace_width + 7, trace_y + trace_height),
        color=MUTED,
        scale=0.31,
        render_scale=render_scale,
    )

    detail_x = 846
    draw_text(
        canvas,
        "MODEL INPUT",
        (detail_x, trace_y - 13),
        color=MUTED,
        scale=0.36,
        render_scale=render_scale,
    )
    preview = cv2.cvtColor(
        np.clip(model_input * 255.0, 0, 255).astype(np.uint8),
        cv2.COLOR_YUV2BGR,
    )
    preview_size = (round(326 * render_scale), round(108 * render_scale))
    preview = cv2.resize(preview, preview_size, interpolation=cv2.INTER_NEAREST)
    detail_left = round(detail_x * render_scale)
    canvas[
        trace_top : trace_top + preview.shape[0],
        detail_left : detail_left + preview.shape[1],
    ] = preview
    cv2.rectangle(
        canvas,
        (detail_left, trace_top),
        (detail_left + preview.shape[1], trace_top + preview.shape[0]),
        LINE,
        max(1, round(render_scale)),
    )

    latest_recorded = recorded_trace[-1]
    latest_predicted = predicted_trace[-1]
    draw_text(
        canvas,
        "RECORDED STEERING",
        (detail_x, 481),
        color=MUTED,
        scale=0.34,
        render_scale=render_scale,
    )
    draw_text(
        canvas,
        f"{latest_recorded:+.4f}",
        (detail_x, 511),
        color=RECORDED,
        scale=0.74,
        weight=2,
        render_scale=render_scale,
    )
    draw_text(
        canvas,
        "MODEL STEERING",
        (detail_x, 548),
        color=MUTED,
        scale=0.34,
        render_scale=render_scale,
    )
    draw_text(
        canvas,
        f"{latest_predicted:+.4f}",
        (detail_x, 578),
        color=PREDICTED,
        scale=0.74,
        weight=2,
        render_scale=render_scale,
    )
    draw_text(
        canvas,
        "66 x 200 YUV input / NVIDIA-style CNN",
        (detail_x, 612),
        color=MUTED,
        scale=0.32,
        render_scale=render_scale,
    )
    draw_text(
        canvas,
        "Replay uses recorded simulator frames; it is not a closed-loop driving benchmark.",
        (28, 652),
        color=MUTED,
        scale=0.34,
        render_scale=render_scale,
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
    parser.add_argument("--scale", type=float, default=1.6)
    args = parser.parse_args()

    if args.start < 0 or args.frames <= 1 or args.fps <= 0 or args.scale <= 0:
        raise ValueError("Start must be non-negative and frames/fps must be positive")
    csv_path = args.data_dir / "driving_log.csv"
    records = read_driving_log(csv_path, args.data_dir / "IMG")
    selected = records[args.start : args.start + args.frames]
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
        f"{round(WIDTH * args.scale)}x{round(HEIGHT * args.scale)}",
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
        for index, record in enumerate(selected, start=1):
            camera_paths = replay_camera_paths(record)
            rgb_by_path = {path: load_rgb(path) for path in camera_paths}
            rgb_frames = tuple(rgb_by_path[path] for path in camera_paths)
            bgr_frames = tuple(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in rgb_frames)
            center_frame = rgb_by_path[replay_model_camera_path(record)]
            model_input = image_preprocess(center_frame)
            prediction = float(
                np.asarray(
                    model.predict(np.expand_dims(model_input, axis=0), verbose=0)
                ).reshape(-1)[0]
            )
            recorded_trace.append(record.steering)
            predicted_trace.append(prediction)
            frame = compose_frame(
                bgr_frames,
                model_input,
                list(recorded_trace),
                list(predicted_trace),
                index,
                len(selected),
                args.scale,
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
