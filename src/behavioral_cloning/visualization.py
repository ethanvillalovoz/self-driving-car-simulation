"""Evidence-backed public figures for the behavioral-cloning project."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image

INK = "#202421"
MUTED = "#667069"
LINE = "#d8ddd8"
SOFT = "#f1f3f0"
COBALT = "#315bd6"
AMBER = "#a96b2d"
GREEN = "#39735e"
CORAL = "#d85f4b"
DARK = "#111714"

LEFT_CAMERA_CROP = (37, 104, 529, 350)
CENTER_CAMERA_CROP = (550, 104, 1041, 350)
RIGHT_CAMERA_CROP = (1062, 104, 1554, 350)
MODEL_INPUT_CROP = (1128, 451, 1564, 596)
STEERING_TRACE_CROP = (37, 470, 1095, 745)
SIMULATOR_CONTEXT_CROP = (0, 125, 1600, 775)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "text.color": INK,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "behavioral-cloning-system-overview",
        }
    )


def _load_crop(path: Path, crop: tuple[int, int, int, int]) -> np.ndarray:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        left, top, right, bottom = crop
        if not (0 <= left < right <= rgb.width and 0 <= top < bottom <= rgb.height):
            raise ValueError(f"Crop {crop!r} exceeds {path} dimensions {rgb.size!r}")
        return np.asarray(rgb.crop(crop))


def _place_image(
    axis,
    image: np.ndarray,
    extent: tuple[float, float, float, float],
    *,
    edgecolor: str = LINE,
    linewidth: float = 0.65,
) -> None:
    x0, x1, y0, y1 = extent
    axis.imshow(image, extent=extent, aspect="auto", zorder=1)
    axis.add_patch(
        Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            transform=axis.transAxes,
            facecolor="none",
            edgecolor=edgecolor,
            linewidth=linewidth,
            zorder=2,
        )
    )


def _arrow(
    axis,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str,
    connectionstyle: str = "arc3",
    linewidth: float = 1.05,
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            transform=axis.transAxes,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=linewidth,
            color=color,
            connectionstyle=connectionstyle,
            zorder=5,
        )
    )


def _section_label(axis, x: float, y: float, title: str, subtitle: str) -> None:
    axis.text(
        x,
        y,
        title,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        color=INK,
        fontsize=7.2,
        fontweight="bold",
    )
    axis.text(
        x,
        y - 0.004,
        subtitle,
        transform=axis.transAxes,
        ha="left",
        va="top",
        color=MUTED,
        fontsize=6.1,
    )


def _validate_text_insets(figure, axes: Sequence, *, minimum_points: float = 3.0) -> None:
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    minimum_pixels = minimum_points * figure.dpi / 72.0
    for axis in axes:
        boundary = axis.get_window_extent(renderer)
        for artist in axis.texts:
            if not artist.get_visible() or not artist.get_text().strip():
                continue
            bounds = artist.get_window_extent(renderer)
            if (
                bounds.x0 < boundary.x0 + minimum_pixels
                or bounds.x1 > boundary.x1 - minimum_pixels
                or bounds.y0 < boundary.y0 + minimum_pixels
                or bounds.y1 > boundary.y1 - minimum_pixels
            ):
                text = artist.get_text()
                raise ValueError(f"Figure text violates the minimum inset: {text!r}")


def save_behavioral_cloning_overview(
    metrics_path: Path,
    replay_poster_path: Path,
    simulator_frame_path: Path,
    output_stem: Path,
) -> list[Path]:
    """Render a vision-and-control overview at paper width."""
    _style()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    camera_views = [
        _load_crop(replay_poster_path, crop)
        for crop in (LEFT_CAMERA_CROP, CENTER_CAMERA_CROP, RIGHT_CAMERA_CROP)
    ]
    model_input = _load_crop(replay_poster_path, MODEL_INPUT_CROP)
    steering_trace = _load_crop(replay_poster_path, STEERING_TRACE_CROP)
    simulator_context = _load_crop(simulator_frame_path, SIMULATOR_CONTEXT_CROP)

    figure = plt.figure(figsize=(7.16, 4.45), facecolor="white")
    canvas = figure.add_axes((0, 0, 1, 1))
    canvas.set_xlim(0, 1)
    canvas.set_ylim(0, 1)
    canvas.axis("off")

    canvas.text(
        0.025,
        0.968,
        "Behavioral cloning, from recorded views to bounded simulator commands",
        transform=canvas.transAxes,
        ha="left",
        va="top",
        color=INK,
        fontsize=10.8,
        fontweight="bold",
    )
    canvas.text(
        0.025,
        0.924,
        (
            "The same image-to-steering path supports training, runtime inference, "
            "and offline replay."
        ),
        transform=canvas.transAxes,
        ha="left",
        va="top",
        color=MUTED,
        fontsize=6.7,
    )

    _section_label(
        canvas,
        0.025,
        0.868,
        "TRAINING OBSERVATIONS",
        "three synchronized simulator cameras",
    )
    _section_label(
        canvas,
        0.642,
        0.868,
        "SIMULATOR LOOP",
        "center frame + speed in, bounded commands out",
    )

    camera_extents = [
        (0.025, 0.202, 0.660, 0.803),
        (0.208, 0.385, 0.660, 0.803),
        (0.391, 0.568, 0.660, 0.803),
    ]
    camera_labels = [
        ("LEFT", "+0.15 steering"),
        ("CENTER", "recorded steering"),
        ("RIGHT", "-0.15 steering"),
    ]
    for image, extent, (view, assumption) in zip(
        camera_views, camera_extents, camera_labels, strict=True
    ):
        _place_image(canvas, image, extent, edgecolor="white", linewidth=1.2)
        x0, x1, y0, _ = extent
        canvas.add_patch(
            Rectangle(
                (x0, y0),
                x1 - x0,
                0.031,
                transform=canvas.transAxes,
                facecolor=DARK,
                edgecolor="none",
                alpha=0.86,
                zorder=3,
            )
        )
        canvas.text(
            x0 + 0.008,
            y0 + 0.016,
            view,
            transform=canvas.transAxes,
            ha="left",
            va="center",
            color="white",
            fontsize=5.4,
            fontweight="bold",
            zorder=4,
        )
        canvas.text(
            x1 - 0.008,
            y0 + 0.016,
            assumption,
            transform=canvas.transAxes,
            ha="right",
            va="center",
            color="white",
            fontsize=5.1,
            zorder=4,
        )

    _place_image(
        canvas,
        simulator_context,
        (0.642, 0.975, 0.588, 0.803),
        edgecolor=INK,
        linewidth=0.75,
    )
    canvas.text(
        0.025,
        0.621,
        "driving log   →   balance 25 bins   →   seed 42 / 80:20 split   →   augment",
        transform=canvas.transAxes,
        ha="left",
        va="center",
        color=AMBER,
        fontsize=6.1,
        fontweight="bold",
    )
    _arrow(canvas, (0.581, 0.658), (0.581, 0.548), color=AMBER)

    canvas.text(
        0.025,
        0.552,
        "SHARED MODEL SPINE",
        transform=canvas.transAxes,
        ha="left",
        va="bottom",
        color=INK,
        fontsize=7.2,
        fontweight="bold",
    )

    canvas.add_patch(
        Rectangle(
            (0.025, 0.438),
            0.348,
            0.103,
            transform=canvas.transAxes,
            facecolor=SOFT,
            edgecolor=LINE,
            linewidth=0.7,
            zorder=0,
        )
    )
    _place_image(canvas, model_input, (0.034, 0.160, 0.452, 0.527), edgecolor=MUTED)
    canvas.text(
        0.177,
        0.517,
        "FIXED PREPROCESSING",
        transform=canvas.transAxes,
        ha="left",
        va="center",
        color=MUTED,
        fontsize=5.5,
        fontweight="bold",
    )
    canvas.text(
        0.177,
        0.466,
        "crop 60:135  →  RGB to YUV\n3 x 3 blur  →  resize 200 x 66\nnormalize",
        transform=canvas.transAxes,
        ha="left",
        va="center",
        color=INK,
        fontsize=5.4,
        linespacing=1.22,
    )

    canvas.add_patch(
        Rectangle(
            (0.373, 0.438),
            0.271,
            0.103,
            transform=canvas.transAxes,
            facecolor=COBALT,
            edgecolor=COBALT,
            linewidth=0.7,
            zorder=0,
        )
    )
    canvas.text(
        0.5085,
        0.500,
        "NVIDIA-STYLE CNN",
        transform=canvas.transAxes,
        ha="center",
        va="center",
        color="white",
        fontsize=7.5,
        fontweight="bold",
    )
    canvas.text(
        0.5085,
        0.466,
        f"{metrics['model_parameters']:,} parameters  →  scalar steering",
        transform=canvas.transAxes,
        ha="center",
        va="center",
        color="white",
        fontsize=5.8,
    )

    canvas.add_patch(
        Rectangle(
            (0.644, 0.438),
            0.331,
            0.103,
            transform=canvas.transAxes,
            facecolor="#e8f0ec",
            edgecolor=GREEN,
            linewidth=0.75,
            zorder=0,
        )
    )
    canvas.text(
        0.663,
        0.508,
        "RUNTIME COMMANDS",
        transform=canvas.transAxes,
        ha="left",
        va="center",
        color=GREEN,
        fontsize=5.5,
        fontweight="bold",
    )
    canvas.text(
        0.663,
        0.474,
        "CNN steering  →  clamp [-1, 1]\nspeed  →  proportional throttle  →  clamp [0, 1]",
        transform=canvas.transAxes,
        ha="left",
        va="center",
        color=INK,
        fontsize=5.8,
        linespacing=1.35,
    )
    _arrow(canvas, (0.373, 0.4895), (0.392, 0.4895), color="white")
    _arrow(canvas, (0.644, 0.4895), (0.663, 0.4895), color=GREEN)
    _arrow(canvas, (0.920, 0.585), (0.920, 0.546), color=GREEN)
    canvas.text(
        0.906,
        0.565,
        "telemetry",
        transform=canvas.transAxes,
        ha="right",
        va="center",
        color=GREEN,
        fontsize=5.2,
        fontweight="bold",
    )
    _arrow(
        canvas,
        (0.704, 0.546),
        (0.704, 0.585),
        color=GREEN,
        connectionstyle="arc3,rad=-0.05",
    )
    canvas.text(
        0.718,
        0.565,
        "commands",
        transform=canvas.transAxes,
        ha="left",
        va="center",
        color=GREEN,
        fontsize=5.2,
        fontweight="bold",
    )

    canvas.text(
        0.025,
        0.397,
        "OFFLINE REPLAY DIAGNOSTIC",
        transform=canvas.transAxes,
        ha="left",
        va="bottom",
        color=INK,
        fontsize=7.2,
        fontweight="bold",
    )
    canvas.text(
        0.302,
        0.397,
        "180 consecutive recorded frames",
        transform=canvas.transAxes,
        ha="left",
        va="bottom",
        color=MUTED,
        fontsize=6.1,
    )
    _place_image(canvas, steering_trace, (0.025, 0.742, 0.090, 0.366), edgecolor=INK)

    canvas.plot(
        [0.046, 0.078],
        [0.071, 0.071],
        transform=canvas.transAxes,
        color=AMBER,
        linewidth=2.0,
    )
    canvas.text(
        0.084,
        0.071,
        "recorded steering",
        transform=canvas.transAxes,
        ha="left",
        va="center",
        color=AMBER,
        fontsize=5.7,
        fontweight="bold",
    )
    canvas.plot(
        [0.204, 0.236],
        [0.071, 0.071],
        transform=canvas.transAxes,
        color=GREEN,
        linewidth=2.0,
    )
    canvas.text(
        0.242,
        0.071,
        "model steering",
        transform=canvas.transAxes,
        ha="left",
        va="center",
        color=GREEN,
        fontsize=5.7,
        fontweight="bold",
    )

    canvas.plot(
        [0.772, 0.772],
        [0.083, 0.365],
        transform=canvas.transAxes,
        color=LINE,
        linewidth=0.8,
    )
    canvas.text(
        0.792,
        0.355,
        "HISTORICAL NOTEBOOK",
        transform=canvas.transAxes,
        ha="left",
        va="top",
        color=INK,
        fontsize=6.2,
        fontweight="bold",
    )
    canvas.text(
        0.792,
        0.311,
        f"{metrics['epochs']} epochs",
        transform=canvas.transAxes,
        ha="left",
        va="top",
        color=INK,
        fontsize=8.4,
        fontweight="bold",
    )
    canvas.text(
        0.792,
        0.273,
        (
            f"best val. MSE  {metrics['best_validation_mse']:.5f}\n"
            f"at epoch {metrics['best_epoch']}"
        ),
        transform=canvas.transAxes,
        ha="left",
        va="top",
        color=INK,
        fontsize=6.3,
        linespacing=1.35,
    )
    canvas.text(
        0.792,
        0.195,
        "ONE RUN · OFFLINE ONLY",
        transform=canvas.transAxes,
        ha="left",
        va="top",
        color=CORAL,
        fontsize=6.0,
        fontweight="bold",
    )
    canvas.text(
        0.792,
        0.160,
        (
            "Incomplete seed and hardware\nrecord. Not a lane-keeping,\n"
            "recovery, intervention, or\ncompletion metric."
        ),
        transform=canvas.transAxes,
        ha="left",
        va="top",
        color=MUTED,
        fontsize=5.7,
        linespacing=1.35,
    )

    canvas.text(
        0.975,
        0.032,
        "SIMULATOR ONLY  ·  OFFLINE EVIDENCE IS DIAGNOSTIC",
        transform=canvas.transAxes,
        ha="right",
        va="bottom",
        color=MUTED,
        fontsize=5.6,
        fontweight="bold",
    )
    _validate_text_insets(figure, [canvas])

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = [output_stem.with_suffix(suffix) for suffix in (".svg", ".pdf", ".png")]
    for path in outputs:
        metadata = {"Title": "Behavioral-cloning training, runtime, and evidence overview"}
        if path.suffix == ".pdf":
            metadata.update(
                {
                    "Author": "Ethan Villalovoz",
                    "CreationDate": None,
                    "ModDate": None,
                }
            )
        elif path.suffix == ".svg":
            metadata.update({"Creator": "self-driving-car-simulation", "Date": None})
        figure.savefig(path, dpi=300, facecolor="white", metadata=metadata)
        if path.suffix == ".svg":
            lines = path.read_text(encoding="utf-8").splitlines()
            path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")
    plt.close(figure)
    return outputs
