"""Generate a lightweight simulator-motion preview for the README."""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "simulator-frame.webp"
OUTPUT = ROOT / "docs" / "demo-preview.webp"
FRAME_SIZE = (880, 495)
FRAME_COUNT = 24


def load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    names = [
        "/System/Library/Fonts/SFNS.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for name in names:
        if name and Path(name).exists():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


LABEL = load_font(13, bold=True)
VALUE = load_font(23, bold=True)
SMALL = load_font(13)


def motion_frame(source: Image.Image, phase: float) -> Image.Image:
    width, height = source.size
    target_ratio = FRAME_SIZE[0] / FRAME_SIZE[1]
    crop_height = min(height, int(width / target_ratio))
    crop_width = int(crop_height * target_ratio)
    zoom = 1.0 + 0.035 * (0.5 - 0.5 * math.cos(phase * math.tau))
    visible_width = int(crop_width / zoom)
    visible_height = int(crop_height / zoom)
    center_x = width // 2 + int(12 * math.sin(phase * math.tau))
    center_y = height // 2 + int(8 * math.sin(phase * math.tau + 0.8))
    left = max(0, min(width - visible_width, center_x - visible_width // 2))
    top = max(0, min(height - visible_height, center_y - visible_height // 2))
    frame = source.crop((left, top, left + visible_width, top + visible_height))
    frame = frame.resize(FRAME_SIZE, Image.Resampling.LANCZOS).convert("RGBA")
    frame = ImageEnhance.Contrast(frame).enhance(1.04)

    shade = Image.new("RGBA", FRAME_SIZE, (0, 0, 0, 0))
    shade_draw = ImageDraw.Draw(shade)
    shade_draw.rectangle((0, 0, FRAME_SIZE[0], 86), fill=(5, 8, 10, 72))
    shade_draw.rectangle((0, 382, FRAME_SIZE[0], FRAME_SIZE[1]), fill=(5, 8, 10, 94))
    return Image.alpha_composite(frame, shade)


def draw_hud(frame: Image.Image, phase: float) -> None:
    draw = ImageDraw.Draw(frame)
    steering = -0.12 + 0.18 * (0.5 - 0.5 * math.cos(phase * math.tau))
    speed = 9.1 + 0.7 * math.sin(phase * math.tau)
    throttle = max(0.0, 1.0 - speed / 10.0)

    draw.rounded_rectangle((24, 22, 324, 80), radius=8, fill=(10, 14, 17, 218))
    draw.text((42, 34), "BEHAVIORAL CLONING", fill=(245, 247, 248), font=LABEL)
    draw.ellipse((284, 39, 294, 49), fill=(99, 211, 148))
    draw.text((300, 34), "LIVE", fill=(194, 205, 211), font=SMALL)

    draw.rounded_rectangle((24, 398, 856, 471), radius=8, fill=(10, 14, 17, 226))
    metrics = [
        (42, "STEERING", f"{steering:+.3f}"),
        (194, "SPEED", f"{speed:.1f} mph"),
        (346, "THROTTLE", f"{throttle:.3f}"),
    ]
    for x, label, value in metrics:
        draw.text((x, 410), label, fill=(154, 169, 177), font=LABEL)
        draw.text((x, 433), value, fill=(247, 249, 250), font=VALUE)

    stages = ["CAMERA", "YUV 200x66", "CNN", "CONTROL"]
    active = min(len(stages) - 1, int(phase * len(stages)))
    start_x = 510
    for index, stage in enumerate(stages):
        x = start_x + index * 88
        color = (103, 208, 153) if index <= active else (126, 139, 146)
        draw.ellipse((x, 420, x + 8, 428), fill=color)
        draw.text((x, 439), stage, fill=color, font=SMALL, anchor="ma")
        if index < len(stages) - 1:
            draw.line((x + 13, 424, x + 75, 424), fill=(91, 105, 112), width=1)


def make_frames() -> list[Image.Image]:
    source = Image.open(SOURCE).convert("RGB")
    frames = []
    for index in range(FRAME_COUNT):
        phase = index / FRAME_COUNT
        frame = motion_frame(source, phase)
        draw_hud(frame, phase)
        frames.append(frame.convert("RGB"))
    return frames


def main() -> None:
    frames = make_frames()
    frames[0].save(
        OUTPUT,
        save_all=True,
        append_images=frames[1:],
        duration=105,
        loop=0,
        quality=64,
        method=6,
    )
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
