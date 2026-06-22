"""Generate the README demo preview GIF from the simulator screenshot."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "simulator_screenshot.png"
OUTPUT = ROOT / "docs" / "demo-preview.gif"
FRAME_SIZE = (840, 636)


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


TITLE_FONT = load_font(30, bold=True)
BODY_FONT = load_font(20)
SMALL_FONT = load_font(17)


def base_frame() -> Image.Image:
    image = Image.open(SOURCE).convert("RGB")
    image.thumbnail(FRAME_SIZE, Image.Resampling.LANCZOS)

    frame = Image.new("RGB", FRAME_SIZE, (12, 16, 22))
    x = (FRAME_SIZE[0] - image.width) // 2
    y = (FRAME_SIZE[1] - image.height) // 2
    frame.paste(image, (x, y))

    overlay = Image.new("RGBA", FRAME_SIZE, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle((0, 0, FRAME_SIZE[0], FRAME_SIZE[1]), fill=(0, 0, 0, 42))
    return Image.alpha_composite(frame.convert("RGBA"), overlay)


def panel(
    frame: Image.Image,
    title: str,
    lines: list[str],
    box: tuple[int, int, int, int] = (28, 28, 585, 168),
) -> Image.Image:
    draw = ImageDraw.Draw(frame)
    draw.rounded_rectangle(box, radius=18, fill=(8, 16, 28, 222))
    draw.rounded_rectangle(box, radius=18, outline=(95, 168, 255, 230), width=2)
    draw.text((box[0] + 22, box[1] + 18), title, fill=(255, 255, 255), font=TITLE_FONT)
    y = box[1] + 62
    for line in lines:
        draw.text((box[0] + 24, y), line, fill=(216, 228, 243), font=BODY_FONT)
        y += 28
    return frame


def callout(
    frame: Image.Image,
    text: str,
    xy: tuple[int, int],
    anchor: tuple[int, int],
    color: tuple[int, int, int] = (49, 130, 206),
) -> Image.Image:
    draw = ImageDraw.Draw(frame)
    width = draw.textlength(text, font=SMALL_FONT) + 30
    box = (xy[0], xy[1], xy[0] + int(width), xy[1] + 42)
    draw.rounded_rectangle(box, radius=14, fill=(8, 16, 28, 232))
    draw.rounded_rectangle(box, radius=14, outline=color + (245,), width=2)
    draw.text((xy[0] + 15, xy[1] + 11), text, fill=(255, 255, 255), font=SMALL_FONT)
    draw.line((box[2] - 16, box[3] - 4, anchor[0], anchor[1]), fill=color + (245,), width=4)
    return frame


def highlight(frame: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    draw = ImageDraw.Draw(frame)
    draw.rounded_rectangle(box, radius=16, outline=(255, 209, 102, 250), width=5)
    return frame


def make_frames() -> list[Image.Image]:
    frames: list[Image.Image] = []

    frame = base_frame()
    frames.append(
        panel(
            frame,
            "Self-Driving Car Simulation",
            ["Behavioral cloning with a CNN", "Real-time Socket.IO control loop"],
        )
    )

    frame = base_frame()
    highlight(frame, (96, 140, 740, 460))
    callout(frame, "center-camera telemetry", (52, 496), (338, 408))
    frames.append(
        panel(
            frame,
            "1. Simulator Frame",
            ["Live camera image arrives from the simulator"],
        )
    )

    frame = base_frame()
    highlight(frame, (96, 207, 740, 390))
    callout(frame, "crop -> YUV -> resize", (62, 482), (395, 332))
    frames.append(
        panel(
            frame,
            "2. Image Preprocessing",
            ["The frame is transformed into model-ready input"],
        )
    )

    frame = base_frame()
    callout(frame, "NVIDIA-style CNN predicts steering", (386, 476), (512, 320), (111, 207, 151))
    frames.append(
        panel(
            frame,
            "3. Model Inference",
            ["Keras model predicts steering angle from vision"],
        )
    )

    frame = base_frame()
    callout(frame, "steering + throttle returned", (438, 494), (630, 382), (237, 137, 54))
    frames.append(
        panel(
            frame,
            "4. Autonomous Control",
            ["The server sends commands back to the simulator"],
        )
    )

    return frames


def main() -> None:
    frames = [frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=96) for frame in make_frames()]
    frames[0].save(
        OUTPUT,
        save_all=True,
        append_images=frames[1:],
        duration=[1100, 1200, 1200, 1200, 1400],
        loop=0,
        optimize=True,
    )
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
