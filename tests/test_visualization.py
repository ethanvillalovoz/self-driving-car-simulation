from pathlib import Path

from PIL import Image

from behavioral_cloning.visualization import (
    CENTER_CAMERA_CROP,
    LEFT_CAMERA_CROP,
    MODEL_INPUT_CROP,
    RIGHT_CAMERA_CROP,
    SIMULATOR_CONTEXT_CROP,
    STEERING_TRACE_CROP,
    save_behavioral_cloning_overview,
)

ROOT = Path(__file__).resolve().parents[1]


def test_overview_crop_contract_matches_committed_media():
    with Image.open(ROOT / "docs/media/self-driving-offline-replay.webp") as replay:
        assert replay.size == (1600, 900)
        for crop in (
            LEFT_CAMERA_CROP,
            CENTER_CAMERA_CROP,
            RIGHT_CAMERA_CROP,
            MODEL_INPUT_CROP,
            STEERING_TRACE_CROP,
        ):
            left, top, right, bottom = crop
            assert 0 <= left < right <= replay.width
            assert 0 <= top < bottom <= replay.height

    with Image.open(ROOT / "docs/simulator-frame.webp") as simulator:
        assert simulator.size == (1600, 900)
        left, top, right, bottom = SIMULATOR_CONTEXT_CROP
        assert 0 <= left < right <= simulator.width
        assert 0 <= top < bottom <= simulator.height


def test_overview_exports_vector_and_raster_formats(tmp_path: Path):
    outputs = save_behavioral_cloning_overview(
        ROOT / "examples/original-run-metrics.json",
        ROOT / "docs/media/self-driving-offline-replay.webp",
        ROOT / "docs/simulator-frame.webp",
        tmp_path / "behavioral-cloning-overview",
    )

    assert [path.suffix for path in outputs] == [".svg", ".pdf", ".png"]
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs)
    assert outputs[1].read_bytes().startswith(b"%PDF")

    svg = outputs[0].read_text(encoding="utf-8")
    assert "OFFLINE REPLAY DIAGNOSTIC" in svg
    assert "Not a lane-keeping" in svg
    assert not any(line.endswith(" ") for line in svg.splitlines())

    with Image.open(outputs[2]) as image:
        assert image.size == (2148, 1335)
