"""Regenerate the committed public figure exports from frozen repository artifacts."""

from pathlib import Path

from behavioral_cloning.visualization import save_behavioral_cloning_overview

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    outputs = save_behavioral_cloning_overview(
        ROOT / "examples" / "original-run-metrics.json",
        ROOT / "docs" / "media" / "self-driving-offline-replay.webp",
        ROOT / "docs" / "simulator-frame.webp",
        ROOT / "docs" / "media" / "behavioral-cloning-overview",
    )
    for output in outputs:
        print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
