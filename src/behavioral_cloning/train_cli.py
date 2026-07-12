"""Command-line entry point for reproducible model training."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .training import TrainingConfig, train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the behavioral-cloning CNN.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("model/model.keras"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--max-per-bin", type=int, default=400)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    history = train(
        TrainingConfig(
            data_dir=args.data_dir,
            output_model=args.output,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_samples_per_bin=args.max_per_bin,
        )
    )
    best_epoch = int(np.argmin(history["val_loss"])) + 1
    print(f"Model written to {args.output}")
    print(f"Best validation MSE: {min(history['val_loss']):.6f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
