"""Command-line entry point for simulator inference."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .control import ControlConfig
from .server import create_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve a behavioral-cloning steering model.")
    parser.add_argument("--model", type=Path, default=Path("model/model.h5"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4567)
    parser.add_argument("--speed-limit", type=float, default=10.0)
    parser.add_argument("--steering-limit", type=float, default=1.0)
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING"], default="INFO")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    if not args.model.is_file():
        raise FileNotFoundError(
            f"Missing model at {args.model}. Run `self-driving-artifacts --model` first."
        )
    if not 1 <= args.port <= 65535:
        raise ValueError("Port must be between 1 and 65535")

    import eventlet
    from tensorflow.keras.models import load_model

    config = ControlConfig(
        speed_limit=args.speed_limit,
        steering_limit=args.steering_limit,
    )
    app = create_app(load_model(args.model, compile=False), config)
    logging.getLogger(__name__).info("Listening on %s:%d", args.host, args.port)
    eventlet.wsgi.server(eventlet.listen((args.host, args.port)), app)


if __name__ == "__main__":
    main()
