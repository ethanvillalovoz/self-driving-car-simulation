"""Socket.IO adapter for the Udacity driving simulator."""

from __future__ import annotations

import logging
from typing import Any

from .control import ControlCommand, ControlConfig, PredictionModel, predict_control

LOGGER = logging.getLogger(__name__)


def send_control(sio: Any, sid: str, command: ControlCommand) -> None:
    sio.emit(
        "steer",
        data={
            "steering_angle": str(command.steering_angle),
            "throttle": str(command.throttle),
        },
        room=sid,
    )


def create_app(model: PredictionModel, config: ControlConfig | None = None) -> Any:
    """Create the legacy-compatible Socket.IO WSGI application."""
    import socketio
    from flask import Flask

    active_config = config or ControlConfig()
    sio = socketio.Server(async_mode="eventlet")
    flask_app = Flask(__name__)

    @sio.on("telemetry")
    def telemetry(sid: str, data: dict[str, Any] | None) -> None:
        if not data:
            sio.emit("manual", data={}, room=sid)
            return
        try:
            command = predict_control(data, model, active_config)
        except (KeyError, TypeError, ValueError) as exc:
            LOGGER.warning("Rejected telemetry from %s: %s", sid, exc)
            send_control(sio, sid, ControlCommand(steering_angle=0.0, throttle=0.0))
            return
        LOGGER.info(
            "sid=%s steering=%.4f throttle=%.4f",
            sid,
            command.steering_angle,
            command.throttle,
        )
        send_control(sio, sid, command)

    @sio.on("connect")
    def connect(sid: str, _environ: dict[str, Any]) -> None:
        LOGGER.info("Simulator connected: %s", sid)
        send_control(sio, sid, ControlCommand(steering_angle=0.0, throttle=0.0))

    @sio.on("disconnect")
    def disconnect(sid: str) -> None:
        LOGGER.info("Simulator disconnected: %s", sid)

    return socketio.WSGIApp(sio, flask_app)
