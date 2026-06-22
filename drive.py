"""Inference server for the Udacity self-driving car simulator.

The simulator streams telemetry frames to this process over Socket.IO. The
server preprocesses each camera frame, predicts a steering angle with the
trained behavioral cloning model, and returns steering/throttle commands.
"""

import base64
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


MODEL_PATH = Path("model/model.h5")
SPEED_LIMIT = 10
SIMULATOR_PORT = 4567


def image_preprocess(img):
    """Preprocess an RGB simulator frame for the NVIDIA-style CNN."""
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    return img / 255.0


def decode_telemetry_image(encoded_image):
    """Decode the simulator's base64 camera image into a NumPy array."""
    image = Image.open(BytesIO(base64.b64decode(encoded_image)))
    return np.asarray(image)


def calculate_throttle(speed, speed_limit=SPEED_LIMIT):
    """Match the original proportional throttle controller."""
    return 1.0 - float(speed) / speed_limit


def predict_control(data, model, speed_limit=SPEED_LIMIT):
    """Predict steering and throttle values from one telemetry payload."""
    speed = float(data["speed"])
    image = decode_telemetry_image(data["image"])
    image = image_preprocess(image)
    batch = np.array([image])

    steering_angle = float(np.asarray(model.predict(batch)).squeeze())
    throttle = calculate_throttle(speed, speed_limit)
    return steering_angle, throttle


def send_control(sio, steering_angle, throttle):
    """Send steering and throttle commands back to the simulator."""
    sio.emit(
        "steer",
        data={
            "steering_angle": str(steering_angle),
            "throttle": str(throttle),
        },
        skip_sid=True,
    )


def create_app(model):
    """Create the Socket.IO/Flask application used by the simulator."""
    import socketio
    from flask import Flask

    sio = socketio.Server()
    flask_app = Flask(__name__)

    @sio.on("telemetry")
    def telemetry(sid, data):
        if data:
            steering_angle, throttle = predict_control(data, model)
            print(
                f"Steering Angle: {steering_angle:.4f}, "
                f"Throttle: {throttle:.4f}"
            )
            send_control(sio, steering_angle, throttle)
        else:
            sio.emit("manual", data={}, skip_sid=True)

    @sio.on("connect")
    def connect(sid, environ):
        print("Connected:", sid)
        send_control(sio, 0, 0)

    return socketio.Middleware(sio, flask_app)


def main():
    """Load the trained model and serve the simulator control loop."""
    import eventlet
    from keras.models import load_model

    app = create_app(load_model(MODEL_PATH, compile=False))
    eventlet.wsgi.server(eventlet.listen(("", SIMULATOR_PORT)), app)


if __name__ == "__main__":
    main()
