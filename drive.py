
# --- Self-Driving Car Inference Server ---
# Receives telemetry from Udacity simulator, preprocesses images, predicts steering, and sends control commands.

import socketio  # Real-time communication between server and simulator
import eventlet  # WSGI server for asynchronous networking
import numpy as np
from flask import Flask
from keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2


# SocketIO server for real-time events
sio = socketio.Server()

# Flask app for WSGI
app = Flask(__name__)

# Speed limit for throttle calculation
speed_limit = 10


def image_preprocess(img):
    """
    Preprocess input image for model prediction:
    - Crop sky and car hood
    - Convert to YUV color space
    - Apply Gaussian blur
    - Resize to (200, 66)
    - Normalize pixel values to [0, 1]
    """
    img = img[60:135, :, :]  # Crop
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)  # Convert to YUV
    img = cv2.GaussianBlur(img, (3, 3), 0)  # Gaussian blur
    img = cv2.resize(img, (200, 66))  # Resize
    img = img / 255.0  # Normalize
    return img


@sio.on('telemetry')
def telemetry(sid, data):
    """
    Handles telemetry events from the simulator:
    - Decodes and preprocesses the incoming image
    - Predicts steering angle using the trained model
    - Calculates throttle based on speed
    - Sends control commands back to the simulator
    """
    if data:
        # Parse speed from telemetry
        speed = float(data['speed'])
        # Decode image from base64
        image = Image.open(BytesIO(base64.b64decode(data['image'])))
        image = np.asarray(image)
        # Preprocess image for model
        image = image_preprocess(image)
        image = np.array([image])  # Add batch dimension
        # Predict steering angle
        steering_angle = float(model.predict(image))
        # Simple throttle control: decrease throttle as speed increases
        throttle = 1.0 - speed / speed_limit
        print(f'Steering Angle: {steering_angle:.4f}, Throttle: {throttle:.4f}')
        send_control(steering_angle, throttle)
    else:
        # If no data, switch to manual mode
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    """
    Handles new simulator connections.
    Sends zero steering and throttle to initialize.
    """
    print('Connected:', sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    """
    Sends steering and throttle commands to the simulator.
    """
    sio.emit('steer', data={
        'steering_angle': str(steering_angle),
        'throttle': str(throttle)
    }, skip_sid=True)


if __name__ == '__main__':
    # Load trained model
    model = load_model('model/model.h5', compile=False)
    # Wrap Flask app with SocketIO middleware
    app = socketio.Middleware(sio, app)
    # Start WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)