"""Connect a trained model to the Udacity simulator in Autonomous Mode.

usage: python drive.py model.pt [--speed 12]
Start this FIRST, then open the simulator and pick Autonomous Mode.
The network only steers; a simple PI controller holds the speed.
"""
import argparse
import base64
from io import BytesIO

import eventlet
import eventlet.wsgi
import numpy as np
import socketio
import torch
from flask import Flask
from PIL import Image

from common import NvidiaNet, preprocess, to_tensor

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--speed", type=float, default=12.0)
args = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = NvidiaNet().to(device)
model.load_state_dict(torch.load(args.model, map_location=device))
model.eval()

sio = socketio.Server()
integral = 0.0


@sio.on("telemetry")
def telemetry(sid, data):
    global integral
    if not data:
        sio.emit("manual", data={}, skip_sid=True)
        return
    speed = float(data["speed"])
    img = np.asarray(Image.open(BytesIO(base64.b64decode(data["image"]))).convert("RGB"))
    with torch.no_grad():
        steer = float(model(to_tensor(preprocess(img)[None], device))[0])
    err = args.speed - speed
    integral = float(np.clip(integral + err, -50, 50))
    throttle = float(np.clip(0.1 * err + 0.002 * integral, -1, 1))
    print(f"steer {steer:+.3f}  throttle {throttle:+.2f}  speed {speed:.1f}")
    sio.emit("steer", data={"steering_angle": str(steer), "throttle": str(throttle)},
             skip_sid=True)


@sio.on("connect")
def connect(sid, environ):
    print("simulator connected")
    sio.emit("steer", data={"steering_angle": "0", "throttle": "0"}, skip_sid=True)


app = socketio.WSGIApp(sio, Flask(__name__))
eventlet.wsgi.server(eventlet.listen(("", 4567)), app)
