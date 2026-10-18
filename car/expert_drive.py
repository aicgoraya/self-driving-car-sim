"""A hand-coded 'teacher' driver that records its own training data.

Finds the road (grey, low-saturation pixels) in the camera image and steers toward
its centre. Every few seconds it deliberately swerves for ~1 s, but the label it
saves is always what the teacher WOULD steer, so the data teaches recovery.

usage: python expert_drive.py <out_dir> [--speed 9] [--frames 6000] [--no-record]
Start it, then open the simulator in Autonomous Mode.
"""
import argparse
import base64
import os
import random
import time
from io import BytesIO

import cv2
import eventlet
import eventlet.wsgi
import numpy as np
import socketio
from flask import Flask
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--speed", type=float, default=9.0)
ap.add_argument("--frames", type=int, default=6000)
ap.add_argument("--no-record", action="store_true")
ap.add_argument("--gain", type=float, default=1.6)
args = ap.parse_args()

os.makedirs(os.path.join(args.out, "IMG"), exist_ok=True)
os.makedirs(os.path.join(args.out, "debug"), exist_ok=True)
log = open(os.path.join(args.out, "driving_log.csv"), "a")
status = open(os.path.join(args.out, "status.txt"), "w")


from road_detect import road_mask, target_x


state = {"n": 0, "saved": 0, "integral": 0.0, "prev": 0.0, "swerve_until": 0.0,
         "swerve_val": 0.0, "next_swerve": None, "last_save": 0.0, "t0": None}
RECORD_HZ = 10.0


def teacher(rgb):
    m = road_mask(rgb)
    x = target_x(m)
    if x is None:
        return state["prev"], m
    steer = float(np.clip(args.gain * (x - 160) / 160, -1, 1))
    steer = 0.6 * steer + 0.4 * state["prev"]
    return steer, m


sio = socketio.Server()


@sio.on("telemetry")
def telemetry(sid, data):
    if not data:
        sio.emit("manual", data={}, skip_sid=True)
        return
    st = state
    speed = float(data["speed"])
    img = np.asarray(Image.open(BytesIO(base64.b64decode(data["image"]))).convert("RGB"))
    label, mask = teacher(img)
    st["prev"] = label

    now = time.time()
    if st["t0"] is None:
        st["t0"] = now
        st["next_swerve"] = now + random.uniform(4, 8)
    applied = label
    if not args.no_record:
        if now < st["swerve_until"]:
            applied = float(np.clip(label + st["swerve_val"], -1, 1))
        elif now >= st["next_swerve"]:
            st["swerve_until"] = now + random.uniform(0.4, 0.8)
            st["swerve_val"] = random.choice([-1, 1]) * random.uniform(0.15, 0.35)
            st["next_swerve"] = st["swerve_until"] + random.uniform(4, 8)

    err = args.speed - speed
    st["integral"] = float(np.clip(st["integral"] + err, -50, 50))
    throttle = float(np.clip(0.1 * err + 0.002 * st["integral"], -1, 1))
    sio.emit("steer", data={"steering_angle": str(applied), "throttle": str(throttle)},
             skip_sid=True)

    n = st["n"]
    k = st["saved"]
    if not args.no_record and k < args.frames and now - st["last_save"] >= 1 / RECORD_HZ:
        st["last_save"] = now
        st["saved"] = k + 1
        name = f"center_{k:06d}.jpg"
        p = os.path.join(args.out, "IMG", name)
        Image.fromarray(img).save(p)
        log.write(f"{p},{p},{p},{label:.4f},{throttle:.3f},0,{speed:.2f}\n")
        log.flush()
    if n % 400 == 0:
        dbg = img.copy()
        dbg[mask > 0] = (dbg[mask > 0] * 0.5 + np.array([0, 0, 255]) * 0.5).astype(np.uint8)
        Image.fromarray(dbg).save(os.path.join(args.out, "debug", f"d{n // 400:04d}.jpg"))
        status.seek(0)
        status.write(f"frame {n} t {time.time() - st['t0']:.0f}s speed {speed:.1f} "
                     f"label {label:+.2f} applied {applied:+.2f} recorded {st['saved']}   \n")
        status.flush()
    st["n"] = n + 1


@sio.on("connect")
def connect(sid, environ):
    print("simulator connected")
    sio.emit("steer", data={"steering_angle": "0", "throttle": "0"}, skip_sid=True)


app = socketio.WSGIApp(sio, Flask(__name__))
eventlet.wsgi.server(eventlet.listen(("", 4567)), app, log_output=False)
