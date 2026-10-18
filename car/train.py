"""Train Nvidia's network on your recorded laps.

usage: python train.py <data_dir> [--epochs 15]
<data_dir> holds driving_log.csv and IMG/ (what the simulator writes when you record).
You can pass several data dirs to combine recordings (e.g. laps + recovery clips).
Saves model.pt (best validation loss) and loss.png.
"""
import argparse
import csv
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from common import NvidiaNet, preprocess

SIDE_CAM_CORRECTION = 0.2


def load_log(data_dir):
    """driving_log.csv has no header: center, left, right, steering, throttle, brake, speed.
    Paths are absolute from the machine that recorded, so keep only the file name."""
    rows = []
    skip = set()   # optional exclude.txt: lines like "4725-5095" (row ranges to drop)
    ex = os.path.join(data_dir, "exclude.txt")
    if os.path.exists(ex):
        for line in open(ex):
            if "-" in line:
                a, b = map(int, line.strip().split("-"))
                skip.update(range(a, b + 1))
    with open(os.path.join(data_dir, "driving_log.csv")) as f:
        for i, r in enumerate(csv.reader(f)):
            if i in skip:
                continue
            if not r or r[0].strip() == "center":
                continue
            paths = [os.path.join(data_dir, "IMG", os.path.basename(p.strip().replace("\\", "/")))
                     for p in r[:3]]
            if len(r) > 6 and float(r[6]) < 2.0:   # skip frames where the car is stopped/stuck
                continue
            rows.append((paths, float(r[3])))
    return rows


class DriveData(Dataset):
    def __init__(self, rows, augment):
        self.rows, self.augment = rows, augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        (c, l, r), steer = self.rows[i]
        path = c
        if self.augment and l != c:  # side cameras only exist in human-recorded data
            cam = random.choice([0, 1, 2])
            path = (c, l, r)[cam]
            steer += (0, SIDE_CAM_CORRECTION, -SIDE_CAM_CORRECTION)[cam]
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        if self.augment:
            if random.random() < 0.5:
                img, steer = img[:, ::-1], -steer
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 2] *= random.uniform(0.6, 1.2)
            img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
        return preprocess(img), np.float32(np.clip(steer, -1, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dirs", nargs="*", help="default: every data/* folder with a driving_log.csv")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val", type=float, default=0.2)
    args = ap.parse_args()

    if not args.data_dirs:
        import glob
        args.data_dirs = sorted(os.path.dirname(p) for p in glob.glob(os.path.join("data", "*", "driving_log.csv")))
    print("training on:", args.data_dirs)
    rows = [r for d in args.data_dirs for r in load_log(d)]
    random.seed(0)
    random.shuffle(rows)
    n_val = int(len(rows) * args.val)
    val_rows, train_rows = rows[:n_val], rows[n_val:]
    print(f"{len(train_rows)} training moments, {len(val_rows)} held back for validation")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "")
    workers = 0 if os.name == "nt" else 4
    tr = DataLoader(DriveData(train_rows, True), args.batch, shuffle=True, num_workers=workers)
    va = DataLoader(DriveData(val_rows, False), args.batch, num_workers=workers)

    model = NvidiaNet().to(device)
    print("weights:", sum(p.numel() for p in model.parameters()))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    hist, best = [], float("inf")
    for ep in range(1, args.epochs + 1):
        model.train()
        tl = 0.0
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            tl += loss.item() * len(y)
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for x, y in va:
                vl += loss_fn(model(x.to(device)), y.to(device)).item() * len(y)
        tl, vl = tl / len(train_rows), vl / max(1, len(val_rows))
        hist.append((tl, vl))
        flag = ""
        if vl < best:
            best = vl
            torch.save(model.state_dict(), "model.pt")
            flag = "  (saved)"
        print(f"epoch {ep:2d}  train {tl:.4f}  val {vl:.4f}{flag}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.plot([h[0] for h in hist], label="train")
        plt.plot([h[1] for h in hist], label="validation")
        plt.xlabel("epoch"); plt.ylabel("MSE"); plt.legend(); plt.title("Steering loss")
        plt.savefig("loss.png", dpi=120)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
