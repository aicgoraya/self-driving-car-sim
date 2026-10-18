"""Shared bits for training and driving. Preprocessing MUST be identical in both,
or the car swerves (the #1 bug in this project)."""
import cv2
import numpy as np
import torch
import torch.nn as nn

IMG_H, IMG_W = 66, 200


def preprocess(rgb):
    """160x320x3 RGB simulator frame -> 66x200x3 YUV float, as in Nvidia's paper."""
    img = rgb[60:135, :, :]                      # crop sky and hood
    img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    return img


class NvidiaNet(nn.Module):
    """Bojarski et al. 2016: normalise, 5 conv, 3 dense, 1 output (~250k weights)."""

    def __init__(self, dropout=0.5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2), nn.ELU(),
            nn.Conv2d(24, 36, 5, stride=2), nn.ELU(),
            nn.Conv2d(36, 48, 5, stride=2), nn.ELU(),
            nn.Conv2d(48, 64, 3), nn.ELU(),
            nn.Conv2d(64, 64, 3), nn.ELU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout),
            nn.Linear(64 * 1 * 18, 100), nn.ELU(),
            nn.Linear(100, 50), nn.ELU(),
            nn.Linear(50, 10), nn.ELU(),
            nn.Linear(10, 1),
        )

    def forward(self, x):
        # x: (N, 66, 200, 3) uint8 or float in [0, 255]
        x = x.permute(0, 3, 1, 2).float() / 127.5 - 1.0
        return self.head(self.conv(x)).squeeze(1)


def to_tensor(imgs, device):
    return torch.from_numpy(np.ascontiguousarray(imgs)).to(device)
