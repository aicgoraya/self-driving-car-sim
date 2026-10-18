"""Adaptive road finder for the teacher driver.

Samples the colour of the surface right in front of the car, keeps pixels close to
that colour (in Lab space, after blurring away texture), and keeps only the region
connected to the car. Works on asphalt, the brick bridge and dirt without tuning.
"""
import cv2
import numpy as np

REF = (slice(112, 126), slice(130, 190))   # patch just above the hood


def road_mask(rgb, thresh=26.0):
    lab = cv2.cvtColor(cv2.GaussianBlur(rgb, (9, 9), 0), cv2.COLOR_RGB2LAB).astype(np.float32)
    ref = np.median(lab[REF].reshape(-1, 3), axis=0)
    w = np.array([0.6, 1.4, 1.4], np.float32)          # care more about colour than brightness
    d = np.sqrt((((lab - ref) * w) ** 2).sum(axis=2))
    m = (d < thresh).astype(np.uint8)
    # yellow edge lines are walls: cut them out so the shoulder isn't connected to the lane
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    yellow = ((hsv[:, :, 0] > 15) & (hsv[:, :, 0] < 38) & (hsv[:, :, 1] > 70) & (hsv[:, :, 2] > 90))
    yellow = cv2.dilate(yellow.astype(np.uint8), np.ones((3, 7), np.uint8))
    m[yellow > 0] = 0
    m[:60] = 0                                           # sky / horizon
    m[132:] = 0                                          # hood
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab_cc = cv2.connectedComponents(m)
    seed = lab_cc[REF].ravel()
    seed = seed[seed > 0]
    if len(seed) == 0:
        return m * 0
    keep = np.bincount(seed).argmax()
    return (lab_cc == keep).astype(np.uint8)


def band_center(mask, y0, y1):
    cols = mask[y0:y1].mean(axis=0) > 0.5
    if cols.sum() < 10:
        return None
    idx = np.where(cols)[0]
    mid = 160 if cols[160] else idx[np.argmin(abs(idx - 160))]
    l = mid
    while l > 0 and cols[l - 1]:
        l -= 1
    r = mid
    while r < 319 and cols[r + 1]:
        r += 1
    return (l + r) / 2.0, r - l


def target_x(mask):
    near = band_center(mask, 105, 122)
    far = band_center(mask, 82, 96)
    pts = [(c, 0.4) for c in [near] if c] + [(c, 0.6) for c in [far] if c]
    if not pts:
        return None
    return sum(c[0] * w for c, w in pts) / sum(w for _, w in pts)
