# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 FlyBrain contributors
"""Pixels -> sprout awareness: where visible tokens are, and which field the banner names.

Nothing here reads simulator state. Tokens are found by color and mapped to the ground with
the known fixed third-person camera (monocular ground-plane projection). The notification
banner is read by template matching: a tutor shows each banner once and names its field,
the way text would be calibrated for a real game's notification font.
"""
import json
import math
from pathlib import Path

import numpy as np

TOKEN_HEIGHT = .5  # tokens float this far above the ground
TEMPLATE_SIZE = (16, 128)  # rows, cols the banner text is normalized to


def _pose(geometry):
    _, pitch, distance, fov = geometry
    p = math.radians(pitch)
    return p, distance, fov


def pixel_to_ground(u, v, size, geometry, z=TOKEN_HEIGHT, avatar_z=0.):
    """Offset (dx, dy) of the ground point at pixel (u, v) from the avatar, camera yaw 0."""
    h, w = size
    p, d, fov = _pose(geometry)
    fx = (w / 2) / math.tan(math.radians(fov) / 2)
    xc, yc = (u - w / 2) / fx, (h / 2 - v) / fx
    eye_y, eye_z = -math.cos(p) * d, avatar_z + 1 + math.sin(p) * d
    dir_y, dir_z = math.cos(p) + yc * math.sin(p), -math.sin(p) + yc * math.cos(p)
    t = (z - eye_z) / dir_z
    return xc * t, eye_y + dir_y * t


def ground_to_pixel(dx, dy, size, geometry, z=TOKEN_HEIGHT, avatar_z=0.):
    h, w = size
    p, d, fov = _pose(geometry)
    fx = (w / 2) / math.tan(math.radians(fov) / 2)
    rel_y, rel_z = dy + math.cos(p) * d, z - (avatar_z + 1 + math.sin(p) * d)
    forward = rel_y * math.cos(p) - rel_z * math.sin(p)
    up = rel_y * math.sin(p) + rel_z * math.cos(p)
    return w / 2 + fx * dx / forward, h / 2 - fx * up / forward


def token_mask(image):
    r, g, b = (image[..., i].astype(int) for i in range(3))
    return (r > 230) & (g < 70) & (b > 200)


def ground_points(us, vs, size, geometry, z=TOKEN_HEIGHT, avatar_z=0.):
    """Vectorized pixel -> ground offsets from the avatar; pixels at or above the horizon are dropped."""
    h, w = size
    p, d, fov = _pose(geometry)
    fx = (w / 2) / math.tan(math.radians(fov) / 2)
    xc, yc = (np.asarray(us) - w / 2) / fx, (h / 2 - np.asarray(vs)) / fx
    dir_y, dir_z = math.cos(p) + yc * math.sin(p), -math.sin(p) + yc * math.cos(p)
    ok = dir_z < -1e-6
    t = (z - (avatar_z + 1 + math.sin(p) * d)) / np.where(ok, dir_z, -1.)
    return np.stack([xc * t, -math.cos(p) * d + dir_y * t], axis=1)[ok]


def mask_cue(mask, geometry, z=TOKEN_HEIGHT, min_pixels=4, arrive=0.):
    """[unit dx, unit dy, distance/30, arrived] toward the nearest blob of a pixel mask, or None.
    Within `arrive` ground units the direction is zeroed and the arrived flag set."""
    ys, xs = np.nonzero(mask)
    if len(xs) < min_pixels:
        return None
    step = max(1, len(xs) // 600)
    points = ground_points(xs[::step], ys[::step], mask.shape[:2], geometry, z)
    if not len(points):
        return None
    near = points[np.argmin(np.hypot(points[:, 0], points[:, 1]))]
    cluster = points[np.hypot(points[:, 0] - near[0], points[:, 1] - near[1]) <= 1.5].mean(axis=0)
    d = math.hypot(*cluster) or 1.
    if d <= arrive:
        return np.array([0., 0., min(1., d / 30), 1.])
    return np.array([cluster[0] / d, cluster[1] / d, min(1., d / 30), 0.])


def token_cue(image, geometry, min_pixels=4):
    """[unit dx, unit dy, distance/30, 0] toward the nearest visible token, or None."""
    return mask_cue(token_mask(image), geometry, TOKEN_HEIGHT, min_pixels)


def banner_signature(image):
    """Normalized white-text bitmap from inside the dark notification card, or None."""
    top = image[:110].astype(int)
    dark = (top.max(axis=2) < 50)
    rows, cols = np.nonzero(dark)
    if len(rows) < 2000:  # a card is thousands of pixels; shadows and trees are not
        return None
    card = top[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    text = card.min(axis=2) > 170
    ys, xs = np.nonzero(text)
    if len(xs) < 40:
        return None
    crop = text[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(float)
    r, c = TEMPLATE_SIZE
    y_edges, x_edges = np.linspace(0, crop.shape[0], r + 1).astype(int), np.linspace(0, crop.shape[1], c + 1).astype(int)
    sig = np.array([[crop[y_edges[i]:max(y_edges[i + 1], y_edges[i] + 1), x_edges[j]:max(x_edges[j + 1], x_edges[j] + 1)].mean()
                     for j in range(c)] for i in range(r)]).ravel()
    sig -= sig.mean()
    norm = np.linalg.norm(sig)
    return sig / norm if norm else None


class BannerReader:
    threshold, margin = .6, .1  # on similarity after removing the text every banner shares

    def __init__(self):
        self.templates = {}

    def learn(self, name, image):
        sig = banner_signature(image)
        if sig is None:
            return False
        self.templates[name] = sig
        return True

    def read(self, image):
        """(field name, score) for the best template match, or None."""
        sig = banner_signature(image)
        if sig is None or not self.templates:
            return None
        common = np.mean(list(self.templates.values()), axis=0) if len(self.templates) > 1 else 0.
        probe = sig - common
        norm = np.linalg.norm(probe)
        if not norm:
            return None
        scored = []
        for n, t in self.templates.items():
            t = t - common
            scored.append((float(probe @ t / (norm * (np.linalg.norm(t) or 1.))), n))
        scored.sort(reverse=True)
        if scored[0][0] < self.threshold or (len(scored) > 1 and scored[0][0] - scored[1][0] < self.margin):
            return None
        return scored[0][1], scored[0][0]

    def save(self, path):
        Path(path).write_text(json.dumps({n: t.tolist() for n, t in self.templates.items()}), encoding='utf-8')

    def load(self, path):
        self.templates = {n: np.array(t) for n, t in json.loads(Path(path).read_text(encoding='utf-8')).items()}


class SproutSense:
    """Per-decision awareness: remembers the last banner for a while and follows visible tokens."""
    alert_seconds = 30.

    def __init__(self, reader=None):
        self.reader = reader or BannerReader()
        self.alert = None  # (field, simulated time seen)
        self.tokens_seen = False

    def reset(self):
        self.alert, self.tokens_seen = None, False

    def look(self, image, geometry, seconds):
        read = self.reader.read(image)
        if read:
            self.alert = (read[0], seconds)
        cue = token_cue(image, geometry)
        self.tokens_seen = cue is not None
        return cue

    def alert_field(self, seconds):
        if self.alert and seconds - self.alert[1] <= self.alert_seconds:
            return self.alert[0]
        return None
