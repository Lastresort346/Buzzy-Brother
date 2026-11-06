# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 FlyBrain contributors
"""Where is the player? A small softmax classifier: hive, path, or a named field.

Labels come from recordings automatically: pollen rising while dipping = in the recorded
field, pollen falling = at the hive, anything else = on the path. Features are a coarse
colour thumbnail of the frame. Engineered and measured, not a biological circuit.
"""
from pathlib import Path

import numpy as np

HIVE, PATH = 'hive', 'path'
GRID = (9, 16)  # thumbnail rows, columns
FIELD_SLOTS = 6  # status channels 4-9 carry up to six field probabilities


def thumbnail(frame):
    """Block-mean colour thumbnail in [0, 1], flattened."""
    rows, cols = GRID
    h, w = frame.shape[0] // rows * rows, frame.shape[1] // cols * cols
    blocks = frame[:h, :w].reshape(rows, h // rows, cols, w // cols, 3).mean((1, 3))
    return (blocks / 255.).ravel()


def auto_labels(fills, dipping, field, window=4, moving=None, pads=None):
    """Per-decision place labels from the pollen trace (fills may contain None).

    With `moving`, the hive label also spreads over the standing-still stretch around each
    drop: players wait at the hive before and between conversion steps. With `pads` (the
    honey prompt is on screen), those decisions are the hive for certain.
    """
    labels = _trace_labels(fills, dipping, field, window)
    if pads is not None:
        labels = [HIVE if pad else label for label, pad in zip(labels, pads)]
    if moving is not None:
        for direction in (range(1, len(labels)), range(len(labels) - 2, -1, -1)):
            for t in direction:
                neighbour = t - 1 if direction.step == 1 else t + 1
                if labels[neighbour] == HIVE and labels[t] == PATH and not moving[t]:
                    labels[t] = HIVE
    return labels


def _trace_labels(fills, dipping, field, window):
    n = len(fills)
    known = [f if f is not None else np.nan for f in fills]
    change = np.diff(np.asarray(known, float), prepend=np.nan)
    labels = []
    for t in range(n):
        near = change[max(0, t - window):t + window + 1]
        if np.nanmin(near, initial=0) < -1e-3:
            labels.append(HIVE)
        elif np.nanmax(near, initial=0) > 1e-3 and any(dipping[max(0, t - window):t + window + 1]):
            labels.append(field)
        else:
            labels.append(PATH)
    return labels


class PlaceModel:
    def __init__(self, classes, weights, mean, scale):
        self.classes, self.weights, self.mean, self.scale = list(classes), weights, mean, scale

    @classmethod
    def fit(cls, thumbs, labels, steps=800, lr=.5, l2=1e-3, seed=0):
        classes = [HIVE, PATH] + sorted(set(labels) - {HIVE, PATH})
        if len(classes) - 2 > FIELD_SLOTS:
            raise ValueError(f'At most {FIELD_SLOTS} fields are supported')
        x = np.asarray(thumbs, float)
        mean, scale = x.mean(0), x.std(0) + 1e-6
        x = np.hstack([(x - mean) / scale, np.ones((len(x), 1))])
        y = np.array([classes.index(label) for label in labels])
        # Balance classes: the path dominates recordings.
        weight = len(y) / (len(classes) * np.maximum(1, np.bincount(y, minlength=len(classes))))[y]
        w = np.random.default_rng(seed).normal(0, .01, (x.shape[1], len(classes)))
        target = np.eye(len(classes))[y]
        for _ in range(steps):
            logits = x @ w
            p = np.exp(logits - logits.max(1, keepdims=True))
            p /= p.sum(1, keepdims=True)
            w -= lr * (x.T @ ((p - target) * weight[:, None]) / len(x) + l2 * w)
        return cls(classes, w, mean, scale)

    def probabilities(self, frame):
        return self.probabilities_from_thumb(thumbnail(frame))

    def probabilities_from_thumb(self, thumb):
        x = np.append((thumb - self.mean) / self.scale, 1.)
        logits = x @ self.weights
        p = np.exp(logits - logits.max())
        return p / p.sum()

    def predict(self, frames):
        return [self.classes[int(np.argmax(self.probabilities(f)))] for f in frames]

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, classes=np.array(self.classes), weights=self.weights, mean=self.mean, scale=self.scale)

    @classmethod
    def load(cls, path):
        with np.load(path) as data:
            return cls([str(c) for c in data['classes']], data['weights'], data['mean'], data['scale'])


class Smoother:
    """Exponential average of place probabilities; one odd frame should not flip the place."""

    def __init__(self, keep=.75):  # about 1 s memory at 4 decisions per second
        self.keep, self.probs = keep, None

    def update(self, probs):
        self.probs = probs if self.probs is None else self.keep * self.probs + (1 - self.keep) * probs
        return self.probs


def status(fill, probs, model, pad=False):
    """13 controller status channels: 0 fill, 1 full, 3 hive, 4-9 fields, 12 path (others unused).

    `pad`: the honey prompt is on screen, so the player is on the hive pad: hive is certain.
    """
    s = np.zeros(13)
    if fill is not None:
        s[0], s[1] = fill, float(fill >= .999)
    if pad:
        s[3] = 1.
    elif probs is not None:
        s[3] = probs[model.classes.index(HIVE)]
        s[12] = probs[model.classes.index(PATH)]
        s[4:4 + len(model.classes) - 2] = probs[2:]
    return s
