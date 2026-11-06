# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 FlyBrain contributors
"""Where am I, where is home, where is my field? Dead reckoning for the Roblox bot.

Assumes the camera yaw stays put (the bot never rotates it), so W/A/S/D always move the same way
on the map. The bot integrates its own held keys into a position (odometry, nothing from the game),
re-anchors at (0, 0) whenever the honey prompt shows (that is your hive's pad, so home is the origin),
and remembers where it collected pollen as named field nodes (arena FieldMemory). It turns that into
the brain's 4-value goal cue: home while the bag is nearly full, the nearest remembered field otherwise.
The cue is engineered; how strongly to follow it is learned (`steering_prior`, then `fit_navigation`).
"""
import math
from pathlib import Path

import numpy as np

from .arena.brain import MOVES
from .arena.memory import FieldMemory

SPEED, DT = 6., .25  # arena units per second held; seconds per decision: the scale the arena cues use
FULL_AT, EMPTY_AT = .98, .05  # head home at nearly full; turn back to the field once nearly empty
FIELD_LIMIT = 60.  # how far from the hive the estimate may wander before it is clamped
NOT_FIELDS = ('hive', 'path', '-')  # place names that are not collection fields
HOME_RADIUS, ARRIVED = 8., 2.5  # no field memories this close to the hive pad; 'arrived' distance


def step_vector(wanted):
    """Map displacement per decision from held keys: d/a = +x/-x, w/s = +y/-y, diagonals normalised."""
    x, y = ('d' in wanted) - ('a' in wanted), ('w' in wanted) - ('s' in wanted)
    norm = max(1., math.hypot(x, y))
    return x / norm * SPEED * DT, y / norm * SPEED * DT


class Compass:
    def __init__(self, memory=None):
        self.memory = memory or FieldMemory(speed=SPEED, dt=DT, limit=FIELD_LIMIT)
        self.reset()

    def reset(self):
        """A new session: at the hive, empty-handed. Remembered fields stay."""
        self.memory.reset(0., 0.)
        self.delivering = False

    @property
    def position(self):
        return self.memory.x, self.memory.y

    def update(self, wanted, blocked, fill, at_pad, collected=0., place=None):
        """After a decision: `wanted` keys held, whether the view stayed frozen (`blocked`: no
        movement counted), the bag `fill`, whether the honey prompt shows, and pollen `collected`
        while dipping in `place`."""
        if not blocked:
            self.memory.integrate(*step_vector(wanted))
        if at_pad:
            self.memory.x = self.memory.y = 0.
        if fill is not None:
            if fill >= FULL_AT:
                self.delivering = True
            elif fill <= EMPTY_AT:
                self.delivering = False
        if collected > 0 and place and place not in NOT_FIELDS and math.hypot(*self.position) > HOME_RADIUS:
            self.memory.visit(place)

    def field_center(self):
        """The middle of the nearest remembered field: all its collection sites, weighted by visits."""
        centers = {}
        for n in self.memory.nodes:
            if n['name']:
                c = centers.setdefault(n['name'], [0., 0., 0.])
                c[0] += n['x'] * n['n']; c[1] += n['y'] * n['n']; c[2] += n['n']
        centers = [(x / w, y / w) for x, y, w in centers.values()]
        return min(centers, key=lambda c: math.hypot(c[0] - self.memory.x, c[1] - self.memory.y), default=None)

    def cue(self):
        """[unit dx, unit dy, distance/30, arrived] toward home (bag nearly full) or the middle of the
        nearest remembered field; zeros if no field is known yet."""
        target = (0., 0.) if self.delivering else self.field_center()
        if target is None:
            return np.zeros(4)
        dx, dy = target[0] - self.memory.x, target[1] - self.memory.y
        d = math.hypot(dx, dy)
        arrived = d <= ARRIVED
        return np.array([0. if arrived else dx / d, 0. if arrived else dy / d, min(1., d / 30), float(arrived)])


def open_memory(path):
    """The remembered fields saved at `path` (none yet if the file is missing). Positions are
    relative to your hive's pad, so a new server with a different hive slot shifts them a little;
    collecting pollen keeps correcting them."""
    memory = FieldMemory(speed=SPEED, dt=DT, limit=FIELD_LIMIT)
    if Path(path).exists():
        memory.load(path)
    return memory


def steering_prior(gain=4.):
    """Starting nav weights (9 moves x 4 cue values): each move scores by how well it points along
    the cue's direction, and 'arrived' favours standing still. Learning refines it."""
    nav = np.zeros((9, 4))
    units = MOVES / np.maximum(np.linalg.norm(MOVES, axis=1, keepdims=True), 1e-9)
    nav[:, :2] = gain * units
    nav[8, 3] = gain
    return nav
