# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 FlyBrain contributors
"""Field memory nodes: path-integrated position plus remembered collection sites.

The controller never sees world coordinates. It integrates its own walked displacement
(odometry, like fly path integration), re-anchors at the hive landmark, and writes
a memory node wherever collection succeeds. A goal field name recalls one node and
becomes a 4-value steering cue. Names are given by a tutor during training only.
"""
import json
import math
from pathlib import Path

import numpy as np


class FieldMemory:
    merge_radius = 6.  # fields are 8.4 wide and at least 12 apart
    DISCOVER = 'DISCOVER'  # goal token: sweep toward the nearest unexplored area
    cell, seen_radius = 8., 5.  # coverage grid spacing; how far around itself the bee counts as having looked

    def __init__(self, speed=6., dt=1 / 60, limit=23.5):
        self.speed, self.dt, self.limit = speed, dt, limit
        self.nodes = []  # dicts: name (or None), x, y, n; unnamed nodes are discoveries awaiting a name
        self.x = self.y = 0.
        n = int(2 * (limit + .5) // self.cell)
        self.cells = [(-(limit + .5) + (i + .5) * self.cell, -(limit + .5) + (j + .5) * self.cell)
                      for i in range(n) for j in range(n)]
        self.seen = set()

    def reset(self, x, y):
        self.x, self.y = float(x), float(y)
        self.seen.clear()  # coverage is per episode; nodes persist
        self._look()

    def _look(self):
        self.seen.update(i for i, (cx, cy) in enumerate(self.cells)
                         if math.hypot(cx - self.x, cy - self.y) <= self.seen_radius)

    def integrate(self, dx, dy):
        """Dead-reckon from the displacement the body actually moved (odometry), so being
        stuck against an obstacle adds no drift. Walls clamp the estimate."""
        self.x = max(-self.limit, min(self.limit, self.x + dx))
        self.y = max(-self.limit, min(self.limit, self.y + dy))
        self._look()

    def visit(self, name=None, colors=None):
        """Record a successful collection at the current estimate; merge into a nearby node.
        `colors` (name -> pollen) is what the dipper touched; nodes keep running totals per color."""
        near = [(math.hypot(n['x'] - self.x, n['y'] - self.y), n) for n in self.nodes]
        near = [p for p in near if p[0] <= self.merge_radius]
        if not near:
            node = {'name': name, 'x': self.x, 'y': self.y, 'n': 1, 'colors': dict(colors or {})}
            self.nodes.append(node)
            return node
        node = min(near, key=lambda p: p[0])[1]
        node['n'] += 1
        node['x'] += (self.x - node['x']) / node['n']
        node['y'] += (self.y - node['y']) / node['n']
        node['name'] = node['name'] or name
        totals = node.setdefault('colors', {})
        for color, amount in (colors or {}).items():
            totals[color] = totals.get(color, 0) + amount
        return node

    def recall(self, name):
        return next((n for n in self.nodes if n['name'] == name), None)

    def nearest_field(self):
        """Name of the closest named field to the current estimate (its default place to work), or None."""
        named = [n for n in self.nodes if n['name']]
        return min(named, key=lambda n: math.hypot(n['x'] - self.x, n['y'] - self.y))['name'] if named else None

    def best_field_for(self, color):
        """Named field where it has collected the most pollen of `color`, or None if it never has."""
        known = [n for n in self.nodes if n['name'] and n.get('colors', {}).get(color, 0) > 0]
        return max(known, key=lambda n: n['colors'][color])['name'] if known else None

    def discoveries(self):
        return [n for n in self.nodes if n['name'] is None]

    def name_here(self, name):
        """Name the nearest unnamed (discovered) node within reach, e.g. when told 'this is X'."""
        near = [(math.hypot(n['x'] - self.x, n['y'] - self.y), n) for n in self.discoveries()]
        near = [p for p in near if p[0] <= self.merge_radius]
        if not near or self.recall(name):
            return None
        node = min(near, key=lambda p: p[0])[1]
        node['name'] = name
        return node

    def explore_target(self):
        left = [c for i, c in enumerate(self.cells) if i not in self.seen]
        return min(left, key=lambda c: math.hypot(c[0] - self.x, c[1] - self.y)) if left else None

    def cue(self, name):
        """[unit dx, unit dy, distance/30, arrived]; zeros if the field is not remembered.
        The DISCOVER goal points at the nearest unexplored area (arrived=1 once all is seen)."""
        if name == self.DISCOVER:
            target = self.explore_target()
            if target is None:
                return np.array([0., 0., 0., 1.])
            dx, dy = target[0] - self.x, target[1] - self.y
            d = math.hypot(dx, dy) or 1.
            return np.array([dx / d, dy / d, min(1., d / 30), 0.])
        node = self.recall(name) if name else None
        if node is None:
            return np.zeros(4)
        dx, dy = node['x'] - self.x, node['y'] - self.y
        d = math.hypot(dx, dy)
        arrived = d <= 2.5
        return np.array([0. if arrived else dx / d, 0. if arrived else dy / d, min(1., d / 30), float(arrived)])

    def save(self, path):
        Path(path).write_text(json.dumps(self.nodes, indent=1) + '\n', encoding='utf-8')

    def load(self, path):
        self.nodes = json.loads(Path(path).read_text(encoding='utf-8'))
