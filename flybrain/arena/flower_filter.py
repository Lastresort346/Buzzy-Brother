# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 FlyBrain contributors
"""Shape cleanup for a flower mask. Pure NumPy plus run-length connected components.

Real fields are dense lattices of small blobs. A raw per-pixel mask also lights up things a field
cannot be: one-pixel outlines around big objects, big solid regions, and lone blobs (tokens,
fakes). `clean` removes those and keeps lattice-like mass, including merged dense far-away fields.
"""
import numpy as np


def _shifted(mask, dy, dx):
    out = np.zeros_like(mask)
    h, w = mask.shape
    out[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] = mask[max(-dy, 0):h + min(-dy, 0), max(-dx, 0):w + min(-dx, 0)]
    return out


def opening(mask, radius=1):
    """Binary erosion then dilation with a square kernel: removes structures thinner than 2*radius+1."""
    offsets = [(dy, dx) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)]
    eroded = mask.copy()
    for dy, dx in offsets:
        eroded &= _shifted(mask, dy, dx)
    out = np.zeros_like(mask)
    for dy, dx in offsets:
        out |= _shifted(eroded, dy, dx)
    return out


def components(mask):
    """Connected components (4-neighbour) from row runs + union-find.
    Returns a list of dicts: area, bbox (y0, y1, x0, x1), centroid, runs [(y, x0, x1)]."""
    parent = []

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    previous, all_runs = [], []
    for y in range(mask.shape[0]):
        row = mask[y]
        edges = np.flatnonzero(np.diff(np.concatenate(([0], row.astype(np.int8), [0]))))
        current = []
        for x0, x1 in zip(edges[::2], edges[1::2]):
            idx = len(parent)
            parent.append(idx)
            current.append((x0, x1, idx))
            all_runs.append((y, x0, x1, idx))
        i = 0
        for x0, x1, idx in current:  # union with overlapping runs in the row above
            while i < len(previous) and previous[i][1] <= x0:
                i += 1
            j = i
            while j < len(previous) and previous[j][0] < x1:
                a, b = find(idx), find(previous[j][2])
                if a != b:
                    parent[a] = b
                j += 1
        previous = current
    groups = {}
    for y, x0, x1, idx in all_runs:
        g = groups.setdefault(find(idx), {'area': 0, 'y0': y, 'y1': y, 'x0': x0, 'x1': x1, 'sy': 0., 'sx': 0., 'runs': []})
        n = x1 - x0
        g['area'] += n
        g['y1'] = y
        g['x0'], g['x1'] = min(g['x0'], x0), max(g['x1'], x1)
        g['sy'] += y * n
        g['sx'] += (x0 + x1 - 1) / 2 * n
        g['runs'].append((y, x0, x1))
    return [{'area': g['area'], 'bbox': (g['y0'], g['y1'] + 1, g['x0'], g['x1']), 'centroid': (g['sy'] / g['area'], g['sx'] / g['area']),
             'runs': g['runs']} for g in groups.values()]


def row_fill(part):
    """Filled fraction of each row between its first and last pixel, averaged by area: ~1 for a solid object
    (even a disc), clearly lower for a lattice of flowers with gaps."""
    spans = {}
    for y, x0, x1 in part['runs']:
        lo, hi = spans.get(y, (x0, x1))
        spans[y] = (min(lo, x0), max(hi, x1))
    return part['area'] / max(1, sum(hi - lo for lo, hi in spans.values()))


def clean(masks, flower=1, thin_radius=1, solid_fraction=.012, solid_fill=.94, small_fraction=.0007, neighbours=2, reach=6.):
    """Return a cleaned copy of a class-id mask (flower pixels only are touched).

    1. opening drops outlines thinner than 2*thin_radius+1 pixels
    2. big components that are almost solid (a sunflower, a wall) are dropped; a merged dense field
       has gaps between flowers, so its row fill is lower and it stays
    3. small blobs need at least `neighbours` other blobs within `reach` blob-diameters (lattice test)
    """
    binary = masks == flower
    h, w = binary.shape
    binary = opening(binary, thin_radius) if thin_radius else binary
    parts = components(binary)
    keep = np.zeros_like(binary)
    small = [p for p in parts if p['area'] < small_fraction * h * w]
    centres = np.array([p['centroid'] for p in small]) if small else np.zeros((0, 2))
    sizes = np.array([2 * np.sqrt(p['area'] / np.pi) for p in small]) if small else np.zeros(0)
    small_ids = {id(p): i for i, p in enumerate(small)}
    for p in parts:
        if p['area'] >= solid_fraction * h * w and row_fill(p) > solid_fill:
            continue
        if id(p) in small_ids:
            i = small_ids[id(p)]
            d = np.hypot(centres[:, 0] - centres[i, 0], centres[:, 1] - centres[i, 1])
            if int((d <= reach * max(sizes[i], 2.)).sum()) - 1 < neighbours:
                continue
        for y, a, b in p['runs']:
            keep[y, a:b] = True
    out = masks.copy()
    out[(masks == flower) & ~keep] = 0
    return out
