# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 FlyBrain contributors
"""Synthetic flower-field views for training a flower recognizer, in NumPy.

A top-down ground texture is built from the cropped sprites (real flowers) plus negatives
that must NOT be called flowers: tokens, fake flowers in wrong colors or shapes, recolored real
sprites, and other round or star-shaped objects. It is then rendered through a pinhole camera at a
random pitch, yaw, distance and field of view. Every scene returns exact labels:

    labels   0 background / 1 real flower
    hard     True where a negative object sits (the confusable pixels)
    kinds    0 none, 1 token, 2 fake flower, 3 recolored sprite, 4 other object

    python -m flybrain.arena.flower_scenes --preview reports/flower-scenes-preview.png
"""
import argparse
import math
import zlib
import struct
from pathlib import Path

import numpy as np

from .flower_sheet import KINDS, COLORS, OUT, read_rgb

TEXTURE = 1024  # top-down texture size in pixels
PPU = 32.  # texture pixels per world unit
KIND_NAMES = {1: 'token', 2: 'fake_flower', 3: 'recolored_sprite', 4: 'other'}


def load_sprites(directory=OUT / 'sprites'):
    """name -> float RGBA (h, w, 4) in 0..1. Alpha is stored in a sibling read of the tile."""
    from panda3d.core import Filename, PNMImage
    sprites = {}
    for kind in KINDS:
        for color in COLORS:
            img = PNMImage()
            if not img.read(Filename.fromOsSpecific(str(Path(directory) / f'{kind}-{color}.png'))):
                raise FileNotFoundError(f'Run flybrain.arena.flower_sheet first ({kind}-{color}.png missing)')
            h, w = img.getYSize(), img.getXSize()
            a = np.empty((h, w, 4), dtype=np.float32)
            for y in range(h):
                for x in range(w):
                    a[y, x] = (img.getRed(x, y), img.getGreen(x, y), img.getBlue(x, y), img.getAlpha(x, y))
            sprites[f'{kind}-{color}'] = a
    return sprites


def bilinear(image, u, v):
    """Sample image (h, w, c) at float pixel coordinates; edges clamp."""
    h, w = image.shape[:2]
    u, v = np.clip(u, 0, w - 1.001), np.clip(v, 0, h - 1.001)
    u0, v0 = u.astype(int), v.astype(int)
    fu, fv = (u - u0)[..., None], (v - v0)[..., None]
    return (image[v0, u0] * (1 - fu) * (1 - fv) + image[v0, u0 + 1] * fu * (1 - fv)
            + image[v0 + 1, u0] * (1 - fu) * fv + image[v0 + 1, u0 + 1] * fu * fv)


def transformed(sprite, scale, angle):
    """Rotate/scale an RGBA sprite about its center (inverse mapping, bilinear)."""
    h, w = sprite.shape[:2]
    size = int(math.ceil(max(h, w) * scale * 1.45)) | 1
    ys, xs = np.mgrid[:size, :size] - size // 2
    c, s = math.cos(angle), math.sin(angle)
    u = (c * xs + s * ys) / scale + (w - 1) / 2
    v = (-s * xs + c * ys) / scale + (h - 1) / 2
    inside = (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
    out = bilinear(sprite, u, v)
    out[..., 3] *= inside
    return out


def paste(texture, labels, hard, kinds, rgba, cx, cy, label=0, kind=0):
    """Alpha-composite an RGBA patch centered at (cx, cy); update label maps where alpha > .5."""
    size = rgba.shape[0]
    x0, y0 = int(cx) - size // 2, int(cy) - size // 2
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + size, TEXTURE), min(y0 + size, TEXTURE)
    if xa >= xb or ya >= yb:
        return
    patch = rgba[ya - y0:yb - y0, xa - x0:xb - x0]
    a = patch[..., 3:4]
    texture[ya:yb, xa:xb] = patch[..., :3] * a + texture[ya:yb, xa:xb] * (1 - a)
    solid = patch[..., 3] > .5
    region = (slice(ya, yb), slice(xa, xb))
    labels[region][solid] = label
    hard[region][solid] = kind != 0
    kinds[region][solid] = kind


def polar_patch(radius, color, center_color, petals, depth, ring=None, spin=0.):
    """Procedural flower/star/token: radius r(theta) = R * (1 - depth + depth * |cos(petals*theta/2)|)."""
    size = int(radius * 2.6) | 1
    ys, xs = np.mgrid[:size, :size] - size // 2
    d, theta = np.hypot(xs, ys), np.arctan2(ys, xs) + spin
    edge = radius * (1 - depth + depth * np.abs(np.cos(petals * theta / 2)))
    out = np.zeros((size, size, 4), dtype=np.float32)
    body = np.clip(edge - d + .5, 0, 1)
    out[..., :3] = color
    if ring is not None:
        out[..., :3] = np.where((d > edge - max(2, radius * .16))[..., None], ring, out[..., :3])
    core = d < radius * .3
    out[..., :3] = np.where(core[..., None], center_color, out[..., :3])
    out[..., 3] = body
    shade = (.82 + .18 * np.clip(1 - d / max(radius, 1), 0, 1))[..., None]
    out[..., :3] *= shade
    return out


FAKE_PALETTES = [(.2, .75, .25), (.95, .55, .1), (.6, .25, .8), (.95, .5, .7), (.12, .12, .14), (.1, .8, .8),
                 (.95, .9, .2), (.5, .3, .15)]
TOKEN_PALETTES = [((.98, .78, .1), (1., 1., .8)), ((1., .05, .95), (1., .8, 1.)), ((.2, .85, .95), (.9, 1., 1.)),
                  ((.95, .45, .1), (1., .9, .6)), ((.5, .9, .25), (.9, 1., .8)), ((.65, .3, .85), (.95, .85, 1.)),
                  ((.15, .35, .9), (.9, .95, 1.)), ((.9, .1, .1), (1., .85, .85)), ((.96, .96, .96), (.8, .8, .8))]


def make_token(rng):
    """Round collectible: colored disc with a bright rim and a simple glyph; some mimic flower colors."""
    radius = rng.uniform(9, 17)
    body, rim = TOKEN_PALETTES[rng.integers(len(TOKEN_PALETTES))]
    glyph = rng.integers(3)
    patch = polar_patch(radius, body, rim, petals=2, depth=0., ring=rim)
    size = patch.shape[0]
    ys, xs = np.mgrid[:size, :size] - size // 2
    d, theta = np.hypot(xs, ys), np.arctan2(ys, xs)
    if glyph == 0:  # star
        mark = d < radius * .55 * (.5 + .5 * np.abs(np.cos(2.5 * theta)))
    elif glyph == 1:  # bar
        mark = (np.abs(xs) < radius * .45) & (np.abs(ys) < radius * .14)
    else:  # hexagon-ish
        mark = d < radius * .42 * (1 - .12 * np.abs(np.cos(3 * theta)))
    patch[..., :3] = np.where((mark & (d > radius * .1))[..., None], rim, patch[..., :3])
    return patch


def make_fake_flower(rng, sprites):
    """A flower-shaped object that is not a collectible flower: odd palette/shape, or a recolored sprite."""
    if rng.random() < .35:
        name = list(sprites)[rng.integers(len(sprites))]
        patch = transformed(sprites[name], rng.uniform(.45, .75), rng.uniform(0, 2 * math.pi))
        original = patch[..., :3].copy()
        perm = [(1, 2, 0), (2, 0, 1), (1, 0, 2), (0, 2, 1)][rng.integers(4)]  # channel shuffle: wrong colors
        patch[..., :3] = patch[..., :3][..., perm]
        solid = patch[..., 3] > .5
        if solid.any() and np.abs(patch[..., :3] - original)[solid].mean() < .15:  # white stays white: tint it
            tint = [(1., .45, .2), (.4, 1., .4), (.8, .4, 1.), (1., .9, .2)][rng.integers(4)]
            patch[..., :3] *= np.array(tint, dtype=np.float32)
        return patch, 3
    petals = int(rng.choice([3, 4, 5, 6, 8, 10]))
    radius = rng.uniform(9, 18)
    color = FAKE_PALETTES[rng.integers(len(FAKE_PALETTES))]
    centre = (.98, .78, .12) if rng.random() < .6 else FAKE_PALETTES[rng.integers(len(FAKE_PALETTES))]
    return polar_patch(radius, color, centre, petals, rng.uniform(.25, .55), spin=rng.uniform(0, 6.28)), 2


def make_other(rng):
    """Boxes, crates, mushrooms, buttons: flower-colored sometimes."""
    radius = rng.uniform(8, 18)
    colors = [(.4, .55, .9), (.85, .15, .15), (.96, .96, .92), (.45, .3, .2), (.85, .75, .3)]
    color = colors[rng.integers(len(colors))]
    if rng.random() < .5:
        patch = polar_patch(radius, color, (.97, .97, .9), petals=4, depth=.0)
        size = patch.shape[0]
        ys, xs = np.mgrid[:size, :size] - size // 2
        patch[..., 3] = (np.maximum(np.abs(xs), np.abs(ys)) < radius * .85).astype(np.float32)  # box
        return patch
    patch = polar_patch(radius, color, (.97, .97, .9), petals=2, depth=0.)  # mushroom cap / button
    size = patch.shape[0]
    ys, xs = np.mgrid[:size, :size] - size // 2
    dots = (np.hypot(xs - radius * .35, ys + radius * .3) < radius * .16) | (np.hypot(xs + radius * .4, ys - radius * .2) < radius * .14)
    patch[..., :3] = np.where(dots[..., None], (.98, .98, .95), patch[..., :3])
    return patch


def ground_texture(rng):
    grass = np.array([(.17, .56, .24), (.2, .6, .3), (.25, .62, .26), (.15, .5, .28), (.08, .3, .13), (.1, .36, .16),
                      (.05, .22, .1)][rng.integers(7)], dtype=np.float32)
    tile = 64
    ys, xs = np.mgrid[:TEXTURE, :TEXTURE]
    checker = (((ys // tile) + (xs // tile)) % 2).astype(np.float32)[..., None]
    texture = grass * (1 + rng.uniform(.03, .1) * (checker - .5)) + rng.normal(0, .012, (TEXTURE, TEXTURE, 1))
    for _ in range(rng.integers(0, 3)):  # dirt path
        x0, w = rng.integers(0, TEXTURE), rng.integers(40, 110)
        texture[:, max(0, x0):x0 + w] = (.72, .66, .46)
    return np.clip(texture, 0, 1).astype(np.float32)


def field_layout(rng, taken):
    """An axis-aligned lattice of cells (rows and columns at a constant pitch) that overlaps no earlier field."""
    for _ in range(60):
        scale = rng.uniform(.22, .7)
        pitch = max(8, int(round(50 * scale * rng.uniform(.95, 1.2))))
        cols, rows = int(rng.integers(5, 31)), int(rng.integers(4, 27))
        w, h = cols * pitch, rows * pitch
        if w >= TEXTURE - 100 or h >= TEXTURE - 100:
            continue
        x0, y0 = int(rng.integers(50, TEXTURE - w - 50)), int(rng.integers(50, TEXTURE - h - 50))
        box = (x0 - pitch, y0 - pitch, x0 + w + pitch, y0 + h + pitch)
        if any(not (box[2] <= t[0] or box[0] >= t[2] or box[3] <= t[1] or box[1] >= t[3]) for t in taken):
            continue
        return {'x0': x0, 'y0': y0, 'cols': cols, 'rows': rows, 'pitch': pitch, 'scale': float(scale), 'box': box}
    return None


def build_field(rng, sprites, negatives=True):
    """Fields are straight rows and columns of upright, equally spaced flowers (no jitter or rotation).
    Single flowers, pairs, triples, blooms and stars are mixed per cell in some fields and uniform in others;
    a few cells are empty (harvested) and a few hold a lookalike, the hardest negatives."""
    texture = ground_texture(rng)
    labels = np.zeros((TEXTURE, TEXTURE), np.uint8)
    hard = np.zeros((TEXTURE, TEXTURE), bool)
    kinds = np.zeros((TEXTURE, TEXTURE), np.uint8)
    taken, fields = [], []
    for _ in range(int(rng.integers(1, 4))):
        layout = field_layout(rng, taken)
        if layout is None:
            break
        taken.append(layout['box'])
        color = COLORS[rng.choice(3, p=[.5, .25, .25])]
        uniform_kind = KINDS[rng.integers(5)] if rng.random() < .5 else None
        patches = {}
        cells = []
        for r in range(layout['rows']):
            for c in range(layout['cols']):
                cx = layout['x0'] + (c + .5) * layout['pitch']
                cy = layout['y0'] + (r + .5) * layout['pitch']
                roll = rng.random()
                if roll < .04:
                    continue  # harvested / empty cell
                if negatives and roll < .07:  # a lookalike sitting in the grid
                    patch, kind = make_fake_flower(rng, sprites) if rng.random() < .6 else (make_token(rng), 1)
                    paste(texture, labels, hard, kinds, patch, cx, cy, label=0, kind=kind)
                    continue
                kind = uniform_kind or KINDS[rng.choice(5, p=[.34, .22, .16, .16, .12])]
                shade = color if rng.random() < .82 else COLORS[rng.integers(3)]
                key = f'{kind}-{shade}'
                if key not in patches:
                    patches[key] = transformed(sprites[key], layout['scale'], 0.)
                paste(texture, labels, hard, kinds, patches[key], cx, cy, label=1)
                cells.append((int(cx), int(cy)))
        fields.append({**{k: v for k, v in layout.items() if k != 'box'}, 'color': color, 'uniform_kind': uniform_kind, 'cells': cells})
    if negatives:
        for _ in range(rng.integers(25, 90)):
            x, y = rng.uniform(30, TEXTURE - 30, 2)
            if any(b[0] <= x <= b[2] and b[1] <= y <= b[3] for b in taken) and rng.random() < .7:
                continue  # mostly keep loose objects off the lattice so rows stay straight
            roll = rng.random()
            if roll < .4:
                patch, kind = make_token(rng), 1
            elif roll < .75:
                patch, kind = make_fake_flower(rng, sprites)
            else:
                patch, kind = make_other(rng), 4
            paste(texture, labels, hard, kinds, patch, x, y, label=0, kind=kind)
    return texture, labels, hard, kinds, fields


def render(texture, labels, hard, kinds, rng, size=(400, 640), pitch=None, yaw=None, with_clutter=True):
    """Pinhole view of the ground texture. Returns image uint8, labels, hard, kinds at `size`."""
    h, w = size
    pitch = math.radians(rng.uniform(24, 70) if pitch is None else pitch)
    yaw = rng.uniform(0, 2 * math.pi) if yaw is None else yaw
    dist, fov = rng.uniform(8, 36), rng.uniform(58, 78)
    target = np.array([rng.uniform(-6, 6), rng.uniform(-6, 6), 0.])
    eye = target + dist * np.array([math.sin(yaw) * math.cos(pitch), -math.cos(yaw) * math.cos(pitch), math.sin(pitch)])
    forward = target + np.array([0, 0, 1.]) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1.])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    fx = (w / 2) / math.tan(math.radians(fov) / 2)
    ys, xs = np.mgrid[:h, :w]
    ray = (forward[None, None] + ((xs - w / 2) / fx)[..., None] * right + ((h / 2 - ys) / fx)[..., None] * up)
    t = -eye[2] / np.where(ray[..., 2] < -1e-6, ray[..., 2], -1e-6)
    ground = ray[..., 2] < -1e-6
    wx, wy = eye[0] + t * ray[..., 0], eye[1] + t * ray[..., 1]
    u, v = wx * PPU + TEXTURE / 2, wy * PPU + TEXTURE / 2
    inside = ground & (u >= 0) & (u < TEXTURE - 1) & (v >= 0) & (v < TEXTURE - 1)
    sky = np.array([.62, .78, .9] if rng.random() < .6 else [0., 0., 0.], dtype=np.float32) * (.9 + .1 * (1 - ys / h))[..., None]
    far = np.array([.2, .5, .25], dtype=np.float32)
    image = np.where(ground[..., None], far, sky)
    sampled = bilinear(texture, np.where(inside, u, 0), np.where(inside, v, 0))
    image = np.where(inside[..., None], sampled, image)
    ui, vi = np.clip(u.astype(int), 0, TEXTURE - 1), np.clip(v.astype(int), 0, TEXTURE - 1)
    lab, hd, kd = labels[vi, ui] * inside, hard[vi, ui] & inside, kinds[vi, ui] * inside
    # photometric variation
    image = image * rng.uniform(.8, 1.2) + rng.uniform(-.06, .06)
    if rng.random() < .5:  # dim scenes: much of the real game is dark
        image = image * rng.uniform(.3, .75)
    image = (image - .5) * rng.uniform(.85, 1.2) + .5
    if rng.random() < .5:  # soft focus
        padded = np.pad(image, ((1, 1), (1, 1), (0, 0)), mode='edge')
        image = sum(padded[dy:dy + h, dx:dx + w] for dy in range(3) for dx in range(3)) / 9
    image = image + rng.normal(0, rng.uniform(0, .02), image.shape)
    out = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    lab, hd, kd = lab.astype(np.uint8), hd, kd.astype(np.uint8)
    if with_clutter:
        clutter(out, lab, hd, kd, rng)
    return out, lab, hd, kd


def clutter(image, labels, hard, kinds, rng):
    """Screen-space things that are not flowers: huge foreground petals, fence slats, health bars, icons.
    They cover flowers (label cleared) and are marked as confusable negatives."""
    h, w = image.shape[:2]
    ys, xs = np.mgrid[:h, :w]

    def stamp(mask, color):
        image[mask] = color
        labels[mask] = 0
        hard[mask] = True
        kinds[mask] = 4

    for _ in range(rng.integers(0, 3)):  # large rounded foreground petal / pillar
        cx, cy = rng.uniform(0, w), rng.uniform(.3 * h, 1.1 * h)
        rx, ry = rng.uniform(.06, .2) * w, rng.uniform(.1, .3) * h
        color = [(150, 130, 0), (150, 20, 25), (240, 240, 240), (90, 60, 30), (40, 120, 50)][rng.integers(5)]
        stamp(((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2 < 1, np.array(color, np.uint8))
    if rng.random() < .35:  # white fence slats
        x0, y0 = rng.uniform(.4 * w, .8 * w), rng.uniform(0, .2 * h)
        slat, gap = int(rng.integers(8, 18)), int(rng.integers(6, 14))
        band = (ys > y0) & (ys < y0 + rng.uniform(.08, .2) * h) & (xs > x0) & (((xs - x0) // 1).astype(int) % (slat + gap) < slat)
        stamp(band, np.array((215, 215, 215), np.uint8))
    for _ in range(rng.integers(0, 4)):  # HUD bar or badge
        x, y = rng.uniform(0, w - 80), rng.uniform(0, h - 20)
        bw, bh = rng.uniform(25, 90), rng.uniform(5, 14)
        stamp((xs > x) & (xs < x + bw) & (ys > y) & (ys < y + bh), np.array((150, 40, 40), np.uint8))
        stamp((xs > x) & (xs < x + .6 * bw) & (ys > y) & (ys < y + bh * .35), np.array((235, 235, 235), np.uint8))
    for _ in range(rng.integers(0, 4)):  # small icons / stars
        cx, cy, r = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(5, 11)
        theta = np.arctan2(ys - cy, xs - cx)
        star = np.hypot(xs - cx, ys - cy) < r * (.45 + .55 * np.abs(np.cos(2.5 * theta)))
        stamp(star, np.array([(30, 150, 170), (230, 200, 40), (200, 60, 70)][rng.integers(3)], np.uint8))


def make_scene(rng, sprites, negatives=True, size=(400, 640)):
    texture, labels, hard, kinds, fields = build_field(rng, sprites, negatives)
    image, lab, hd, kd = render(texture, labels, hard, kinds, rng, size, with_clutter=negatives)
    return {'image': image, 'labels': lab, 'hard': hd, 'kinds': kd, 'fields': fields}


def save_png(path, image):
    h, w = image.shape[:2]
    raw = b''.join(b'\x00' + image[y].tobytes() for y in range(h))

    def chunk(tag, data):
        body = struct.pack('>I', len(data)) + tag + data
        return body + struct.pack('>I', zlib.crc32(tag + data))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
                           + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b''))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', type=Path, help='Write a montage of sample views to this PNG')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--count', type=int, default=6)
    args = parser.parse_args()
    sprites = load_sprites()
    rng = np.random.default_rng(args.seed)
    scenes = [make_scene(rng, sprites, size=(200, 320)) for _ in range(args.count)]
    if args.preview:
        rows = []
        for s in scenes:
            overlay = s['image'].copy()
            overlay[s['labels'] == 1] = (overlay[s['labels'] == 1] * .4 + np.array([255, 0, 0]) * .6).astype(np.uint8)
            overlay[s['hard']] = (overlay[s['hard']] * .4 + np.array([255, 255, 0]) * .6).astype(np.uint8)
            rows.append(np.concatenate([s['image'], overlay], axis=1))
        save_png(args.preview, np.concatenate(rows, axis=0))
        print('Wrote', args.preview, '(left: view, right: red = real flower label, yellow = negative object)')


if __name__ == '__main__':
    main()
