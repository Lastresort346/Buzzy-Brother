# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 FlyBrain contributors
"""Cut a flower sprite sheet (5 rows x 3 colors) into named tiles and alpha-keyed sprites.

    python -m flybrain.arena.flower_sheet path/to/sheet.png

Layout assumed (as in the supplied screenshot): rows are single, pair, triple, bloom, star;
columns are white, red, blue. The sheet is third-party game artwork: tiles are written under
data/ (git-ignored) for local training only and must not be committed or redistributed.
"""
import argparse
from pathlib import Path

import numpy as np
from panda3d.core import Filename, PNMImage

KINDS = ('single', 'pair', 'triple', 'bloom', 'star')
COLORS = ('white', 'red', 'blue')
OUT = Path(__file__).resolve().parents[2] / 'data/bss-flowers'


def read_rgb(path):
    img = PNMImage()
    if not img.read(Filename.fromOsSpecific(str(path))):
        raise FileNotFoundError(path)
    w, h = img.getXSize(), img.getYSize()
    rgb = np.empty((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            rgb[y, x] = (round(img.getRed(x, y) * 255), round(img.getGreen(x, y) * 255), round(img.getBlue(x, y) * 255))
    return rgb


def write_rgba(path, rgba):
    h, w = rgba.shape[:2]
    img = PNMImage(w, h, 4)
    for y in range(h):
        for x in range(w):
            r, g, b, a = (int(v) / 255 for v in rgba[y, x])
            img.setXelA(x, y, r, g, b, a)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not img.write(Filename.fromOsSpecific(str(path))):
        raise OSError(path)


def runs(flags, min_length):
    out, start = [], None
    for i, on in enumerate(list(flags) + [False]):
        if on and start is None:
            start = i
        elif not on and start is not None:
            if i - start >= min_length:
                out.append((start, i))
            start = None
    return out


def find_tiles(rgb, min_size=24, tolerance=36):
    """Grid of (y0, y1, x0, x1) boxes: background is the corner color; 1-pixel rules are ignored."""
    background = rgb[2, 2].astype(int)
    different = np.abs(rgb.astype(int) - background).sum(axis=2) > tolerance
    grid = []
    for y0, y1 in runs(different.any(axis=1), min_size):
        grid.append([(y0, y1, x0 + 0, x1) for x0, x1 in runs(different[y0:y1].any(axis=0), min_size)])
    return grid, background


def sprite_alpha(tile, ground, low=28, high=110):
    """Alpha from color distance to the tile's ground (its border median): leaves/petals stay, ground goes."""
    distance = np.abs(tile.astype(int) - ground).sum(axis=2)
    return np.clip((distance - low) / (high - low), 0, 1)


def crop(source, out=OUT):
    rgb = read_rgb(source)
    grid, background = find_tiles(rgb)
    if len(grid) != len(KINDS) or any(len(row) != len(COLORS) for row in grid):
        raise SystemExit(f'Expected {len(KINDS)} rows x {len(COLORS)} columns, found {[len(r) for r in grid]}')
    out = Path(out)
    written = []
    for kind, row in zip(KINDS, grid):
        for color, (y0, y1, x0, x1) in zip(COLORS, row):
            tile = rgb[y0:y1, x0:x1]
            border = np.concatenate([tile[0], tile[-1], tile[:, 0], tile[:, -1]])
            ground = np.median(border, axis=0).astype(int)
            alpha = sprite_alpha(tile, ground)
            opaque = np.dstack([tile, np.full(tile.shape[:2], 255, dtype=np.uint8)])
            sprite = np.dstack([tile, (alpha * 255).astype(np.uint8)])
            write_rgba(out / 'tiles' / f'{kind}-{color}.png', opaque)
            write_rgba(out / 'sprites' / f'{kind}-{color}.png', sprite)
            written.append((kind, color, tile.shape[1], tile.shape[0], tuple(int(v) for v in ground)))
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sheet', type=Path)
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    for kind, color, w, h, ground in crop(args.sheet, args.out):
        print(f'{kind}-{color}: {w}x{h}, ground {ground}')
    print('Wrote tiles and sprites under', args.out.resolve())


if __name__ == '__main__':
    main()
