# SPDX-License-Identifier: GPL-3.0-or-later
import tempfile
import unittest
from pathlib import Path

import numpy as np

from flybrain.arena.flower_scenes import make_scene, save_png, KIND_NAMES
from flybrain.arena.flower_sheet import KINDS, COLORS, crop, find_tiles, read_rgb
from flybrain.arena.perception import PerceptionModel, pixel_features
from flybrain.arena.train_flowers import sample


def fake_sprites():
    """15 synthetic RGBA sprites (colored disc with a yellow center) standing in for the real artwork."""
    ys, xs = np.mgrid[:50, :50] - 25
    d = np.hypot(xs, ys)
    sprites = {}
    for kind in KINDS:
        for i, color in enumerate(COLORS):
            rgba = np.zeros((50, 50, 4), np.float32)
            rgba[..., :3] = [(.95, .95, .95), (.85, .1, .1), (.15, .25, .85)][i]
            rgba[d < 6, :3] = (.98, .8, .1)
            rgba[..., 3] = (d < 20).astype(np.float32)
            sprites[f'{kind}-{color}'] = rgba
    return sprites


def fake_sheet():
    sheet = np.zeros((504, 234, 3), np.uint8)
    sheet[:] = (19, 30, 30)
    for r, y in enumerate((28, 153, 278, 376, 452)):
        for c, x in enumerate((13, 95, 177)):
            sheet[y:y + 50, x:x + 50] = (43, 143, 61)
            sheet[y + 15:y + 35, x + 15:x + 35] = [(240, 240, 240), (220, 20, 20), (30, 60, 220)][c]
    sheet[118, :] = (60, 70, 70)  # thin separator rules must be ignored
    return sheet


class FlowerDataTests(unittest.TestCase):
    def test_sheet_is_cut_into_five_rows_of_three_named_tiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'sheet.png'
            save_png(path, fake_sheet())
            grid, _ = find_tiles(read_rgb(path))
            self.assertEqual([len(row) for row in grid], [3] * 5)
            written = crop(path, Path(tmp) / 'out')
            self.assertEqual(len(written), 15)
            self.assertTrue((Path(tmp) / 'out/sprites/star-blue.png').exists())
            self.assertTrue((Path(tmp) / 'out/tiles/single-white.png').exists())

    def test_scene_labels_are_consistent_and_deterministic(self):
        sprites = fake_sprites()
        a = make_scene(np.random.default_rng(5), sprites, size=(100, 160))
        b = make_scene(np.random.default_rng(5), sprites, size=(100, 160))
        for key in ('image', 'labels', 'hard', 'kinds'):
            np.testing.assert_array_equal(a[key], b[key])
        self.assertEqual(a['fields'], b['fields'])
        self.assertEqual(a['image'].shape, (100, 160, 3))
        self.assertFalse((a['labels'].astype(bool) & a['hard']).any())  # a pixel is flower or negative, never both
        self.assertTrue(set(np.unique(a['kinds'])) <= {0, *KIND_NAMES})
        self.assertTrue((a['kinds'][a['hard']] > 0).all())

    def test_fields_are_straight_rows_and_columns_of_identical_upright_flowers(self):
        from flybrain.arena.flower_scenes import build_field
        sprites = {k: v.copy() for k, v in fake_sprites().items()}
        checked = 0
        for seed in range(8):
            _, labels, _, _, fields = build_field(np.random.default_rng(seed), sprites, negatives=False)
            for f in fields:
                xs = sorted({x for x, _ in f['cells']})
                ys = sorted({y for _, y in f['cells']})
                self.assertTrue(all((x - xs[0]) % f['pitch'] == 0 for x in xs))  # constant column pitch
                self.assertTrue(all((y - ys[0]) % f['pitch'] == 0 for y in ys))  # constant row pitch
                self.assertGreaterEqual(len(f['cells']), f['cols'] * f['rows'] * .8)
                if f['uniform_kind']:  # same sprite in every cell: label patches must match exactly
                    half = int(25 * f['scale'])
                    patches = [labels[y - half:y + half, x - half:x + half] for x, y in f['cells']]
                    self.assertTrue(all(np.array_equal(p, patches[0]) for p in patches))
                    checked += 1
        self.assertGreater(checked, 0)

    def test_fields_do_not_overlap(self):
        from flybrain.arena.flower_scenes import build_field
        for seed in range(10):
            fields = build_field(np.random.default_rng(seed), fake_sprites(), negatives=False)[4]
            for i, a in enumerate(fields):
                for b in fields[i + 1:]:
                    sep_x = a['x0'] + a['cols'] * a['pitch'] <= b['x0'] or b['x0'] + b['cols'] * b['pitch'] <= a['x0']
                    sep_y = a['y0'] + a['rows'] * a['pitch'] <= b['y0'] or b['y0'] + b['rows'] * b['pitch'] <= a['y0']
                    self.assertTrue(sep_x or sep_y)

    def test_clean_scenes_have_flowers_and_no_negatives_when_disabled(self):
        sprites = fake_sprites()
        seen = 0
        for seed in range(6):
            s = make_scene(np.random.default_rng(seed), sprites, negatives=False, size=(100, 160))
            self.assertFalse(s['hard'].any())
            seen += int((s['labels'] == 1).any())
        self.assertGreater(seen, 0)

    def test_two_class_recognizer_learns_a_flower_versus_lookalike_split(self):
        sprites = fake_sprites()
        rng = np.random.default_rng(1)
        data = [sample(make_scene(np.random.default_rng(100 + i), sprites, size=(400, 640)), rng) for i in range(4)]
        x, y, h = (np.concatenate(part) for part in zip(*data))
        self.assertTrue(h.any() and (y == 1).any())
        model = PerceptionModel(classes=2, hidden=16)
        model.fit(x, y, steps=300, hard=h)
        probability = model.probabilities(x)
        self.assertEqual(probability.shape[1], 2)
        self.assertGreater(float((probability.argmax(axis=1) == y).mean()), .85)


if __name__ == '__main__':
    unittest.main()
