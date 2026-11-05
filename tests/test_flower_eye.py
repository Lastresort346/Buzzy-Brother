# SPDX-License-Identifier: GPL-3.0-or-later
import unittest

import numpy as np

from flybrain.arena.flower_eye import COLORS, FlowerEye, color_masks, FLOWER_Z
from flybrain.arena.sprouts import ground_to_pixel

GEOMETRY = (0., 34., 18., 68.)
SIZE = (400, 640)


class Stub:
    """Stands in for a trained model: returns a fixed flower/hive mask."""
    def __init__(self, masks):
        self.masks = masks
        self.metadata = {'version': 'stub'}

    def segment(self, image):
        p = np.zeros((*self.masks.shape, 3), np.float32)
        p[..., 1] = (self.masks == 1) * .9
        p[..., 2] = (self.masks == 2) * .9
        return p, self.masks


def scene(placements):
    image = np.full((*SIZE, 3), (30, 90, 40), np.uint8)
    mask = np.zeros(SIZE, np.uint8)
    for (dx, dy), color in placements:
        u, v = ground_to_pixel(dx, dy, SIZE, GEOMETRY, z=FLOWER_Z)
        y, x = int(v), int(u)
        image[y - 4:y + 5, x - 4:x + 5] = {'white': (225, 225, 220), 'red': (200, 25, 25), 'blue': (40, 70, 210)}[color]
        mask[y - 4:y + 5, x - 4:x + 5] = 1
    return image, mask


class FlowerEyeTests(unittest.TestCase):
    def test_colors_are_split_from_pixels(self):
        image, mask = scene([((-4, 6), 'white'), ((0, 6), 'red'), ((4, 6), 'blue')])
        found = color_masks(image, mask == 1)
        self.assertEqual({c: int(found[c].sum()) > 50 for c in COLORS}, {'white': True, 'red': True, 'blue': True})
        self.assertFalse((found['red'] & found['blue']).any())
        self.assertEqual(int(sum(found[c].sum() for c in COLORS)), int((mask == 1).sum()))

    def test_segment_has_the_arena_recognizer_interface_and_hive_loses_to_flower(self):
        image, flowers = scene([((0, 6), 'red')])
        hive = np.zeros((200, 320), np.uint8)
        hive[:, :] = 2  # hive everywhere at half resolution
        eye = FlowerEye(Stub(flowers), Stub(hive))
        probability, semantic = eye.segment(image)
        self.assertEqual((probability.shape, semantic.shape), ((400, 640, 3), (400, 640)))
        self.assertTrue((semantic[flowers == 1] == 1).all())
        self.assertTrue((semantic[flowers == 0] == 2).all())
        self.assertIn('flower-eye/stub', eye.metadata['version'])

    def test_cleanup_hook_is_applied(self):
        image, flowers = scene([((0, 6), 'red')])
        eye = FlowerEye(Stub(flowers), post=lambda masks: np.zeros_like(masks))
        _, semantic = eye.segment(image)
        self.assertEqual(int(semantic.sum()), 0)

    def test_color_cue_points_at_the_nearest_flower_of_that_color(self):
        image, flowers = scene([((-4, 6), 'white'), ((5, 7), 'blue'), ((-3, 12), 'blue')])
        eye = FlowerEye(Stub(flowers))
        eye.segment(image)
        cue = eye.color_cue('blue', GEOMETRY)
        expected = np.array([5, 7]) / np.hypot(5, 7)
        self.assertTrue(np.allclose(cue[:2], expected, atol=.1), cue)
        self.assertEqual(cue[3], 0)
        self.assertIsNone(eye.color_cue('red', GEOMETRY))

    def test_cue_reports_arrival_when_the_flower_is_underfoot(self):
        image, flowers = scene([((0.3, 1.0), 'red')])
        eye = FlowerEye(Stub(flowers))
        eye.segment(image)
        cue = eye.color_cue('red', GEOMETRY)
        self.assertEqual((cue[3], cue[0], cue[1]), (1., 0., 0.))


if __name__ == '__main__':
    unittest.main()
