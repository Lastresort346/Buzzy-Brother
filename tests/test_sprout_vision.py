# SPDX-License-Identifier: GPL-3.0-or-later
import unittest

import numpy as np

from flybrain.arena.sprouts import (BannerReader, SproutSense, ground_to_pixel, pixel_to_ground, token_cue)

GEOMETRY = (0., 34., 18., 68.)
SIZE = (400, 640)


def frame(tokens=(), banner=None, offset=(0, 0)):
    image = np.full((*SIZE, 3), (90, 150, 80), dtype=np.uint8)
    for dx, dy in tokens:
        u, v = ground_to_pixel(dx, dy, SIZE, GEOMETRY)
        image[int(v) - 2:int(v) + 3, int(u) - 2:int(u) + 3] = (255, 10, 240)
    if banner is not None:
        y, x = 10 + offset[0], 90 + offset[1]
        image[y:y + 56, x:x + 460] = (13, 15, 25)
        for row, col, h, w in banner:
            image[y + row:y + row + h, x + col:x + col + w] = (255, 255, 255)
    return image


# Crude "text": different layouts stand in for different notification wording.
BLUE = [(10, 30, 30, 8), (10, 60, 30, 8), (10, 90, 30, 40), (30, 150, 10, 200), (22, 380, 14, 6)]
EAST = [(10, 30, 30, 100), (28, 160, 12, 12), (10, 200, 30, 8), (10, 240, 30, 8), (35, 300, 10, 30)]


class SproutVisionTests(unittest.TestCase):
    def test_ground_projection_round_trips(self):
        for point in ((0, 4), (3, 7), (-5, 2), (6, 12)):
            u, v = ground_to_pixel(*point, SIZE, GEOMETRY)
            back = pixel_to_ground(u, v, SIZE, GEOMETRY)
            self.assertAlmostEqual(back[0], point[0], places=6)
            self.assertAlmostEqual(back[1], point[1], places=6)

    def test_token_cue_points_at_the_nearest_token_from_pixels_only(self):
        cue = token_cue(frame(tokens=[(4, 5), (-9, 14)]), GEOMETRY)
        expected = np.array([4, 5]) / np.hypot(4, 5)
        self.assertTrue(np.allclose(cue[:2], expected, atol=.08), cue)
        self.assertIsNone(token_cue(frame(), GEOMETRY))

    def test_banner_reader_tells_fields_apart_wherever_the_card_sits(self):
        reader = BannerReader()
        self.assertTrue(reader.learn('BLUE FIELD', frame(banner=BLUE)))
        self.assertTrue(reader.learn('EAST FIELD', frame(banner=EAST)))
        for offset in ((0, 0), (6, 30)):
            self.assertEqual(reader.read(frame(banner=BLUE, offset=offset))[0], 'BLUE FIELD')
            self.assertEqual(reader.read(frame(banner=EAST, offset=offset))[0], 'EAST FIELD')
        self.assertIsNone(reader.read(frame()))

    def test_alert_is_remembered_after_the_banner_disappears_then_expires(self):
        reader = BannerReader()
        reader.learn('EAST FIELD', frame(banner=EAST))
        reader.learn('BLUE FIELD', frame(banner=BLUE))
        sense = SproutSense(reader)
        sense.look(frame(banner=EAST), GEOMETRY, 10.)
        sense.look(frame(), GEOMETRY, 14.)
        self.assertEqual(sense.alert_field(14.), 'EAST FIELD')
        self.assertIsNone(sense.alert_field(45.))


if __name__ == '__main__':
    unittest.main()
