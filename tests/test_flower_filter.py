# SPDX-License-Identifier: GPL-3.0-or-later
import unittest

import numpy as np

from flybrain.arena.flower_filter import clean, components, opening


def blobs(shape, centres, size=6):
    mask = np.zeros(shape, np.uint8)
    for y, x in centres:
        mask[y:y + size, x:x + size] = 1
    return mask


class FlowerFilterTests(unittest.TestCase):
    def test_opening_drops_thin_lines_and_keeps_blocks(self):
        mask = np.zeros((40, 40), bool)
        mask[5, 2:38] = True  # 1-pixel outline
        mask[20:26, 10:16] = True  # 6x6 blob
        out = opening(mask)
        self.assertFalse(out[5].any())
        self.assertTrue(out[20:26, 10:16].all())

    def test_components_report_area_bbox_and_centroid(self):
        mask = np.zeros((30, 30), bool)
        mask[2:6, 3:9] = True
        mask[10:14, 10:14] = True
        mask[10:14, 14:18] = True  # touches the previous block: one component
        parts = sorted(components(mask), key=lambda p: p['bbox'][0])
        self.assertEqual([p['area'] for p in parts], [24, 32])
        self.assertEqual(parts[0]['bbox'], (2, 6, 3, 9))
        self.assertAlmostEqual(parts[1]['centroid'][1], 13.5)

    def test_diagonal_neighbours_are_not_merged(self):
        mask = np.zeros((10, 10), bool)
        mask[2, 2] = mask[3, 3] = True
        self.assertEqual(len(components(mask)), 2)

    def test_lattice_survives_but_lone_blob_outline_and_solid_object_do_not(self):
        shape = (200, 320)
        lattice = [(20 + 14 * r, 20 + 14 * c) for r in range(5) for c in range(8)]
        mask = blobs(shape, lattice)
        mask[150:152, 100:300] = 1  # thin outline
        mask[170:174, 20:26] = 1  # lone blob far from everything
        mask[100:180, 200:300] = 1  # big solid object
        out = clean(mask)
        self.assertEqual(int(out[20:100, 20:140].sum()), int(mask[20:100, 20:140].sum()))  # lattice intact
        self.assertEqual(int(out[150:152, 100:300].sum()), 0)
        self.assertEqual(int(out[170:174, 20:26].sum()), 0)
        self.assertEqual(int(out[100:180, 200:300].sum()), 0)

    def test_round_solid_object_is_dropped_even_though_a_disc_fills_only_78_percent_of_its_box(self):
        ys, xs = np.mgrid[:200, :320]
        mask = (np.hypot(ys - 100, xs - 160) < 45).astype(np.uint8)
        self.assertEqual(int(clean(mask).sum()), 0)

    def test_merged_dense_field_with_gaps_is_kept(self):
        mask = np.zeros((200, 320), np.uint8)
        mask[20:110, 20:260] = 1
        for r in range(22, 108, 6):
            for c in range(22, 258, 6):
                mask[r:r + 2, c:c + 2] = 0  # a net with holes: flowers merged into one connected mass
        out = clean(mask, thin_radius=0)
        self.assertGreater(int(out.sum()), .9 * int(mask.sum()))

    def test_other_classes_are_untouched(self):
        mask = np.zeros((50, 50), np.uint8)
        mask[5, 5:45] = 1
        mask[20:30, 20:30] = 2
        out = clean(mask)
        self.assertTrue((out[20:30, 20:30] == 2).all())


if __name__ == '__main__':
    unittest.main()
