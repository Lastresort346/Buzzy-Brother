# SPDX-License-Identifier: GPL-3.0-or-later
import unittest

import numpy as np

from flybrain.places import HIVE, PATH, PlaceModel, Smoother, auto_labels, status, thumbnail


def scene(color, seed):
    rng = np.random.default_rng(seed)
    frame = np.clip(np.array(color) + rng.normal(0, 12, (90, 160, 3)), 0, 255)
    return frame.astype(np.uint8)


class PlaceTests(unittest.TestCase):
    def test_auto_labels_from_pollen_trace(self):
        fills = [0, 0, 0, .1, .2, .3, .3, .3, .3, .3, .3, .3, .2, .1, 0, 0, 0, 0, 0, 0, 0]
        dipping = [False] * 3 + [True] * 4 + [False] * 14
        labels = auto_labels(fills, dipping, 'dandelion', window=1)
        self.assertEqual(labels[4], 'dandelion')
        self.assertEqual(labels[13], HIVE)
        self.assertEqual(labels[9], PATH)
        self.assertEqual(labels[20], PATH)

    def test_prompt_on_screen_means_hive(self):
        labels = auto_labels([0.] * 4, [False] * 4, 'f', window=1, pads=[False, True, True, False])
        self.assertEqual(labels, [PATH, HIVE, HIVE, PATH])
        model = PlaceModel([HIVE, PATH, 'f'], np.zeros((10, 3)), np.zeros(9), np.ones(9))
        s = status(.4, np.array([0., 1., 0.]), model, pad=True)
        self.assertEqual((s[0], s[3], s[12]), (.4, 1., 0.))  # the pad overrides a "path" guess

    def test_hive_spreads_over_standing_still(self):
        fills = [0, .5, .5, .5, .5, .4, .5 - .2, .5 - .2, .5 - .2, .5 - .2]
        moving = [True, False, False, False, False, False, False, False, False, True]
        labels = auto_labels(fills, [False] * 10, 'f', window=0, moving=moving)
        self.assertEqual(labels[5], HIVE)
        self.assertEqual(labels[2:9], [HIVE] * 7)  # waiting before and after the drop
        self.assertEqual(labels[9], PATH)  # walking away

    def test_smoother_ignores_one_odd_frame(self):
        smooth = Smoother(keep=.75)
        for _ in range(8):
            steady = smooth.update(np.array([0., 0., 1.]))
        odd = smooth.update(np.array([0., 1., 0.]))
        self.assertEqual(int(np.argmax(odd)), 2)
        self.assertAlmostEqual(odd.sum(), 1)

    def test_unknown_fills_do_not_crash(self):
        # No change can be measured from an unknown reading, so the first known one is still "path".
        self.assertEqual(auto_labels([None, .1, .2], [True] * 3, 'x', window=1), [PATH, 'x', 'x'])

    def test_classifier_separates_distinct_scenes(self):
        colors = {HIVE: (220, 180, 40), PATH: (60, 160, 60), 'dandelion': (230, 230, 230)}
        thumbs, labels = [], []
        for i in range(30):
            for name, color in colors.items():
                thumbs.append(thumbnail(scene(color, i)))
                labels.append(name)
        model = PlaceModel.fit(thumbs, labels, steps=200)
        test = [scene(c, 100 + i) for i, c in enumerate(colors.values())]
        self.assertEqual(model.predict(test), list(colors))
        s = status(.5, model.probabilities(test[0]), model)
        self.assertEqual(s[0], .5)
        self.assertGreater(s[3], .5)  # hive channel
        self.assertAlmostEqual(s[3] + s[4] + s[12], 1)


if __name__ == '__main__':
    unittest.main()
