# SPDX-License-Identifier: GPL-3.0-or-later
import math
import unittest

import numpy as np

from flybrain.arena.brain import MOVES
from flybrain.nav import DT, SPEED, Compass, step_vector, steering_prior


def walk(compass, keys, decisions, **kw):
    for _ in range(decisions):
        compass.update(keys, False, kw.get('fill', 0.), kw.get('pad', False), kw.get('collected', 0.), kw.get('place'))


class NavTests(unittest.TestCase):
    def test_held_keys_move_the_estimate_and_diagonals_are_not_faster(self):
        self.assertEqual(step_vector({'w'}), (0., SPEED * DT))
        self.assertEqual(step_vector({'a'}), (-SPEED * DT, 0.))
        self.assertEqual(step_vector({'w', 's'}), (0., 0.))
        self.assertAlmostEqual(math.hypot(*step_vector({'w', 'd'})), SPEED * DT)
        c = Compass()
        walk(c, {'w'}, 4)
        self.assertEqual(c.position, (0., 4 * SPEED * DT))

    def test_blocked_decisions_add_no_drift_and_the_pad_reanchors(self):
        c = Compass()
        c.update({'w'}, True, 0., False)
        self.assertEqual(c.position, (0., 0.))
        walk(c, {'d'}, 5)
        self.assertNotEqual(c.position, (0., 0.))
        walk(c, {'d'}, 1, pad=True)
        self.assertEqual(c.position, (0., 0.))

    def test_cue_points_home_when_full_and_to_the_field_otherwise(self):
        c = Compass()
        walk(c, {'w'}, 20)  # 30 units north, dipping in dandelion
        walk(c, set(), 1, collected=5., place='dandelion')
        self.assertEqual([n['name'] for n in c.memory.nodes], ['dandelion'])
        walk(c, set(), 1, fill=1.)  # bag full: deliver
        cue = c.cue()
        self.assertTrue(c.delivering)
        self.assertEqual((round(cue[0], 3), round(cue[1], 3)), (0., -1.))  # home is south
        walk(c, {'s'}, 20, fill=.99)
        walk(c, set(), 1, fill=0.)  # emptied: back to collecting
        self.assertFalse(c.delivering)
        self.assertGreater(c.cue()[1], .9)  # field is north again

    def test_arrival_and_unknown_fields(self):
        c = Compass()
        self.assertTrue((c.cue() == 0).all())  # no field remembered yet
        walk(c, set(), 1, fill=1.)
        self.assertEqual(c.cue()[3], 1.)  # delivering, standing at home
        walk(c, set(), 1, collected=3., place='path')
        walk(c, set(), 1, collected=3., place='hive', pad=True)
        self.assertEqual(c.memory.nodes, [])  # only named fields are remembered

    def test_the_prior_turns_every_direction_into_the_matching_move(self):
        nav = steering_prior()
        for k in range(16):
            a = 2 * math.pi * (k + .2) / 16  # off the exact tie between two moves
            want = int(np.argmax(MOVES[:8] @ [math.cos(a), math.sin(a)] / np.linalg.norm(MOVES[:8], axis=1)))
            self.assertEqual(int(np.argmax(nav @ [math.cos(a), math.sin(a), .5, 0.])), want, k)
        self.assertEqual(int(np.argmax(nav @ [0., 0., 0., 1.])), 8)  # arrived: stand


if __name__ == '__main__':
    unittest.main()
