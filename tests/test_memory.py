# SPDX-License-Identifier: GPL-3.0-or-later
import math
import unittest

import numpy as np

from flybrain.arena import Action, ArenaWorld
from flybrain.arena.memory import FieldMemory


class FieldMemoryTests(unittest.TestCase):
    def test_path_integration_tracks_unobstructed_world_motion(self):
        w = ArenaWorld()
        w.x, w.y = 0., 8.  # clear of obstacles
        m = FieldMemory()
        m.reset(w.x, w.y)
        for i in range(120):
            a = Action(forward=1, strafe=1 if i < 60 else -1)
            x, y = w.x, w.y
            w.step(a)
            m.integrate(w.x - x, w.y - y)
        self.assertAlmostEqual(m.x, w.x, places=6)
        self.assertAlmostEqual(m.y, w.y, places=6)

    def test_no_drift_while_stuck_against_an_obstacle(self):
        w = ArenaWorld()
        w.x, w.y = -4., -8.  # south of the obstacle at (-4, -5)
        m = FieldMemory()
        m.reset(w.x, w.y)
        for _ in range(120):
            x, y = w.x, w.y
            w.step(Action(forward=1))
            m.integrate(w.x - x, w.y - y)
        self.assertAlmostEqual(m.y, w.y, places=6)

    def test_visits_merge_by_place_and_names_stick(self):
        m = FieldMemory()
        m.reset(-10, 1)
        m.visit('BLUE')
        m.reset(-9, 2)
        node = m.visit()
        self.assertEqual((len(m.nodes), node['n'], node['name']), (1, 2, 'BLUE'))
        m.reset(10, 5)
        m.visit('WHITE')
        self.assertEqual(len(m.nodes), 2)

    def test_cue_points_at_remembered_field_and_is_zero_if_unknown(self):
        m = FieldMemory()
        m.reset(0, 15)
        m.visit('ROSE')
        m.reset(0, 0)
        self.assertTrue(np.allclose(m.cue('ROSE')[:2], (0, 1)))
        self.assertFalse(m.cue('MISSING').any())
        m.reset(0, 14)
        self.assertEqual(m.cue('ROSE')[3], 1)

    def test_discover_cue_sweeps_unexplored_area_and_discoveries_can_be_named(self):
        m = FieldMemory()
        m.reset(-9, -15)
        first = m.cue(FieldMemory.DISCOVER)
        self.assertAlmostEqual(np.hypot(*first[:2]), 1)
        for _ in range(3000):  # follow the cue; should cover the arena and then report done
            c = m.cue(FieldMemory.DISCOVER)
            if c[3]:
                break
            m.integrate(c[0] * .1, c[1] * .1)
        self.assertEqual(len(m.seen), len(m.cells))
        self.assertEqual(m.cue(FieldMemory.DISCOVER)[3], 1)
        m.reset(18, -4)
        m.visit()  # collected somewhere unknown: an unnamed discovery
        self.assertEqual(len(m.discoveries()), 1)
        self.assertIsNone(m.recall('LAVA FIELD'))
        self.assertIsNotNone(m.name_here('LAVA FIELD'))
        self.assertEqual(len(m.discoveries()), 0)
        self.assertIsNotNone(m.recall('LAVA FIELD'))

    def test_default_work_field_is_the_nearest_named_one(self):
        m = FieldMemory()
        self.assertIsNone(m.nearest_field())
        for name, x, y in (('BLUE', -10, 1), ('WHITE', 10, 5)):
            m.reset(x, y)
            m.visit(name)
        m.visit()  # extra visits must not change which is nearest
        m.reset(-10, 8)
        self.assertEqual(m.nearest_field(), 'BLUE')
        m.reset(12, 9)
        self.assertEqual(m.nearest_field(), 'WHITE')

    def test_every_field_has_room_for_its_own_node(self):
        centers = [(p[0], p[1]) for p in ArenaWorld.patches]
        for i, a in enumerate(centers):
            for b in centers[i + 1:]:
                self.assertGreater(math.dist(a, b), 2 * FieldMemory.merge_radius)
        for x, y in centers:
            self.assertLess(max(abs(x), abs(y)) + 4.3, 24)


if __name__ == '__main__':
    unittest.main()
