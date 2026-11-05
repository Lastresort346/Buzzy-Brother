# SPDX-License-Identifier: GPL-3.0-or-later
import unittest

from flybrain.arena import Action, ArenaWorld
from flybrain.arena.drives import ThinkingCenter, DIM
from flybrain.arena.memory import FieldMemory


def memory_with_fields():
    m = FieldMemory()
    for name, x, y, colors in (('BLUE FIELD', -10, 1, {'blue': 300}), ('WHITE FIELD', 10, 5, {'white': 300}),
                               ('ROSE FIELD', 0, 15, {'red': 250, 'white': 20})):
        m.reset(x, y)
        m.visit(name, colors)
    m.reset(0, 0)
    return m


class DriveTests(unittest.TestCase):
    def test_world_reports_the_color_of_the_pollen_just_collected(self):
        w = ArenaWorld()
        w.x, w.y = -10, 1  # blue field
        for _ in range(60):
            w.step(Action(dip=True))
            if w.last_color_gain:
                break
        self.assertEqual(list(w.last_color_gain), [0])  # color index 0 = blue
        self.assertEqual(sum(w.last_color_gain.values()), w.last_gain)

    def test_memory_keeps_color_totals_and_recalls_the_best_field(self):
        m = memory_with_fields()
        self.assertEqual(m.best_field_for('blue'), 'BLUE FIELD')
        self.assertEqual(m.best_field_for('red'), 'ROSE FIELD')
        self.assertEqual(m.best_field_for('white'), 'WHITE FIELD')  # 300 beats the 20 at the rose field
        m.reset(0, 15)
        m.visit(colors={'white': 500})  # merges into ROSE FIELD; now it has the most white
        self.assertEqual(m.best_field_for('white'), 'ROSE FIELD')

    def test_old_memory_files_without_color_stats_still_load(self):
        m = FieldMemory()
        m.nodes = [{'name': 'BLUE FIELD', 'x': -10, 'y': 1, 'n': 5}]
        self.assertIsNone(m.best_field_for('blue'))
        m.reset(-10, 1)
        m.visit('BLUE FIELD', {'blue': 10})
        self.assertEqual(m.best_field_for('blue'), 'BLUE FIELD')

    def test_wanted_field_color_and_default_goals(self):
        m = memory_with_fields()
        mind = ThinkingCenter()
        self.assertEqual(mind.goal(m, .2), 'BLUE FIELD')  # nothing wanted: nearest field
        mind.want(field='ROSE FIELD')
        self.assertEqual(mind.goal(m, .2), 'ROSE FIELD')
        mind.want(color='white')
        self.assertEqual(mind.goal(m, .2), 'WHITE FIELD')
        mind.want(color='red')
        self.assertEqual(mind.goal(m, .2), 'ROSE FIELD')

    def test_unknown_color_means_go_explore(self):
        mind = ThinkingCenter(want_color='blue')
        self.assertEqual(mind.goal(FieldMemory(), .1), FieldMemory.DISCOVER)

    def test_a_full_bag_converting_or_the_hive_overrides_every_want(self):
        m = memory_with_fields()
        mind = ThinkingCenter(want_field='ROSE FIELD')
        self.assertIsNone(mind.goal(m, 1.0))
        self.assertIsNone(mind.goal(m, .4, converting=True))
        self.assertIsNone(mind.goal(m, .4, in_hive=True))
        self.assertEqual(mind.mode(.4), 'collect')
        self.assertEqual(mind.mode(1.0), 'deliver')

    def test_needs_grow_and_shrink_with_the_bag(self):
        mind = ThinkingCenter()
        low, high = mind.needs(.1), mind.needs(.9)
        self.assertLess(low['deliver'], high['deliver'])
        self.assertGreater(low['collect'], high['collect'])

    def test_attention_boosts_the_wanted_color_only(self):
        mind = ThinkingCenter()
        self.assertEqual(mind.attention(), {})
        mind.want(color='blue')
        self.assertEqual(mind.attention(), {'white': DIM, 'red': DIM, 'blue': 1.})
        with self.assertRaises(ValueError):
            mind.want(color='green')


if __name__ == '__main__':
    unittest.main()
