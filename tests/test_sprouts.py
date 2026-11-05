# SPDX-License-Identifier: GPL-3.0-or-later
import math
import unittest

from dataclasses import replace

from flybrain.arena import Action, ArenaConfig, ArenaWorld


def run(w, seconds, action=Action()):
    for _ in range(int(seconds * 60)):
        w.step(action)


def harvest(w, field_center, orders=True, tries=40):
    """Stand in the field and dip until it is harvested (or give up)."""
    w.sprout_orders = orders
    w.x, w.y = field_center
    for _ in range(tries):
        run(w, 1, Action(dip=True))
        if w.sprout and w.sprout['harvested']:
            return True
    return False


class SproutTests(unittest.TestCase):
    def test_no_tokens_until_harvested_even_if_the_bee_is_right_there(self):
        w = ArenaWorld()
        w.sprout_orders = True
        w.start_sprout('BLUE FIELD')
        w.x, w.y = -10, 1  # standing in the field, but not collecting
        run(w, 30)
        self.assertEqual((len(w.tokens), w.sprout['progress'], w.sprout['harvested']), (0, 0, False))

    def test_collecting_pollen_in_the_field_harvests_and_then_tokens_drop(self):
        w = ArenaWorld()
        w.start_sprout('EAST FIELD')
        self.assertTrue(harvest(w, (18, -4)))
        self.assertGreaterEqual(w.sprout['progress'], w.config.sprout_pollen)
        run(w, 7)
        self.assertEqual(len(w.tokens), 12)
        self.assertTrue(all(math.hypot(t['x'] - 18, t['y'] + 4) < 5.2 for t in w.tokens))
        run(w, 25)
        self.assertEqual((w.tokens, w.sprout), ([], None))

    def test_pollen_from_another_field_does_not_count(self):
        w = ArenaWorld()
        w.sprout_orders = True
        w.start_sprout('EAST FIELD')
        w.x, w.y = -10, 1  # collecting in the BLUE field
        run(w, 20, Action(dip=True))
        self.assertGreater(w.total_collected, 0)
        self.assertEqual(w.sprout['progress'], 0)

    def test_no_orders_means_the_sprout_is_ignored(self):
        w = ArenaWorld()
        w.start_sprout('EAST FIELD')
        self.assertFalse(harvest(w, (18, -4), orders=False, tries=15))
        self.assertEqual(w.sprout['progress'], 0)

    def test_unharvested_sprout_fades_after_the_timeout(self):
        w = ArenaWorld(replace(ArenaConfig(), sprout_timeout=5))
        w.start_sprout('BLUE FIELD')
        run(w, 4.9)
        self.assertIsNotNone(w.sprout)
        run(w, .3)
        self.assertIsNone(w.sprout)

    def test_tokens_are_only_collected_when_ordered(self):
        for ordered, expected in ((False, 0), (True, 20)):
            w = ArenaWorld()
            w.sprout_orders = ordered
            w.start_sprout('BLUE FIELD')
            w.sprout['harvested'], w.sprout['spawned'] = True, 99
            w.tokens = [{'x': w.x, 'y': w.y + .5, 'ttl': 9}]
            reward = w.step(Action())[0]
            self.assertEqual((w.honey, len(w.tokens)), (expected, 1 - bool(expected)))
            self.assertEqual(reward, 2 * expected)

    def test_orders_survive_reset_and_events_are_seeded(self):
        a, b = ArenaWorld(), ArenaWorld()
        a.sprout_orders = True
        a.reset()
        self.assertTrue(a.sprout_orders)
        for w in (a, b):
            w.start_sprout()
        self.assertEqual(a.telemetry()['sprout_field'], b.telemetry()['sprout_field'])

    def test_automatic_sprouts_follow_the_interval(self):
        w = ArenaWorld(replace(ArenaConfig(), sprout_seconds=5))
        run(w, 4.5)
        self.assertEqual(w.sprouts_started, 0)
        run(w, 1)
        self.assertEqual(w.sprouts_started, 1)


if __name__ == '__main__':
    unittest.main()
