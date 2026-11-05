# SPDX-License-Identifier: GPL-3.0-or-later
import unittest
from types import SimpleNamespace

from flybrain.arena import Action, ArenaWorld
from flybrain.arena.comms import COOLDOWN, PRESETS, CommunicationCenter
from flybrain.arena.memory import FieldMemory


def make(**kw):
    w = ArenaWorld()
    return SimpleNamespace(world=w, memory=FieldMemory(), goal=None, _brain_action=Action(), **kw)


def tick(app, seconds):
    app.world.tick += int(seconds / app.world.config.dt)


def ids(msgs):
    return [m['id'] for m in msgs]


def drain(a, c, n=8):
    """Run update() n times 1.1 s apart (under the cooldown); return {id: message}."""
    got = {}
    for _ in range(n):
        got.update({m['id']: m for m in c.update(a)})
        tick(a, 1.1)
    return got


def cond(a, c):
    return c.conditions(a, a.world.tick * a.world.config.dt)


class CommsTests(unittest.TestCase):
    def test_bag_and_honey_presets_only_under_their_condition(self):
        a, c = make(), CommunicationCenter()
        self.assertEqual(set(cond(a, c)), {'bag_empty', 'low_honey', 'saving_for'})
        a.world.pollen = 10
        self.assertEqual(set(cond(a, c)), {'low_honey', 'saving_for'})
        a.world.pollen = a.world.config.pollen_capacity
        a.world.honey = 500
        self.assertEqual(set(cond(a, c)), {'bag_full', 'want_item'})
        self.assertEqual(cond(a, c)['bag_full'][2]['pollen'], a.world.config.pollen_capacity)
        self.assertEqual(drain(a, c)['bag_full']['text'], PRESETS['bag_full'])

    def test_item_priority_is_utility_per_price_with_evidence(self):
        a, c = make(), CommunicationCenter()
        a.world.honey = 500
        m = cond(a, c)
        self.assertNotIn('saving_for', m)
        key, fields, ev = m['want_item']
        self.assertEqual(fields['item'], 'bee sticker')  # 2/25 beats 6/80 and 12/200
        self.assertAlmostEqual(ev['utility_per_price'], .08)
        self.assertEqual(drain(a, c)['want_item']['text'], 'I want a bee sticker.')

    def test_purchase_reported_once(self):
        a, c = make(), CommunicationCenter()
        a.world.honey = 500
        c.update(a)
        tick(a, 2)
        a.world.honey, _, _ = a.world.shop.buy(1, a.world.honey)
        self.assertEqual(drain(a, c)['bought']['text'], 'I bought a bee sticker.')
        self.assertNotIn('bought', drain(a, c))

    def test_dedupe_and_cooldown(self):
        a, c = make(), CommunicationCenter()
        a.world.pollen = a.world.config.pollen_capacity
        self.assertIn('bag_full', drain(a, c, 1))
        self.assertNotIn('bag_full', drain(a, c, 4))  # same state, no repeat
        a.world.pollen = 0
        drain(a, c, 1)
        a.world.pollen = a.world.config.pollen_capacity
        self.assertNotIn('bag_full', drain(a, c, 1))  # recurred inside the cooldown
        tick(a, COOLDOWN)
        self.assertIn('bag_full', drain(a, c))

    def test_global_rate_limit(self):
        a, c = make(), CommunicationCenter()
        self.assertEqual(len(c.update(a)), 1)  # three conditions hold, one per MIN_GAP
        self.assertEqual(len(c.update(a)), 0)
        tick(a, 1.1)
        self.assertEqual(len(c.update(a)), 1)

    def collect(self, a, c, color, pollen):
        f = next(f for f in a.world.flowers if f.color == color)
        a.world.x, a.world.y, a.world.heading = f.x, f.y - .8, 0.
        a.world.total_collected += pollen
        c.update(a)

    def test_color_preference_needs_evidence(self):
        a, c = make(), CommunicationCenter()
        self.collect(a, c, 0, 150)  # plenty of pollen but only one colour
        self.assertNotIn('color_pref', cond(a, c))
        self.collect(a, c, 1, 30)
        m = drain(a, c)['color_pref']
        self.assertEqual(m['text'], 'I prefer blue flowers.')
        self.assertEqual(m['evidence']['pollen_by_color'], {'blue': 150, 'white': 30})
        c2 = CommunicationCenter()
        a2 = make()
        self.collect(a2, c2, 0, 40)
        self.collect(a2, c2, 1, 40)  # enough pollen but a tie
        self.assertNotIn('color_pref', cond(a2, c2))

    def test_color_preference_not_below_minimum_pollen(self):
        a, c = make(), CommunicationCenter()
        self.collect(a, c, 0, 60)
        self.collect(a, c, 1, 10)
        self.assertNotIn('color_pref', cond(a, c))

    def test_goal_vs_memory(self):
        a, c = make(), CommunicationCenter()
        a.goal = 'ROSE FIELD'
        self.assertEqual(drain(a, c)['new_field']['text'], 'I need to find a new field.')
        a.memory.x, a.memory.y = 0., 15.
        a.memory.visit('ROSE FIELD')
        a.memory.x = 5.
        got = drain(a, c)
        self.assertEqual(got['go_field']['text'], 'I want to go to the Rose Field.')
        self.assertEqual(got['go_field']['evidence']['odometry_distance'], 5.)
        self.assertNotIn('new_field', got)

    def test_discovery_awaiting_name(self):
        a, c = make(), CommunicationCenter()
        a.memory.visit()
        self.assertEqual(drain(a, c)['name_field']['evidence'], {'unnamed_fields': 1})
        self.assertNotIn('name_field', cond(make(), CommunicationCenter()))

    def test_stuck_requires_commanded_motion_without_odometry_progress(self):
        a, c = make(), CommunicationCenter()
        a._brain_action = Action(forward=1)
        cond(a, c)
        tick(a, 2)
        self.assertNotIn('stuck', cond(a, c))
        tick(a, 2)
        self.assertGreaterEqual(cond(a, c)['stuck'][2]['seconds_without_progress'], 3)
        b, d = make(), CommunicationCenter()  # not commanding movement: never stuck
        cond(b, d)
        tick(b, 10)
        self.assertNotIn('stuck', cond(b, d))
        e, f = make(), CommunicationCenter()  # moving freely: never stuck
        e._brain_action = Action(forward=1)
        for _ in range(10):
            e.memory.x += 1
            tick(e, 1)
            self.assertNotIn('stuck', cond(e, f))

    def test_every_message_has_evidence_and_known_preset(self):
        a, c = make(), CommunicationCenter()
        for m in drain(a, c).values():
            self.assertIn(m['id'], PRESETS)
            self.assertTrue(m['evidence'])


if __name__ == '__main__':
    unittest.main()
