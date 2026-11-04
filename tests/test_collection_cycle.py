# SPDX-License-Identifier: GPL-3.0-or-later
import unittest

from flybrain.arena import ArenaConfig,ArenaWorld,Action
from flybrain.arena.reward import RewardSystem
import numpy as np


class CollectionCycleTests(unittest.TestCase):
    def test_partial_flower_collection_preserves_pollen_at_capacity(self):
        w=ArenaWorld(ArenaConfig(pollen_capacity=7,spawn_x=-10,spawn_y=1))
        before=sum(f.pollen for f in w.flowers)
        reward,_,_,info=w.step(Action(dip=True))
        self.assertEqual(w.pollen,7)
        self.assertEqual(w.total_collected,7)
        self.assertEqual(reward,7)
        self.assertTrue(info['backpack_full'])
        self.assertAlmostEqual(before-sum(f.pollen for f in w.flowers),.7)
        for _ in range(120):
            w.step(Action(dip=True))
        self.assertEqual(w.pollen,7)
        self.assertEqual(w.total_collected,7)

    def test_conversion_requires_hive_and_action(self):
        w=ArenaWorld()
        w.pollen=120
        for _ in range(60):
            w.step(Action(convert=True))
        self.assertEqual((w.pollen,w.honey),(120,0))
        w.x,w.y=w.config.hive_x,w.config.hive_y
        for _ in range(30):
            w.step()
        self.assertEqual(w.honey,0)
        for _ in range(60):
            w.step(Action(convert=True))
        self.assertEqual((w.pollen,w.honey),(60,60))

    def test_conversion_conserves_inventory_and_never_mints_empty_honey(self):
        w=ArenaWorld(ArenaConfig(spawn_x=-9,spawn_y=-17))
        w.pollen=37
        rewards=0
        for _ in range(180):
            reward,*_=w.step(Action(convert=True))
            rewards+=reward
        self.assertEqual((w.pollen,w.honey,w.total_converted),(0,37,37))
        self.assertEqual(rewards,74)
        w.reset()
        self.assertEqual((w.pollen,w.honey,w.total_converted),(0,0,0))

    def test_fractional_bee_batches_continue_after_key_release(self):
        w=ArenaWorld(ArenaConfig(conversion_rate=7.5,spawn_x=-9,spawn_y=-17))
        w.pollen=100
        w.step(Action(convert=True))
        for _ in range(14):
            w.step()
        self.assertEqual(w.honey,1)
        for _ in range(15):
            w.step()
        self.assertEqual(w.honey,3)
        self.assertTrue(w.conversion_active)
        # A new E press stops it; mere release does not.
        w.step(Action(convert=True))
        for _ in range(120):
            w.step()
        self.assertEqual(w.honey,3)
        self.assertFalse(w.conversion_active)

    def test_airborne_conversion_is_unavailable(self):
        w=ArenaWorld(ArenaConfig(spawn_x=-9,spawn_y=-17))
        w.pollen=50
        w.step(Action(jump=True,convert=True))
        self.assertEqual(w.honey,0)

    def test_honey_has_a_distinct_positive_learning_signal(self):
        reward=RewardSystem(1)
        delta=reward.evaluate(np.ones(1),np.ones(1),0,honey=20,learning=False)
        self.assertAlmostEqual(delta,.4)
        self.assertEqual(reward.components['honey'],.4)
        self.assertEqual(reward.components['pollen'],0)

    def test_complete_collect_walk_convert_cycle_is_feasible_without_relocation(self):
        w=ArenaWorld(ArenaConfig(pollen_capacity=7,spawn_x=-10,spawn_y=1))
        w.step(Action(dip=True))
        for _ in range(180):
            w.step(Action(forward=-1))
        self.assertTrue(w.in_conversion_zone())
        for _ in range(60):
            w.step(Action(convert=True))
        self.assertEqual((w.total_collected,w.pollen,w.honey),(7,0,7))

    def test_one_press_runs_bee_steps_until_empty(self):
        w=ArenaWorld(ArenaConfig(spawn_x=-9,spawn_y=-17))
        w.pollen=37
        w.step(Action(convert=True))
        self.assertEqual(w.honey,0)
        self.assertTrue(w.conversion_active)
        for _ in range(14):
            w.step()
        self.assertEqual((w.pollen,w.honey,w.last_conversion_bee),(22,15,1))
        for _ in range(15):
            w.step()
        self.assertEqual((w.pollen,w.honey,w.last_conversion_bee),(7,30,2))
        for _ in range(15):
            w.step()
        self.assertEqual((w.pollen,w.honey,w.last_conversion_bee),(0,37,3))
        self.assertEqual(w.last_conversion_batch,7)
        self.assertFalse(w.conversion_active)
        self.assertEqual(w.conversion_status,'backpack_empty')

    def test_held_e_does_not_toggle_conversion_repeatedly(self):
        w=ArenaWorld(ArenaConfig(spawn_x=-9,spawn_y=-17))
        w.pollen=120
        for _ in range(60):
            w.step(Action(convert=True))
        self.assertEqual((w.pollen,w.honey),(60,60))
        self.assertTrue(w.conversion_active)
        self.assertEqual(w.conversion_steps,4)

    def test_leaving_hive_cancels_without_remote_or_backdated_conversion(self):
        w=ArenaWorld(ArenaConfig(spawn_x=-9,spawn_y=-17))
        w.pollen=100
        w.step(Action(convert=True))
        w.x=0
        for _ in range(60):
            w.step()
        self.assertEqual(w.honey,0)
        self.assertEqual(w.conversion_status,'left_hive')
        w.x=w.config.hive_x
        w.step()
        self.assertFalse(w.conversion_active)
        w.step(Action(convert=True))
        self.assertEqual(w.honey,0)

    def test_reset_clears_toggle_and_progress(self):
        w=ArenaWorld(ArenaConfig(spawn_x=-9,spawn_y=-17))
        w.pollen=100
        for _ in range(30):
            w.step(Action(convert=True))
        w.reset()
        self.assertFalse(w.conversion_active)
        self.assertFalse(w.previous_convert)
        self.assertEqual(w.conversion_steps,0)
        self.assertEqual(w.last_conversion_bee,0)


if __name__=='__main__':
    unittest.main()
