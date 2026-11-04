# SPDX-License-Identifier: GPL-3.0-or-later
import math
import unittest

from flybrain.arena import Action, ArenaConfig, ArenaWorld


class ArenaWorldTests(unittest.TestCase):
    def test_diagonal_speed_matches_straight_speed(self):
        straight,diagonal=ArenaWorld(),ArenaWorld()
        x,y=straight.x,straight.y
        for _ in range(60):
            straight.step(Action(forward=1))
            diagonal.step(Action(forward=1,strafe=1))
        self.assertAlmostEqual(math.hypot(straight.x-x,straight.y-y),6)
        self.assertAlmostEqual(math.hypot(diagonal.x-x,diagonal.y-y),6)

    def test_fixed_camera_ignores_all_camera_actions(self):
        w=ArenaWorld()
        for _ in range(20):
            w.step(Action(camera_yaw=50,camera_pitch=50,zoom=-100))
        self.assertEqual((w.camera_yaw,w.camera_pitch,w.camera_distance),(0,34,18))

    def test_unlocked_camera_never_reaches_first_person(self):
        w=ArenaWorld(ArenaConfig(camera_locked=False))
        w.step(Action(zoom=-1000,camera_pitch=-1000,camera_yaw=1000))
        self.assertEqual(w.camera_distance,8)
        self.assertGreaterEqual(w.camera_pitch,15)
        w.step(Action(zoom=1000,camera_pitch=1000))
        self.assertEqual(w.camera_distance,20)
        self.assertLessEqual(w.camera_pitch,65)

    def test_jump_has_gravity_and_cannot_be_held_to_fly(self):
        w=ArenaWorld()
        heights=[]
        for _ in range(120):
            w.step(Action(jump=True))
            heights.append(w.z)
        self.assertGreater(max(heights),1)
        self.assertLess(max(heights),1.3)
        self.assertEqual(w.z,0)
        self.assertTrue(w.grounded)
        self.assertEqual(w.vz,0)

    def test_walls_and_obstacles_block_ground_movement(self):
        w=ArenaWorld()
        w.x=-4
        w.y=-8
        for _ in range(60):
            w.step(Action(forward=1))
        self.assertLessEqual(w.y,-6-w.config.radius+.001)
        w.reset()
        for _ in range(1000):
            w.step(Action(strafe=1000))
        self.assertAlmostEqual(w.x,w.config.half_size-w.config.radius)

    def test_lands_on_obstacle_then_falls_when_walking_off(self):
        w=ArenaWorld()
        w.x,w.y,w.z=-4,-5,2
        w.grounded=False
        for _ in range(60):
            w.step()
        self.assertAlmostEqual(w.z,1)
        for _ in range(90):
            w.step(Action(strafe=-1))
        self.assertAlmostEqual(w.z,0)

    def test_dipper_collects_only_in_range_and_obeys_cooldown(self):
        w=ArenaWorld()
        w.step(Action(dip=True))
        self.assertEqual(w.pollen,0)
        self.assertEqual(w.swings,1)
        w.reset()
        w.x,w.y=-10,1
        reward,_,_,info=w.step(Action(dip=True))
        self.assertGreater(reward,0)
        pollen=w.pollen
        for _ in range(47):
            w.step(Action(dip=True))
        self.assertEqual(w.swings,1)
        self.assertEqual(w.pollen,pollen)
        w.step(Action(dip=True))
        self.assertEqual(w.swings,2)

    def test_seed_and_action_sequence_are_reproducible(self):
        a,b=ArenaWorld(),ArenaWorld()
        for action in [Action(forward=1)]*30+[Action(jump=True,strafe=1)]*30:
            self.assertEqual(a.step(action),b.step(action))
        initial=[(f.x,f.y) for f in a.flowers]
        a.reset(seed=123)
        self.assertNotEqual(initial,[(f.x,f.y) for f in a.flowers])
        self.assertEqual(a.pollen,0)
        self.assertEqual(a.tick,0)

    def test_rewards_and_episode_truncation(self):
        w=ArenaWorld(ArenaConfig(episode_seconds=1))
        for _ in range(59):
            reward,terminated,truncated,_=w.step()
            self.assertFalse(terminated or truncated)
            self.assertEqual(reward,0)
        _,terminated,truncated,_=w.step()
        self.assertFalse(terminated)
        self.assertTrue(truncated)

    def test_invalid_inputs_fail_before_state_changes(self):
        with self.assertRaises(ValueError):
            Action(forward=float('nan'))
        with self.assertRaises(ValueError):
            ArenaConfig(camera_distance=0)
        with self.assertRaises(ValueError):
            ArenaConfig(dt=1)


if __name__=='__main__':
    unittest.main()
