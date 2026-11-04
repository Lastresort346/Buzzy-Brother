# SPDX-License-Identifier: GPL-3.0-or-later
import unittest
from pathlib import Path
import tempfile
import json

import numpy as np

from flybrain.arena.brain import ConnectomeGraph,NeuralController
from flybrain.arena.telemetry import BehaviorTelemetry
from flybrain.arena import Action,ArenaWorld


def fixture():
    pre=np.arange(16,dtype=np.int32)
    post=(16+np.arange(16)%8).astype(np.int32)
    return ConnectomeGraph([str(720575940000000000+i) for i in range(40)],
        ['optic']*16+['descending']*8+['central']*16,
        pre,post,np.full(16,.5),np.arange(16),np.arange(16,24),'synthetic-test-only',16)


class NeuralControllerTests(unittest.TestCase):
    def test_public_shop_state_changes_sensory_firing_without_image_changes(self):
        graph=fixture()
        graph.groups[-13:]=['sensory_ascending']*13
        a,b=NeuralController(graph),NeuralController(graph)
        image=np.zeros((100,160,3),dtype=np.uint8)
        a.act(image,inventory=np.zeros(13))
        b.act(image,inventory=np.ones(13))
        self.assertGreater(b.rates[b.proprio].sum(),a.rates[a.proprio].sum())
        self.assertFalse(np.array_equal(a.last_features,b.last_features))

    def test_frozen_greedy_deployment_does_not_update_paid_reward_weights(self):
        brain=NeuralController(fixture(),learning=False,sample_actions=False)
        brain.weights.fill(0)
        brain.weights[8,-1]=10  # idle
        brain.weights[14,-1]=10  # buy snack
        action=brain.act(np.zeros((100,160,3),dtype=np.uint8))
        self.assertEqual((action.strafe,action.forward,action.buy),(0,0,2))
        actor=brain.weights.copy()
        critic=brain.reward_system.value_weights.copy()
        brain.feedback(0,purchases=6)
        np.testing.assert_array_equal(actor,brain.weights)
        np.testing.assert_array_equal(critic,brain.reward_system.value_weights)

    def test_camera_pixels_cause_different_spikes(self):
        a,b=NeuralController(fixture()),NeuralController(fixture())
        a.act(np.zeros((100,160,3),dtype=np.uint8))
        b.act(np.full((100,160,3),255,dtype=np.uint8))
        self.assertGreater(b.last_counts.sum(),a.last_counts.sum())
        self.assertFalse(np.array_equal(a.rates,b.rates))

    def test_real_recurrent_edges_influence_motor_activity(self):
        connected=fixture()
        cut=fixture()
        cut.weights.fill(0)
        a,b=NeuralController(connected),NeuralController(cut)
        bright=np.full((100,160,3),255,dtype=np.uint8)
        a.act(bright)
        b.act(bright)
        self.assertGreater(a.rates[connected.motor].sum(),b.rates[cut.motor].sum())

    def test_seeded_dynamics_and_actions_repeat(self):
        a,b=NeuralController(fixture(),seed=7),NeuralController(fixture(),seed=7)
        image=np.full((100,160,3),128,dtype=np.uint8)
        for _ in range(4):
            self.assertEqual(a.act(image),b.act(image))
            np.testing.assert_array_equal(a.last_counts,b.last_counts)
            self.assertTrue(np.isfinite(a.v).all())
            self.assertTrue(np.isfinite(a.rates).all())

    def test_reward_changes_readout_but_not_connectome_weights(self):
        b=NeuralController(fixture())
        b.act(np.full((100,160,3),128,dtype=np.uint8))
        before=b.weights.copy()
        anatomy=b.graph.weights.copy()
        b.feedback(0)
        np.testing.assert_array_equal(before,b.weights)
        b.feedback(100)
        self.assertFalse(np.array_equal(before,b.weights))
        np.testing.assert_array_equal(anatomy,b.graph.weights)
        self.assertEqual(b.update_count,1)

    def test_reset_clears_transient_credit_but_preserves_learned_readout(self):
        b=NeuralController(fixture())
        b.act(np.full((100,160,3),128,dtype=np.uint8))
        b.feedback(100)
        learned=b.weights.copy()
        b.reset_dynamics()
        self.assertEqual(b.rates.sum(),0)
        self.assertEqual(b.eligibility.sum(),0)
        self.assertIsNone(b.previous_image)
        np.testing.assert_array_equal(learned,b.weights)

    def test_saved_checkpoint_has_source_and_exact_ids_without_pickle(self):
        b=NeuralController(fixture())
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            path=Path(folder)/'readout.npz'
            b.save(path)
            with np.load(path,allow_pickle=False) as stored:
                info=json.loads(str(stored['metadata']))
                self.assertEqual(info['root_ids'],b.graph.root_ids)
                self.assertEqual(info['source_identity'],'synthetic-test-only')
                np.testing.assert_array_equal(stored['weights'],b.weights)


class BehaviorTelemetryTests(unittest.TestCase):
    def advance(self,t,w,action,steps):
        for _ in range(steps):
            before=(w.x,w.y,w.z)
            reward,*_=w.step(action)
            t.observe(before,w,action,reward)

    def test_wall_push_counts_once_until_rearmed(self):
        w=ArenaWorld()
        t=BehaviorTelemetry()
        w.x=w.config.half_size-w.config.radius
        self.advance(t,w,Action(strafe=1),120)
        self.assertEqual(t.escape_attempts,1)
        self.advance(t,w,Action(),1)
        self.advance(t,w,Action(strafe=1),60)
        self.assertEqual(t.escape_attempts,2)
        t.reset_episode()
        self.assertEqual(t.escape_attempts,2)
        self.assertEqual(t.episode_escapes,0)

    def test_interior_blockage_is_not_an_escape(self):
        w=ArenaWorld()
        t=BehaviorTelemetry()
        w.x,w.y=-4,-8
        self.advance(t,w,Action(forward=1),180)
        self.assertEqual(t.escape_attempts,0)
        self.assertGreater(t.blocked_fraction,.5)

    def test_happiness_is_bounded_and_reacts_to_actual_collection(self):
        w=ArenaWorld()
        t=BehaviorTelemetry()
        self.assertEqual(t.happiness,50)
        w.x,w.y=-10,1
        self.advance(t,w,Action(dip=True),1)
        self.assertGreater(t.happiness,50)
        self.assertEqual(t.state,'COLLECTING')
        w.x=w.config.half_size-w.config.radius
        self.advance(t,w,Action(strafe=1),900)
        self.assertLess(t.happiness,50)
        self.assertGreaterEqual(t.happiness,0)
        self.assertLessEqual(t.happiness,100)


if __name__=='__main__':
    unittest.main()
