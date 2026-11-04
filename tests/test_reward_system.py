# SPDX-License-Identifier: GPL-3.0-or-later
import unittest
import numpy as np

from flybrain.arena.reward import RewardSystem
from flybrain.arena.brain import NeuralController
from test_neural_controller import fixture


class RewardSystemTests(unittest.TestCase):
    def test_prediction_error_uses_next_state_and_terminal_mask(self):
        reward=RewardSystem(2)
        reward.value_weights[:]=[2,3]
        previous=np.array([1.,0.])
        following=np.array([0.,1.])
        delta=reward.evaluate(previous,following,100,learning=False)
        self.assertAlmostEqual(delta,1+.98*3-2)
        terminal=reward.evaluate(previous,following,100,terminal=True,learning=False)
        self.assertAlmostEqual(terminal,-1)

    def test_anticipated_reward_reduces_surprise_and_omission_is_negative(self):
        reward=RewardSystem(1)
        x=np.ones(1)
        first=reward.evaluate(x,x,100,terminal=True)
        for _ in range(100):
            last=reward.evaluate(x,x,100,terminal=True)
        self.assertLess(abs(last),abs(first)*.1)
        missed=reward.evaluate(x,x,0,terminal=True)
        self.assertLess(missed,0)
        self.assertGreater(reward.aversive,0)
        self.assertEqual(reward.appetitive,0)

    def test_task_event_components_are_separate_and_no_dip_means_no_dip_penalty(self):
        reward=RewardSystem(1)
        x=np.ones(1)
        reward.evaluate(x,x,0,empty_dips=2,blocked_fraction=.5,learning=False)
        for key,value in {'pollen':0.,'honey':0.,'empty_dips':-.05,'blocked':-.025}.items():
            self.assertEqual(reward.components[key],value)
        self.assertAlmostEqual(reward.task_reward,-.075)
        reward.evaluate(x,x,0,learning=False)
        self.assertEqual(reward.task_reward,0)

    def test_readout_credit_waits_for_the_next_observation(self):
        b=NeuralController(fixture())
        image=np.full((100,160,3),128,dtype=np.uint8)
        b.act(image)
        before=b.weights.copy()
        b.queue_feedback(100)
        np.testing.assert_array_equal(before,b.weights)
        with self.assertRaises(ValueError):
            b.queue_feedback(100)
        b.act(image)
        self.assertFalse(np.array_equal(before,b.weights))
        self.assertIsNone(b._pending_feedback)
        self.assertEqual(b.last_reward,100)

    def test_frozen_evaluation_does_not_train_actor_or_critic(self):
        b=NeuralController(fixture(),learning=False)
        b.act(np.full((100,160,3),128,dtype=np.uint8))
        before=b.weights.copy()
        b.feedback(100)
        np.testing.assert_array_equal(before,b.weights)
        self.assertEqual(b.reward_system.value_weights.sum(),0)
        self.assertGreater(b.reward_system.phasic,0)

    def test_reset_clears_dopamine_traces_but_retains_value_memory(self):
        reward=RewardSystem(1)
        reward.evaluate(np.ones(1),np.ones(1),100)
        learned=reward.value_weights.copy()
        reward.reset_transient()
        np.testing.assert_array_equal(learned,reward.value_weights)
        self.assertEqual(reward.tonic,0)
        self.assertEqual(reward.phasic,0)
        self.assertEqual(reward.value_trace.sum(),0)

    def test_invalid_events_cannot_update_critic(self):
        reward=RewardSystem(1)
        for values in [dict(pollen=float('nan')),dict(pollen=-1),dict(pollen=0,blocked_fraction=2)]:
            with self.assertRaises(ValueError):
                reward.evaluate(np.ones(1),np.ones(1),**values)
        self.assertEqual(reward.value_weights.sum(),0)


if __name__=='__main__':
    unittest.main()
