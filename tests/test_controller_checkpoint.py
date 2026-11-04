# SPDX-License-Identifier: GPL-3.0-or-later
from pathlib import Path
import tempfile
import unittest
import numpy as np

from flybrain.arena.brain import NeuralController
from test_neural_controller import fixture


class ControllerCheckpointTests(unittest.TestCase):
    def test_actor_and_value_learning_reload_without_pickle(self):
        a=NeuralController(fixture())
        a.act(np.full((80,120,3),128,dtype=np.uint8),inventory=[.5,.2])
        a.feedback(20,honey=10)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            path=Path(folder)/'controller.npz'
            a.save(path)
            b=NeuralController(fixture())
            b.load(path)
            np.testing.assert_array_equal(a.weights,b.weights)
            np.testing.assert_array_equal(a.reward_system.value_weights,b.reward_system.value_weights)
            self.assertEqual(a.update_count,b.update_count)
            with self.assertRaises(ValueError):
                NeuralController(fixture(),seed=13).load(path)


if __name__=='__main__':
    unittest.main()
