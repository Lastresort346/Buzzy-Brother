# SPDX-License-Identifier: GPL-3.0-or-later
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from flybrain.arena.perception import PerceptionModel,pixel_features,metrics


class PerceptionTests(unittest.TestCase):
    def test_context_features_distinguish_equal_center_pixels(self):
        a=np.zeros((40,64,3),dtype=np.uint8)
        b=a.copy()
        b[19:22,30:35]=255
        b[20,32]=0
        fa,shape,*_=pixel_features(a)
        fb,*_=pixel_features(b)
        i=20*shape[1]+32
        self.assertFalse(np.array_equal(fa[i],fb[i]))
        self.assertEqual(fa.shape[1],35)

    def test_supervised_fit_and_safe_round_trip(self):
        rng=np.random.default_rng(7)
        x=np.concatenate([rng.normal(c*2,.1,(100,35)) for c in range(3)]).astype(np.float32)
        y=np.repeat(np.arange(3),100)
        model=PerceptionModel()
        model.fit(x,y,steps=120)
        p=model.probabilities(x)
        self.assertGreater(np.mean(np.argmax(p,axis=1)==y),.98)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            path=Path(folder)/'model.npz'
            model.save(path)
            restored=PerceptionModel.load(path)
            np.testing.assert_allclose(p,restored.probabilities(x))

    def test_metrics_do_not_hide_false_positives_in_background_accuracy(self):
        truth=np.array([1,1,0,0,2,2])
        prediction=np.array([1,0,1,0,2,2])
        result=metrics(truth,prediction)
        self.assertEqual(result['flower']['precision'],.5)
        self.assertEqual(result['flower']['recall'],.5)
        self.assertAlmostEqual(result['flower']['iou'],1/3)
        self.assertEqual(result['hive']['iou'],1)


if __name__=='__main__':
    unittest.main()
