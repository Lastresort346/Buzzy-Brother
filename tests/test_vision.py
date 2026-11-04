# SPDX-License-Identifier: GPL-3.0-or-later
import unittest
import numpy as np

from flybrain.arena.vision import ThirdPersonVision


def adapter():
    return ThirdPersonVision([str(i) for i in range(512)],np.arange(512),['left']*256+['right']*256)


def stripe(column,row=None):
    image=np.zeros((80,120,3),dtype=np.uint8)
    if row is None:
        image[:,column*10:(column+1)*10]=255
    else:
        image[row*10:(row+1)*10,column*10:(column+1)*10]=255
    return image


class VisionTests(unittest.TestCase):
    def test_spatial_mapping_has_overlap_and_keeps_single_camera(self):
        v=adapter()
        self.assertEqual(len(v.mapping),512)
        self.assertEqual(len({m[0] for m in v.mapping}),512)
        for slot,row,column,profile in v.mapping:
            self.assertIn(column,range(16) if slot<256 else range(8,24))
            self.assertIn(row,range(16))
            self.assertIn(profile,range(4))
        self.assertEqual(v.snapshot()['camera_count'],1)
        self.assertFalse(v.snapshot()['stereo_depth'])

    def test_local_stimulus_stays_on_corresponding_side(self):
        a,b=adapter(),adapter()
        dark=np.zeros((80,120,3),dtype=np.uint8)
        dark_drive=a.encode(dark)
        left_drive=b.encode(stripe(0))
        self.assertGreater(np.linalg.norm(dark_drive[:256]-left_drive[:256]),.1)
        np.testing.assert_array_equal(dark_drive[256:],left_drive[256:])

    def test_camera_input_does_not_depend_on_rng_or_neuron_id_magnitude(self):
        a,b=adapter(),adapter()
        np.testing.assert_array_equal(a.encode(stripe(2)),b.encode(stripe(2)))
        self.assertEqual(a.mapping_record(),b.mapping_record())

    def test_masked_avatar_changes_do_not_create_sensory_motion(self):
        v=adapter()
        dark=np.zeros((80,120,3),dtype=np.uint8)
        before=v.encode(dark).copy()
        changed=dark.copy()
        changed[v.last_mask]=255
        after=v.encode(changed)
        np.testing.assert_allclose(before,after)
        self.assertEqual(sum(v.snapshot()['motion_energy'].values()),0)

    def test_stationary_image_has_no_temporal_or_directional_activity(self):
        v=adapter()
        image=stripe(2)
        v.encode(image)
        v.encode(image)
        self.assertEqual(v.features[:,:,4:10].sum(),0)

    def test_spatial_edges_detect_boundaries_without_mask_artifacts(self):
        v=adapter()
        uniform=np.full((80,120,3),128,dtype=np.uint8)
        v.encode(uniform)
        self.assertAlmostEqual(v.features[:,:,3].sum(),0)
        boundary=np.zeros_like(uniform)
        boundary[:,:60]=255
        v.reset()
        v.encode(boundary)
        self.assertGreater(v.features[:,:,3].sum(),0)

    def test_rightward_and_leftward_motion_have_distinct_channels(self):
        right,left=adapter(),adapter()
        right.encode(stripe(1))
        right.encode(stripe(2))
        left.encode(stripe(2))
        left.encode(stripe(1))
        self.assertGreater(right.features[:,:,6].sum(),right.features[:,:,7].sum())
        self.assertGreater(left.features[:,:,7].sum(),left.features[:,:,6].sum())

    def test_downward_motion_and_brightness_change_are_distinct(self):
        v=adapter()
        v.encode(stripe(2,1))
        v.encode(stripe(2,2))
        self.assertGreater(v.features[:,:,8].sum(),v.features[:,:,9].sum())
        v.reset()
        v.encode(np.zeros((80,120,3),dtype=np.uint8))
        v.encode(np.full((80,120,3),255,dtype=np.uint8))
        self.assertGreater(v.features[:,:,4].sum(),0)
        self.assertEqual(v.features[:,:,6:10].sum(),0)

    def test_zoom_or_rotation_change_resets_motion_history(self):
        v=adapter()
        v.encode(stripe(1),camera_geometry=(0,34,18,68))
        v.encode(stripe(2),camera_geometry=(10,34,18,68))
        self.assertTrue(v.motion_reset)
        self.assertEqual(v.features[:,:,4:10].sum(),0)

    def test_reset_removes_ghost_motion_and_previews_are_rgb(self):
        v=adapter()
        v.encode(stripe(1))
        previews=v.previews()
        self.assertEqual(len(previews),3)
        for image in previews:
            self.assertEqual(image.shape,(80,120,3))
            self.assertEqual(image.dtype,np.uint8)
        v.reset()
        self.assertIsNone(v.previous_luminance)
        self.assertIsNone(v.previews())
        v.encode(stripe(3))
        self.assertEqual(v.features[:,:,4:10].sum(),0)

    def test_avatar_mask_scales_with_known_third_person_zoom(self):
        v=adapter()
        image=np.zeros((80,120,3),dtype=np.uint8)
        v.encode(image,camera_geometry=(0,34,18,68))
        far=v.snapshot()['avatar_mask_fraction']
        v.encode(image,camera_geometry=(0,34,8,68))
        self.assertGreater(v.snapshot()['avatar_mask_fraction'],far)
        self.assertTrue(v.motion_reset)

    def test_invalid_pixels_are_rejected(self):
        v=adapter()
        for image in (np.zeros((5,5,3)),np.zeros((80,120)),np.full((80,120,3),np.nan)):
            with self.assertRaises(ValueError):
                v.encode(image)


if __name__=='__main__':
    unittest.main()
