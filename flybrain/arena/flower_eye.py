# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 FlyBrain contributors
"""The flower eye: detector + shape cleanup + flower color, as one drop-in perception for the fly.

It answers the same `segment(image)` call as the arena recognizer (class 1 flower, class 2 hive), so it
plugs straight into `ThirdPersonVision` and from there into the optic neurons. On top of that it can
say which detected flowers are white, red or blue, and where the nearest flower of a wanted color is.
Runtime inputs are RGB pixels only.
"""
import numpy as np

from .sprouts import mask_cue

COLORS = ('white', 'red', 'blue')
FLOWER_Z = .03  # flowers lie on the ground


def color_masks(image, flower):
    """Split a flower mask into white / red / blue pixels by color. `image` and `flower` share a shape.
    Yellow centers and leaf green fall into none of the three."""
    rgb = np.asarray(image).astype(np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    top, bottom = rgb.max(axis=2), rgb.min(axis=2)
    saturation = (top - bottom) / np.maximum(top, 1.)
    red = flower & (r > 1.35 * g) & (r > 1.35 * b)
    blue = flower & (b > 1.25 * r) & (b > 1.05 * g)
    white = flower & (saturation < .3) & (top > 70)
    return {'white': white & ~red & ~blue, 'red': red, 'blue': blue}


class FlowerEye:
    """`flower_model` is a PerceptionModel with a flower class 1 (e.g. the BSS-trained one); `hive_model` is
    optional and supplies class 2; `post` cleans the flower mask (e.g. flower_filter.clean)."""

    def __init__(self, flower_model, hive_model=None, post=None):
        self.flower_model, self.hive_model, self.post = flower_model, hive_model, post
        self.metadata = {'version': 'flower-eye/' + str(flower_model.metadata.get('version', 'model'))
                         + ('+clean' if post else '') + ('+hive' if hive_model is not None else '')}
        self.image = None
        self.flower = None
        self._colors = None
        self._last = None

    def segment(self, image):
        if image is self.image and self._last is not None:  # same frame already analysed (e.g. by the cue)
            return self._last
        probability, masks = self.flower_model.segment(image)
        if self.post is not None:
            masks = self.post(masks)
        flower = masks == 1
        h, w = flower.shape
        out = np.zeros((h, w, 3), np.float32)
        out[..., 1] = probability[..., 1] * flower
        semantic = flower.astype(np.uint8)
        if self.hive_model is not None:
            ph, hm = self.hive_model.segment(image)
            yi, xi = np.arange(h) * hm.shape[0] // h, np.arange(w) * hm.shape[1] // w
            hive = (hm[yi[:, None], xi[None, :]] == 2) & ~flower
            out[..., 2] = ph[yi[:, None], xi[None, :], 2] * hive
            semantic[hive] = 2
        out[..., 0] = 1 - out[..., 1] - out[..., 2]
        self.image, self.flower, self._colors, self._last = image, flower, None, (out, semantic)
        return out, semantic

    def colors(self):
        """White/red/blue masks of the last segmented image, at the flower-mask resolution."""
        if self.flower is None:
            raise ValueError('segment() an image first')
        if self._colors is None:
            h, w = self.flower.shape
            yi, xi = np.arange(h) * self.image.shape[0] // h, np.arange(w) * self.image.shape[1] // w
            self._colors = color_masks(np.asarray(self.image)[yi[:, None], xi[None, :]], self.flower)
        return self._colors

    def color_cue(self, color, geometry, arrive=1.3, min_pixels=4):
        """Steering cue toward the nearest visible flower of `color` (see sprouts.mask_cue), or None."""
        mask = self.colors()[color]
        # mask_cue projects with the mask's own pixel grid, so scale-free: it only needs the shape.
        return mask_cue(mask, geometry, FLOWER_Z, min_pixels, arrive)
