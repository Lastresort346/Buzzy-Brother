# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 FlyBrain contributors
"""What the BSS HUD says, read from Roblox window pixels: the pollen bar and the hive prompt."""
import numpy as np

# Pollen bar in the BSS HUD: anchored top-centre at fixed pixel size (measured at 1920 wide,
# same position at 1080 and 1009 high). Offsets are from the window's horizontal centre.
BAR_LEFT, BAR_RIGHT, BAR_ROWS = 59, 280, (27, 51)  # rows above/below the "0/200" text
BAR_EMPTY = np.array([115, 112, 101])
PANEL_PROBE, PANEL_COLOR = (40, 285), np.array([248, 240, 213])  # light frame just right of the bar


def read_pollen_fill(frame):
    """Backpack fill in [0, 1] from the pollen bar, or None if the HUD is not visible."""
    cx = frame.shape[1] // 2
    y, dx = PANEL_PROBE
    if frame.shape[0] <= max(BAR_ROWS) or np.abs(frame[y, cx + dx].astype(int) - PANEL_COLOR).sum() > 60:
        return None
    bar = frame[list(BAR_ROWS), cx + BAR_LEFT:cx + BAR_RIGHT].astype(int)
    # The empty bar is translucent grey (scenery shows through but stays desaturated);
    # the fill is a saturated colour (green at 6/200). Require it in both rows.
    filled = (bar.max(-1) - bar.min(-1) > 60).all(0)
    return float(filled.mean())


def confirmed_fill(raw_before, raw, accepted):
    """Accept a reading only when the previous frame was within 10 bar pixels of it.

    One-frame glitches (seen: 5 -> 80 -> 5 pollen) are rejected; steady collection is not.
    """
    if raw is None or raw_before is None or abs(raw - raw_before) > 10 / (BAR_RIGHT - BAR_LEFT):
        return accepted
    return raw


def pollen_events(before, after, capacity):
    """(collected, converted) pollen between two fills. Pollen only drops by conversion."""
    if before is None or after is None:
        return 0., 0.
    change = (after - before) * capacity
    if abs(change) < capacity / (BAR_RIGHT - BAR_LEFT):  # under one bar pixel: noise
        return 0., 0.
    return (change, 0.) if change > 0 else (0., -change)


def honey_prompt(frame):
    """'make' (blue pill) or 'stop' (red pill) of the E prompt, shown only on your hive's pad; else None.

    The prompt sits under the HUD bars at top centre. Looks for one wide, solid pill of that
    colour there; sizes are fractions of the frame, so full-size and 320-wide frames both work.
    """
    h, w = frame.shape[:2]
    band = frame[int(.055 * h):int(.14 * h), int(.3 * w):int(.7 * w)].astype(np.int16)
    r, g, b = band[..., 0], band[..., 1], band[..., 2]
    masks = {'make': (b > 170) & (r < 110) & (g > 80) & (g < 190) & (b - r > 110),
             'stop': (r > 150) & (g < 90) & (b < 90)}
    for name, mask in masks.items():
        ys, xs = np.nonzero(mask)
        if len(ys) < .008 * mask.size:
            continue
        width, height = np.ptp(xs) + 1, np.ptp(ys) + 1
        if width > .12 * mask.shape[1] and width / height > 2.5 and len(ys) / (width * height) > .4:
            return name
    return None


class PromptWatch:
    """Debounced honey prompt: a change needs two matching frames. `arrived` is True on the first
    frame of a visit to the pad."""

    def __init__(self):
        self.raw = self.state = None

    def update(self, raw):
        before = self.state
        if raw == self.raw:
            self.state = raw
        self.raw = raw
        return self.state, before is None and self.state is not None
