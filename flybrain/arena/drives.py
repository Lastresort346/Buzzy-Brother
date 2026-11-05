# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 FlyBrain contributors
"""The thinking center: wants and needs that decide where the bee goes and what it looks at.

Wants are set by the trainer (a named field, or a flower color). Needs come from its own public
status: a full or converting bag means deliver to the hive; otherwise collect. The center turns that
into (1) a goal field name for the memory-based steering cue and (2) an attention gain per flower
color that boosts the wanted color's drive into the optic neurons and dims the rest. These are
engineered motivational variables, not a model of feeling.
"""
from .flower_eye import COLORS
from .memory import FieldMemory

DIM = .4  # relative drive from flowers of an unwanted color while a color is wanted


class ThinkingCenter:
    def __init__(self, want_field=None, want_color=None):
        self.want_field = None
        self.want_color = None
        self.want(want_field, want_color)

    def want(self, field=None, color=None):
        """Set what the bee wants now. A field outranks a color; both None clears the want."""
        if color is not None and color not in COLORS:
            raise ValueError(f'Unknown flower color {color!r}; expected one of {COLORS}')
        self.want_field, self.want_color = field, color

    def needs(self, pollen_fraction, converting=False, in_hive=False):
        """Pressures in 0..1: delivering grows as the bag fills; collecting shrinks."""
        busy = pollen_fraction >= .999 or converting or in_hive
        return {'deliver': 1. if busy else float(pollen_fraction) ** 2,
                'collect': 0. if busy else 1. - float(pollen_fraction)}

    def mode(self, pollen_fraction, converting=False, in_hive=False):
        return 'deliver' if self.needs(pollen_fraction, converting, in_hive)['deliver'] >= 1. else 'collect'

    def goal(self, memory, pollen_fraction, converting=False, in_hive=False):
        """Field name to steer toward, FieldMemory.DISCOVER to look for one, or None (let the hive trip run)."""
        if self.mode(pollen_fraction, converting, in_hive) == 'deliver':
            return None
        if self.want_field:
            return self.want_field
        if self.want_color:
            return memory.best_field_for(self.want_color) or FieldMemory.DISCOVER
        return memory.nearest_field()

    def attention(self):
        """Gain per flower color for the visual drive: wanted color 1, others dimmed; empty = no bias."""
        if not self.want_color:
            return {}
        return {c: 1. if c == self.want_color else DIM for c in COLORS}

    def describe(self, pollen_fraction=0., converting=False, in_hive=False):
        return {'want_field': self.want_field, 'want_color': self.want_color,
                'mode': self.mode(pollen_fraction, converting, in_hive),
                'needs': self.needs(pollen_fraction, converting, in_hive)}
