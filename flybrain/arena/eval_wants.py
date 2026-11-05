# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 FlyBrain contributors
"""Does the bee collect the flower color / go to the field it wants? Start at the hive, give it a want,
let the trained controller run with no teacher, and count the pollen it gathers per flower color.

    python -m flybrain.arena.eval_wants --eye bss --checkpoint models/controller-cycle-v1.npz
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from .app import ArenaApp, COLOR_NAMES
from .train_quest import configure, DECISION_SECONDS


def run(app, want_color=None, want_field=None, decisions=120):
    configure(app, 9900, 300, decisions * DECISION_SECONDS + 1)
    app.goal = None
    app.mind.want(want_field, want_color)
    w = app.world
    app.color_totals = {}
    started = time.perf_counter()
    for _ in range(decisions):
        app.choose_brain_action()
        app.step(app._brain_action, frames=app.brain.decision_frames)
    elapsed = time.perf_counter() - started
    app.finish_brain_feedback(terminal=True)
    total = w.total_collected
    field = min(w.patches, key=lambda p: math.hypot(p[0] - w.x, p[1] - w.y))[3]
    return {'want_color': want_color, 'want_field': want_field, 'pollen_collected': total, 'pollen_by_color': {k: int(app.color_totals.get(k, 0)) for k in COLOR_NAMES},
            'ended_near': field, 'seconds_per_decision': round(elapsed / decisions, 3),
            'mind': app.mind.describe(w.pollen / w.config.pollen_capacity)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eye', choices=('arena', 'bss'), default='bss')
    parser.add_argument('--checkpoint', type=Path, default=Path('models/controller-cycle-v1.npz'))
    parser.add_argument('--decisions', type=int, default=120)
    parser.add_argument('--report', type=Path, default=Path('reports/wants-v1.json'))
    args = parser.parse_args()
    app = ArenaApp(offscreen=True, width=640, height=400)
    app.connect_brain(monitor=False, checkpoint=args.checkpoint, eye=args.eye)
    app.brain.learning = app.brain.sample_actions = False
    app.memory.load('models/field-memory.json')
    wants = [dict(want_color='blue'), dict(want_color='white'), dict(want_color='red'),
             dict(want_field='ROSE FIELD'), dict(want_field='SOUTH FIELD'), dict()]
    rows = []
    try:
        for want in wants:
            row = run(app, decisions=args.decisions, **want)
            rows.append(row)
            print(row, flush=True)
    finally:
        app.destroy()
    args.report.write_text(json.dumps({'eye': args.eye, 'checkpoint': str(args.checkpoint), 'rows': rows}, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
