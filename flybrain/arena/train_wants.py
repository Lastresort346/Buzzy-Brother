# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 FlyBrain contributors
"""Train the goal steering to follow WANTS (a field or a flower color) with the flower eye in the loop.

Training-only teacher: knows the true field centers; for a color want it goes to the field the bee's own
memory says yields that color. The student gets RGB through the flower eye, public status, and the cue
the thinking center produces. Rounds of teacher drive, mixed drive, then student drive (DAgger), each
followed by a refit of the steering weights; then the same six wants as eval_wants, with no teacher.

    python -m flybrain.arena.train_wants --eye bss --base models/controller-cycle-v1.npz
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .app import ArenaApp
from .eval_wants import run as evaluate_want
from .train_quest import configure, teacher, random_start, DECISION_SECONDS
from .world import ArenaWorld

COLORS = ('blue', 'white', 'red')


def episode(app, want, seed, start, decisions, mix=0, use_teacher=False, record=None):
    kind, value = want
    field = value if kind == 'field' else app.memory.best_field_for(value)
    target = next(p[:2] for p in app.world.patches if p[3] == field)
    configure(app, seed, 300, decisions * DECISION_SECONDS + 1, start)
    app.goal = None
    app.mind.want(field=value if kind == 'field' else None, color=value if kind == 'color' else None)
    for step in range(1, decisions + 1):
        predicted = app.choose_brain_action()
        label = teacher(app.world, target)
        if record is not None:
            record.append((app.brain.last_features.copy(), app.last_cue.copy(), label))
        drive = use_teacher or (mix and step % mix == 1)
        app.step(label if drive else predicted, frames=app.brain.decision_frames)
        if app.world.pollen >= app.world.config.pollen_capacity:
            break
    app.finish_brain_feedback(terminal=True)
    return {'want': want, 'seed': seed, 'filled': app.world.pollen >= app.world.config.pollen_capacity}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eye', choices=('arena', 'bss'), default='bss')
    parser.add_argument('--base', type=Path, default=Path('models/controller-cycle-v1.npz'))
    parser.add_argument('--output', type=Path, default=Path('models/controller-wants-v1.npz'))
    parser.add_argument('--report', type=Path, default=Path('reports/wants-v2.json'))
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--decisions', type=int, default=80)
    parser.add_argument('--fit-steps', type=int, default=3000)
    args = parser.parse_args()
    app = ArenaApp(offscreen=True, width=640, height=400)
    app.connect_brain(monitor=False, checkpoint=args.base, eye=args.eye)
    brain = app.brain
    brain.learning = brain.sample_actions = False
    fields = [p[3] for p in ArenaWorld.patches]
    wants = [('field', f) for f in fields] + [('color', c) for c in COLORS]
    rng = np.random.default_rng(31)
    try:
        app.memory.load('models/field-memory.json')
        data = []
        for rnd in range(args.rounds):
            for i, want in enumerate(wants):
                start = None if i % 3 else random_start(app, rng)  # a third of the episodes start away from the hive
                kwargs = dict(use_teacher=True) if rnd == 0 else dict(mix=4) if rnd == 1 else {}
                r = episode(app, want, 8200 + 100 * rnd + i, start, args.decisions, record=data, **kwargs)
                print('ROUND', rnd, want, 'filled', r['filled'], flush=True)
            x, c, a = [d[0] for d in data], [d[1] for d in data], [d[2] for d in data]
            print('Refit:', brain.fit_navigation(x, c, a, steps=args.fit_steps), flush=True)
        rows = []
        for want in ([dict(want_color=c) for c in COLORS] + [dict(want_field=f) for f in ('ROSE FIELD', 'SOUTH FIELD')] + [dict()]):
            rows.append(evaluate_want(app, decisions=120, **want))
            print('EVAL', rows[-1], flush=True)
        args.report.write_text(json.dumps({'eye': args.eye, 'base': str(args.base), 'rounds': args.rounds, 'rows': rows}, indent=2) + '\n', encoding='utf-8')
        brain.save(args.output)
    finally:
        app.destroy()


if __name__ == '__main__':
    main()
