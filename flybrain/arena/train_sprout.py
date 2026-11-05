# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 FlyBrain contributors
"""Sprout events: harvest a sprout, then collect its tokens, but only when ordered to.

The bee sees the notification banner and the tokens as pixels. Banner -> field name by
template matching, field name -> place by field memory, tokens -> direction by color and the
fixed camera. Training-only teacher: reads true token positions to label movement. Only the
goal-steering weights are fitted; collection is gated by the trainer's order, and the
evaluation compares ordered+banner, ordered but banner-blind, and not ordered.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from .app import ArenaApp
from .world import Action, ArenaWorld
from .train_quest import configure, DECISION_SECONDS

MAX_DECISIONS = 320  # 80 s: reach the field, harvest 100 pollen, then 12 tokens over 6 s, 15 s lifetime each


def token_teacher(world):
    if world.tokens:
        t = min(world.tokens, key=lambda t: math.hypot(t['x'] - world.x, t['y'] - world.y))
        tx, ty = t['x'], t['y']
    elif world.sprout:
        tx, ty = world.sprout['x'], world.sprout['y']
    else:
        return Action()
    harvesting = bool(world.sprout) and not world.sprout['harvested']  # collect pollen in the field first
    dx, dy = tx - world.x, ty - world.y
    d = math.hypot(dx, dy) or 1.
    if harvesting and d <= 1.:
        return Action(dip=True)
    sx = float(np.sign(dx)) if abs(dx) > .38 * d else 0.
    sy = float(np.sign(dy)) if abs(dy) > .38 * d else 0.
    if world.blocked(world.x + sx * .9, world.y + sy * .9):  # slide around obstacles
        sx, sy = (0., sy or 1.) if sx else (1., 0.)
    return Action(strafe=sx, forward=sy, dip=harvesting)


def sprout_run(app, field, seed, orders=True, banner=True, use_teacher=False, mix=0, record=None):
    configure(app, seed, 300, MAX_DECISIONS * DECISION_SECONDS + 1)
    app.world.sprout_orders, app.goal = orders, None
    app.world.start_sprout(field)
    templates = app.sense.reader.templates
    if not banner:
        app.sense.reader.templates = {}
    steps = 0
    try:
        for steps in range(1, MAX_DECISIONS + 1):
            predicted = app.choose_brain_action()
            label = token_teacher(app.world)
            if record is not None:
                record.append((app.brain.last_features.copy(), app.last_cue.copy(), label))
            drive = use_teacher or (mix and steps % mix == 1)
            app.step(label if drive else predicted, frames=app.brain.decision_frames)
            if app.world.sprout is None and not app.world.tokens:
                break
    finally:
        app.sense.reader.templates = templates
        app.finish_brain_feedback(terminal=True)
    w = app.world
    return {'field': field, 'seed': seed, 'orders': orders, 'banner_visible_to_policy': banner,
            'token_honey': w.token_honey_total, 'tokens_collected': w.token_honey_total // w.config.token_value,
            'of': w.config.sprout_tokens, 'seconds': steps * DECISION_SECONDS}


def calibrate(app, path):
    """Tutor shows each field's banner once; the reader stores its template."""
    for p in ArenaWorld.patches:
        app.reset(seed=42)
        app.world.start_sprout(p[3])
        app.world.sprout['age'] = 1.
        app.step(Action(), frames=2)
        if not app.sense.reader.learn(p[3], app.observe()):
            raise SystemExit(f'Could not read the banner for {p[3]}')
    app.sense.reader.save(path)


def evaluate(app, names, seeds=(9800,)):
    rows = []
    for seed in seeds:
        for name in names:
            for kwargs in (dict(), dict(banner=False), dict(orders=False)):
                rows.append(sprout_run(app, name, seed, **kwargs))
            print('EVAL', name, seed, [r['tokens_collected'] for r in rows[-3:]], flush=True)
    return rows


def summarize(rows, **match):
    chosen = [r for r in rows if all(r[k] == v for k, v in match.items())]
    return {'runs': len(chosen), 'mean_tokens': float(np.mean([r['tokens_collected'] for r in chosen])) if chosen else None,
            'of': chosen[0]['of'] if chosen else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=None, help='Quest checkpoint (default: v2 if present, else v1)')
    parser.add_argument('--output', type=Path, default=Path('models/controller-sprout-v1.npz'))
    parser.add_argument('--demo-seeds', type=int, default=2)
    parser.add_argument('--rounds', type=int, default=1, help='Student-driven correction rounds')
    args = parser.parse_args()
    base = args.base or next(p for p in (Path('models/controller-quest-v2.npz'), Path('models/controller-quest-v1.npz')) if p.exists())
    app = ArenaApp(offscreen=True, width=640, height=400)
    app.connect_brain(monitor=False, checkpoint=base)
    brain = app.brain
    brain.learning = brain.sample_actions = False
    names = [p[3] for p in ArenaWorld.patches]
    try:
        app.memory.load('models/field-memory.json')
        calibrate(app, 'models/sprout-banners.json')
        data = []
        for i in range(args.demo_seeds):
            for name in names:
                r = sprout_run(app, name, 7900 + i, use_teacher=True, record=data)
                print('DEMO', name, r['tokens_collected'], '/', r['of'], flush=True)

        def fit(steps):
            return brain.fit_navigation([d[0] for d in data], [d[1] for d in data], [d[2] for d in data], steps=steps)

        print('Fit 1:', fit(2000), flush=True)
        for rnd in range(args.rounds):
            for i, name in enumerate(names):
                r = sprout_run(app, name, 8000 + 10 * rnd + i, record=data)
                print('ROLLOUT', name, r['tokens_collected'], '/', r['of'], flush=True)
            print('Refit:', fit(3000), flush=True)
        rows = evaluate(app, names)
        report = {'stage': 'collect sprout tokens when ordered', 'base_checkpoint': str(base),
                  'policy_inputs': 'RGB (banner and tokens as pixels), public status, recalled memory cue; no world coordinates',
                  'teacher': 'training-only, reads true token positions',
                  'ordered_with_banner': summarize(rows, orders=True, banner_visible_to_policy=True),
                  'ordered_banner_blind': summarize(rows, orders=True, banner_visible_to_policy=False),
                  'not_ordered': summarize(rows, orders=False), 'episodes': rows,
                  'scope': 'simulated sprouts, hive start, fixed 6-field layout; imitation only, no reinforcement fine-tuning'}
        Path('reports/sprout-v1.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        brain.save(args.output)
        for key in ('ordered_with_banner', 'ordered_banner_blind', 'not_ordered'):
            print(key, report[key], flush=True)
    finally:
        app.destroy()


if __name__ == '__main__':
    main()
