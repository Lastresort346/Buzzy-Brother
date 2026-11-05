# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 FlyBrain contributors
"""Retrain the collect -> hive -> convert -> shop cycle so the bee stops idling and swinging at grass.

Training-only teacher (reads true flower state): swing the dipper ONLY when a ready flower is
within reach, otherwise walk to the nearest ready flower; go to the hive when the bag is full
(or something is affordable), convert, buy. The student sees RGB, public status and the default
field cue. Rounds of teacher demonstrations, then student-driven rollouts with teacher labels
(DAgger), each followed by a refit of the whole readout and the goal-steering weights.
Evaluated before and after from the same starts; no teacher at evaluation.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from .app import ArenaApp
from .world import Action
from .train_market import preferred_item
from .train_quest import configure, random_start, DECISION_SECONDS

REACH = 1.6  # swing when a ready flower is this close


def toward(world, tx, ty):
    dx, dy = tx - world.x, ty - world.y
    d = math.hypot(dx, dy) or 1.
    sx = float(np.sign(dx)) if abs(dx) > .38 * d else 0.
    sy = float(np.sign(dy)) if abs(dy) > .38 * d else 0.
    if world.blocked(world.x + sx * .9, world.y + sy * .9):  # slide around obstacles
        sx, sy = (0., sy or 1.) if sx else (1., 0.)
    return Action(strafe=sx, forward=sy)


def cycle_teacher(world):
    slot = preferred_item(world)
    afford = slot and world.honey >= world.shop.prices[slot - 1]
    if world.in_conversion_zone():
        if afford:
            return Action(buy=slot)
        if world.conversion_active:
            return Action()
        if world.pollen > 0:
            return Action(convert=True)
    if world.conversion_active:
        return Action()
    c = world.config
    if world.pollen >= c.pollen_capacity or afford:
        return toward(world, c.hive_x, c.hive_y)
    ready = [f for f in world.flowers if f.pollen >= .25]
    if not ready:
        return Action()
    f = min(ready, key=lambda f: math.hypot(f.x - world.x, f.y - world.y))
    if math.hypot(f.x - world.x, f.y - world.y) <= REACH:
        return Action(dip=True)
    return toward(world, f.x, f.y)


def scenario(app, rng, hive_start):
    start = None if hive_start else random_start(app, rng)
    return start, int(rng.choice([0, 0, 90, 150, 220, 300])), int(rng.choice([0, 0, 120, 400]))


def episode(app, seed, start, pollen, wallet, decisions, mix=0, use_teacher=False, record=None):
    configure(app, seed, 300, decisions * DECISION_SECONDS + 1, start)
    app.goal, app.world.sprout_orders = None, False
    app.world.pollen, app.world.honey = pollen, wallet
    for step in range(1, decisions + 1):
        predicted = app.choose_brain_action()
        label = cycle_teacher(app.world)
        if record is not None:
            record.append((app.brain.last_features.copy(), app.last_cue.copy(), label))
        drive = use_teacher or (mix and step % mix == 1)
        app.step(label if drive else predicted, frames=app.brain.decision_frames)
    app.finish_brain_feedback(terminal=True)


def measure(app, seed, start, pollen, decisions):
    configure(app, seed, 300, decisions * DECISION_SECONDS + 1, start)
    app.goal, app.world.sprout_orders = None, False
    app.world.pollen = pollen
    w = app.world
    idle = longest = 0
    for _ in range(decisions):
        before = (w.x, w.y, w.total_collected, w.total_converted)
        app.choose_brain_action()
        app.step(app._brain_action, frames=app.brain.decision_frames)
        stuck = math.hypot(w.x - before[0], w.y - before[1]) < .05 and w.total_collected == before[2] and w.total_converted == before[3]
        idle = idle + 1 if stuck else 0
        longest = max(longest, idle)
    app.finish_brain_feedback(terminal=True)
    return {'start': start or 'hive', 'initial_pollen': pollen, 'seconds': decisions * DECISION_SECONDS,
            'pollen_collected': w.total_collected, 'pollen_converted': w.total_converted, 'honey_earned': w.honey_earned,
            'swings': w.swings, 'dry_swing_fraction': round((w.swings - w.successful_swings) / w.swings, 3) if w.swings else 0.,
            'longest_idle_seconds': longest * DECISION_SECONDS}


def evaluate(app, decisions):
    rng = np.random.default_rng(2024)
    rows = [measure(app, 9600, None, 0, decisions)]
    for i in range(2):
        rows.append(measure(app, 9601 + i, random_start(app, rng), (0, 150)[i], decisions))
    for r in rows:
        print('  eval', r, flush=True)
    return {'episodes': rows, 'honey_earned': sum(r['honey_earned'] for r in rows),
            'pollen_collected': sum(r['pollen_collected'] for r in rows),
            'mean_dry_swing_fraction': round(float(np.mean([r['dry_swing_fraction'] for r in rows])), 3),
            'worst_idle_seconds': max(r['longest_idle_seconds'] for r in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=Path('models/controller-quest-v2.npz'))
    parser.add_argument('--output', type=Path, default=Path('models/controller-cycle-v1.npz'))
    parser.add_argument('--rounds', type=int, default=3, help='Student-driven correction rounds after the teacher round')
    parser.add_argument('--episodes', type=int, default=12)
    parser.add_argument('--decisions', type=int, default=120)
    parser.add_argument('--eval-decisions', type=int, default=300)
    parser.add_argument('--fit-steps', type=int, default=1200)
    parser.add_argument('--dip-balance', type=float, default=1.0, help='<1 softens up-weighting of rare dip labels (fewer dry swings)')
    parser.add_argument('--report', type=Path, default=Path('reports/cycle-v1.json'))
    args = parser.parse_args()
    app = ArenaApp(offscreen=True, width=640, height=400)
    app.connect_brain(monitor=False, checkpoint=args.base)
    brain = app.brain
    brain.learning = brain.sample_actions = False
    try:
        app.memory.load('models/field-memory.json')
        print('BEFORE', flush=True)
        before = evaluate(app, args.eval_decisions)
        rng = np.random.default_rng(777)
        data = []

        def fit():
            x, c, a = [d[0] for d in data], [d[1] for d in data], [d[2] for d in data]
            print('  fit', brain.fit_demonstrations(x, a, steps=args.fit_steps, balance=args.dip_balance)['training_movement_accuracy'],
                  brain.fit_navigation(x, c, a, steps=2000), flush=True)

        for rnd in range(args.rounds + 1):
            for i in range(args.episodes):
                start, pollen, wallet = scenario(app, rng, hive_start=i % 3 == 0)
                kwargs = dict(use_teacher=True) if rnd == 0 else dict(mix=4 if i % 2 else 0)
                episode(app, 6000 + 100 * rnd + i, start, pollen, wallet, args.decisions, record=data, **kwargs)
            print(f'ROUND {rnd}: {len(data)} labelled decisions', flush=True)
            fit()
        print('AFTER', flush=True)
        after = evaluate(app, args.eval_decisions)
        report = {'stage': 'cycle retrain: swing only when a ready flower is in reach, DAgger from varied starts',
                  'base_checkpoint': str(args.base), 'rounds': args.rounds, 'labelled_decisions': len(data),
                  'before': before, 'after': after,
                  'teacher': 'training-only, reads true flower state',
                  'scope': 'imitation only; fixed 6-field layout; no sprouts in this stage'}
        args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        brain.save(args.output)
        for k in ('honey_earned', 'pollen_collected', 'mean_dry_swing_fraction', 'worst_idle_seconds'):
            print(f'{k}: before {before[k]} -> after {after[k]}', flush=True)
    finally:
        app.destroy()


if __name__ == '__main__':
    main()
