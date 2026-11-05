# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 FlyBrain contributors
"""Stage 1 quests: remember the fields, then go to the named one and fill the bag.

Training-only teacher: reads true field centers to label actions. The student gets only
RGB, public status, and the cue recalled from its own FieldMemory (path integration plus
collection sites). Only the 4->9 goal-steering weights are fitted; the market readout
(dipping, conversion, shop) is left untouched. Evaluation has no teacher, and each quest
is repeated with the goal cue removed to show what the memory contributes.
"""
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path

import numpy as np

from .app import ArenaApp
from .world import Action, ArenaWorld
from .train_market import cardinal

CAPACITY = 300
DECISION_SECONDS = .25


def teacher(world, target):
    dx, dy = target[0] - world.x, target[1] - world.y
    if math.hypot(dx, dy) <= 1.:
        return Action(dip=True)
    a = cardinal(dx, dy, dip=True)
    if world.blocked(world.x + a.strafe * .9, world.y + a.forward * .9):  # slide around obstacles
        a = Action(strafe=float(np.sign(dx) or 1.), dip=True) if a.forward else Action(forward=float(np.sign(dy) or 1.), dip=True)
    return a


def configure(app, seed, capacity, seconds, start=None):
    c = app.world.config
    app.training_task = 'quest'
    sx, sy = start if start else (c.hive_x, c.hive_y + 1.5)
    app.world.config = replace(c, spawn_x=sx, spawn_y=sy, pollen_capacity=capacity, episode_seconds=seconds)
    app.reset(seed=seed)


def nearest_field(world):
    return min(world.patches, key=lambda p: math.hypot(p[0] - world.x, p[1] - world.y))[3]


def quest(app, name, seed, use_teacher=False, mix=0, record=None, decisions=400, goal=True, start=None):
    """Run one go-to-field-and-fill episode; `mix` > 0 lets the teacher drive every mix-th step."""
    target = next(p for p in app.world.patches if p[3] == name)[:2]
    configure(app, seed, CAPACITY, decisions * DECISION_SECONDS + 1, start)
    app.goal = name if goal else None
    steps = 0
    for steps in range(1, decisions + 1):
        predicted = app.choose_brain_action()
        label = teacher(app.world, target)
        if record is not None:
            record.append((app.brain.last_features.copy(), app.memory.cue(app.goal), label))
        drive = use_teacher or (mix and steps % mix == 1)
        app.step(label if drive else predicted, frames=app.brain.decision_frames)
        if app.world.pollen >= CAPACITY:
            break
    app.finish_brain_feedback(terminal=True)
    w = app.world
    return {'goal': name, 'seed': seed, 'cue': goal, 'filled': w.pollen >= CAPACITY, 'decisions': steps,
            'seconds_to_fill': steps * DECISION_SECONDS if w.pollen >= CAPACITY else None,
            'collected': w.total_collected, 'ended_at': nearest_field(w),
            'at_goal': w.pollen >= CAPACITY and nearest_field(w) == name}


def tour(app, decisions=140):
    """Teaching tour: a tutor names each field as the bee first collects there."""
    configure(app, 7400, 10 ** 5, 1000)
    app.teach, app.goal = True, None
    for name, cx, cy in [(p[3], p[0], p[1]) for p in app.world.patches]:
        for _ in range(decisions):
            app.choose_brain_action()
            app.step(teacher(app.world, (cx, cy)), frames=app.brain.decision_frames)
    app.finish_brain_feedback(terminal=True)
    app.teach = False


def discover(app, hidden, seed, decisions=480):
    """Forget one field, explore with the DISCOVER goal, then name the discovery and recall it."""
    target = next(p for p in app.world.patches if p[3] == hidden)
    saved = list(app.memory.nodes)
    app.memory.nodes = [n for n in saved if n['name'] != hidden]
    configure(app, seed, 10 ** 5, decisions * DECISION_SECONDS + 1)
    app.goal, app.teach = app.memory.DISCOVER, False
    found = None
    for steps in range(1, decisions + 1):
        app.choose_brain_action()
        app.step(app._brain_action, frames=app.brain.decision_frames)
        found = next((n for n in app.memory.discoveries() if math.dist((n['x'], n['y']), target[:2]) <= 6), None)
        if found:
            break
    app.finish_brain_feedback(terminal=True)
    named = bool(found) and app.memory.name_here(hidden) is not None and app.memory.recall(hidden) is not None
    row = {'hidden': hidden, 'seed': seed, 'found': bool(found), 'seconds_to_find': steps * DECISION_SECONDS if found else None,
           'cells_seen': f'{len(app.memory.seen)}/{len(app.memory.cells)}', 'named_and_recalled': named,
           'spurious_discoveries': len([n for n in app.memory.discoveries()])}
    app.memory.nodes, app.goal = saved, None
    return row


def evaluate(app, names, seeds=(9500, 9501)):
    with_memory, without = [], []
    for seed in seeds:
        for name in names:
            with_memory.append(quest(app, name, seed))
            without.append(quest(app, name, seed, goal=False))
            print('EVAL', name, seed, with_memory[-1]['at_goal'], without[-1]['at_goal'], flush=True)
    return with_memory, without


def random_start(app, rng):
    """A free spot anywhere in the arena (not inside an obstacle), for recovery training."""
    limit = app.world.config.half_size - 1.5
    while True:
        x, y = rng.uniform(-limit, limit), rng.uniform(-limit, limit)
        if not app.world.blocked(x, y):
            return float(x), float(y)


def recovery(app, field='BLUE FIELD', decisions=48):
    """With no order, a bag 150 full, from spots around the field edge: does it get back to collecting?"""
    cx, cy = next(p[:2] for p in app.world.patches if p[3] == field)
    rows = []
    for dx, dy in ((0, 5.2), (0, -5.2), (5.2, 0), (-5.2, 0), (0, 8), (6, 6)):
        configure(app, 9900, CAPACITY, decisions * DECISION_SECONDS + 1, (cx + dx, cy + dy))
        app.goal = None
        app.world.pollen = 150
        for _ in range(decisions):
            app.choose_brain_action()
            app.step(app._brain_action, frames=app.brain.decision_frames)
        app.finish_brain_feedback(terminal=True)
        rows.append({'start_offset': [dx, dy], 'pollen_gained': app.world.pollen - 150})
    return {'rows': rows, 'recovered': sum(r['pollen_gained'] > 0 for r in rows), 'of': len(rows)}


def summarize(rows):
    done = [r for r in rows if r['at_goal']]
    return {'quests': len(rows), 'filled_at_goal': len(done),
            'mean_seconds_to_fill': float(np.mean([r['seconds_to_fill'] for r in done])) if done else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=Path('models/controller-market-v6.npz'))
    parser.add_argument('--output', type=Path, default=Path('models/controller-quest-v1.npz'))
    parser.add_argument('--demo-seeds', type=int, default=2)
    parser.add_argument('--refine', type=int, default=0, metavar='ROUNDS', help='Start from the saved quest checkpoint: student drives, teacher labels, refit')
    parser.add_argument('--start-from', type=Path, default=None, help='Refine from this checkpoint (default: --output)')
    parser.add_argument('--save-as', type=Path, default=None, help='Refined checkpoint name (default: controller-quest-v2.npz)')
    parser.add_argument('--random-starts', type=int, default=0, help='Extra student rollouts per field from random spots')
    parser.add_argument('--discover-only', action='store_true', help='Evaluate discovery with the saved quest checkpoint and memory')
    args = parser.parse_args()
    app = ArenaApp(offscreen=True, width=640, height=400)
    app.connect_brain(monitor=False, checkpoint=args.base)
    brain = app.brain
    brain.learning = brain.sample_actions = False
    names = [p[3] for p in ArenaWorld.patches]
    if args.discover_only:
        try:
            app.brain.load(args.output)
            app.memory.load('models/field-memory.json')
            rows = [discover(app, n, 9700 + i) for i, n in enumerate(names)]
            for r in rows:
                print('DISCOVER', r, flush=True)
            report = {'stage': 'find a forgotten field by exploring, then name it', 'episodes': rows,
                      'found': sum(r['found'] for r in rows), 'of': len(rows),
                      'scope': 'hive start, noise-free odometry; coverage grid 8 units, look radius 5'}
            Path('reports/discovery-v1.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        finally:
            app.destroy()
        return
    if args.refine:
        try:
            app.brain.load(args.start_from or args.output)
            app.memory.load('models/field-memory.json')
            data = []
            rng = np.random.default_rng(5150)
            for rnd in range(args.refine):
                for i, name in enumerate(names):
                    r = quest(app, name, 7700 + 10 * rnd + i, record=data, decisions=240)  # pure student rollout
                    print('ROLLOUT', rnd, name, 'filled', r['filled'], 'at', r['ended_at'], flush=True)
                    for k in range(args.random_starts):
                        r = quest(app, name, 7750 + 10 * rnd + i, record=data, decisions=160, start=random_start(app, rng))
                        print('ROLLOUT', rnd, name, 'random start, filled', r['filled'], 'at', r['ended_at'], flush=True)
                fit = app.brain.fit_navigation([d[0] for d in data], [d[1] for d in data], [d[2] for d in data], steps=3000)
                print('Refit:', fit, flush=True)
            with_memory, without = evaluate(app, names)
            edge = recovery(app)
            print('Edge recovery:', edge['recovered'], '/', edge['of'], flush=True)
            report = {'stage': 'refined by student-driven rollouts (teacher labels only)', 'rounds': args.refine, 'fit': fit,
                      'random_starts_per_field_per_round': args.random_starts, 'edge_recovery_no_order': edge,
                      'with_memory': summarize(with_memory), 'goal_cue_removed': summarize(without),
                      'episodes': with_memory + without}
            target = args.save_as or args.output.with_name('controller-quest-v2.npz')
            Path('reports/' + target.stem.replace('controller-quest-', 'quest-fill-') + '.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
            app.brain.save(target)
            print('With memory:', report['with_memory'], '| Cue removed:', report['goal_cue_removed'], flush=True)
        finally:
            app.destroy()
        return
    try:
        tour(app)
        errors = {n['name']: round(math.dist((n['x'], n['y']), next(p[:2] for p in app.world.patches if p[3] == n['name'])), 2)
                  for n in app.memory.nodes if n['name']}
        missing = [n for n in names if app.memory.recall(n) is None]
        print('Memory nodes (estimate error in units):', errors, 'missing:', missing, flush=True)
        if missing:
            raise SystemExit(f'Tour did not form memory nodes for {missing}')
        app.memory.save('models/field-memory.json')
        data = []
        for i in range(args.demo_seeds):
            for name in names:
                r = quest(app, name, 7500 + i, use_teacher=True, record=data, decisions=240)
                print('DEMO', r['goal'], r['seed'], 'filled', r['filled'], flush=True)
        fit = brain.fit_navigation([d[0] for d in data], [d[1] for d in data], [d[2] for d in data])
        print('Fit 1:', fit, flush=True)
        for name in names:  # one aggregation round: student drives, teacher labels
            quest(app, name, 7600, mix=4, record=data, decisions=240)
        fit = brain.fit_navigation([d[0] for d in data], [d[1] for d in data], [d[2] for d in data])
        print('Fit 2:', fit, flush=True)
        with_memory, without = evaluate(app, names)
        report = {'stage': 'go to commanded field and fill a 300 bag; no hive',
                  'policy_inputs': 'RGB, public status, recalled memory cue; no world coordinates',
                  'teacher': 'training-only, reads true field centers', 'memory_node_error_units': errors,
                  'fit': fit, 'with_memory': summarize(with_memory), 'goal_cue_removed': summarize(without),
                  'episodes': with_memory + without,
                  'scope': 'fixed 6-field layout, hive start; drift is only corrected at the hive'}
        Path('reports/quest-fill-v1.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        brain.save(args.output)
        print('With memory:', report['with_memory'], '\nCue removed:', report['goal_cue_removed'], flush=True)
    finally:
        app.destroy()


if __name__ == '__main__':
    main()
