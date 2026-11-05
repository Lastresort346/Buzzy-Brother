# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 FlyBrain contributors
"""Train a flower / not-flower recognizer on synthetic views built from the cropped sprites.

    python -m flybrain.arena.flower_sheet <sheet.png>      # once: cut the sprite sheet
    python -m flybrain.arena.train_flowers                 # generate scenes, train, evaluate

Classes: 0 background (incl. tokens, fake flowers, recolored sprites, boxes), 1 real flower.
Evaluation uses scenes the model never saw and reports, per negative kind, how many of its
pixels are wrongly called flower. The previous arena recognizer is scored on the same scenes.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .flower_scenes import load_sprites, make_scene, KIND_NAMES
from .perception import PerceptionModel, pixel_features

MODEL_PATH = Path(__file__).resolve().parents[2] / 'models/perception-bss-v1.npz'


def take(rng, indices, cap):
    return indices if len(indices) <= cap else rng.choice(indices, cap, replace=False)


def sample(scene, rng, resolution=(200, 320), context=False):
    """Training pixels: flower, edge, hard-negative, and random background samples."""
    features, shape, ys, xs = pixel_features(scene['image'], resolution, context)
    labels = scene['labels'][ys[:, None], xs[None, :]]
    hard = scene['hard'][ys[:, None], xs[None, :]]
    edge = np.zeros(shape, bool)
    edge[:-1] |= labels[:-1] != labels[1:]
    edge[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    flat_labels, flat_hard, flat_edge = labels.ravel(), hard.ravel(), edge.ravel()
    chosen = np.unique(np.concatenate([
        take(rng, np.flatnonzero(flat_labels == 1), 2500), take(rng, np.flatnonzero(flat_hard), 2500),
        take(rng, np.flatnonzero(flat_edge), 1500), take(rng, np.flatnonzero((flat_labels == 0) & ~flat_hard), 2500)]))
    return features[chosen], flat_labels[chosen], flat_hard[chosen]


def evaluate(model, scenes, flower_class=1, post=None):
    tp = fp = fn = 0
    kind_total, kind_wrong = {k: 0 for k in KIND_NAMES}, {k: 0 for k in KIND_NAMES}
    for scene in scenes:
        _, masks = model.segment(scene['image'])
        if post is not None:
            masks = post(masks)
        ys = np.linspace(0, 399, masks.shape[0]).astype(int)
        xs = np.linspace(0, 639, masks.shape[1]).astype(int)
        truth = scene['labels'][ys[:, None], xs[None, :]] == 1
        kinds = scene['kinds'][ys[:, None], xs[None, :]]
        predicted = masks == flower_class
        tp += int((truth & predicted).sum())
        fp += int((~truth & predicted).sum())
        fn += int((truth & ~predicted).sum())
        for k in KIND_NAMES:
            mask = kinds == k
            kind_total[k] += int(mask.sum())
            kind_wrong[k] += int((mask & predicted).sum())
    result = {'flower_precision': round(tp / max(1, tp + fp), 4), 'flower_recall': round(tp / max(1, tp + fn), 4),
              'flower_iou': round(tp / max(1, tp + fp + fn), 4),
              'negatives_called_flower': {KIND_NAMES[k]: round(kind_wrong[k] / max(1, kind_total[k]), 4) for k in KIND_NAMES}}
    return result


def detect_real(models, folder, out='data/real-detect'):
    """Run each model on real screenshots (PNG) and save overlays; no labels, so report coverage only."""
    from .flower_scenes import bilinear, save_png
    from .flower_sheet import read_rgb
    results = {}
    Path(out).mkdir(parents=True, exist_ok=True)
    for path in sorted(Path(folder).glob('*.png')):
        img = read_rgb(path)
        ys, xs = np.mgrid[:400, :640]
        view = np.clip(bilinear(img.astype(np.float32), (xs + .5) * img.shape[1] / 640 - .5, (ys + .5) * img.shape[0] / 400 - .5), 0, 255).astype(np.uint8)
        panels, stats = [view], {}
        for name, (model, post) in models.items():
            _, masks = model.segment(view)
            if post is not None:
                masks = post(masks)
            big = np.kron(masks, np.ones((400 // masks.shape[0], 640 // masks.shape[1]), np.uint8))
            over = view.copy()
            over[big == 1] = (over[big == 1] * .35 + np.array([255, 0, 0]) * .65).astype(np.uint8)
            panels.append(over)
            stats[name] = round(float((masks == 1).mean()), 4)
        save_png(Path(out) / f'{path.stem}-{"-".join(models)}.png', np.concatenate(panels, axis=1))
        results[path.name] = stats
    return results


def arena_token_check(models):
    """Real arena frames with sprout tokens: how much of the token color is called flower?"""
    from .app import ArenaApp
    from .world import Action
    from .sprouts import token_mask
    app = ArenaApp(offscreen=True, width=640, height=400)
    try:
        app.reset(seed=42)
        app.world.sprout_orders = False
        counts = {name: [0, 0, 0, 0] for name in models}  # token px, token called flower, flower px, flower found
        for field, spot in (('BLUE FIELD', (-10, -3)), ('WHITE FIELD', (10, 2)), ('ROSE FIELD', (0, 12))):
            app.world.x, app.world.y = spot
            app.world.start_sprout(field)
            app.world.sprout.update(harvested=True, spawned=0, drop_age=3.)
            app.step(Action(), frames=2)
            for _ in range(4):
                app.step(Action(), frames=30)
            image = app.observe()
            labels = app.semantic_labels()
            for name, (model, cls, post) in models.items():
                _, masks = model.segment(image)
                if post is not None:
                    masks = post(masks)
                ys, xs = np.linspace(0, 399, masks.shape[0]).astype(int), np.linspace(0, 639, masks.shape[1]).astype(int)
                tokens = token_mask(image)[ys[:, None], xs[None, :]]
                flowers = labels[ys[:, None], xs[None, :]] == 1
                called = masks == cls
                c = counts[name]
                c[0] += int(tokens.sum()); c[1] += int((tokens & called).sum())
                c[2] += int(flowers.sum()); c[3] += int((flowers & called).sum())
        return {name: {'token_pixels': c[0], 'token_pixels_called_flower': round(c[1] / max(1, c[0]), 4),
                       'arena_flower_recall': round(c[3] / max(1, c[2]), 4)} for name, c in counts.items()}
    finally:
        app.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-scenes', type=int, default=120)
    parser.add_argument('--test-scenes', type=int, default=30)
    parser.add_argument('--steps', type=int, default=2500)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--output', type=Path, default=MODEL_PATH)
    parser.add_argument('--report', type=Path, default=Path('reports/flower-training-v1.json'))
    parser.add_argument('--resolution', choices=('half', 'full'), default='half', help='Inference grid: 200x320 or the full 400x640')
    parser.add_argument('--real', type=Path, default=None, help='Folder of real PNG screenshots to run and overlay')
    parser.add_argument('--context', action='store_true', help='Add wide-window context features (new models only)')
    parser.add_argument('--filter', action='store_true', help='Also report results after shape cleanup (flower_filter.clean)')
    parser.add_argument('--eval-only', action='store_true', help='Skip training; evaluate the model already saved at --output')
    parser.add_argument('--skip-arena', action='store_true')
    args = parser.parse_args()
    sprites = load_sprites()
    rng = np.random.default_rng(11)
    resolution = (400, 640) if args.resolution == 'full' else (200, 320)
    if args.eval_only:
        model = PerceptionModel.load(args.output)
        samples = 0
    else:
        xs, ys, hs = [], [], []
        for i in range(args.train_scenes):
            scene = make_scene(np.random.default_rng(1000 + i), sprites, negatives=i % 7 != 0)
            x, y, h = sample(scene, rng, resolution, args.context)
            xs.append(x), ys.append(y), hs.append(h)
            if (i + 1) % 20 == 0:
                print(f'generated {i + 1}/{args.train_scenes} training scenes', flush=True)
        features, labels, hard = np.concatenate(xs), np.concatenate(ys), np.concatenate(hs)
        print(f'{len(labels)} samples: {int((labels == 1).sum())} flower, {int(hard.sum())} hard negative', flush=True)
        model = PerceptionModel(classes=2, hidden=args.hidden, context=args.context)
        model.metadata = {'version': 'bss-flower-mlp-v1', 'classes': ['not_flower', 'flower'],
                          'runtime_inputs': 'RGB patches only', 'trained_on': 'synthetic views of cropped BSS sprites + negatives',
                          'status': 'trained'}
        model.resolution = resolution
        model.metadata['resolution'] = list(resolution)
        model.metadata['context'] = args.context
        model.fit(features, labels, steps=args.steps, hard=hard)
        model.save(args.output)
        samples = int(len(labels))
    test = [make_scene(np.random.default_rng(9000 + i), sprites) for i in range(args.test_scenes)]
    report = {'train_scenes': args.train_scenes, 'samples': samples, 'steps': args.steps, 'hidden': args.hidden,
              'context_features': model.context, 'held_out_scenes': args.test_scenes, 'new_model': evaluate(model, test)}
    from .flower_filter import clean
    if args.filter:
        report['new_model_after_shape_cleanup'] = evaluate(model, test, post=clean)
    old_path = Path(__file__).resolve().parents[2] / 'models/perception-v3.npz'
    old = PerceptionModel.load(old_path) if old_path.exists() else None
    if old is not None:
        report['previous_arena_model_on_same_scenes'] = evaluate(old, test)
    if args.real:
        compare = {'raw': (model, None)}
        if args.filter:
            compare['cleaned'] = (model, clean)
        if old is not None:
            compare['previous'] = (old, None)
        report['real_screenshots_fraction_called_flower'] = detect_real(compare, args.real)
    if not args.skip_arena:
        models = {'new_model': (model, 1, None)}
        if args.filter:
            models['new_model_cleaned'] = (model, 1, clean)
        if old is not None:
            models['previous_arena_model'] = (old, 1, None)
        report['arena_frames_with_sprout_tokens'] = arena_token_check(models)
    report['scope'] = 'synthetic-to-synthetic held-out scenes; not measured on real Roblox frames'
    args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ('scope',)}, indent=2))


if __name__ == '__main__':
    main()
