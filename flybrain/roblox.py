# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 FlyBrain contributors
"""Experimental bridge: Roblox Player window pixels -> neural controller -> allowlisted inputs.

Sandbox rules, enforced here and nowhere else:
- Only ALLOWED inputs can ever be sent (movement, jump, dipper, convert). Esc, Alt+F4,
  menus, chat and every other key are unreachable from the controller.
- Input is sent only while the Roblox Player window has focus and is released
  immediately when focus leaves. (Roblox ignores keys posted to an unfocused window;
  tested 2026-11-03.) Screen capture still works while the window is covered.
- F8 stops the bot and releases everything. Inputs are also released on any exit.
Windows only. Uses ctypes; no extra dependencies.
"""
import argparse
import ctypes
import ctypes.wintypes as wt
import sys
import time
from pathlib import Path

import numpy as np

from .hud import (BAR_EMPTY, BAR_LEFT, BAR_RIGHT, BAR_ROWS, PANEL_COLOR, PANEL_PROBE, PromptWatch, confirmed_fill,  # noqa: F401
                  honey_prompt, pollen_events, read_pollen_fill)

# shortcut: fixed key bindings (BSS defaults), upgrade if users rebind keys in-game.
ALLOWED = {'w': 0x57, 'a': 0x41, 's': 0x53, 'd': 0x44, 'space': 0x20, 'e': 0x45, 'mouse': None}
STOP_KEY = 0x77  # F8
RECORD_ORDER = ('d', 'a', 'w', 's', 'space', 'mouse', 'e')  # column order in demonstrations.npz


class WallReflex:
    """Engineered escape reflex, not learned: after 1 s stuck, back up 0.5 s, then sidestep 1 s.

    Recordings never show a wall, so there is nothing to imitate. Its decisions are not
    credited to the brain. Only allowlisted movement keys are used.
    """

    def __init__(self, patience=4, rng=None):
        self.patience, self.count, self.plan = patience, 0, []
        self.rng = rng or np.random.default_rng()

    def step(self, blocked):
        """Inputs to use instead of the brain's for this decision, or None."""
        if self.plan:
            return self.plan.pop(0)
        self.count = self.count + 1 if blocked else 0
        if self.count < self.patience:
            return None
        self.count = 0
        side = str(self.rng.choice(['a', 'd']))
        self.plan = [{'s'}] * 2 + [{side}] * 4
        return self.plan.pop(0)


def stuck(before, after, wanted):
    """True when movement keys were held but the view barely changed (walking into a wall)."""
    if before is None or not wanted & {'w', 'a', 's', 'd'}:
        return False
    # Share of pixels that changed clearly. Measured on recordings (2026-11-03): walking >= 0.26
    # in 90% of frames, standing <= 0.02 in 90%; flying bees move few pixels, so they barely count.
    changed = (np.abs(before.astype(np.int16) - after).max(-1) > 25).mean()
    return float(changed) < .08


def held_inputs(u32):
    """Allowlisted inputs the player is pressing. Taps since the last check count for jump/dipper/E."""
    held = set()
    for name, vk in ALLOWED.items():
        state = u32.GetAsyncKeyState(0x01 if vk is None else vk)  # 0x01: left mouse button
        if state & (0x8000 if name in 'wasd' else 0x8001):
            held.add(name)
    return held


def action_from_inputs(held):
    """Inverse of inputs_for: a recorded player label in the controller's Action space."""
    from .arena.world import Action
    return Action(strafe=float(('d' in held) - ('a' in held)), forward=float(('w' in held) - ('s' in held)),
                  jump='space' in held, dip='mouse' in held, convert='e' in held)


def balance_walking(features, labels):
    """Repeat walking decisions until they match standing ones.

    Players spend most recorded time standing (dipping, waiting at the hive), so an
    unbalanced fit learns "never move" (seen 2026-11-03: 163 of 182 test decisions).
    """
    walking = [i for i, a in enumerate(labels) if a.strafe or a.forward]
    standing = len(labels) - len(walking)
    if not walking or len(walking) >= standing:
        return features, labels
    extra = [walking[i % len(walking)] for i in range(standing - len(walking))]
    return list(features) + [features[i] for i in extra], list(labels) + [labels[i] for i in extra]


def inputs_for(action):
    """Map a controller Action to the set of inputs to hold. Never leaves ALLOWED."""
    held = set()
    if action.forward > 0: held.add('w')
    if action.forward < 0: held.add('s')
    if action.strafe < 0: held.add('a')
    if action.strafe > 0: held.add('d')
    if action.jump: held.add('space')
    if action.dip: held.add('mouse')
    if action.convert: held.add('e')
    # action.buy and camera fields are deliberately dropped: shops need menu clicks.
    assert held <= ALLOWED.keys(), held
    return held


class RobloxWindow:
    def __init__(self):
        self.u32 = ctypes.windll.user32
        self.g32 = ctypes.windll.gdi32
        self.u32.SetProcessDPIAware()
        self.hwnd = self._find()

    def _find(self):
        found = []

        @ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
        def visit(h, _):
            title = ctypes.create_unicode_buffer(256)
            cls = ctypes.create_unicode_buffer(256)
            self.u32.GetWindowTextW(h, title, 256)
            self.u32.GetClassNameW(h, cls, 256)
            if self.u32.IsWindowVisible(h) and title.value == 'Roblox' and cls.value == 'WINDOWSCLIENT':
                found.append(h)
            return True
        self.u32.EnumWindows(visit, 0)
        if len(found) != 1:
            raise RuntimeError(f'Expected exactly one Roblox Player window, found {len(found)}')
        return found[0]

    def focused(self):
        return self.u32.GetForegroundWindow() == self.hwnd

    def size(self):
        r = wt.RECT()
        self.u32.GetClientRect(self.hwnd, ctypes.byref(r))
        return r.right, r.bottom

    def capture(self):
        """RGB client pixels via PrintWindow; works while covered, not while minimized."""
        if self.u32.IsIconic(self.hwnd):
            raise RuntimeError('Roblox window is minimized; restore it (it may stay behind other windows)')
        w, h = self.size()
        dc = self.u32.GetDC(self.hwnd)
        mdc = self.g32.CreateCompatibleDC(dc)
        bmp = self.g32.CreateCompatibleBitmap(dc, w, h)
        self.g32.SelectObject(mdc, bmp)
        try:
            if not self.u32.PrintWindow(self.hwnd, mdc, 3):  # client only + full GPU content
                raise RuntimeError('PrintWindow failed')
            header = (ctypes.c_uint32 * 10)(40, w, (-h) & 0xffffffff, 1 | (32 << 16), 0, 0, 0, 0, 0, 0)
            buf = (ctypes.c_ubyte * (w * h * 4))()
            self.g32.GetDIBits(mdc, bmp, 0, h, buf, header, 0)
        finally:
            self.g32.DeleteObject(bmp)
            self.g32.DeleteDC(mdc)
            self.u32.ReleaseDC(self.hwnd, dc)
        return np.frombuffer(buf, np.uint8).reshape(h, w, 4)[:, :, 2::-1].copy()


class SandboxedInput:
    """Holds allowlisted inputs on the Roblox window; releases on focus loss, stop, or exit."""

    def __init__(self, window):
        self.window, self.held = window, set()

    def _send(self, name, down):
        u32 = self.window.u32
        if name == 'mouse':
            u32.mouse_event(0x2 if down else 0x4, 0, 0, 0, 0)
        else:
            u32.keybd_event(ALLOWED[name], u32.MapVirtualKeyW(ALLOWED[name], 0), 0 if down else 0x2, 0)

    def apply(self, wanted):
        if not wanted <= ALLOWED.keys():
            raise ValueError(f'Input outside the sandbox allowlist: {wanted - ALLOWED.keys()}')
        if not self.window.focused():
            wanted = set()  # never type into whatever other window the user switched to
        elif 'mouse' in wanted and not self._cursor_inside():
            wanted = wanted - {'mouse'}
        for name in self.held - wanted:
            self._send(name, False)
        for name in wanted - self.held:
            self._send(name, True)
        self.held = set(wanted)

    def _cursor_inside(self):
        p = wt.POINT()
        self.window.u32.GetCursorPos(ctypes.byref(p))
        return self.window.u32.WindowFromPoint(p) == self.window.hwnd

    def release(self):
        self.apply(set())


STUCK_PENALTY = -.1  # worth about 10 pollen in the reward system's units (pollen/100)
ARRIVAL_BONUS = .3  # worth about 30 pollen: reaching the hive pad with a mostly full bag


def reward_events(collected, converted, blocked, previous_inputs, arrived=False):
    """Reward for the previous decision, in RewardSystem.evaluate's keyword terms.

    Bees gather on their own, so pollen only counts while the bot was dipping. Walls cost
    about 10 pollen via the signed shaping channel; the shared 'blocked' term (-0.05) is
    too weak to outweigh bee-made pollen. Arriving at the hive pad with a mostly full bag
    pays through the same channel, so walking home is what gets learned.
    """
    progress = STUCK_PENALTY if blocked else ARRIVAL_BONUS if arrived else 0.
    return {'reward': collected if 'mouse' in previous_inputs else 0., 'honey': converted,
            'blocked_fraction': float(blocked), 'progress': progress}


def brain_frame(frame):
    """About 320 px wide: what the brain and place recognizer see, live and in recordings."""
    step = max(1, frame.shape[1] // 320)
    return np.ascontiguousarray(frame[::step, ::step])


def fit_places(paths, output):
    from . import places
    train_x, train_y, test_x, test_y = [], [], [], []
    for path in paths:
        with np.load(path) as data:
            if 'frames' not in data:
                raise SystemExit(f'{path} has no frames; record it again with this version')
            fills = [None if np.isnan(f) else float(f) for f in data['fills']]
            inputs = data['inputs']
            labels = places.auto_labels(fills, inputs[:, RECORD_ORDER.index('mouse')], str(data['field']),
                                        moving=inputs[:, :4].any(1),
                                        pads=[honey_prompt(f) is not None for f in data['frames']])
            thumbs = [places.thumbnail(f) for f in data['frames']]
        cut = int(len(thumbs) * .8)  # hold out the last 20% of each recording in time
        train_x += thumbs[:cut]; train_y += labels[:cut]
        test_x += thumbs[cut:]; test_y += labels[cut:]
    model = places.PlaceModel.fit(train_x, train_y)
    predicted = [model.classes[int(np.argmax(model.probabilities_from_thumb(x)))] for x in test_x]
    for name in model.classes:
        rows = [p == t for p, t in zip(predicted, test_y) if t == name]
        print(f'{name:>12}: train {train_y.count(name):4d}, held-out {len(rows):4d}, '
              f'held-out accuracy {np.mean(rows) if rows else float("nan"):.2f}', flush=True)
    model.save(output)
    print(f'Saved place recognizer to {output}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', choices=['foreground', 'none'], default='none',
                        help='none = dry run (default): read the screen and print actions only')
    parser.add_argument('--checkpoint', default='models/controller-market-v5.npz')
    parser.add_argument('--seconds', type=float, default=60)
    parser.add_argument('--capacity', type=int, default=200, help='bag capacity shown in the HUD (Pouch: 200)')
    parser.add_argument('--learn', action='store_true',
                        help='explore and update the readout from on-screen pollen reward (needs --input foreground)')
    parser.add_argument('--explore', type=float,
                        help='with --learn: scale on loaded movement preferences (lower = more random moves); '
                             'default 0.3 for the arena checkpoint, 1 (unchanged) for others')
    parser.add_argument('--record', action='store_true',
                        help='you play; save frames as brain features plus your allowlisted keys (sends no input)')
    parser.add_argument('--fit', nargs='+', metavar='DEMOS',
                        help='train the readout on recorded demonstrations.npz files, then exit')
    parser.add_argument('--fit-steps', type=int, default=1200)
    parser.add_argument('--field', help='with --record: the field you collect in this session, e.g. dandelion')
    parser.add_argument('--places', default='models/bss-places.npz', help='place recognizer (hive/path/fields)')
    parser.add_argument('--fit-places', nargs='+', metavar='DEMOS',
                        help='train the place recognizer on recordings that include frames, then exit')
    parser.add_argument('--memory', default='models/bss-field-memory.json',
                        help='remembered field positions relative to your hive pad (read, and written by live runs)')
    parser.add_argument('--fit-nav', action='store_true',
                        help='with --fit: also fit cue-following on the recordings (only sound if you never turned the camera in them)')
    parser.add_argument('--no-auto-convert', action='store_true',
                        help='do not press E by itself on the hive pad with a half-full bag (the brain must learn to)')
    args = parser.parse_args(argv)
    if args.record and not args.field:
        parser.error('--record needs --field NAME (the field you will collect in)')
    default_checkpoint = parser.get_default('checkpoint')
    if args.explore is None:
        args.explore = .3 if args.checkpoint == default_checkpoint else 1.
    if not 0 <= args.explore <= 1:
        parser.error('--explore must be between 0 and 1')
    if args.learn and args.input != 'foreground':
        parser.error('--learn needs --input foreground: unexecuted actions cannot be credited')
    if args.record and (args.learn or args.input != 'none'):
        parser.error('--record sends no input; do not combine it with --learn or --input')
    if args.capacity <= 0:
        parser.error('--capacity must be positive')
    session = Path('runs/roblox') / time.strftime('%Y%m%dT%H%M%S')
    if args.fit_places:
        fit_places(args.fit_places, args.places)
        return
    from .arena.brain import load_graph, NeuralController
    from .arena.perception import PerceptionModel
    from . import nav, places
    place_model = places.PlaceModel.load(args.places) if Path(args.places).exists() else None
    brain = NeuralController(load_graph(), perception=PerceptionModel.load(), learning=args.learn,
                             sample_actions=args.learn)
    brain.load(args.checkpoint, allow_migration=True)

    def inventory_for(fill, small, smoother, pad=False):
        if place_model is None:
            return None if fill is None else [fill, 0.]  # honey not read yet
        return places.status(fill, smoother.update(place_model.probabilities(small)), place_model, pad)

    memory = nav.open_memory(args.memory)
    compass = nav.Compass(memory)
    if not brain.nav_weights.any():
        brain.nav_weights[:] = nav.steering_prior()  # start by following the cue; learning refines it

    if args.fit:
        features, labels, cues = [], [], []
        for path in args.fit:
            with np.load(path) as data:
                inputs = [{n for n, on in zip(RECORD_ORDER, row) if on} for row in data['inputs']]
                if 'frames' in data:  # replay so features include the place recognizer, pad and goal cue
                    brain.reset_dynamics()
                    smoother, watch, compass = places.Smoother(), PromptWatch(), nav.Compass(memory)
                    previous, last_fill = set(), None
                    for small, fill, keys in zip(data['frames'], data['fills'], inputs):
                        fill = None if np.isnan(fill) else float(fill)
                        pad = watch.update(honey_prompt(small))[0] is not None
                        collected = pollen_events(last_fill, fill, args.capacity)[0] if 'mouse' in previous else 0.
                        compass.update(previous, False, fill, pad, collected, str(data['field']))
                        cues.append(compass.cue())
                        brain.act(small, seconds=.25, inventory=inventory_for(fill, small, smoother, pad),
                                  goal_cue=cues[-1])
                        features.append(brain.last_features.copy())
                        previous, last_fill = keys, fill
                else:
                    features += list(data['features'])
                    cues += [np.zeros(4)] * len(inputs)
                labels += [action_from_inputs(keys) for keys in inputs]
        paired, labels = balance_walking(list(zip(features, cues)), labels)
        features, cues = [p[0] for p in paired], [p[1] for p in paired]
        print('Fit:', brain.fit_demonstrations(features, labels, steps=args.fit_steps), flush=True)
        if args.fit_nav:
            print('Navigation:', brain.fit_navigation(features, cues, labels, steps=args.fit_steps), flush=True)
        brain.save(session / 'readout.npz')
        print(f'Saved readout trained on {len(labels)} recorded decisions to {session / "readout.npz"}', flush=True)
        return
    window = RobloxWindow()
    if args.learn:
        # Arena habits are near-certain ("forward"); soften movement choices so learning can explore.
        brain.weights[:9] *= args.explore
    control = None if args.input == 'none' else SandboxedInput(window)
    recorded_features, recorded_inputs, recorded_frames, recorded_fills = [], [], [], []
    mode = f'recording you in {args.field}' if args.record else f'input {args.input}, learning {args.learn}'
    places_note = 'none yet' if place_model is None else ', '.join(place_model.classes)
    print(f'Roblox window found ({mode}). Places known: {places_note}. Press F8 to stop.', flush=True)
    place = '-'
    smoother = places.Smoother()
    reflex = WallReflex()
    reflexed = False  # was the previous decision the reflex's, not the brain's?
    watch, prompt, arrived, e_sent = PromptWatch(), None, False, False
    compass.reset()
    end = time.monotonic() + args.seconds
    fill = raw = view = None
    wanted = set()
    totals = [0., 0., 0]
    try:
        window.u32.GetAsyncKeyState(STOP_KEY)  # clear any earlier press
        # 0x8001: held now, or tapped since the last check (a quick tap falls between checks).
        while time.monotonic() < end and not window.u32.GetAsyncKeyState(STOP_KEY) & 0x8001:
            tick = time.monotonic()
            frame = window.capture()
            raw_before, raw = raw, read_pollen_fill(frame)
            previous, fill = fill, confirmed_fill(raw_before, raw, fill)
            collected, converted = pollen_events(previous, fill, args.capacity)
            before, view = view, frame[::8, ::8]
            blocked = stuck(before, view, wanted)
            totals[0] += collected
            totals[1] += converted
            totals[2] += blocked
            prompt, arrived = watch.update(honey_prompt(frame))
            arrived = arrived and fill is not None and fill >= .5  # home with a mostly full bag
            if args.learn and window.focused() and not reflexed:
                # Credited to the previous decision; applied inside the next act().
                brain.queue_feedback(**reward_events(collected, converted, blocked, wanted, arrived))
            small = brain_frame(frame)
            inventory = inventory_for(fill, small, smoother, prompt is not None)
            if place_model is not None:
                place = 'hive' if prompt else place_model.classes[int(np.argmax(smoother.probs))]
            # Where the last decision's keys took it, then the cue: home when nearly full, else its field.
            compass.update(wanted, blocked, fill, prompt is not None, collected if 'mouse' in wanted else 0., place)
            action = brain.act(small, seconds=.25, inventory=inventory, goal_cue=compass.cue())
            wanted = inputs_for(action)
            if prompt is None:
                e_sent = False
            if 'e' in wanted and prompt != 'make':
                wanted = wanted - {'e'}  # E only starts honey on the pad; elsewhere, or mid-conversion, it would do nothing or stop it
            elif 'e' in wanted:
                e_sent = True
            elif (prompt == 'make' and not e_sent and not args.no_auto_convert
                  and fill is not None and fill >= .5):
                wanted = wanted | {'e'}  # engineered: start honey once per pad visit with a half-full bag
                e_sent = True
            escape = reflex.step(blocked) if control is not None else None
            reflexed = escape is not None
            if reflexed:
                wanted = escape
            if args.record:
                wanted = held_inputs(window.u32)  # the player's keys label this frame's features
                if window.focused():
                    recorded_features.append(brain.last_features.copy())
                    recorded_inputs.append([n in wanted for n in RECORD_ORDER])
                    recorded_frames.append(small)
                    recorded_fills.append(np.nan if fill is None else fill)
            if control is not None:
                control.apply(wanted)
            shown = 'no HUD' if fill is None else f'{fill * args.capacity:.0f}/{args.capacity}'
            print(f'{time.strftime("%H:%M:%S")} pollen={shown} place={place} inputs={sorted(wanted)} '
                  f'pad={prompt} goal={"home" if compass.delivering else "field"} '
                  f'at=({compass.position[0]:.0f},{compass.position[1]:.0f}) '
                  f'focused={window.focused()} stuck={blocked} reflex={reflexed} collected={totals[0]:.0f} '
                  f'converted={totals[1]:.0f} stuck_total={totals[2]}', flush=True)
            time.sleep(max(0., .25 - (time.monotonic() - tick)))
    finally:
        if control is not None:
            control.release()
        if args.learn:
            brain.save(session / 'readout.npz')
            print(f'Saved learned readout to {session / "readout.npz"}', flush=True)
        if memory.nodes:
            memory.save(args.memory)
        if recorded_features:
            session.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(session / 'demonstrations.npz', features=np.array(recorded_features),
                                inputs=np.array(recorded_inputs, dtype=bool), frames=np.array(recorded_frames),
                                fills=np.array(recorded_fills), field=np.array(args.field.lower()))
            print(f'Saved {len(recorded_features)} recorded decisions to {session / "demonstrations.npz"}', flush=True)
        print('Stopped; all inputs released.', flush=True)


if __name__ == '__main__':
    sys.exit(main())
