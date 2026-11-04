# SPDX-License-Identifier: GPL-3.0-or-later
import unittest

import numpy as np

from flybrain.arena.world import Action
from flybrain.roblox import (ALLOWED, BAR_EMPTY, BAR_LEFT, BAR_RIGHT, PANEL_COLOR, PANEL_PROBE,
                             SandboxedInput, inputs_for, pollen_events, read_pollen_fill, stuck, confirmed_fill,
                             RECORD_ORDER, action_from_inputs, balance_walking, held_inputs, reward_events, WallReflex)


def hud_frame(fill):
    frame = np.zeros((300, 1920, 3), np.uint8)
    cx = 960
    frame[PANEL_PROBE[0], cx + PANEL_PROBE[1]] = PANEL_COLOR
    frame[20:60, cx + BAR_LEFT:cx + BAR_RIGHT] = BAR_EMPTY
    width = round((BAR_RIGHT - BAR_LEFT) * fill)
    frame[20:60, cx + BAR_LEFT:cx + BAR_LEFT + width] = (230, 190, 40)
    return frame


class PollenHudTests(unittest.TestCase):
    def test_reads_bar_fill(self):
        for fill in (0, .25, .5, 1):
            self.assertAlmostEqual(read_pollen_fill(hud_frame(fill)), fill, delta=.01)

    def test_missing_hud_returns_none(self):
        self.assertIsNone(read_pollen_fill(np.zeros((300, 1920, 3), np.uint8)))

    def test_one_frame_spikes_are_ignored(self):
        accepted = None
        readings = [.025, .025, .40, .025, .30, .34, .34]  # 5 -> spike 80 -> 5, then real collection
        seen = []
        for before, raw in zip([None] + readings, readings):
            accepted = confirmed_fill(before, raw, accepted)
            seen.append(accepted)
        self.assertEqual(seen, [None, .025, .025, .025, .025, .34, .34])
        growing = [.10, .12, .14, .16]  # steady growth of about 4 bar pixels per frame
        accepted = None
        for before, raw in zip([None] + growing, growing):
            accepted = confirmed_fill(before, raw, accepted)
        self.assertEqual(accepted, .16)

    def test_reward_ignores_bee_pollen_and_punishes_walls(self):
        self.assertEqual(reward_events(20, 0, False, {'mouse'})['reward'], 20)
        self.assertEqual(reward_events(20, 0, False, {'w'})['reward'], 0)  # bees gathered it
        self.assertEqual(reward_events(0, 30, False, set())['honey'], 30)  # converting needs no dip
        self.assertLess(reward_events(0, 0, True, {'w'})['progress'], 0)

    def test_wall_reflex_backs_up_then_sidesteps(self):
        reflex = WallReflex(patience=4, rng=np.random.default_rng(0))
        self.assertEqual([reflex.step(True) for _ in range(3)], [None] * 3)
        plan = [reflex.step(False) if i else reflex.step(True) for i in range(6)]
        self.assertEqual(plan[:2], [{'s'}, {'s'}])
        self.assertEqual(len({frozenset(x) for x in plan[2:]}), 1)
        self.assertLessEqual(plan[2], {'a', 'd'})
        self.assertIsNone(reflex.step(False))  # back to the brain
        for escape in plan:
            self.assertLessEqual(escape, ALLOWED.keys())

    def test_wall_reflex_needs_continuous_stuck(self):
        reflex = WallReflex(patience=4)
        self.assertEqual([reflex.step(b) for b in (True, True, True, False, True)], [None] * 5)

    def test_stuck_needs_movement_and_a_still_view(self):
        still = np.full((10, 10, 3), 100, np.uint8)
        moved = np.full((10, 10, 3), 140, np.uint8)
        self.assertTrue(stuck(still, still, {'w', 'mouse'}))
        self.assertFalse(stuck(still, moved, {'w'}))
        self.assertFalse(stuck(still, still, {'mouse'}))  # standing still dipping is not stuck
        self.assertFalse(stuck(None, still, {'w'}))

    def test_events_split_collection_and_conversion(self):
        self.assertEqual(pollen_events(.1, .6, 200), (100., 0.))
        self.assertEqual(pollen_events(.6, .1, 200), (0., 100.))
        self.assertEqual(pollen_events(.5, .501, 200), (0., 0.))
        self.assertEqual(pollen_events(None, .5, 200), (0., 0.))


class FakeUser32:
    def __init__(self):
        self.sent = []
        self.foreground = 1

    def GetForegroundWindow(self):
        return self.foreground

    def PostMessageW(self, hwnd, msg, wparam, lparam):
        self.sent.append((msg, wparam))

    def MapVirtualKeyW(self, vk, kind):
        return 0

    def keybd_event(self, vk, scan, flags, extra):
        self.sent.append(('key', vk, flags))

    def mouse_event(self, *args):
        self.sent.append(('mouse',) + args)

    def GetCursorPos(self, point):
        return True

    def WindowFromPoint(self, point):
        return 1


class FakeWindow:
    hwnd = 1

    def __init__(self):
        self.u32 = FakeUser32()

    def focused(self):
        return self.u32.foreground == self.hwnd

    def size(self):
        return 100, 80


class SandboxTests(unittest.TestCase):
    def test_every_action_maps_inside_allowlist(self):
        for strafe in (-1, 0, 1):
            for forward in (-1, 0, 1):
                action = Action(strafe=strafe, forward=forward, jump=True, dip=True, convert=True, buy=3, camera_yaw=5)
                self.assertLessEqual(inputs_for(action), ALLOWED.keys())
        self.assertEqual(inputs_for(Action(forward=1, strafe=-1)), {'w', 'a'})
        self.assertEqual(inputs_for(Action(buy=2)), set())

    def test_escape_and_other_keys_are_rejected(self):
        control = SandboxedInput(FakeWindow())
        for forbidden in ({'escape'}, {'alt', 'f4'}, {'w', 'tab'}):
            with self.assertRaises(ValueError):
                control.apply(forbidden)
        self.assertEqual(control.window.u32.sent, [])

    def test_foreground_mode_releases_when_focus_leaves(self):
        window = FakeWindow()
        control = SandboxedInput(window)
        control.apply({'w'})
        self.assertEqual(control.held, {'w'})
        window.u32.foreground = 2  # user switched to another app
        control.apply({'w', 'space'})
        self.assertEqual(control.held, set())
        self.assertEqual(window.u32.sent[-1], ('key', ALLOWED['w'], 0x2))

    def test_release_lifts_everything(self):
        control = SandboxedInput(FakeWindow())
        control.apply({'w', 'mouse', 'e'})
        control.release()
        self.assertEqual(control.held, set())

class RecordingTests(unittest.TestCase):
    def test_recorded_keys_round_trip_to_inputs(self):
        for held in ({'w', 'a'}, {'s', 'd', 'mouse'}, {'e'}, {'space', 'w'}, set()):
            self.assertEqual(inputs_for(action_from_inputs(held)), held)
        self.assertEqual(action_from_inputs({'w', 's'}).forward, 0)  # opposite keys cancel

    def test_held_inputs_reads_only_allowlisted_keys(self):
        class Keys:
            def GetAsyncKeyState(self, vk):
                return {0x57: 0x8000, 0x45: 0x0001, 0x01: 0x8000, 0x1B: 0x8000}.get(vk, 0)  # W held, E tapped, mouse, Esc
        self.assertEqual(held_inputs(Keys()), {'w', 'e', 'mouse'})
        self.assertEqual(set(RECORD_ORDER), set(ALLOWED))

    def test_balance_walking_matches_standing(self):
        labels = [Action()] * 6 + [Action(forward=1), Action(strafe=-1)]
        features, balanced = balance_walking(list(range(8)), labels)
        walking = sum(1 for a in balanced if a.forward or a.strafe)
        self.assertEqual((walking, len(balanced) - walking), (6, 6))
        self.assertEqual(len(features), len(balanced))
        self.assertEqual(balance_walking([1], [Action()]), ([1], [Action()]))  # nothing to repeat


if __name__ == '__main__':
    unittest.main()
