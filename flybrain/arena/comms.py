# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 FlyBrain contributors
"""Read-only communication center: preset sentences chosen from measured state.

Nothing here feeds the controller or changes training. Each message is a fixed template
plus the numbers that triggered it. "I want X" is shorthand for a measured priority
(highest current utility per price among in-stock items), not a feeling or a decision
the neural controller made. Inputs are the public status (pack, wallet, shop), the
bee's own odometry memory and goal, and history tracked here. The only world read is
the colour of the flower under the dipper when pollen arrives (contact sensing).
"""
import json
import math
from collections import deque

from .economy import CATALOG

PRESETS = {
    'want_item': 'I want a {item}.',
    'saving_for': 'I am saving honey for a {item}.',
    'bought': 'I bought a {item}.',
    'low_honey': 'My honey is low.',
    'bag_full': 'My bag is full.',
    'bag_empty': 'My bag is empty.',
    'stuck': 'I am stuck.',
    'color_pref': 'I prefer {color} flowers.',  # preference = most pollen collected, not a feeling
    'new_field': 'I need to find a new field.',
    'go_field': 'I want to go to the {field}.',
    'name_field': 'I found a field that needs a name.',
    'wants_color': 'I want {color} flowers.',  # set by the trainer through the thinking center
}
COLORS = {0: 'blue', 1: 'white', 2: 'rose'}  # flower colour ids used by ArenaWorld.patches
COOLDOWN = 10.   # simulated seconds before the same preset may be said again
MIN_GAP = 1.     # simulated seconds between any two messages
STUCK_SECONDS, STUCK_MOVE = 3., .5
PREF_MIN_POLLEN, PREF_MARGIN = 100, 1.2  # evidence needed before a colour preference is reported


class CommunicationCenter:
    def __init__(self):
        self.pollen_by_color = {}
        self.log = deque(maxlen=50)
        self._said = {}   # id -> state key, for presets whose condition still holds
        self._cool = {}   # id -> time last said (survives the condition ending)
        self._last_t = -1e9
        self._total = 0
        self._owned = self._now_owned = None
        self._anchor = None

    def latest(self, n=4):
        return list(self.log)[-n:]

    def report(self, app):
        """update() plus print and session telemetry; the single hook the app calls."""
        msgs = self.update(app)
        for m in msgs:
            print(f'[bee] {m["text"]}  {m["evidence"]}', flush=True)
            if getattr(app, '_brain_log', None) is not None:
                app._brain_log.write(json.dumps({'simulation_seconds': m['t'], 'message': m}) + '\n')
        return msgs

    def _track(self, w):
        gain, self._total = w.total_collected - self._total, w.total_collected  # negative after a world reset
        if gain > 0 and w.flowers:
            h = math.radians(w.heading)
            cx, cy = w.x - .8 * math.sin(h), w.y + .8 * math.cos(h)
            f = min(w.flowers, key=lambda f: math.hypot(f.x - cx, f.y - cy))
            self.pollen_by_color[f.color] = self.pollen_by_color.get(f.color, 0) + gain

    def _stuck(self, app, t):
        m, a = getattr(app, 'memory', None), getattr(app, '_brain_action', None)
        if m is None or a is None or not (a.forward or a.strafe) or app.world.conversion_active:
            self._anchor = None
            return None
        if self._anchor is None or math.hypot(m.x - self._anchor[0], m.y - self._anchor[1]) > STUCK_MOVE:
            self._anchor = (m.x, m.y, t)
        return t - self._anchor[2] if t - self._anchor[2] >= STUCK_SECONDS else None

    def conditions(self, app, t):
        """{preset id: (state key, template fields, evidence)} for every condition now true."""
        w, out = app.world, {}
        shop = w.shop.snapshot(w.honey)
        items = shop['items']
        if w.pollen >= w.config.pollen_capacity:
            out['bag_full'] = (None, {}, {'pollen': w.pollen, 'capacity': w.config.pollen_capacity})
        elif w.pollen == 0 and not w.conversion_active:
            out['bag_empty'] = (None, {}, {'pollen': 0})
        stock = [i for i in items if i['stock'] != 0]
        for i in stock:  # current marginal utility per honey, as in HoneyShop.buy
            item = CATALOG[i['slot'] - 1]
            u = item.utility / (1 + .3 * i['owned']) if item.stock is None else item.utility
            i['upp'] = u / i['price']
        if stock:
            ev = lambda i: {'item': i['name'], 'price': i['price'], 'honey': w.honey, 'stock': i['stock'],
                            'owned': i['owned'], 'utility_per_price': round(i['upp'], 4)}
            ok = [i for i in stock if i['affordable']]
            if ok:
                b = max(ok, key=lambda i: i['upp'])
                out['want_item'] = (b['name'], {'item': b['name'].lower()}, ev(b))
            else:
                b = max(stock, key=lambda i: i['upp'])
                out['saving_for'] = (b['name'], {'item': b['name'].lower()}, ev(b))
            cheapest = min(i['price'] for i in stock)
            if w.honey < cheapest:
                out['low_honey'] = (None, {}, {'honey': w.honey, 'cheapest_price': cheapest})
        owned = [i['owned'] for i in items]
        if self._owned is not None and owned != self._owned:
            n = next(k for k, (a, b) in enumerate(zip(owned, self._owned)) if a != b)
            out['bought'] = (shop['purchases'], {'item': items[n]['name'].lower()},
                             {'item': items[n]['name'], 'purchases': shop['purchases'], 'honey_spent': shop['honey_spent']})
        self._now_owned = owned
        total = sum(self.pollen_by_color.values())
        if total >= PREF_MIN_POLLEN and len(self.pollen_by_color) >= 2:
            r = sorted(self.pollen_by_color.items(), key=lambda kv: -kv[1])
            if r[0][1] >= PREF_MARGIN * r[1][1]:
                out['color_pref'] = (r[0][0], {'color': COLORS.get(r[0][0], f'colour {r[0][0]}')},
                                     {'pollen_by_color': {COLORS.get(k, k): v for k, v in r}, 'total': total,
                                      'note': 'preference = highest pollen yield'})
        m, goal = getattr(app, 'memory', None), getattr(app, 'goal', None)
        mind = getattr(app, 'mind', None)
        if mind is not None and mind.want_field and not goal:
            goal = mind.want_field
        if mind is not None and mind.want_color and m is not None:
            out['wants_color'] = (mind.want_color, {'color': mind.want_color},
                                  {'want_color': mind.want_color, 'best_known_field': m.best_field_for(mind.want_color),
                                   'note': 'a want set by the trainer; steers toward the best-yield field and boosts that color in vision'})
        if m is not None:
            node = m.recall(goal) if goal and goal != m.DISCOVER else None
            if node:
                out['go_field'] = (goal, {'field': goal.title()},
                                   {'goal': goal, 'remembered_visits': node['n'],
                                    'odometry_distance': round(math.hypot(node['x'] - m.x, node['y'] - m.y), 1)})
            elif goal:  # DISCOVER, or a name memory has no node for
                out['new_field'] = (goal, {}, {'goal': goal, 'remembered_fields': sum(1 for n in m.nodes if n['name'])})
            d = m.discoveries()
            if d:
                out['name_field'] = (len(d), {}, {'unnamed_fields': len(d)})
        s = self._stuck(app, t)
        if s is not None:
            out['stuck'] = (None, {}, {'seconds_without_progress': round(s, 1), 'min_move': STUCK_MOVE})
        return out

    def update(self, app):
        """Return the new messages (dicts: id, text, evidence, t) for this moment."""
        w = app.world
        t = w.tick * w.config.dt
        self._track(w)
        now = self.conditions(app, t)
        for k in [k for k in self._said if k not in now]:
            del self._said[k]  # condition ended: it may be said again after the cooldown
        out = []
        for k, (key, fields, ev) in now.items():
            if k in self._said and self._said[k] == key:
                continue
            if t - self._cool.get(k, -1e9) < COOLDOWN or t - self._last_t < MIN_GAP:
                continue
            m = {'id': k, 'text': PRESETS[k].format(**fields), 'evidence': ev, 't': round(t, 2)}
            self._said[k], self._cool[k], self._last_t = key, t, t
            self.log.append(m)
            out.append(m)
        if 'bought' not in now or any(m['id'] == 'bought' for m in out):
            self._owned = self._now_owned  # a purchase held back by a cooldown is retried
        return out
