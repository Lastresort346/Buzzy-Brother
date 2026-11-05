# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 FlyBrain contributors
"""Fixed-step simulation, independent of the renderer and biological network."""
from dataclasses import dataclass, asdict
import math
import random
from .economy import HoneyShop


@dataclass(frozen=True)
class ArenaConfig:
    seed: int = 42
    dt: float = 1 / 60
    half_size: float = 24
    speed: float = 6
    radius: float = .42
    height: float = 1.55
    jump_speed: float = 7
    gravity: float = 20
    dipper_cooldown: float = .8
    dipper_radius: float = 2
    camera_locked: bool = True
    camera_distance: float = 18
    camera_pitch: float = 34
    min_camera_distance: float = 8
    max_camera_distance: float = 20
    episode_seconds: float = 600
    spawn_x: float = 0
    spawn_y: float = -14
    pollen_capacity: int = 300
    conversion_rate: float = 60
    honey_per_pollen: int = 1
    hive_x: float = -9
    hive_y: float = -17
    conversion_radius: float = 2.5
    conversion_step_seconds: float = .25
    simulated_bees: int = 5
    sprout_seconds: float = 0.  # >0 spawns a sprout event this often (0 = events only when started)
    sprout_tokens: int = 12
    sprout_pollen: int = 100  # pollen collected inside the field to harvest the sprout
    sprout_timeout: float = 90.  # an unharvested sprout fades after this long
    token_radius: float = 1.3
    token_value: int = 20  # honey per token
    token_seconds: float = 15.

    def __post_init__(self):
        for name in ('dt', 'half_size', 'speed', 'radius', 'height', 'jump_speed',
                     'gravity', 'dipper_cooldown', 'dipper_radius', 'episode_seconds',
                     'conversion_rate','conversion_radius','conversion_step_seconds'):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not (0 < self.min_camera_distance <= self.camera_distance <= self.max_camera_distance):
            raise ValueError('Camera must stay within strictly third-person distance limits')
        if not 15 <= self.camera_pitch <= 65:
            raise ValueError('Camera pitch must be between 15 and 65 degrees')
        if self.dt > 1 / 30:
            raise ValueError('Use a timestep at or below 1/30 second for collision stability')
        if not all(math.isfinite(v) and abs(v)<self.half_size-self.radius for v in (self.spawn_x,self.spawn_y)):
            raise ValueError('Spawn must be finite and inside the arena')
        if not isinstance(self.pollen_capacity,int) or self.pollen_capacity<1:
            raise ValueError('Capacity must be a positive integer')
        if not isinstance(self.honey_per_pollen,int) or self.honey_per_pollen<1:
            raise ValueError('Honey ratio must be a positive integer')
        if not isinstance(self.simulated_bees,int) or self.simulated_bees<1:
            raise ValueError('Simulated bee count must be a positive integer')


@dataclass(frozen=True)
class Action:
    # Standard camera-relative controls: strafe left/right, backward/forward.
    strafe: float = 0
    forward: float = 0
    jump: bool = False
    dip: bool = False
    camera_yaw: float = 0  # degrees per step, ignored in the locked stage
    camera_pitch: float = 0
    zoom: float = 0  # distance delta; strictly third-person when unlocked
    convert: bool = False
    buy: int = 0

    def __post_init__(self):
        if not isinstance(self.buy,int) or not 0<=self.buy<=3:
            raise ValueError('buy must be 0 (none) or item slot 1, 2, 3')
        for key in ('strafe', 'forward', 'camera_yaw', 'camera_pitch', 'zoom'):
            if not math.isfinite(getattr(self, key)):
                raise ValueError(f'Action {key} must be finite')


@dataclass(frozen=True)
class Obstacle:
    x: float
    y: float
    width: float
    depth: float
    height: float


@dataclass
class Flower:
    x: float
    y: float
    color: int
    pollen: float = 1


class ArenaWorld:
    # Fixed obstacles make resets reproducible; flower variation uses an isolated RNG.
    obstacles = (Obstacle(-4, -5, 3.5, 2, 1), Obstacle(5, -1, 2, 5, 1.4),
                 Obstacle(-15, 10, 3, 3, 1.7), Obstacle(14, 14, 4, 2, .8))
    # The first three are the original fields (kept first so seeds/layouts stay reproducible).
    # Later fields reuse the recognized flower colors, so identity must come from location.
    patches = ((-10, 1, 0, 'BLUE FIELD'), (10, 5, 1, 'WHITE FIELD'), (0, 15, 2, 'ROSE FIELD'),
               (18, -4, 0, 'EAST FIELD'), (10, -16, 1, 'SOUTH FIELD'), (-17, 18, 2, 'NORTH-WEST FIELD'))

    def __init__(self, config=None):
        self.config = config or ArenaConfig()
        self.reset()

    def reset(self, seed=None, preserve_economy=False):
        wallet=getattr(self,'honey',0) if preserve_economy else 0
        earned=getattr(self,'honey_earned',0) if preserve_economy else 0
        shop=getattr(self,'shop',None) if preserve_economy else None
        self.seed = self.config.seed if seed is None else seed
        rng = random.Random(self.seed)
        self.flowers = [Flower(cx + (i - 3) * 1.05 + rng.uniform(-.12, .12),
                               cy + (j - 3) * 1.05 + rng.uniform(-.12, .12), color)
                        for cx, cy, color, _ in self.patches for i in range(7) for j in range(7)]
        self.x, self.y, self.z = float(self.config.spawn_x), float(self.config.spawn_y), 0.
        self.vz = 0.
        self.heading = 0.
        self.grounded = True
        self.tick = 0
        self.pollen = 0
        self.total_collected = 0
        self.honey = wallet
        self.honey_earned = earned
        self.shop = shop if shop is not None else HoneyShop()
        self.previous_buy = 0
        self.last_purchase_utility = 0.
        self.last_purchase_status = 'none'
        self.total_converted = 0
        self.last_converted = 0
        self.conversion_credit = 0.
        self.conversion_active = False
        self.conversion_timer = 0.
        self.previous_convert = False
        self.conversion_steps = 0
        self.last_conversion_bee = 0
        self.last_conversion_batch = 0
        self.conversion_status = 'idle'
        self.swings = 0
        self.successful_swings = 0
        self.cooldown = 0.
        self.previous_jump = False
        self.camera_yaw = 0.
        self.camera_pitch = self.config.camera_pitch
        self.camera_distance = self.config.camera_distance
        self.last_gain = 0
        self.last_color_gain = {}  # pollen just collected per flower color index (what the dipper touched)
        self.last_swing_tick = -1000
        # Sprout events: a sprout appears in a field and drops collectible tokens. Tokens are only
        # picked up while `sprout_orders` is set (the trainer's instruction); otherwise they are ignored.
        self.sprout = None
        self.tokens = []
        self.sprout_orders = getattr(self, 'sprout_orders', False)
        self.event_rng = random.Random(self.seed * 7919 + 1)
        self.next_sprout = self.config.sprout_seconds
        self.sprouts_started = 0
        self.token_honey_total = 0
        self.last_token_honey = 0
        self.speed_fraction = 0.
        return self.telemetry()

    def start_sprout(self, field=None):
        """Begin a sprout event in a named (or random) field."""
        names = [p[3] for p in self.patches]
        name = names[self.event_rng.randrange(len(names))] if field is None else field
        cx, cy = next(p[:2] for p in self.patches if p[3] == name)
        self.sprout = {'field': name, 'x': cx, 'y': cy, 'age': 0., 'spawned': 0, 'progress': 0, 'harvested': False,
                       'drop_age': 0.}
        self.tokens = []
        self.sprouts_started += 1

    def credit_sprout(self, gained):
        """Pollen collected inside the sprout's field (not merely near it) counts toward harvesting it."""
        s = self.sprout
        if s is None or s['harvested'] or not self.sprout_orders or gained <= 0:
            return
        if max(abs(self.x - s['x']), abs(self.y - s['y'])) <= 4.5:
            s['progress'] += gained

    def update_sprout(self):
        c = self.config
        if c.sprout_seconds and self.sprout is None and self.tick * c.dt >= self.next_sprout:
            self.start_sprout()
        s = self.sprout
        if s is not None:
            s['age'] += c.dt
            if not s['harvested'] and s['progress'] >= c.sprout_pollen:
                s['harvested'] = True
            if not s['harvested'] and s['age'] >= c.sprout_timeout:
                self.sprout = s = None
                self.next_sprout = self.tick * c.dt + c.sprout_seconds
        if s is not None and s['harvested']:
            s['drop_age'] += c.dt
            if s['spawned'] < c.sprout_tokens and s['drop_age'] >= .5 * (s['spawned'] + 1):
                s['spawned'] += 1
                self.tokens.append({'x': s['x'] + self.event_rng.uniform(-3.6, 3.6),
                                    'y': s['y'] + self.event_rng.uniform(-3.6, 3.6), 'ttl': c.token_seconds})
        for t in self.tokens:
            t['ttl'] -= c.dt
        self.tokens = [t for t in self.tokens if t['ttl'] > 0]
        self.last_token_honey = 0
        if self.sprout_orders and self.z < .6:
            near = [t for t in self.tokens if math.hypot(t['x'] - self.x, t['y'] - self.y) <= c.token_radius]
            if near:
                self.tokens = [t for t in self.tokens if t not in near]
                self.last_token_honey = len(near) * c.token_value
                self.honey += self.last_token_honey
                self.honey_earned += self.last_token_honey
                self.token_honey_total += self.last_token_honey
        if s is not None and s['harvested'] and s['spawned'] >= c.sprout_tokens and not self.tokens:
            self.sprout = None
            self.next_sprout = self.tick * c.dt + c.sprout_seconds

    def overlaps(self, x, y, obstacle):
        # Circle against rectangle footprint, avoiding invisible square corner collisions.
        qx = max(abs(x - obstacle.x) - obstacle.width / 2, 0)
        qy = max(abs(y - obstacle.y) - obstacle.depth / 2, 0)
        return qx * qx + qy * qy < self.config.radius ** 2

    def blocked(self, x, y):
        return any(self.overlaps(x, y, o) and self.z < o.height - .001 for o in self.obstacles)

    def step(self, action=Action()):
        if not isinstance(action, Action):
            raise TypeError('Expected an Action')
        c = self.config
        self.tick += 1
        if not c.camera_locked:
            self.camera_yaw = (self.camera_yaw + max(-8, min(8, action.camera_yaw))) % 360
            self.camera_pitch = max(15, min(65, self.camera_pitch + max(-5, min(5, action.camera_pitch))))
            self.camera_distance = max(c.min_camera_distance, min(c.max_camera_distance,
                                                                 self.camera_distance + action.zoom))
        forward = max(-1, min(1, action.forward))
        strafe = max(-1, min(1, action.strafe))
        norm = max(1, math.hypot(forward, strafe))
        forward /= norm
        strafe /= norm
        yaw = math.radians(self.camera_yaw)
        dx = (strafe * math.cos(yaw) - forward * math.sin(yaw)) * c.speed * c.dt
        dy = (strafe * math.sin(yaw) + forward * math.cos(yaw)) * c.speed * c.dt
        self.speed_fraction = math.hypot(dx, dy) / (c.speed * c.dt)
        if dx or dy:
            self.heading = math.degrees(math.atan2(-dx, dy))
        if action.jump and not self.previous_jump and self.grounded:
            self.vz = c.jump_speed
            self.grounded = False
        self.previous_jump = action.jump
        boundary = c.half_size - c.radius
        nx = max(-boundary, min(boundary, self.x + dx))
        if not self.blocked(nx, self.y):
            self.x = nx
        ny = max(-boundary, min(boundary, self.y + dy))
        if not self.blocked(self.x, ny):
            self.y = ny
        floor = max([0.] + [o.height for o in self.obstacles
                            if self.overlaps(self.x, self.y, o) and self.z >= o.height - .001])
        old_z = self.z
        self.vz -= c.gravity * c.dt
        self.z += self.vz * c.dt
        if self.z <= floor and old_z >= floor:
            self.z = floor
            self.vz = 0
            self.grounded = True
        else:
            self.grounded = False
        for flower in self.flowers:
            flower.pollen = min(1., flower.pollen + c.dt / 12)
        self.update_sprout()
        self.cooldown = max(0., self.cooldown - c.dt)
        self.last_gain = 0
        self.last_color_gain = {}
        swung = False
        if action.dip and self.cooldown <= 1e-8:
            swung = True
            self.swings += 1
            self.cooldown = c.dipper_cooldown
            self.last_swing_tick = self.tick
            if self.z < .35:
                # The dipper collects in a small forward-facing footprint, not through a remote action.
                h = math.radians(self.heading)
                center_x, center_y = self.x - .8 * math.sin(h), self.y + .8 * math.cos(h)
                for flower in self.flowers:
                    if math.hypot(flower.x - center_x, flower.y - center_y) <= c.dipper_radius and flower.pollen >= .25:
                        gained = min(int(flower.pollen * 10),c.pollen_capacity-self.pollen-self.last_gain)
                        if gained<=0:
                            continue
                        self.last_gain += gained
                        self.last_color_gain[flower.color] = self.last_color_gain.get(flower.color, 0) + gained
                        flower.pollen = max(0.,flower.pollen-gained/10)
                self.pollen += self.last_gain
                self.total_collected += self.last_gain
                self.successful_swings += self.last_gain > 0
                self.credit_sprout(self.last_gain)
        self.last_converted=0
        if action.convert and not self.previous_convert and self.in_conversion_zone():
            if self.conversion_active:
                self.conversion_active=False
                self.conversion_status='stopped'
            elif self.pollen>0:
                self.conversion_active=True
                self.conversion_status='converting'
        self.previous_convert=bool(action.convert)
        if self.conversion_active and not self.in_conversion_zone():
            self.conversion_active=False
            self.conversion_status='left_hive'
        if self.conversion_active and self.pollen<=0:
            self.conversion_active=False
            self.conversion_status='backpack_empty'
        if self.conversion_active:
            self.conversion_timer+=c.dt
            while self.conversion_timer+1e-9>=c.conversion_step_seconds and self.pollen>0:
                self.conversion_timer=max(0.,self.conversion_timer-c.conversion_step_seconds)
                self.conversion_credit+=c.conversion_rate*c.conversion_step_seconds
                batch=min(self.pollen,int(self.conversion_credit+1e-8))
                self.conversion_credit-=batch
                self.pollen-=batch
                self.honey+=batch*c.honey_per_pollen
                self.honey_earned+=batch*c.honey_per_pollen
                self.total_converted+=batch
                self.last_converted+=batch
                self.last_conversion_bee=self.conversion_steps%c.simulated_bees+1
                self.last_conversion_batch=batch
                self.conversion_steps+=1
            if self.pollen==0:
                self.conversion_active=False
                self.conversion_status='backpack_empty'
        if not self.conversion_active:
            self.conversion_timer=0.
            self.conversion_credit=0.
        self.last_purchase_utility=0.
        if action.buy and action.buy!=self.previous_buy:
            if self.in_conversion_zone():
                self.honey,self.last_purchase_utility,self.last_purchase_status=self.shop.buy(action.buy,self.honey)
            else:
                self.last_purchase_status='not_at_shop'
        self.previous_buy=action.buy
        truncated = self.tick * c.dt >= c.episode_seconds
        info = self.telemetry()
        info.update(swung=swung, collected=self.last_gain,converted_pollen=self.last_converted,
                    honey_gain=self.last_converted*c.honey_per_pollen+self.last_token_honey,token_honey=self.last_token_honey,purchase_utility=self.last_purchase_utility)
        return float(self.last_gain+2*(self.last_converted*c.honey_per_pollen+self.last_token_honey)+100*self.last_purchase_utility), False, truncated, info

    def in_conversion_zone(self):
        return self.grounded and self.z<.35 and math.hypot(self.x-self.config.hive_x,self.y-self.config.hive_y)<=self.config.conversion_radius

    def camera_pose(self):
        yaw, pitch = math.radians(self.camera_yaw), math.radians(self.camera_pitch)
        distance = self.camera_distance
        target = (self.x, self.y, self.z + 1.0)
        eye = (self.x + math.sin(yaw) * math.cos(pitch) * distance,
               self.y - math.cos(yaw) * math.cos(pitch) * distance,
               self.z + 1.0 + math.sin(pitch) * distance)
        return eye, target

    def telemetry(self):
        return {'position': [self.x, self.y, self.z], 'grounded': self.grounded,
                'heading_degrees': self.heading, 'pollen': self.pollen,
                'pollen_capacity':self.config.pollen_capacity,'backpack_full':self.pollen>=self.config.pollen_capacity,
                'total_collected':self.total_collected,'honey':self.honey,'honey_earned':self.honey_earned,
                'total_converted':self.total_converted,'shop':self.shop.snapshot(self.honey),
                'last_purchase_status':self.last_purchase_status,
                'sprout_field':self.sprout['field'] if self.sprout else None,
                'sprout_progress':self.sprout['progress'] if self.sprout else 0,
                'sprout_harvested':bool(self.sprout and self.sprout['harvested']),'tokens':len(self.tokens),
                'token_honey_total':self.token_honey_total,'sprout_orders':self.sprout_orders,
                'in_conversion_zone':self.in_conversion_zone(),'converting':self.conversion_active,
                'conversion_status':self.conversion_status,'conversion_steps':self.conversion_steps,
                'conversion_step_progress':self.conversion_timer/self.config.conversion_step_seconds,
                'simulated_bees':self.config.simulated_bees,'last_conversion_bee':self.last_conversion_bee,
                'last_conversion_batch':self.last_conversion_batch,
                'dipper_ready': self.cooldown <= 1e-8, 'cooldown_seconds': self.cooldown,
                'tick': self.tick, 'simulation_seconds': self.tick * self.config.dt,
                'seed': self.seed, 'swings': self.swings, 'successful_swings': self.successful_swings,
                'camera_locked': self.config.camera_locked, 'camera_distance': self.camera_distance,
                'camera_yaw_degrees': self.camera_yaw, 'camera_pitch_degrees': self.camera_pitch}

    def configuration(self):
        return asdict(self.config)
