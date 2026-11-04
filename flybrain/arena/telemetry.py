# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 FlyBrain contributors
"""Observer-only behavioral measurements; does not feed coordinates to the policy."""
import math


class BehaviorTelemetry:
    def __init__(self):
        self.escape_attempts=0
        self.episodes=1
        self.reset_episode(increment=False)

    def reset_episode(self,increment=True):
        self.episode_escapes=0
        self.outward_seconds=0.
        self.escape_latched=False
        self.blocked_fraction=0.
        self.recent_reward=0.
        self.collect_seconds=0.
        self.purchase_seconds=0.
        self.simulation_seconds=0.
        self.state='EXPLORING'
        self.happiness=50.
        if increment:
            self.episodes+=1

    def observe(self,before,world,action,reward):
        dt=world.config.dt
        self.simulation_seconds+=dt
        moving=bool(action.strafe or action.forward)
        moved=math.hypot(world.x-before[0],world.y-before[1])
        blocked=moving and moved<world.config.speed*dt*.1
        alpha=1-math.exp(-dt/2)
        self.blocked_fraction+=(float(blocked)-self.blocked_fraction)*alpha
        yaw=math.radians(world.camera_yaw)
        dx=action.strafe*math.cos(yaw)-action.forward*math.sin(yaw)
        dy=action.strafe*math.sin(yaw)+action.forward*math.cos(yaw)
        limit=world.config.half_size-world.config.radius
        outward=((world.x>=limit-.05 and dx>.1) or (world.x<=-limit+.05 and dx<-.1) or
                 (world.y>=limit-.05 and dy>.1) or (world.y<=-limit+.05 and dy<-.1))
        if outward:
            self.outward_seconds+=dt
            if self.outward_seconds>=.5 and not self.escape_latched:
                self.escape_attempts+=1
                self.episode_escapes+=1
                self.escape_latched=True
        else:
            self.outward_seconds=0
            self.escape_latched=False
        self.recent_reward*=math.exp(-dt/10)
        self.recent_reward+=max(0,reward)/100
        self.collect_seconds=2 if reward>0 and world.last_converted==0 else max(0,self.collect_seconds-dt)
        self.purchase_seconds=2 if world.last_purchase_utility>0 else max(0,self.purchase_seconds-dt)
        self.happiness=max(0,min(100,50+45*math.tanh(self.recent_reward)-25*self.blocked_fraction))
        self.state=('PURCHASED' if self.purchase_seconds>0 else 'CONVERTING' if world.conversion_active else 'COLLECTING' if self.collect_seconds>0 else 'BLOCKED' if self.blocked_fraction>.5
                    else 'JUMPING' if not world.grounded else 'EXPLORING' if moving else 'IDLE')

    def snapshot(self):
        return {'state':self.state,'happiness_proxy':self.happiness,'recent_reward_trace':self.recent_reward,
                'blocked_fraction':self.blocked_fraction,'escape_attempts':self.escape_attempts,
                'episode_escape_attempts':self.episode_escapes,'episodes':self.episodes,
                'escape_definition':'>=0.5 seconds of continuous outward movement against an outer wall; rearmed when outward input stops',
                'happiness_definition':'clip(50 + 45*tanh(10-second decayed environment-reward/100 trace) - 25*2-second blocked EMA, 0, 100)'}
