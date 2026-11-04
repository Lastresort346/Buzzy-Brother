# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 FlyBrain contributors
"""Playable wingless fly arena and pixel-based controller interface."""
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import time
from datetime import datetime, timezone

from panda3d.core import (AmbientLight, CardMaker, DirectionalLight,
                         Filename, Fog, TextNode, TransparencyAttrib,
                         loadPrcFileData)
from direct.showbase.ShowBase import ShowBase
from direct.gui.OnscreenText import OnscreenText

from .geometry import box_mesh, sphere_mesh, cylinder_mesh
from .world import Action, ArenaConfig, ArenaWorld


GRASS = (.36, .58, .32, 1)
LEAF = (.22, .40, .25, 1)
CREAM = (.94, .89, .72, 1)
CHARCOAL = (.16, .20, .21, 1)
FIELD_COLORS = ((.38,.56,.88,1),(.96,.93,.82,1),(.87,.44,.51,1))


class ArenaApp(ShowBase):
    """`step(Action, frames)` returns RGB pixels, collection reward, flags, and diagnostics.

    Diagnostics contain privileged world state and must not be fed to a vision-only
    policy. `connect_brain()` enables the experimental neural controller.
    """
    def __init__(self, config=None, offscreen=False, software=False, width=1280, height=800):
        loadPrcFileData('', '\n'.join([
            'window-title FlyBrain - Wingless Motor Arena', f'win-size {width} {height}',
            'audio-library-name null', 'sync-video true', 'show-frame-rate-meter false',
            'model-cache-dir', 'textures-power-2 none',
            'notify-level warning', 'framebuffer-multisample 1', 'multisamples 4',
            'load-display p3tinydisplay' if software else 'load-display pandagl',
            'window-type offscreen' if offscreen else 'window-type onscreen']))
        super().__init__(windowType='offscreen' if offscreen else None)
        if self.win is None:
            raise RuntimeError('No graphics buffer. Try --software for offscreen diagnostics.')
        self.disableMouse()
        self.world = ArenaWorld(config)
        self.brain=None
        self.monitor=None
        self.behavior=None
        self.brain_enabled=False
        self._brain_reward=0.
        self._brain_honey=0.
        self._brain_purchases=0.
        self._brain_full_wait_steps=0
        self._brain_progress=0.
        self.training_progress_shaping=False
        self._brain_empty_dips=0
        self._brain_blocked_steps=0
        self._brain_feedback_steps=0
        self.training_task='manual'
        self._brain_next_tick=0
        self._brain_action=Action()
        self._brain_log=None
        self._brain_session=None
        self._brain_last_logged=-1
        self._brain_checkpoint_bucket=0
        self.offscreen = offscreen
        self.paused = False
        self.keys = {}
        self._interact_pending=False
        self._purchase_pending=0
        self.accumulator = 0.
        self.last_time = time.perf_counter()
        self.screenshot_counter = 0
        self.observation_size = (640, 400)
        self._observation_buffer=None
        self._observation_camera=None
        self.box = box_mesh()
        self.sphere = sphere_mesh()
        self.cylinder = cylinder_mesh()
        self.setBackgroundColor(.73,.84,.86,1)
        self.camLens.setFov(68)
        self.camLens.setNearFar(.15,180)
        ambient = AmbientLight('sky-light')
        ambient.setColor((.67,.72,.72,1))
        self.render.setLight(self.render.attachNewNode(ambient))
        sunlight = DirectionalLight('sun')
        sunlight.setColor((.78,.73,.60,1))
        sun = self.render.attachNewNode(sunlight)
        sun.setHpr(-35,-52,0)
        self.render.setLight(sun)
        fog = Fog('distance-haze')
        fog.setColor(.73,.84,.86)
        fog.setLinearRange(48,105)
        self.render.setFog(fog)
        self.build_world()
        self.build_avatar()
        self.build_hud()
        self.bind_inputs()
        self.render_state()
        if not offscreen:
            self.taskMgr.add(self.interactive_update, 'arena-update')

    def shape(self, kind, parent, position, scale, color, heading=0):
        node = getattr(self,kind).copyTo(parent)
        node.setPos(*position)
        if isinstance(scale,(float,int)):
            node.setScale(scale)
        else:
            node.setScale(*scale)
        node.setColor(*color)
        node.setH(heading)
        return node

    def label_3d(self, parent, text, position, scale=.55, color=CREAM):
        label = TextNode(text)
        label.setText(text)
        label.setAlign(TextNode.ACenter)
        label.setTextColor(*color)
        label.setShadow(.03,.03)
        node = parent.attachNewNode(label)
        node.setPos(*position)
        node.setScale(scale)
        node.setBillboardPointEye()
        node.setLightOff()
        return node

    def build_world(self):
        terrain = self.render.attachNewNode('static-arena')
        self.world_labels = self.render.attachNewNode('arena-labels')
        self.flower_geometry=self.render.attachNewNode('flower-geometry')
        self.home_geometry=self.render.attachNewNode('home-geometry')
        self.shape('box',terrain,(0,0,-.35),(60,60,.7),GRASS)
        # Soft tiled meadow gives movement an obvious visual reference.
        for x in range(-24,24,4):
            for y in range(-24,24,4):
                if ((x+y)//4)%2:
                    self.shape('box',terrain,(x+2,y+2,.006),(4,4,.01),(.38,.60,.34,1))
        self.shape('box',terrain,(0,-6,.015),(3.6,35,.02),(.70,.67,.49,1))
        self.shape('box',terrain,(0,3,.018),(30,3,.02),(.70,.67,.49,1))
        # Low continuous walls have the same inner boundary as the simulation.
        half = self.world.config.half_size
        for x,y,sx,sy in [(0,-half-.35,2*half+1.4,.7),(0,half+.35,2*half+1.4,.7),
                           (-half-.35,0,.7,2*half),(half+.35,0,.7,2*half)]:
            self.shape('box',terrain,(x,y,.48),(sx,sy,.96),LEAF)
            self.shape('box',terrain,(x,y,.99),(sx+.12,sy+.12,.12),(.36,.49,.30,1))
        for obstacle in self.world.obstacles:
            self.shape('box',terrain,(obstacle.x,obstacle.y,obstacle.height/2),
                       (obstacle.width,obstacle.depth,obstacle.height),(.51,.57,.53,1))
            self.shape('box',terrain,(obstacle.x,obstacle.y,obstacle.height+.04),
                       (obstacle.width+.08,obstacle.depth+.08,.08),(.66,.69,.58,1))
        for cx,cy,color,name in self.world.patches:
            self.shape('box',terrain,(cx,cy,.025),(8.4,8.4,.05),(.26,.43,.26,1))
            for dx,dy,sx,sy in [(0,-4.3,8.8,.18),(0,4.3,8.8,.18),(-4.3,0,.18,8.4),(4.3,0,.18,8.4)]:
                self.shape('box',terrain,(cx+dx,cy+dy,.12),(sx,sy,.24),CREAM)
            self.shape('box',terrain,(cx,cy+4.5,1.3),(.12,.12,2.6),CHARCOAL)
            self.shape('box',terrain,(cx,cy+4.5,2.5),(5,.18,.9),CHARCOAL)
            self.label_3d(self.world_labels,name,(cx,cy+4.5,3.3),.36)
        for flower in self.world.flowers:
            self.shape('cylinder',terrain,(flower.x,flower.y,.15),(.035,.035,.3),LEAF)
            for angle in range(0,360,72):
                theta = math.radians(angle)
                self.shape('sphere',self.flower_geometry,(flower.x+.17*math.cos(theta),flower.y+.17*math.sin(theta),.33),
                           (.19,.13,.055),FIELD_COLORS[flower.color],angle)
            self.shape('sphere',self.flower_geometry,(flower.x,flower.y,.37),(.10,.10,.05),(.99,.72,.20,1))
        # Wingless player's spawn pad and hive-like landmark.
        self.shape('cylinder',terrain,(0,-14,.02),(1.8,1.8,.04),(.83,.77,.50,1))
        self.shape('box',self.home_geometry,(-9,-19,1.4),(5,.5,2.8),(.53,.35,.17,1))
        for x in range(5):
            self.shape('cylinder',self.home_geometry,(-10.7+x*.85,-19.3,1.4),(.32,.32,.08),(.92,.66,.22,1)) .setP(90)
        c=self.world.config
        self.shape('cylinder',self.home_geometry,(c.hive_x,c.hive_y,.035),
                   (c.conversion_radius,c.conversion_radius,.07),(.86,.62,.20,1))
        self.label_3d(self.world_labels,'HIVE / PRESS E TO MAKE HONEY',(c.hive_x,c.hive_y,3.8),.32)
        # Hard negatives for recognition: colored shapes that are not flowers.
        for x,y,color in [(-18,-9,FIELD_COLORS[0]),(17,-10,FIELD_COLORS[1]),(17,15,FIELD_COLORS[2])]:
            self.shape('box',terrain,(x,y,.6),(.7,.7,1.2),color)
        self.label_3d(self.world_labels,'HOME',(-9,-19,3.1),.48)
        for i in range(18):
            angle = i*2*math.pi/18
            tx,ty=math.cos(angle)*34,math.sin(angle)*34
            self.shape('cylinder',terrain,(tx,ty,2),( .42,.42,4),(.45,.33,.21,1))
            self.shape('sphere',terrain,(tx,ty,5),(2.3,2.3,3.3),(.28,.45,.32,1))
        terrain.flattenStrong()
        self.flower_geometry.flattenStrong()
        self.home_geometry.flattenStrong()
        self.flower_markers = [self.shape('sphere',self.flower_geometry,(f.x,f.y,.405),(.105,.105,.05),(.99,.72,.20,1))
                               for f in self.world.flowers]
        self.harvest_ring = self.render.attachNewNode('dipper-footprint')
        for angle in range(0,360,15):
            a=math.radians(angle)
            self.shape('sphere',self.harvest_ring,(2*math.cos(a),2*math.sin(a),.02),(.08,.08,.025),(.98,.85,.34,1))
        self.harvest_ring.hide()

    def build_avatar(self):
        self.avatar = self.render.attachNewNode('wingless-fly')
        self.body = self.avatar.attachNewNode('body')
        # Six legs and antennae are appearance only; the action interface never controls them.
        self.shape('sphere',self.body,(0,-.25,.95),(.46,.62,.47),(.33,.38,.36,1))
        self.shape('sphere',self.body,(0,.28,1.05),(.43,.43,.43),CHARCOAL)
        for side in (-1,1):
            self.shape('sphere',self.body,(side*.28,.56,1.19),(.23,.15,.25),(.73,.28,.23,1))
            self.shape('sphere',self.body,(side*.29,.66,1.25),(.06,.035,.07),(.98,.85,.70,1))
            antenna=self.shape('cylinder',self.body,(side*.18,.42,1.65),(.027,.027,.42),CHARCOAL)
            antenna.setR(side*20)
            for y in (-.50,-.10,.30):
                leg=self.shape('cylinder',self.body,(side*.47,y,.47),(.055,.055,.67),CHARCOAL)
                leg.setR(side*33)
                self.shape('sphere',self.body,(side*.65,y+.05,.17),(.13,.21,.075),CHARCOAL)
        self.shape('box',self.body,(0,-.65,1.02),(.56,.35,.62),(.86,.64,.25,1))
        self.shape('box',self.body,(0,-.84,1.06),(.36,.045,.28),CREAM)
        self.tool = self.body.attachNewNode('dipper')
        self.tool.setPos(.58,.1,.88)
        self.shape('cylinder',self.tool,(0,.43,.08),(.045,.045,.95),(.58,.38,.21,1)).setP(75)
        self.shape('sphere',self.tool,(0,.92,-.07),(.28,.32,.12),(.78,.80,.68,1))
        self.shape('sphere',self.tool,(0,.92,.015),(.21,.25,.06),(.35,.44,.43,1))
        self.shadow=self.shape('sphere',self.render,(0,0,.025),(.7,.75,.018),(.12,.22,.16,.30))
        self.shadow.setTransparency(TransparencyAttrib.MAlpha)

    def hud_text(self,text,pos,scale=.04,color=(.95,.96,.90,1),align=TextNode.ALeft):
        return OnscreenText(text=text,pos=pos,scale=scale,fg=color,align=align,mayChange=True)

    def panel(self,name,bottom,top,color):
        cm=CardMaker(name)
        cm.setFrame(-self.getAspectRatio(),self.getAspectRatio(),bottom,top)
        panel=self.aspect2d.attachNewNode(cm.generate())
        panel.setColor(*color)
        panel.setTransparency(TransparencyAttrib.MAlpha)
        return panel

    def build_hud(self):
        aspect=self.getAspectRatio()
        self.panel('top-bar',.76,1,(.10,.17,.16,.94))
        self.panel('bottom-bar',-1,-.69,(.10,.17,.16,.94))
        self.shop_status=self.hud_text('',(-aspect+.07,-.755),.031,(.91,.84,.58,1))
        self.hud_text('F L Y B R A I N   /   M O T O R   L A B',(-aspect+.07,.91),.038,(.81,.85,.71,1))
        self.hud_text('Wingless meadow',(-aspect+.07,.815),.065)
        self.camera_status=self.hud_text('THIRD PERSON / LOCKED',(aspect-.07,.91),.032,align=TextNode.ARight)
        self.pollen_status=self.hud_text('POLLEN   0',(aspect-.07,.815),.055,align=TextNode.ARight)
        self.dipper_status=self.hud_text('DIPPER READY',(-aspect+.07,-.855),.044,(1,.84,.41,1))
        self.hud_text('WASD move  CLICK dip  E hive  1/2/3 buy  SPACE jump',(-aspect+.07,-.944),.031)
        self.hud_text('R  reset    ESC  pause    F12  screenshot',(aspect-.07,-.944),.031,align=TextNode.ARight)
        self.world_status=self.hud_text('',(aspect-.07,-.855),.032,align=TextNode.ARight)
        self.pause_status=self.hud_text('',(0,.1),.065,align=TextNode.ACenter)

    def bind_inputs(self):
        for key in ('w','a','s','d','arrow_up','arrow_down','arrow_left','arrow_right','space','e','1','2','3','mouse1','mouse3'):
            self.accept(key,self.set_key,[key,True])
            self.accept(key+'-up',self.set_key,[key,False])
        self.accept('r',self.reset_interactive)
        self.accept('escape',self.toggle_pause)
        self.accept('f12',self.save_screenshot)
        self.accept('tab',self.toggle_brain)
        self.accept('focus-out',self.release_inputs)
        self.accept('window-event',self.on_window_event)
        self.mouse_last=None
        self.zoom_input=0
        self.accept('wheel_up',self.zoom,[-1])
        self.accept('wheel_down',self.zoom,[1])

    def on_window_event(self, window):
        # Preserve ShowBase's close/resize handling when adding focus-loss behavior.
        self.windowEvent(window)
        if not window.getProperties().getForeground():
            self.release_inputs()

    def set_key(self,key,pressed):
        if key=='e' and pressed and not self.keys.get('e',False):
            self._interact_pending=True
        if key in ('1','2','3') and pressed and not self.keys.get(key,False):
            self._purchase_pending=int(key)
        self.keys[key]=pressed

    def release_inputs(self):
        self.keys.clear()
        self._interact_pending=False
        self._purchase_pending=0
        self.mouse_last=None

    def zoom(self,amount):
        self.zoom_input += amount

    def reset_interactive(self,preserve_economy=False):
        self.reset(preserve_economy=preserve_economy)
        self.accumulator=0
        self.release_inputs()
        self.render_state()

    def toggle_pause(self):
        self.paused=not self.paused
        self.release_inputs()
        self.accumulator=0
        if self.monitor is not None:
            self.monitor.refresh()

    def connect_brain(self,monitor=True,task='cycle',checkpoint='auto'):
        from .brain import load_graph,NeuralController
        from .telemetry import BehaviorTelemetry
        if self.brain is not None:
            raise ValueError('A controller is already connected')
        if task not in ('pollen','convert','cycle','explore'):
            raise ValueError('Unknown training task')
        self.training_task=task
        if task in ('pollen','cycle'):
            # Explicit collection curriculum: begin at flowers, learn the dipper, then evaluate navigation separately.
            self.world.config=replace(self.world.config,spawn_x=-10,spawn_y=-1,episode_seconds=600)
            self.world.reset(self.world.seed)
        elif task=='convert':
            self.world.config=replace(self.world.config,spawn_x=self.world.config.hive_x,
                                      spawn_y=self.world.config.hive_y,episode_seconds=30)
            self.world.reset(self.world.seed)
            self.world.pollen=self.world.config.pollen_capacity
        graph=load_graph()
        from .perception import PerceptionModel
        self.brain=NeuralController(graph,seed=self.world.seed,perception=PerceptionModel.load(),
                                    learning=self.offscreen,sample_actions=self.offscreen)
        if checkpoint=='auto':
            candidate=Path(__file__).resolve().parents[2]/'models/controller-market-v5.npz'
            checkpoint=candidate if candidate.exists() else None
        if checkpoint is not None:
            self.brain.load(checkpoint,allow_migration=True)
        self.behavior=BehaviorTelemetry()
        self.brain_enabled=True
        self._brain_session=Path('runs/arena-brain')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        self._brain_session.mkdir(parents=True,exist_ok=True)
        self._brain_log=(self._brain_session/'telemetry.jsonl').open('w',encoding='utf-8')
        metadata={'controller':'experimental LIF connectome reservoir + reward-modulated action readout',
                  'neurons':len(graph.root_ids),'directed_connections':len(graph.pre),
                  'anatomical_synapses':graph.anatomical_synapses,'optic_inputs':len(graph.sensory),
                  'motor_readout_neurons':len(graph.motor),'source_identity':graph.source_identity,
                  'config':self.world.configuration(),'neural_dt':self.brain.neural_dt,
                  'control_frames':self.brain.decision_frames,'seed':self.world.seed,
                  'policy_inputs':'RGB plus visible hive/shop affordances, inventory, prices, and stocks; no target/world coordinates',
                  'status':'training mode' if self.offscreen else 'deterministic deployment; learning in offline trainer'}
        metadata['training_task']=task
        metadata['vision']=self.brain.vision.snapshot()
        metadata['optic_receptive_fields']=self.brain.vision.mapping_record()
        metadata['reward_rules']={'pollen_per_100':1.,'honey_per_100':2.,'successful_purchase':'item utility',
                                  'full_wait_interval':-.01,'empty_dip':-.025,'fully_blocked_interval':-.05,
                                  'prediction':'TD error r + 0.98 V(next) - V(previous)'}
        (self._brain_session/'configuration.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
        if monitor:
            from .monitor import NeuralMonitor
            self.monitor=NeuralMonitor(self)
        print(f'Neural controller connected: {len(graph.root_ids):,} neurons / {len(graph.pre):,} connections',flush=True)
        print(f'Session archive: {self._brain_session.resolve()}',flush=True)
        self.accumulator=0
        self.last_time=time.perf_counter()

    def toggle_brain(self):
        if self.brain is None:
            return
        if self.brain_enabled:
            self.finish_brain_feedback(terminal=True)
        self.brain.eligibility.fill(0)
        self.brain.reward_system.value_trace.fill(0)
        self.brain_enabled=not self.brain_enabled
        self._brain_next_tick=self.world.tick
        self.release_inputs()
        if self.monitor is not None:
            self.monitor.refresh()

    def choose_brain_action(self):
        self.finish_brain_feedback()
        self.render_state()
        camera_geometry=(self.world.camera_yaw,self.world.camera_pitch,self.world.camera_distance,float(self.camLens.getFov()[0]))
        self._brain_action=self.brain.act(self.observe(),self.brain.decision_frames*self.world.config.dt,
                                          camera_geometry=camera_geometry,
                                          inventory=self.public_status())
        self._brain_next_tick=self.world.tick+self.brain.decision_frames
        if self.monitor is not None:
            self.monitor.capture()
        self.log_brain()
        return self._brain_action

    def finish_brain_feedback(self,terminal=False):
        events={'reward':self._brain_reward,'honey':self._brain_honey,'purchases':self._brain_purchases,
                'full_wait':self._brain_full_wait_steps/max(1,self._brain_feedback_steps),'progress':self._brain_progress,
                'empty_dips':self._brain_empty_dips,
                'blocked_fraction':self._brain_blocked_steps/max(1,self._brain_feedback_steps)}
        if self.brain.decisions and self._brain_feedback_steps:
            if terminal:
                self.brain.feedback(**events,terminal=True)
            else:
                self.brain.queue_feedback(**events)
        self._brain_reward=0.
        self._brain_honey=0.
        self._brain_purchases=0.
        self._brain_full_wait_steps=0
        self._brain_progress=0.
        self._brain_empty_dips=0
        self._brain_blocked_steps=0
        self._brain_feedback_steps=0

    def advance_world(self,action):
        before=(self.world.x,self.world.y,self.world.z)
        result=self.world.step(action)
        if self.behavior is not None:
            self.behavior.observe(before,self.world,action,result[0])
        if self.brain_enabled:
            self._brain_reward+=result[3]['collected']
            self._brain_honey+=result[3]['honey_gain']
            self._brain_purchases+=result[3]['purchase_utility']
            self._brain_full_wait_steps+=int(self.world.pollen>=self.world.config.pollen_capacity and not self.world.conversion_active)
            if self.training_progress_shaping and self.world.pollen>=self.world.config.pollen_capacity:
                c=self.world.config
                old_distance=math.hypot(before[0]-c.hive_x,before[1]-c.hive_y)
                new_distance=math.hypot(self.world.x-c.hive_x,self.world.y-c.hive_y)
                self._brain_progress+=(old_distance-new_distance)*.1
            self._brain_empty_dips+=int(result[3]['swung'] and result[3]['collected']==0)
            moving=bool(action.strafe or action.forward)
            moved=math.hypot(self.world.x-before[0],self.world.y-before[1])
            self._brain_blocked_steps+=int(moving and moved<self.world.config.speed*self.world.config.dt*.1)
            self._brain_feedback_steps+=1
        return result

    def public_status(self):
        w=self.world
        items=w.shop.snapshot(w.honey)['items']
        return [w.pollen/w.config.pollen_capacity,float(w.pollen>=w.config.pollen_capacity),
                float(w.conversion_active),float(w.in_conversion_zone()),
                *[float(item['affordable']) for item in items],
                (w.shop.stock[1] or 0)/3,float(bool(w.shop.stock[2])),
                min(1,w.shop.prices[0]/300),min(1,w.honey/300),0.,min(1,w.shop.purchases/10)]

    def brain_step(self):
        """One neural decision from RGB and public UI state; usable offscreen."""
        if self.brain is None:
            raise ValueError('Connect a neural controller first')
        action=self.choose_brain_action()
        result=self.step(action,frames=self.brain.decision_frames)
        if self.monitor is not None:
            self.monitor.refresh()
            self.monitor.pump()
        return result

    def log_brain(self,force=False):
        second=int(self.world.tick*self.world.config.dt)
        if self._brain_log is not None and (force or second!=self._brain_last_logged):
            record={'simulation_seconds':self.world.tick*self.world.config.dt,'neural':self.brain.snapshot(),
                    'behavior':self.behavior.snapshot(),'diagnostics':self.world.telemetry()}
            self._brain_log.write(json.dumps(record)+'\n')
            self._brain_log.flush()
            self._brain_last_logged=second
        bucket=self.brain.decisions//120
        if self._brain_session is not None and bucket>self._brain_checkpoint_bucket:
            self.brain.save(self._brain_session/f'checkpoint-d{self.brain.decisions:08d}.npz')
            self._brain_checkpoint_bucket=bucket

    def destroy(self):
        if getattr(self,'brain',None) is not None and getattr(self,'_brain_session',None) is not None:
            self.finish_brain_feedback(terminal=True)
            self.log_brain(force=True)
            self.brain.save(self._brain_session/'readout.npz')
            self._brain_session=None
        if getattr(self,'_brain_log',None) is not None:
            self._brain_log.close()
            self._brain_log=None
        if getattr(self,'monitor',None) is not None:
            self.monitor.close()
        super().destroy()

    def human_action(self):
        keys=self.keys
        yaw=pitch=0
        if not self.world.config.camera_locked and self.mouseWatcherNode.hasMouse():
            p=self.mouseWatcherNode.getMouse()
            current=(p.x,p.y)
            if keys.get('mouse3') and self.mouse_last is not None:
                yaw=-(current[0]-self.mouse_last[0])*140
                pitch=-(current[1]-self.mouse_last[1])*90
            self.mouse_last=current
        zoom,self.zoom_input=self.zoom_input,0
        interact,self._interact_pending=self._interact_pending,False
        buy,self._purchase_pending=self._purchase_pending,0
        return Action(strafe=int(bool(keys.get('d') or keys.get('arrow_right')))-int(bool(keys.get('a') or keys.get('arrow_left'))),
                      forward=int(bool(keys.get('w') or keys.get('arrow_up')))-int(bool(keys.get('s') or keys.get('arrow_down'))),
                      jump=bool(keys.get('space')),dip=bool(keys.get('mouse1')),
                      camera_yaw=yaw,camera_pitch=pitch,zoom=zoom,convert=interact,buy=buy)

    def interactive_update(self,task):
        now=time.perf_counter()
        elapsed=min(.1,now-self.last_time)
        self.last_time=now
        if not self.paused:
            self.accumulator+=elapsed
            action=None
            first=True
            while self.accumulator>=self.world.config.dt:
                # Mouse deltas apply once; held buttons apply every fixed simulation step.
                if self.brain_enabled:
                    if self.world.tick>=self._brain_next_tick:
                        self.choose_brain_action()
                    current=self._brain_action
                else:
                    if action is None:
                        action=self.human_action()
                    current=action if first else replace(action,camera_yaw=0,camera_pitch=0,zoom=0,convert=False,buy=0)
                _,_,truncated,_=self.advance_world(current)
                if self.brain_enabled and (self._brain_action.convert or self._brain_action.buy):
                    self._brain_action=replace(self._brain_action,convert=False,buy=0)
                self.accumulator-=self.world.config.dt
                first=False
                if truncated and self.brain_enabled:
                    self.reset_interactive(preserve_economy=True)
                    break
        self.render_state()
        if self.monitor is not None:
            self.monitor.pump()
        return task.cont

    def render_state(self):
        w=self.world
        self.avatar.setPos(w.x,w.y,w.z)
        self.avatar.setH(w.heading)
        self.body.setZ(math.sin(w.tick*.22)*.025*w.speed_fraction if w.grounded else 0)
        self.shadow.setPos(w.x,w.y,.025)
        self.shadow.setScale(.7+w.z*.1,.75+w.z*.1,.018)
        elapsed=(w.tick-w.last_swing_tick)*w.config.dt
        swing=math.sin(math.pi*elapsed/.32) if elapsed<.32 else 0
        self.tool.setP(-swing*55)
        if elapsed<.32:
            h=math.radians(w.heading)
            self.harvest_ring.setPos(w.x-.8*math.sin(h),w.y+.8*math.cos(h),.03)
            self.harvest_ring.show()
        else:
            self.harvest_ring.hide()
        eye,target=w.camera_pose()
        self.camera.setPos(*eye)
        self.camera.lookAt(*target)
        self.camera_status.setText('THIRD PERSON / '+('LOCKED' if w.config.camera_locked else 'RMB ORBIT / SCROLL ZOOM'))
        self.pollen_status.setText(f'PACK {w.pollen}/{w.config.pollen_capacity}    HONEY {w.honey:,}')
        shop=w.shop.snapshot(w.honey)
        offers='   '.join(f'{i["slot"]} {i["name"]}: {i["price"]}h '+('unlimited' if i['stock'] is None else f'stock {i["stock"]}') for i in shop['items'])
        self.shop_status.setText(('HIVE SHOP / ' if w.in_conversion_zone() else 'Rewards at hive / ')+offers)
        self.dipper_status.setText('DIPPER READY' if w.cooldown<=1e-8 else f'DIPPER  {w.cooldown:.1f}s')
        self.world_status.setText(f'{w.x:5.1f}, {w.y:5.1f}     '+('ON GROUND' if w.grounded else 'AIRBORNE'))
        if w.in_conversion_zone():
            if w.conversion_active:
                bee=w.conversion_steps%w.config.simulated_bees+1
                progress=int(100*w.conversion_timer/w.config.conversion_step_seconds)
                self.world_status.setText(f'BEE {bee}/{w.config.simulated_bees}  {progress}% / E STOP')
            else:
                self.world_status.setText('HIVE / E START CONVERTING' if w.pollen else 'HIVE / BACKPACK EMPTY')
        elif w.pollen>=w.config.pollen_capacity:
            self.world_status.setText('BACKPACK FULL / RETURN TO HIVE')
        self.pause_status.setText('PAUSED\nESC to resume' if self.paused else '')
        if self.brain is not None:
            mode='NEURAL CONTROL' if self.brain_enabled else 'MANUAL CONTROL'
            self.camera_status.setText(f'{mode} / CAMERA LOCKED / TAB TO SWITCH')
        for flower,marker in zip(w.flowers,self.flower_markers):
            amount=flower.pollen
            marker.setColor(.36+.63*amount,.36+.36*amount,.25-.05*amount,1)

    def sync_flower_positions(self):
        # Petals are baked into the seed-specific static scene. Rebuild the scene for a different seed.
        for flower,marker in zip(self.world.flowers,self.flower_markers):
            marker.setPos(flower.x,flower.y,.405)

    def observe(self):
        """A single third-person RGB camera frame, without the human HUD."""
        import numpy as np
        if self._observation_buffer is None:
            self._observation_buffer=self.win.makeTextureBuffer('policy-rgb',*self.observation_size,to_ram=True)
            if self._observation_buffer is None:
                raise RuntimeError('Could not create the policy camera buffer')
            self._observation_buffer.setClearColor((.73,.84,.86,1))
            self._observation_buffer.setSort(-100)
            # A scene-only render of the same camera pose/lens avoids flashing the human HUD off.
            self._observation_camera=self.makeCamera(self._observation_buffer,scene=self.render,
                                                     lens=self.camLens,camName='policy-rgb-camera')
        self._observation_buffer.setActive(True)
        try:
            self.graphicsEngine.renderFrame()
            image=self._observation_buffer.getTexture()
            # Convert using Panda's explicit RGB channel order and bottom-up row order.
            rgb=np.frombuffer(image.getRamImageAs('RGB'),dtype=np.uint8)
            rgb=rgb.reshape(image.getYSize(),image.getXSize(),3)[::-1].copy()
            return rgb
        finally:
            self._observation_buffer.setActive(False)

    def reset(self,seed=None,preserve_economy=False):
        previous_seed=self.world.seed
        self.world.reset(seed,preserve_economy=preserve_economy)
        if self.training_task=='convert':
            self.world.pollen=self.world.config.pollen_capacity
        if self.world.seed!=previous_seed:
            # Resetting with a new seed must change geometry as well as collision/resource positions.
            self.render.find('**/static-arena').removeNode()
            self.world_labels.removeNode()
            self.flower_geometry.removeNode()
            self.home_geometry.removeNode()
            self.harvest_ring.removeNode()
            self.build_world()
        self.sync_flower_positions()
        if self.brain is not None:
            self.finish_brain_feedback(terminal=True)
            self.brain.reset_dynamics()
            self._brain_reward=0.
            self._brain_next_tick=0
            self._brain_action=Action()
            self.behavior.reset_episode()
            if self.monitor is not None:
                self.monitor.history.clear()
                self.monitor.refresh()
        self.render_state()
        return self.observe(), {'diagnostics':self.world.telemetry(),'config':self.world.configuration()}

    def semantic_labels(self):
        """Training/evaluation oracle only. Never called by the live policy."""
        import numpy as np
        nodes=(self.render,self.flower_geometry,self.home_geometry)
        states=[node.getState() for node in nodes]
        try:
            self.render.setColor(0,0,0,1,100)
            self.render.setLightOff(100)
            self.render.setTextureOff(100)
            self.render.setShaderOff(100)
            self.render.setFogOff(100)
            self.flower_geometry.setColor(1,0,0,1,200)
            self.home_geometry.setColor(0,1,0,1,200)
            image=self.observe()
            labels=np.zeros(image.shape[:2],dtype=np.uint8)
            labels[(image[:,:,0]>200)&(image[:,:,1]<30)]=1
            labels[(image[:,:,1]>200)&(image[:,:,0]<30)]=2
            return labels
        finally:
            for node,state in zip(nodes,states):
                node.setState(state)

    def step(self,action=Action(),frames=4):
        if not isinstance(frames,int) or frames<1:
            raise ValueError('frames must be a positive integer')
        reward=0
        for i in range(frames):
            per_frame=action if i==0 else replace(action,camera_yaw=0,camera_pitch=0,zoom=0,convert=False,buy=0)
            gain,terminated,truncated,info=self.advance_world(per_frame)
            reward+=gain
            if terminated or truncated:
                break
        self.render_state()
        return self.observe(),reward,terminated,truncated,{'diagnostics':info}

    def save_screenshot(self,path=None):
        if path is None:
            self.screenshot_counter+=1
            path=Path('runs/arena')/f'screenshot-{int(time.time())}-{self.screenshot_counter}.png'
        path=Path(path).resolve()
        path.parent.mkdir(parents=True,exist_ok=True)
        self.render_state()
        self.graphicsEngine.renderFrame()
        self.graphicsEngine.renderFrame()
        if not self.win.saveScreenshot(Filename.fromOsSpecific(str(path))):
            raise RuntimeError('Screenshot save failed')
        print(f'Screenshot: {path}',flush=True)
        return path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--free-camera',action='store_true',help='Optional later stage: RMB orbit and third-person-only zoom')
    parser.add_argument('--offscreen',action='store_true')
    parser.add_argument('--software',action='store_true',help='Software-rendered graphics diagnostic fallback')
    parser.add_argument('--screenshot',type=Path)
    parser.add_argument('--smoke-test',action='store_true',help='Scripted walk/collect/render validation; not a trained controller')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--brain',action='store_true',help='Connect the experimental spiking neural controller and open its monitor')
    parser.add_argument('--no-monitor',action='store_true',help='Run neural control without the second window')
    parser.add_argument('--brain-smoke-test',action='store_true',help='Check the actual camera/neuron/action loop offscreen')
    parser.add_argument('--task',choices=['pollen','convert','cycle','explore'],default='cycle',help='Collection/conversion curriculum')
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--fresh',action='store_true',help='Do not load a trained controller checkpoint')
    args=parser.parse_args()
    app=ArenaApp(ArenaConfig(seed=args.seed,camera_locked=not args.free_camera),offscreen=args.offscreen,software=args.software)
    if args.brain or args.brain_smoke_test:
        app.connect_brain(monitor=not args.no_monitor,task=args.task,
                          checkpoint=None if args.fresh or args.brain_smoke_test else args.checkpoint or 'auto')
    if args.brain_smoke_test:
        import numpy as np
        try:
            initial=app.observe()
            actions=[]
            for _ in range(32):
                observation,_,_,_,_=app.brain_step()
                actions.append(app.brain.snapshot()['action'])
            assert app.brain.total_spikes>0
            assert len(set(actions))>1
            assert np.any(initial!=observation)
            assert app.world.camera_distance==app.world.config.camera_distance
            # Check sustained pixel sensitivity and a recurrent-edge ablation on the real graph.
            from .brain import NeuralController
            graph=app.brain.graph
            dark_controller=NeuralController(graph,seed=args.seed,perception=app.brain.vision.perception)
            bright_controller=NeuralController(graph,seed=args.seed,perception=app.brain.vision.perception)
            cut_controller=NeuralController(replace(graph,weights=np.zeros_like(graph.weights)),seed=args.seed,perception=app.brain.vision.perception)
            for _ in range(8):
                dark_controller.act(np.zeros_like(initial))
                bright_controller.act(np.full_like(initial,255))
                cut_controller.act(np.full_like(initial,255))
            pixel_effect=float(np.linalg.norm(dark_controller.last_features-bright_controller.last_features))
            edge_effect=float(np.linalg.norm(bright_controller.last_features-cut_controller.last_features))
            assert pixel_effect>.001 and edge_effect>.001
            # A real collection transition tests the reward path separately from untrained exploration.
            app.world.x,app.world.y,app.world.z=-10,1,0
            app.world.vz=0
            app.world.grounded=True
            app.world.cooldown=0
            app.world.pollen=0
            _,harvest,_,_,_=app.step(Action(dip=True),frames=60)
            old=app.brain.weights.copy()
            app.brain_step()
            assert harvest>0 and not np.array_equal(old,app.brain.weights)
            report={'status':'passed','neural':app.brain.snapshot(),'behavior':app.behavior.snapshot(),
                    'observation_shape':list(observation.shape),'action_variety':sorted(set(actions)),
                    'vision_changed':True,'reward_updates_readout':True,
                    'real_graph_pixel_effect_l2':pixel_effect,'real_graph_recurrent_ablation_effect_l2':edge_effect,
                    'reward_check':'test-only relocation to flower patch; autonomous collection not established'}
            Path('reports/brain-smoke.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
            if args.screenshot:
                app.save_screenshot(args.screenshot)
            print(json.dumps(report,indent=2))
        finally:
            app.destroy()
    elif args.smoke_test:
        import numpy as np
        try:
            initial,_=app.reset()
            assert initial.shape==(400,640,3) and initial.dtype==np.uint8
            app.step(Action(strafe=-1),frames=100)
            walked,_,_,_,_=app.step(Action(forward=1),frames=150)
            assert np.any(walked!=initial), 'Camera pixels must change after walking'
            image,reward,_,_,info=app.step(Action(dip=True),frames=1)
            assert reward>0, 'The scripted route must reach a harvestable flower patch'
            app.step(Action(dip=True),frames=47)
            assert app.world.swings==1, 'Held input must respect the dipper cooldown'
            assert app.world.camera_distance==app.world.config.camera_distance
            checks={'status':'passed','controller':'scripted acceptance route; no learned policy',
                    'observation_shape':list(image.shape),'camera_pixels_changed':True,
                    'first_harvest_reward':reward,'dipper_cooldown_respected':True,
                    'config':app.world.configuration(),'final_diagnostics':app.world.telemetry()}
            report=Path('reports/arena-smoke.json')
            report.parent.mkdir(parents=True,exist_ok=True)
            report.write_text(json.dumps(checks,indent=2)+'\n',encoding='utf-8')
            if args.screenshot:
                app.save_screenshot(args.screenshot)
            print(json.dumps(checks,indent=2))
        finally:
            app.destroy()
    elif args.screenshot:
        app.save_screenshot(args.screenshot)
        app.destroy()
    elif args.offscreen:
        frame,info=app.reset()
        print(json.dumps({'observation_shape':list(frame.shape),'diagnostics':info['diagnostics']}))
        app.destroy()
    else:
        try:
            app.run()
        finally:
            app.destroy()


if __name__=='__main__':
    main()
