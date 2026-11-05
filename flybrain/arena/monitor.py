# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 FlyBrain contributors
"""Native second window. Graphs use actual simulated spikes, never decorative activity."""
from collections import deque
import math
import tkinter as tk
from tkinter import ttk

import numpy as np

BG='#111d22'
PANEL='#1b2b32'
TEXT='#eaf0e9'
MUTED='#a2b5b7'
ACCENT='#eacf82'
GREEN='#80c9ae'


class NeuralMonitor:
    def __init__(self,app):
        self.app=app
        self.closed=False
        self.history=deque(maxlen=120)
        self.root=tk.Tk()
        self.root.title('FlyBrain - Neural Activity & State')
        x=1280 if self.root.winfo_screenwidth()>=2240 else 60
        self.root.geometry(f'960x940+{x}+35')
        self.root.minsize(860,940)
        self.root.configure(bg=BG)
        self.root.protocol('WM_DELETE_WINDOW',self.close)
        self.root.bind('<Escape>',lambda e:app.toggle_pause())
        self.root.bind('<space>',lambda e:app.toggle_pause())
        font=('Segoe UI',11)
        self.root.option_add('*Font',font)
        header=tk.Frame(self.root,bg=BG)
        header.pack(fill='x',padx=22,pady=(18,10))
        tk.Label(header,text='NEURAL ACTIVITY / FLYBRAIN',bg=BG,fg=TEXT,font=('Segoe UI',18)).pack(anchor='w')
        tk.Label(header,text='Experimental spiking model  ·  camera + public status → connectome → trained controls',bg=BG,fg=MUTED).pack(anchor='w')
        controls=tk.Frame(header,bg=BG)
        controls.pack(anchor='e',pady=(8,0))
        for text,command in [('Pause / resume',app.toggle_pause),('Manual / neural',app.toggle_brain),('Reset episode',app.reset_interactive)]:
            tk.Button(controls,text=text,command=command,bg=PANEL,fg=TEXT,activebackground=BG,activeforeground=TEXT,
                      relief='flat',padx=10,pady=4).pack(side='left',padx=(6,0))
        status=tk.Frame(self.root,bg=BG)
        status.pack(fill='x',padx=22)
        self.state=self.card(status,'CURRENT STATE')
        self.happy=self.card(status,'HAPPINESS / REWARD PROXY')
        self.escape=self.card(status,'ESCAPE ATTEMPTS')
        self.definition=tk.Label(self.root,text='Escape attempt: ≥0.5 s pushing outward against an outer wall; counts episodes of pushing.',
                                 bg=BG,fg=MUTED,anchor='w')
        self.definition.pack(fill='x',padx=22,pady=(8,0))
        self.metrics=tk.Label(self.root,text='',bg=BG,fg=TEXT,anchor='w',justify='left')
        self.metrics.pack(fill='x',padx=22,pady=(12,6))
        graph_header=tk.Frame(self.root,bg=BG)
        graph_header.pack(fill='x',padx=22,pady=(4,0))
        tk.Label(graph_header,text='Binned spike raster · up to 72 sampled neurons · 250 ms bins',bg=BG,fg=TEXT).pack(side='left')
        self.group=tk.StringVar(value='Overview')
        choice=ttk.Combobox(graph_header,textvariable=self.group,values=['Overview','Optic inputs','Motor / descending'],
                            state='readonly',width=20)
        choice.pack(side='right')
        choice.bind('<<ComboboxSelected>>',lambda e:self.draw())
        self.raster=tk.Canvas(self.root,height=180,bg=BG,highlightthickness=0)
        self.raster.pack(fill='x',padx=22)
        self.raster.bind('<Button-1>',self.select_neuron)
        self.selected=tk.Label(self.root,text='Select a raster row to inspect its exact neuron ID.',bg=BG,fg=MUTED,anchor='w')
        self.selected.pack(fill='x',padx=22)
        tk.Label(self.root,text='Population rate: green, left axis / Hz · dopamine-like RPE: amber, right axis',bg=BG,fg=TEXT,anchor='w').pack(fill='x',padx=22,pady=(14,0))
        self.trace=tk.Canvas(self.root,height=95,bg=BG,highlightthickness=0)
        self.trace.pack(fill='x',padx=22)
        preview_frame=tk.Frame(self.root,bg=BG)
        preview_frame.pack(fill='x',padx=22,pady=(6,6))
        self.preview_labels=[]
        self.preview_images=[]
        for title in ('640 × 400 camera / mask','Flowers green / hive amber','Motion: right R / left B'):
            panel=tk.Frame(preview_frame,bg=BG)
            panel.pack(side='left',padx=(0,24))
            tk.Label(panel,text=title,bg=BG,fg=MUTED,font=('Segoe UI',10)).pack(anchor='w')
            label=tk.Label(panel,bg=BG)
            label.pack(anchor='w')
            self.preview_labels.append(label)
        self.talk=tk.Label(self.root,text='',bg=PANEL,fg=TEXT,anchor='w',justify='left',padx=14,pady=6)
        self.talk.pack(fill='x',padx=22,pady=(4,0))
        self.footer=tk.Label(self.root,text='Happiness is a measured reward/blockage proxy, not evidence of subjective feelings.\n'
                                  'Real source connectivity; experimental encoding, unsigned LIF dynamics, and action learning.',
                             bg=BG,fg=MUTED,justify='left',anchor='w')
        self.footer.pack(fill='x',padx=22,pady=(4,16))
        self.raster.bind('<Configure>',lambda e:self.draw())
        self.trace.bind('<Configure>',lambda e:self.draw_trace())
        self.selected_index=None
        self.image=None
        self.sample=np.array([],dtype=int)
        self.raster_box=(90,10,1,2)
        self.root.update_idletasks()
        self.refresh()

    def card(self,parent,title):
        frame=tk.Frame(parent,bg=PANEL,padx=14,pady=10)
        frame.pack(side='left',fill='x',expand=True,padx=(0,8))
        tk.Label(frame,text=title,bg=PANEL,fg=MUTED,font=('Segoe UI',9)).pack(anchor='w')
        value=tk.Label(frame,text='—',bg=PANEL,fg=ACCENT,font=('Segoe UI',19))
        value.pack(anchor='w')
        return value

    def close(self):
        if not self.closed:
            self.closed=True
            self.root.destroy()

    def pump(self):
        if not self.closed:
            try:
                self.root.update_idletasks()
                self.root.update()
            except tk.TclError:
                self.closed=True

    def capture(self):
        if self.closed:
            return
        b=self.app.brain
        self.history.append({'counts':b.last_counts.copy(),'rate':float(b.rates.mean()),
                             'dopamine':b.reward_system.phasic})
        self.refresh()

    def refresh(self):
        if self.closed:
            return
        a=self.app
        b=a.brain
        s=b.snapshot()
        t=a.behavior.snapshot()
        mode='PAUSED' if a.paused else t['state'] if a.brain_enabled else 'MANUAL'
        self.state.config(text=mode)
        self.happy.config(text=f'{t["happiness_proxy"]:.0f} / 100')
        self.escape.config(text=f'{t["escape_attempts"]}  total · {t["episode_escape_attempts"]} episode')
        if a.world.conversion_active:
            phase=(f'bee {a.world.conversion_steps%a.world.config.simulated_bees+1}/{a.world.config.simulated_bees} · '
                   f'step {a.world.conversion_steps+1} · {int(100*a.world.conversion_timer/a.world.config.conversion_step_seconds)}%')
        else:
            phase=f'{"learning" if s["learning_enabled"] else "deployed"} · {s["action"]}{" + dip" if s["dip"] else ""}{" + E" if s["convert"] else ""}{" + buy "+str(s["buy"]) if s["buy"] else ""}'
        self.metrics.config(text=f'{s["neurons"]:,} neurons · {s["connections"]:,} connections · {s["active_neurons"]:,} spiked in latest bin\n'
              f'Pack {a.world.pollen}/{a.world.config.pollen_capacity} · honey {a.world.honey} · spent {a.world.shop.spent} · bought {a.world.shop.purchases} · {phase}\n'
              f'COLLECT → CONVERT → BUY · V {b.reward_system.expected_value:+.2f} · '
              f'prediction error {b.reward_system.prediction_error:+.3f} · phasic signal {b.reward_system.phasic:+.3f}\n'
              f'Reward: purchases {b.reward_system.components["purchases"]:+.2f} · honey {b.reward_system.components["honey"]:+.2f} · '
              f'full wait {b.reward_system.components["full_wait"]:+.3f} · next sticker {a.world.shop.prices[0]}h')
        c=getattr(a,'comms',None)
        self.talk.config(text='BEE SAYS (preset sentences from measured state)\n'+('\n'.join(f'{m["t"]:7.1f}s  {m["text"]}' for m in c.latest(3)) if c is not None and c.log else '—'))
        previews=b.vision.previews()
        if previews is not None:
            self.preview_images=[]
            for label,pixels in zip(self.preview_labels,previews):
                ys=np.linspace(0,pixels.shape[0]-1,100).astype(int)
                xs=np.linspace(0,pixels.shape[1]-1,160).astype(int)
                pixels=pixels[ys[:,None],xs[None,:]]
                height,width=pixels.shape[:2]
                ppm=f'P6\n{width} {height}\n255\n'.encode()+pixels.tobytes()
                image=tk.PhotoImage(data=ppm,format='PPM')
                label.config(image=image)
                self.preview_images.append(image)
        if self.selected_index is not None:
            i=self.selected_index
            self.selected.config(text=f'Neuron {b.graph.root_ids[i]} · {b.graph.groups[i]} · {b.last_counts[i]} spikes / bin · {b.rates[i]:.1f} Hz EMA')
        self.draw()

    def choose_sample(self):
        graph=self.app.brain.graph
        candidates=(graph.sensory if self.group.get()=='Optic inputs' else graph.motor if self.group.get()=='Motor / descending'
                    else np.arange(len(graph.root_ids)))
        self.sample=candidates[np.linspace(0,len(candidates)-1,min(72,len(candidates))).astype(int)]

    def draw(self):
        if self.closed:
            return
        self.choose_sample()
        canvas=self.raster
        canvas.delete('all')
        width=max(860,canvas.winfo_width())
        zoom=max(1,(width-100)//120)
        x,y=90,8
        values=np.zeros((len(self.sample),120),dtype=int)
        if self.history:
            recent=np.stack([h['counts'][self.sample] for h in self.history],axis=1)
            values[:,-recent.shape[1]:]=recent
        palette=['#203039']+[f'#{int(45+190*k/30):02x}{int(75+132*k/30):02x}{int(82+48*k/30):02x}' for k in range(1,31)]
        raw=tk.PhotoImage(width=120,height=len(self.sample))
        raw.put(' '.join('{'+' '.join(palette[min(30,int(v))] for v in row)+'}' for row in values))
        self.image=raw.zoom(zoom,2)
        canvas.create_image(x,y,image=self.image,anchor='nw')
        self.raster_box=(x,y,zoom,2)
        for row in (0,len(self.sample)//2,len(self.sample)-1):
            i=self.sample[row]
            canvas.create_text(x-7,y+row*2+4,text='…'+self.app.brain.graph.root_ids[i][-8:],fill=MUTED,anchor='e',font=('Consolas',9))
        for fraction,label in [(0,'−30 s'),(.5,'−15 s'),(1,'now')]:
            canvas.create_text(x+fraction*120*zoom,y+len(self.sample)*2+15,text=label,fill=MUTED,
                               anchor='w' if fraction==0 else 'e' if fraction==1 else 'center',font=('Segoe UI',9))
        self.draw_trace()

    def draw_trace(self):
        if self.closed:
            return
        canvas=self.trace
        canvas.delete('all')
        width=max(860,canvas.winfo_width())
        left,right,top,bottom=50,width-45,10,77
        rates=[h['rate'] for h in self.history]
        ceiling=max(20,math.ceil(max(rates,default=0)/20)*20)
        for value in (0,ceiling/2,ceiling):
            yy=bottom-(bottom-top)*value/ceiling
            canvas.create_line(left,yy,right,yy,fill=PANEL)
            canvas.create_text(left-8,yy,text=f'{value:g}',fill=MUTED,anchor='e',font=('Segoe UI',9))
        coords=[]
        for j,value in enumerate(rates):
            xx=right-(len(rates)-1-j)*(right-left)/119
            coords.extend((xx,bottom-(bottom-top)*value/ceiling))
        if len(coords)>=4:
            canvas.create_line(*coords,fill=GREEN,width=2)
        elif len(coords)==2:
            canvas.create_oval(coords[0]-2,coords[1]-2,coords[0]+2,coords[1]+2,fill=GREEN,outline='')
        dopamine=[]
        for j,h in enumerate(self.history):
            xx=right-(len(self.history)-1-j)*(right-left)/119
            dopamine.extend((xx,bottom-(bottom-top)*(h['dopamine']+1)/2))
        if len(dopamine)>=4:
            canvas.create_line(*dopamine,fill=ACCENT,width=2,dash=(4,3))
        for value in (-1,0,1):
            canvas.create_text(right+7,bottom-(bottom-top)*(value+1)/2,text=f'{value:+d}',fill=ACCENT,anchor='w',font=('Segoe UI',9))
        canvas.create_text(left,96,text='−30 s',fill=MUTED,anchor='w',font=('Segoe UI',9))
        canvas.create_text(right,96,text='now',fill=MUTED,anchor='e',font=('Segoe UI',9))

    def select_neuron(self,event):
        x,y,zoom,row_height=self.raster_box
        row=int((event.y-y)//row_height)
        if x<=event.x<x+120*zoom and 0<=row<len(self.sample):
            self.selected_index=int(self.sample[row])
            self.refresh()
