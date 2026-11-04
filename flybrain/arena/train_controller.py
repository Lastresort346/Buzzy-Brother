# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 FlyBrain contributors
"""Bounded actor/critic training with conversion, collection, and full-cycle lessons."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import numpy as np

from .app import ArenaApp
from .brain import NeuralController
from .world import Action


def lesson(app,name,seed):
    c=app.world.config
    app.training_task=name
    app.world.config=replace(c,spawn_x=c.hive_x if name=='convert' else -10,
                            spawn_y=c.hive_y if name=='convert' else -1,episode_seconds=120)
    app.reset(seed=seed)


def run_episode(app,name,seed,steps,random_policy=False):
    lesson(app,name,seed)
    rng=np.random.default_rng(seed)
    for _ in range(steps):
        if random_policy:
            move=int(rng.integers(9))
            choices=[(0,1),(-1,1),(1,1),(-1,0),(1,0),(-1,-1),(1,-1),(0,-1),(0,0)]
            x,y=choices[move]
            app.step(Action(strafe=x,forward=y,dip=bool(rng.random()<.5),
                            convert=bool(rng.random()<.5),jump=bool(rng.random()<.05)),frames=15)
        else:
            app.brain_step()
    if not random_policy:
        app.finish_brain_feedback(terminal=True)
    w=app.world
    return {'lesson':name,'seed':seed,'seconds':steps*.25,'collected':w.total_collected,
            'honey':w.honey,'backpack':w.pollen,'capacity':w.config.pollen_capacity,
            'successful_dips':w.successful_swings,'full_cycle':bool(name=='cycle' and w.total_collected>=w.config.pollen_capacity
                                                                   and w.honey>=w.config.pollen_capacity)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes-per-stage',type=int,default=4)
    parser.add_argument('--steps',type=int,default=64)
    parser.add_argument('--output',type=Path,default=Path('models/controller-v3.npz'))
    args=parser.parse_args()
    if args.episodes_per_stage<1 or args.steps<1:
        parser.error('Episode and step counts must be positive')
    app=ArenaApp(offscreen=True,width=640,height=400)
    app.connect_brain(monitor=False,task='cycle',checkpoint=None)
    trained=app.brain
    graph=trained.graph
    perception=trained.vision.perception
    initial_actor=trained.weights.copy()
    training=[]
    try:
        for stage,name in enumerate(('convert','pollen','cycle')):
            for episode in range(args.episodes_per_stage):
                result=run_episode(app,name,1000+stage*100+episode,args.steps)
                training.append(result)
                print(f'TRAIN {name} {episode+1}/{args.episodes_per_stage}: pollen {result["collected"]}, honey {result["honey"]}',flush=True)
            trained.save(args.output.with_name(args.output.stem+f'-{name}.npz'))
        trained.save(args.output)
        evaluation={}
        for condition in ('random','untrained','trained'):
            policy=NeuralController(graph,seed=42,learning=False,perception=perception)
            if condition=='trained':
                policy.load(args.output)
            app.brain=policy
            app.brain_enabled=condition!='random'
            results=[]
            for stage,name in enumerate(('convert','pollen','cycle')):
                for k in range(2):
                    results.append(run_episode(app,name,9000+stage*100+k,args.steps,random_policy=condition=='random'))
            evaluation[condition]={'episodes':results,'mean_honey':float(np.mean([r['honey'] for r in results])),
                                   'complete_cycles':sum(r['full_cycle'] for r in results)}
            print(f'EVAL {condition}: mean honey {evaluation[condition]["mean_honey"]:.1f}, '
                  f'complete cycles {evaluation[condition]["complete_cycles"]}',flush=True)
        report={'algorithm':'experimental on-policy TD actor/critic readout with eligibility traces; fixed connectome reservoir',
                'perception':'frozen supervised flower/hive recognizer; runtime RGB only',
                'policy_inputs':'RGB plus public inventory status; no world coordinates or oracle labels',
                'training_episodes':training,'held_out_evaluation':evaluation,
                'readout_changed':bool(np.any(initial_actor!=trained.weights)),
                'updates':trained.update_count,'checkpoint':str(args.output),
                'scope':'initial bounded training batch; reliable full collection-return-conversion behavior is not assumed'}
        Path('reports/controller-training-v3.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({'checkpoint':str(args.output),'updates':trained.update_count,
                          'trained_complete_cycles':evaluation['trained']['complete_cycles']},indent=2))
    finally:
        app.finish_brain_feedback(terminal=True)
        app.brain=trained
        app.brain_enabled=True
        app.destroy()


if __name__=='__main__':
    main()
