# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 FlyBrain contributors
"""Training-only expert warm-up, reward fine-tuning, and unguided held-out evaluation.

The expert reads simulator positions ONLY here to produce demonstrations. The deployed
controller receives RGB and publicly visible inventory/shop state through neural inputs.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import numpy as np

from .app import ArenaApp
from .world import Action
from .brain import NeuralController
from .economy import CATALOG


def preferred_item(world):
    choices=[]
    for i,item in enumerate(CATALOG):
        if world.shop.stock[i]==0:
            continue
        utility=item.utility/(1+.3*world.shop.owned[i]) if item.stock is None else item.utility
        choices.append((utility/world.shop.prices[i],i+1))
    return max(choices)[1] if choices else 0


def cardinal(dx,dy,dip=False):
    if abs(dy)>=abs(dx):
        return Action(forward=float(np.sign(dy)),dip=dip)
    return Action(strafe=float(np.sign(dx)),dip=dip)


def expert(world):
    """Explicit oracle teacher. This function is never imported by live control."""
    slot=preferred_item(world)
    afford=slot and world.honey>=world.shop.prices[slot-1]
    if world.in_conversion_zone():
        if afford:
            return Action(buy=slot)
        if world.conversion_active:
            return Action()
        if world.pollen>0:
            return Action(convert=True)
    if world.conversion_active:
        return Action()
    if world.pollen>=world.config.pollen_capacity or afford:
        return cardinal(world.config.hive_x-world.x,world.config.hive_y-world.y)
    # Initial skill: collect from a known flower patch. Waiting for regrowth is valid
    # but deliberately not claimed to be an optimal farming strategy.
    cx,cy,_,_=world.patches[0]
    if abs(world.x-cx)<=2.0 and abs(world.y-cy)<=2.0:
        return Action(dip=True)
    return cardinal(cx-world.x,cy-world.y,dip=True)


def configure(app,seed,capacity=60,gap=None,wallet=0,initial_pollen=0,
              stickers=0,limited_sold_out=False,spawn_y=None):
    c=app.world.config
    if gap is None:
        x,y=-10.,-1.
    else:
        x,y=c.hive_x-1,c.hive_y+gap
    if spawn_y is not None:
        y=spawn_y
    app.training_task='market'
    app.world.config=replace(c,spawn_x=x,spawn_y=y,pollen_capacity=capacity,episode_seconds=180)
    app.reset(seed=seed)
    app.world.pollen=initial_pollen
    app.world.honey=wallet
    # Explicit initial economy states for training/evaluation; no free runtime purchases.
    for slot,count in ((1,stickers),(2,3 if limited_sold_out else 0),(3,int(limited_sold_out))):
        for _ in range(count):
            app.world.shop.buy(slot,10**12)


def evaluate(app,weights,value,scenarios,seed=42):
    policy=NeuralController(app.brain.graph,seed=seed,learning=False,perception=app.brain.vision.perception,sample_actions=False)
    policy.weights[:]=weights
    policy.reward_system.value_weights[:]=value
    app.finish_brain_feedback(terminal=True)
    app.brain=policy
    app.brain_enabled=True
    app.training_progress_shaping=False
    results=[]
    for spec in scenarios:
        configure(app,**{k:v for k,v in spec.items() if k!='steps'})
        app.world.config=replace(app.world.config,episode_seconds=max(180,spec['steps']/4+1))
        positions=[]
        resource_history=[]
        starting_purchases=app.world.shop.purchases
        starting_spent=app.world.shop.spent
        starting_utility=app.world.shop.total_utility
        full_wait=longest_full_wait=boundary_wait=longest_boundary_wait=0
        conversion_fills=[]
        for _ in range(spec['steps']):
            was_converting,pollen_before=app.world.conversion_active,app.world.pollen
            app.brain_step()
            if app.world.conversion_active and not was_converting:
                conversion_fills.append(pollen_before/app.world.config.pollen_capacity)
            positions.append([round(app.world.x,2),round(app.world.y,2)])
            resource_history.append((app.world.total_collected,app.world.total_converted))
            full_wait=full_wait+1 if app.world.pollen>=app.world.config.pollen_capacity and not app.world.conversion_active else 0
            longest_full_wait=max(longest_full_wait,full_wait)
            boundary=app.world.config.half_size-app.world.config.radius-.02
            boundary_wait=boundary_wait+1 if max(abs(app.world.x),abs(app.world.y))>=boundary else 0
            longest_boundary_wait=max(longest_boundary_wait,boundary_wait)
        app.finish_brain_feedback(terminal=True)
        w=app.world
        row={'seed':spec['seed'],'capacity':spec.get('capacity',60),'initial_pollen':spec.get('initial_pollen',0),
             'collected':w.total_collected,'converted':w.total_converted,'honey_earned':w.honey_earned,
             'honey_wallet':w.honey,'purchases':w.shop.purchases-starting_purchases,'owned':w.shop.owned,
             'initial_purchases':starting_purchases,'purchase_utility':w.shop.total_utility-starting_utility,
             'honey_spent':w.shop.spent-starting_spent,
             'full_cycle':w.total_collected>=w.config.pollen_capacity and w.total_converted>=w.config.pollen_capacity and w.shop.purchases>starting_purchases,
             'last_position':positions[-1],'position_samples':positions[::16]}
        # Backpack fraction when each conversion began; low values mean short partial trips.
        row['mean_fill_at_conversion']=float(np.mean(conversion_fills)) if conversion_fills else 0.
        row['longest_full_bag_wait_seconds']=longest_full_wait/4
        row['longest_boundary_stay_seconds']=longest_boundary_wait/4
        earlier=resource_history[max(0,len(resource_history)-121)]
        row['last_30_seconds_collected']=w.total_collected-earlier[0]
        row['last_30_seconds_converted']=w.total_converted-earlier[1]
        results.append(row)
        print(f'EVAL seed {row["seed"]}: collected {row["collected"]}, converted {row["converted"]}, '
              f'purchases {row["purchases"]}, full cycle {row["full_cycle"]}, '
              f'fill at conversion {row["mean_fill_at_conversion"]:.2f}',flush=True)
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('models/controller-market-v6.npz'))
    parser.add_argument('--demo-steps',type=int,default=96)
    parser.add_argument('--fit-steps',type=int,default=1200)
    parser.add_argument('--rl-episodes',type=int,default=4)
    args=parser.parse_args()
    app=ArenaApp(offscreen=True,width=640,height=400)
    app.connect_brain(monitor=False,checkpoint=None)
    trained=app.brain
    trained.learning=False
    features=[]
    labels=[]
    demo_stats=[]
    scenarios=[dict(seed=7100+i,capacity=60,gap=gap,initial_pollen=60) for i,gap in enumerate((1,4,10,17))]
    scenarios += [dict(seed=7200,capacity=60,gap=1,wallet=300),dict(seed=7201,capacity=60,gap=1,wallet=700)]
    scenarios += [dict(seed=7300,capacity=60),dict(seed=7301,capacity=120),
                  dict(seed=7302,capacity=300,initial_pollen=200),dict(seed=7303,capacity=300,initial_pollen=240)]
    # Full trips from an empty 300 bag. Without these, every 300-capacity demo began
    # 2/3 full, so the student learned "dip ~80 pollen, then go home".
    scenarios += [dict(seed=7304,capacity=300,steps=320),dict(seed=7305,capacity=300,steps=320)]
    try:
        for n,spec in enumerate(scenarios):
            configure(app,**{k:v for k,v in spec.items() if k!='steps'})
            for _ in range(spec.get('steps',args.demo_steps)):
                app.choose_brain_action()
                target=expert(app.world)
                features.append(app.brain.last_features.copy())
                labels.append(target)
                app.step(target,frames=15)
            app.finish_brain_feedback(terminal=True)
            demo_stats.append(dict(spec,collected=app.world.total_collected,converted=app.world.total_converted,purchases=app.world.shop.purchases))
            print(f'DEMO {n+1}/{len(scenarios)}: collected {app.world.total_collected}, '
                  f'converted {app.world.total_converted}, purchases {app.world.shop.purchases}',flush=True)
        Path('data').mkdir(exist_ok=True)
        np.savez_compressed('data/market-demonstrations-v6.npz',features=np.array(features),
                            actions=np.array([[a.strafe,a.forward,a.dip,a.jump,a.convert,a.buy] for a in labels]))
        fit=trained.fit_demonstrations(features,labels,steps=args.fit_steps)
        print('Supervised warm-up:',json.dumps(fit),flush=True)
        trained.save(args.output.with_name(args.output.stem+'-demonstrations.npz'))
        validation=[dict(seed=8101,capacity=60,gap=16,initial_pollen=60,steps=128),
                    dict(seed=8102,capacity=60,steps=192),dict(seed=8103,capacity=300,steps=320)]
        reference_actor=trained.weights.copy()
        reference_value=trained.reward_system.value_weights.copy()
        validation_result=evaluate(app,reference_actor,reference_value,validation)
        # Correct states actually visited by the student, then refit. The teacher is
        # still training-only; held-out evaluation below has no teacher calls.
        if not all(r['purchases']>0 for r in validation_result):
            app.brain=trained
            trained.learning=False
            for n,spec in enumerate(validation):
                configure(app,**{k:v for k,v in spec.items() if k!='steps'})
                for t in range(160):
                    predicted=app.choose_brain_action()
                    target=expert(app.world)
                    features.append(trained.last_features.copy())
                    labels.append(target)
                    # Alternate student and teacher trajectories during aggregation.
                    app.step(target if t%4==0 else predicted,frames=15)
                app.finish_brain_feedback(terminal=True)
                print(f'AGGREGATE {n+1}: {app.world.total_converted} converted / {app.world.shop.purchases} purchases',flush=True)
            fit=trained.fit_demonstrations(features,labels,steps=args.fit_steps)
            reference_actor=trained.weights.copy()
            reference_value=trained.reward_system.value_weights.copy()
            validation_result=evaluate(app,reference_actor,reference_value,validation)
        app.brain=trained
        trained.learning=True
        trained.sample_actions=True
        app.training_progress_shaping=True
        rl=[]
        for ep in range(args.rl_episodes):
            configure(app,seed=8400+ep,capacity=60 if ep<2 else 300,
                      gap=12 if ep<2 else None,initial_pollen=60 if ep<2 else 0)
            for _ in range(128 if ep<2 else 320):
                app.brain_step()
            app.finish_brain_feedback(terminal=True)
            rl.append({'seed':8400+ep,'collected':app.world.total_collected,'converted':app.world.total_converted,
                       'purchases':app.world.shop.purchases})
            print(f'RL {ep+1}/{args.rl_episodes}: {rl[-1]}',flush=True)
        trained.save(args.output.with_name(args.output.stem+'-rl-candidate.npz'))
        rl_actor=trained.weights.copy()
        rl_value=trained.reward_system.value_weights.copy()
        rl_validation=evaluate(app,rl_actor,rl_value,validation)
        def score(rows):
            return (sum(r['full_cycle'] for r in rows),round(min(r['mean_fill_at_conversion'] for r in rows),1),
                    sum(r['purchases'] for r in rows),sum(r['converted'] for r in rows))
        keep_rl=score(rl_validation)>=score(validation_result)
        trained.weights[:]=rl_actor if keep_rl else reference_actor
        trained.reward_system.value_weights[:]=rl_value if keep_rl else reference_value
        final_tests=[dict(seed=9101,capacity=60,gap=17,initial_pollen=60,steps=160),
                     dict(seed=9102,capacity=60,steps=240),dict(seed=9103,capacity=300,steps=320)]
        final=evaluate(app,trained.weights,trained.reward_system.value_weights,final_tests)
        app.brain=trained
        trained.save(args.output)
        report={'policy_inputs':'RGB and public UI state encoded into neurons; no runtime world coordinates',
                'teacher':'privileged simulator teacher used only for demonstration warm-up and aggregation',
                'demonstrations':demo_stats,'fit':fit,'rl_episodes':rl,'validation_bc':validation_result,
                'validation_rl':rl_validation,'promoted_rl_candidate':keep_rl,'held_out':final,
                'economy':'escalating sticker prices with diminishing novelty; 3 snacks; 1 charm',
                'checkpoint':str(args.output),'scope':'fixed arena/corridor curriculum; economic optimality not established'}
        Path('reports/market-training-v6.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        print('Saved market policy and evaluation report.',flush=True)
    finally:
        app.finish_brain_feedback(terminal=True)
        app.brain=trained
        app.destroy()


if __name__=='__main__':
    main()
