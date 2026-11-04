# SPDX-License-Identifier: GPL-3.0-or-later
"""Training-only student-state corrections for late-economy failures."""
import json
import sys
from pathlib import Path
import numpy as np
from .app import ArenaApp
from .world import Action
from .train_market import configure,evaluate,expert


def main():
    V=sys.argv[1] if len(sys.argv)>1 else 'v6'  # model version to read and write
    app=ArenaApp(offscreen=True,width=640,height=400)
    app.connect_brain(monitor=False,checkpoint=f'models/controller-market-{V}.npz')
    trained=app.brain
    trained.learning=False
    trained.sample_actions=False
    with np.load(f'data/market-demonstrations-{V}.npz') as data:
        features=list(data['features'])
        labels=[Action(strafe=float(a[0]),forward=float(a[1]),dip=bool(a[2]),jump=bool(a[3]),convert=bool(a[4]),buy=int(a[5])) for a in data['actions']]
    trained.save(f'models/controller-market-{V}-before-refinement.npz')
    corrections=[]
    try:
        scenarios=[dict(seed=9400,capacity=300,stickers=3,limited_sold_out=True,wallet=23,initial_pollen=180,spawn_y=23.5),
                   dict(seed=9401,capacity=300,stickers=4,limited_sold_out=True,wallet=7),
                   dict(seed=9402,capacity=300,stickers=6,limited_sold_out=True,wallet=30),
                   dict(seed=9403,capacity=300)]
        for n,spec in enumerate(scenarios):
            configure(app,**spec)
            for t in range(192):
                predicted=app.choose_brain_action()
                target=expert(app.world)
                features.append(trained.last_features.copy())
                labels.append(target)
                app.step(predicted if t%4==3 else target,frames=15)
            app.finish_brain_feedback(terminal=True)
            fit=trained.fit_demonstrations(features,labels,steps=800)
            row=dict(spec,collected=app.world.total_collected,converted=app.world.total_converted,fit=fit)
            corrections.append(row)
            print('CORRECTION',n+1,json.dumps(row),flush=True)
        trained.save(f'models/controller-market-{V}-refined-candidate.npz')
        tests=[dict(seed=9500,capacity=300,steps=480),
               dict(seed=9501,capacity=300,steps=480,stickers=3,limited_sold_out=True,wallet=23),
               dict(seed=9502,capacity=300,steps=480,stickers=6,limited_sold_out=True,wallet=30),
               dict(seed=9503,capacity=60,gap=17,initial_pollen=60,steps=128)]
        results=evaluate(app,trained.weights.copy(),trained.reward_system.value_weights.copy(),tests)
        passed=all(r['converted']>=r['capacity'] and r['purchases']>0 and abs(r['last_position'][1])<22 for r in results)
        if passed:
            trained.save(f'models/controller-market-{V}.npz')
        report={'corrections':corrections,'held_out':results,'promoted':passed,
                'method':'Supervised corrections of student-visited states after initial BC + RL; no teacher in held-out evaluation',
                'scope':'Fixed corridor; expensive items require saving across trips. No proof of economic optimality.'}
        Path(f'reports/market-refinement-{V}.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        print('Refinement promoted:',passed,flush=True)
    finally:
        app.brain=trained
        app.destroy()


if __name__=='__main__':
    main()
