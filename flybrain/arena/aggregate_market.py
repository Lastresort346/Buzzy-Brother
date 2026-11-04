# SPDX-License-Identifier: GPL-3.0-or-later
"""Collect teacher labels on fully student-driven trajectories, then refit offline."""
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
    app.connect_brain(monitor=False,checkpoint=f'models/controller-market-{V}-refined-candidate.npz')
    trained=app.brain
    trained.learning=False
    trained.sample_actions=False
    with np.load(f'data/market-demonstrations-{V}.npz') as data:
        features=list(data['features'])
        labels=[Action(strafe=float(a[0]),forward=float(a[1]),dip=bool(a[2]),jump=bool(a[3]),convert=bool(a[4]),buy=int(a[5])) for a in data['actions']]
    rounds=[]
    best_score=(-1,-1)
    try:
        for iteration in range(4):
            configure(app,seed=9600+iteration,capacity=300)
            for _ in range(480):
                predicted=app.choose_brain_action()
                features.append(trained.last_features.copy())
                labels.append(expert(app.world))
                app.step(predicted,frames=15)  # Teacher supplies labels, never actions here.
            app.finish_brain_feedback(terminal=True)
            w=app.world
            row={'iteration':iteration+1,'collected':w.total_collected,'converted':w.total_converted,
                 'purchases':w.shop.purchases,'last_position':[w.x,w.y]}
            score=(w.total_converted,w.shop.purchases)
            if score>best_score:
                best_score=score
                # Preserve the policy that produced the rollout BEFORE another fit.
                trained.save(f'models/controller-market-{V}-student-success.npz')
            row['fit']=trained.fit_demonstrations(features,labels,steps=1000)
            rounds.append(row)
            trained.save(f'models/controller-market-{V}-aggregated-candidate.npz')
            print('STUDENT ROUND',json.dumps(row),flush=True)
        np.savez_compressed(f'data/market-aggregated-{V}.npz',features=np.array(features),
                            actions=np.array([[a.strafe,a.forward,a.dip,a.jump,a.convert,a.buy] for a in labels]))
        tests=[dict(seed=9700,capacity=300,steps=800),dict(seed=9701,capacity=300,steps=800)]
        results=evaluate(app,trained.weights.copy(),trained.reward_system.value_weights.copy(),tests)
        def passes(rows):
            return all(r['converted']>=900 and r['purchases']>=8 and r['longest_boundary_stay_seconds']<5 and r['mean_fill_at_conversion']>=.9 and
                       r['last_30_seconds_collected']+r['last_30_seconds_converted']>0 for r in rows)
        post_fit_results=results
        passed=passes(results)
        rolled_back=False
        if not passed:
            trained.load(f'models/controller-market-{V}-student-success.npz')
            results=evaluate(app,trained.weights.copy(),trained.reward_system.value_weights.copy(),tests)
            passed=passes(results)
            rolled_back=True
        if passed:
            trained.save(f'models/controller-market-{V}.npz')
        report={'student_rollouts':rounds,'validation_final_fit':post_fit_results,
                'validation_selected':results,'rolled_back_final_fit':rolled_back,'promoted':passed,
                'method':'Training-only labels on fully student-driven trajectories; frozen teacher-free held-out policy',
                'scope':'Fixed arena corridor; no proof of economically optimal actions or general navigation'}
        Path(f'reports/market-aggregation-{V}.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        print('Aggregated policy promoted:',passed,flush=True)
    finally:
        app.brain=trained
        app.destroy()


if __name__=='__main__':
    main()
