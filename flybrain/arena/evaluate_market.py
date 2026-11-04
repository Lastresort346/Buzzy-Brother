# SPDX-License-Identifier: GPL-3.0-or-later
"""Frozen untrained comparison and a longer deployment-policy economy trial."""
import json
import argparse
from pathlib import Path
from .app import ArenaApp
from .train_market import evaluate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',default='models/controller-market-v5.npz')
    parser.add_argument('--report',default='reports/market-endurance-v5.json')
    parser.add_argument('--steps',type=int,default=1200)
    parser.add_argument('--seeds',type=int,nargs='+',default=[9301])
    parser.add_argument('--skip-baseline',action='store_true')
    args=parser.parse_args()
    app=ArenaApp(offscreen=True,width=640,height=400)
    app.connect_brain(monitor=False,checkpoint=None)
    try:
        brain=app.brain
        baseline=[] if args.skip_baseline else evaluate(app,brain.weights.copy(),brain.reward_system.value_weights.copy(),
                                                       [dict(seed=9300,capacity=300,steps=320)])
        brain.load(args.checkpoint)
        trained=evaluate(app,brain.weights.copy(),brain.reward_system.value_weights.copy(),
                         [dict(seed=seed,capacity=300,steps=args.steps) for seed in args.seeds])
        result={'frozen_untrained_80_seconds':baseline,'trained_trials':trained,
                'checkpoint':args.checkpoint,'requested_steps':args.steps,
                'caveat':'Different durations; compare completion, not raw totals. No purchase-reward ablation; this tests the combined demonstration/RL curriculum, not the causal effect of the shop alone.'}
        Path(args.report).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({k:[{key:row[key] for key in ('collected','converted','purchases','last_position',
                         'longest_boundary_stay_seconds','last_30_seconds_collected','last_30_seconds_converted')}
                            for row in rows] for k,rows in result.items() if isinstance(rows,list)},indent=2),flush=True)
    finally:
        app.destroy()


if __name__=='__main__':
    main()
