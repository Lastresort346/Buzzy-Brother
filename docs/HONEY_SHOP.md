# Honey purchases and the return-to-hive curriculum

The current experiment adds goal-valued collectibles to the hive. A purchase creates
a positive reward event only when honey is actually paid and an available item is
received. This is a programmed utility/dopamine-inspired signal, not a measurement
of subjective pleasure or a biological dopamine concentration.

| Key | Collectible | Starting price | Availability | Initial purchase utility |
|---|---|---:|---|---:|
| 1 | Bee sticker | 25 honey | Unlimited; price rises by 65%, rounded up, each purchase | 2 |
| 2 | Royal snack | 80 honey | Three in stock | 6 |
| 3 | Field charm | 200 honey | One in stock | 12 |

Sticker utility diminishes as `2 / (1 + 0.3 * stickers_already_owned)`. Limited items
give their configured utility once per successful purchase. These are collectibles;
they do not currently give equipment/stat boosts. Failed, unaffordable, sold-out, or
remote purchases give no utility and never deduct honey. Holding a key buys once;
another press is required for another transaction.

The shop uses the hive interaction area. Prices, stock, owned counts, wallet, and
purchase rewards are visible in the HUD/monitor and telemetry. E still starts the
automatic simulated-bee conversion process; buying uses keys 1/2/3 independently.
Automatic live episode resets preserve the wallet and shop scarcity. Manual R starts
a new economy. Offline training episodes start independent economies for fair comparisons.

## Why reward alone was insufficient

The previous readout compressed activity into 32 motor averages, and continued to
sample noisy actions during live gameplay. A larger purchase bonus cannot teach a
return route that is never experienced. The new readout retains:

- 32 motor-neuron activity pools;
- 13 public status channels measured through sensory-ascending neuron activity;
- 16 visual-projection and 12 spatial optic-neuron pools;
- pairwise interactions of status-neuron activity, to distinguish combinations such
  as a full backpack at the hive versus a full backpack elsewhere.

Inputs are still neural firing rates. Public status comprises inventory, the visible
hive/shop interaction prompt, affordability/stock information, and a camera-derived
nearby-flower signal. Raw world coordinates and teacher directions are not runtime
policy inputs. Live gameplay uses deterministic trained actions; exploration occurs
in the offline trainer. The anatomical graph remains fixed and preserved.

## Training and evaluation

`python -m flybrain.arena.train_market` generates labelled demonstrations using an
explicit privileged teacher **only during training**. Demonstrations begin with
buying/converting at the hive, then full bags at increasing distances, and finally
empty/partially filled bags at the flower patch. A supervised warm-up teaches the
action readout those examples, with optional student-state aggregation.

Reinforcement fine-tuning then uses collection, conversion, and purchase utility.
A small cost discourages waiting with a full bag. Distance-progress shaping is used
only in training return lessons; deployed and evaluated policies receive no direction
or distance shaping. A stronger demonstration policy is retained if the RL candidate
regresses on validation. Both checkpoints remain recoverable.

Held-out evaluation has no teacher intervention and scores actual collected pollen,
converted pollen, paid purchases, spending, and complete cycles. Preloaded-bag return
lessons are reported separately from empty-start full cycles. See
`reports/market-training-v5.json` for measured results and any remaining failures.
The curriculum uses the present fixed arena/corridor; broad navigation generalization
and economically optimal purchasing are not assumed.

The initial v5 run passed all three short held-out cycle checks. Its empty-start
300-capacity trial collected 780 pollen, converted 600, and made seven purchases in
80 simulated seconds. However, it ended at the north edge with a partial bag. A
separate 300-second trial converted only one bag before stalling. These short-cycle
successes therefore **did not establish sustained farming**.

`python -m flybrain.arena.refine_market` adds student-state corrections from the
observed edge failure and economies with depleted limited stock and expensive
stickers. Its report, `reports/market-refinement-v5.json`, records the frozen,
teacher-free tests and whether the candidate was promoted. The initial policy is
preserved as `models/controller-market-v5-before-refinement.npz`. Use
`python -m flybrain.arena.evaluate_market` for a longer trial of the deployed model.
The untrained comparison has a shorter duration, so totals are not a matched-rate
comparison; no ablation currently isolates the effect of purchase rewards alone.

`python -m flybrain.arena.aggregate_market` then labels fully student-driven
trajectories and refits offline. The teacher supplies training labels but never
executes an action in these rollouts. This exposes missed E presses and return
decisions at intermediate stock/wallet combinations. Aggregated features/labels
are saved in `data/market-aggregated-v5.npz`.

Its final promotion gate (`reports/market-aggregation-v5.json`) requires both
held-out 200-second, empty-start, 300-capacity trials to convert at least 900 pollen,
make at least eight paid purchases, spend less than five continuous seconds at an
outer boundary, and still collect or convert in the last 30 seconds. Only a passing
candidate replaces the default checkpoint. These tests measure sustained behavior
in this arena, not economic optimality.

## Deployed result

The final aggregation fit regressed and was rejected. The preceding checkpoint was
recovered from its immutable run archive and evaluated separately. In two 200-second
empty-start trials (300 capacity), it converted **2,683 / 2,593 pollen** and made
**11 paid purchases in each**. Neither trial spent time at an outer boundary; both
continued gathering and converting during the final 30 seconds. This checkpoint is
now the default `models/controller-market-v5.npz`.

`reports/market-deployment-v5.json` records the selected file, its SHA-256, validation
metrics, and limitations; `reports/market-fallback-v5.json` contains the detailed
trials. These results validate the combined demonstration/reward/correction training
procedure in two fixed-corridor cases. They do not establish optimal purchase choices,
general navigation, or a causal benefit of shop rewards alone. Trainer checkpoints
are now preserved before subsequent fits so a regression can be rolled back.
