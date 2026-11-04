# Flower recognition and collection/conversion training

The current system combines supervised visual recognition with a connectome-derived
spiking controller. Recognition and behavioral learning are measured separately.

## Image processing

The camera renders 640 × 400 RGB. A small NumPy MLP processes local RGB patches at
320 × 200 to segment background, visible flowers, and the hive/conversion station.
It uses color, neighboring pixels, local texture, and gradients. It does not read
object IDs, labels, or target coordinates at runtime.

`train_perception` renders an oracle class pass only for supervised training and
evaluation. Scene state is restored after that pass. Training, validation, and test
use separate seeds/views: 30 training frames, 12 validation frames for selecting
confidence thresholds, and 15 held-out test frames. Colored non-flower objects are
included as negative examples. The measured held-out pixel results are:

| Class | Precision | Recall | Intersection over union |
|---|---:|---:|---:|
| Flower | 86.4% | 89.8% | 78.6% |
| Hive/conversion station | 98.7% | 98.2% | 96.9% |

These results apply to this procedural arena. They do not establish recognition in
Bee Swarm Simulator or general scene understanding. Exact metrics and split seeds
are in `reports/perception-v3.json`; the trained recognizer is `models/perception-v3.npz`.

Recognition confidence is pooled with local brightness, color, edges, and motion on
a 24 × 16 grid. Small confidently recognized flowers are preserved by max pooling,
rather than diluted into a mostly green tile average. A reversible selection contains
3,304 source neurons, including 2,048 optic inputs (908 left, 1,140 right), and 146,364
directed connections. Screen receptive fields remain engineered, not biologically
validated retinotopy. The two optic sides share one camera with central overlap.

## Backpack and hive rules

- Backpack capacity defaults to 300 pollen and is configurable.
- Collection stops at capacity. Partial collection removes only accepted pollen
  from a flower; overflow does not destroy its remaining resource.
- Press **E once** inside the gold conversion pad at the hive while grounded.
  Conversion latches on and continues automatically until the backpack is empty.
- Five simulated bee work slots take turns processing a batch every 0.25 seconds.
  At the default rate each batch converts 15 pollen into 15 honey; the last batch
  can be smaller. Worker identity, batch count, progress, and honey are observable.
  These are simulated work steps, not additional bee agents or rendered bee bodies.
- Conversion averages 60 pollen/second. Fractional custom rates accumulate across
  batches. Releasing E does not stop the process; pressing E again or leaving the
  hive area cancels it. A held key does not repeatedly toggle. Reset clears it.
- Empty inventory cannot produce honey. Honey and remaining pollen are shown on
  the human HUD and monitor. Automatic live episode resets retain the wallet and
  shop scarcity; manual resets and independent training episodes start fresh.
- Raw environment reward is accepted pollen plus twice newly produced honey.
  The neural reward module tracks these components separately and also applies
  the existing small empty-dip and blocked-movement costs.

The policy receives RGB and its own normalized backpack/honey status, encoded into
an engineered proprioceptive input on selected sensory-ascending neurons. These are
public inventory values visible to a player, not target locations or navigation hints.
Motor decisions still come from the network's motor/descending activity.

## Reinforcement learning

The initial training method is an on-policy TD actor/critic readout with eligibility
traces. It trains the action/value heads while the anatomical connectivity and visual
recognizer stay fixed. There are movement, jump, dipper, and conversion actions.

The bounded curriculum has three explicitly different lessons:

1. **Convert:** start at the hive with a preloaded backpack to learn the conversion
   control. Honey from this lesson is not evidence of finding or collecting pollen.
2. **Pollen:** start on a flower patch with an empty backpack to learn collection.
3. **Cycle:** start empty at the flower patch and attempt collection, return, and
   conversion without relocation or a scripted route during the episode.

The training tool evaluates random, frozen untrained, and frozen trained controllers
on separate held-out episode seeds. It reports collected pollen, honey, and complete
cycles. A complete cycle requires both collecting and converting at least one full
backpack in the cycle lesson. Updates or partial conversion alone do not establish
successful cycle learning. Read `reports/controller-training-v3.json` for actual
results; no successful full cycle is assumed.

The first bounded run, using the earlier hold-to-convert control, completed
12 training episodes and 767 readout updates. Its metrics below are historical,
not an evaluation of the corrected E-toggle interaction. Those weights can serve
as a warm start, but the conversion policy needs further training under the new control.
Across two held-out conversion lessons, random controls produced 115 honey, the
frozen untrained readout produced 182, and the trained readout produced 203. All
three produced zero honey in the held-out full-cycle lessons. This is a small
initial comparison, not evidence of robust improvement or successful return-to-hive
learning. Further work should lengthen the cycle curriculum and evaluate more seeds.

## Reproduce

```powershell
.\.venv\Scripts\python.exe -m flybrain.arena.train_perception
.\.venv\Scripts\python.exe -m flybrain.arena.train_controller --episodes-per-stage 4 --steps 64
.\.venv\Scripts\python.exe -m flybrain.arena.train_market
.\launch_brain.ps1
```

Recognition training must precede controller training. Default neural launches load
the compatible `models/controller-market-v5.npz` when present. This newer market
curriculum supersedes the historical v3 run above; see `docs/HONEY_SHOP.md` for
demonstration warm-up, reinforcement fine-tuning, and unguided cycle evaluation.
Use `--fresh` to start a new
readout, or `--checkpoint PATH` for an explicit compatible model. Loading validates
source IDs, graph identity, seed, action-head shape, and encoder/model versions.
This is a weight warm start, not an exact simulation-state resume.

The trainer preserves per-stage checkpoints and a final checkpoint. Interactive runs
retain telemetry and periodic checkpoints in `runs/arena-brain/`. Model versions from
the earlier graph or 12 × 8 encoder are incompatible and remain separate artifacts.

Human controls: WASD/arrows move, Space jumps, click dips, E toggles conversion, R resets,
Esc pauses, and Tab switches neural/manual control. The camera remains locked by default.
