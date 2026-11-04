# Experimental embodied neural controller

Run `launch_brain.ps1`, or `python -m flybrain.arena --brain`, after installing arena
dependencies and building the local connectome database. This opens the arena and a
second native Tk window with measured neural activity and state. Tkinter is part of
standard Windows Python installations. `--no-monitor` omits that second window.

## What is connected

1. The arena supplies a 640 × 400 third-person RGB image, with the human HUD hidden.
2. A trained RGB-patch recognizer identifies flowers and the hive at 320 × 200.
   A third-person spatial adapter preserves a 24 × 16 screen grid and computes
   luminance, color opponency, edges, ON/OFF changes, and directional motion. It
   excludes a known avatar screen region and maps local responses to 2,048 annotated
   optic neurons: 908 left and 1,140 right, with overlapping central camera regions.
   This is an **engineered input mapping**, not established biological retinotopy or
   stereo vision. Recognized classes come from RGB inference, not runtime oracle
   segmentation or map locations. See `docs/COLLECTION_TRAINING.md` for evaluation.
3. The original 1,000-neuron core is preserved. A new selection adds 256 incoming
   visual-projection/visual-centrifugal neighbors, then 2,048 optic neighbors with strong
   connections into that expanded core. The induced graph contains 3,304 neurons,
   146,364 directed pairs, and real source synapse-count weights.
4. Adaptive-threshold leaky integrate-and-fire units run at a 5 ms timestep, with a
   20 ms voltage time constant and a 250 ms threshold-adaptation decay. Each spike
   adds 0.15 to its threshold; there are two refractory simulation steps.
5. Incoming-normalized **unsigned positive** recurrent weights drive the model with
   gain 4. Background drive is 0.95; optic drive is 0.55 plus 3.8 times a local visual response,
   with seeded Gaussian input noise of standard deviation 0.10. No physiological
   inhibitory signs or transmitter-specific neuromodulation have been established.
6. The firing rates of 297 annotated descending/motor neurons are pooled into 32
   features. Additional pools retain visual-projection, regional optic, and public-status
   activity, with status interactions (151 features plus bias). Thirteen public status
   channels drive 47 sensory-ascending neurons. The readout chooses eight movement
   directions or idle, dipper/jump/E decisions, and no purchase or one of three items.
   Training samples actions; live deployment uses deterministic choices. These are game controls, never limb
   joints. Actions last 15 physics steps (250 ms); camera rotation/zoom remain locked.
7. An experimental actor/critic readout learns from a collect/convert/buy reward system.
   Reward is `min(pollen/100, 5) + 2*min(honey/100, 5) + purchase_utility - 0.01*full_wait_fraction - 0.025*empty_dips - 0.05*blocked_fraction` per control
   interval. A learned value head estimates future task return. Credit is assigned
   after the next camera observation using `delta = r + 0.98*V(next) - V(previous)`.
   Terminal transitions use zero next value. Separate appetitive/aversive channels
   track positive/negative prediction error; the bounded phasic signal is `tanh(delta)`.
   A slow 0.975-retention reward trace modulates motor excitability by at most 10%.
   Eligibility traces retain recent action credit. The anatomical connectivity itself
   is not trained here; a learned value function replaces the earlier scalar baseline.

The connectome genuinely participates in the signal path, but this is a hybrid
experimental controller, not a validated reconstruction of a fly brain. An untrained
controller initially explores stochastically. There is no hard-coded route to fields,
and no claim of general navigation, purposeful escape, or broad game competence.
The market curriculum uses privileged demonstrations only in training, then evaluates
the learned policy without a teacher. See `docs/HONEY_SHOP.md` and its linked report.

## Pollen collection lesson

Neural launches default to `--task cycle`: start at `(-10, -1)` on the blue
flower patch, with 600-second episodes. Automatic resets preserve shop scarcity and
the honey wallet; manual R starts a new economy. This is an explicit tool-use curriculum,
not a navigation achievement. The backpack holds 300 pollen; conversion requires
pressing E once inside the hive pad. Five simulated bee workers process timed
batches until empty; E again or leaving the hive cancels it. `--task pollen` is the collection lesson and
`--task convert` starts at the hive with a preloaded pack for tool learning.
All actions still come from the neural readout;
there is no scripted collection route. Real collection provides positive feedback,
executed empty swings get a small cost, and attempted movement with negligible
horizontal displacement gets a small blocked-movement cost. Idle periods and merely
holding a dipper during its cooldown do not incur an empty-swing penalty.

Use `--task explore` to retain the original navigation start and episode duration.
The objective is collecting pollen, converting it into honey, and buying reward-valued
collectibles at the hive. The simulator has no fruit objects, fruit
reward, or fruit-search policy. No olfactory or reproductive circuitry was pruned
to change its behavior.

## Monitor definitions

- **Raster:** actual spike counts in 250 ms neural bins for up to 72 sampled neurons.
  Choose overview, optic input, or motor/descending samples. Select a row for its exact
  source ID and rate. This is a sampled display, not every neuron's raster.
- **Visual previews:** thumbnails of the higher-resolution single camera with the
  avatar exclusion region marked, RGB-based flower/hive recognition, and directional motion. These previews are computed from real
  camera pixels; the mask overlay is display-only.
- **Population trace:** mean exponentially smoothed firing rate across all selected
  neurons in green (left axis, Hz), plus the bounded phasic dopamine-inspired signal
  in amber (right axis, -1 to +1). Pauses freeze these graphs. The monitor also shows
  predicted value, raw prediction error, and separate pollen/honey/cost components.
- **Active neurons:** neurons with at least one spike in the last decision interval.
- **State:** observable action/outcome labels: exploring, idle, collecting, jumping,
  blocked, manual, or paused. These labels do not establish psychological experience.
- **Happiness / reward proxy:** bounded 0–100 indicator:
  `clip(50 + 45*tanh(reward_trace) - 25*blocked_fraction, 0, 100)`. The reward trace
  adds environment reward/100 (collection, conversion, and paid-purchase utility)
  and decays with a 10-second time constant. Blockage uses a 2-second
  exponential average. Baseline is 50. This is observer telemetry, not a subjective
  emotion measurement and not a second reward input to the controller.
- **Escape attempt:** at least 0.5 seconds of continuous outward movement against an
  outer boundary. One continuous push counts once; stopping outward input rearms it.
  Both session and episode totals are shown. Interior obstacle collisions do not
  count, and the counter does not imply an intention to escape.

Raw positions and boundary geometry are not policy inputs. The environment supplies
a scalar blocked-movement outcome for reinforcement, separately from the camera.
Boundary geometry is used for the observer's escape counter. Manual control does not
train the readout. Reset clears neural dynamics and eligibility traces while retaining
the current session's learned readout. Session spike/escape totals remain cumulative.

## Controls, archives, and limitations

Tab switches between neural and manual control. Esc pauses simulation and learning.
The monitor has pause/resume, manual/neural, and episode-reset buttons. Closing the
monitor leaves the arena running. Closing the arena also closes the monitor.

Session files in `runs/arena-brain/<UTC timestamp>/` include the configuration,
observer/neural telemetry at roughly one-second intervals, immutable checkpoints
every 120 decisions (about 30 simulated seconds), and `readout.npz` saved on orderly
exit. Checkpoints store readout weights, exact source IDs, source identity,
seed, counters, RNG state, value-head weights, and the current reward-system metrics.
They are not a full exact-resume neural snapshot.
Neural launches automatically load a compatible `models/controller-market-v5.npz` if present.
Use `--fresh` for a new readout. Force-killing a process may bypass the exit-time save; periodic
checkpoints limit the amount of unsaved learning after the first 120 decisions.

The reversible selection is `selections/arena-vision-002.json`. It does not alter the
source files or earlier benchmark selection. Loading fails if source identity or the
base selection differs.

## Checks

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m flybrain.arena --offscreen --brain-smoke-test --no-monitor
```

The smoke check verifies that the real graph spikes, camera images change after
actions, multiple actions occur, third-person camera limits hold, sustained dark vs.
bright images affect motor features, and removing recurrent edges affects motor
features. It separately relocates the avatar to flowers **only inside the test** to
verify the reward-to-readout update path. This does not demonstrate autonomous
flower discovery or learned competence. Results are saved to `reports/brain-smoke.json`.

Next research work: evaluate learning against random, untrained, shuffled-connectome,
and conventional-policy controls; test inhibitory/transmitter hypotheses; validate
sensory/motor mappings and generalization. The new monitor makes failures visible.

## Biological reference and scope

Fly reinforcement research connects dopaminergic signaling, mushroom-body circuitry,
and prediction-dependent memory updates; it does not justify equating one scalar
with a biological dopamine center. Relevant primary sources include
[Felsenberg et al. (2017)](https://www.nature.com/articles/nature21716),
[a computational mushroom-body model (2021)](https://www.nature.com/articles/s41467-021-22592-4),
and [Huang et al. (2024)](https://www.nature.com/articles/s41586-024-07819-w).
Our current upgrade is an engineered TD reward module, not a reconstruction of those
circuits, receptor effects, dopamine concentration, or subjective happiness.
