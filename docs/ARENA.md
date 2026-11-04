# Wingless 3D motor arena

The first embodied environment is a standalone Panda3D application, separate from
Roblox. It uses original procedural geometry and a wingless fly avatar. Only standard
movement, jumping, dipper use, and staged camera actions are exposed. Six cosmetic
legs move with the body; there are no limb-action or flight channels.

## Run

From the project root after installing `requirements-arena.txt`:

```powershell
.\.venv\Scripts\python.exe -m flybrain.arena
```

Controls: WASD/arrows move relative to the camera; Space jumps; held left click
uses the dipper on its cooldown. Press E once at the hive to start automatic bee conversion.
R resets the seed-42 world. Esc pauses and clears
held inputs. F12 saves screenshots to `runs/arena/`. Window close exits.

The default camera follows position but keeps fixed rotation, elevation, and distance.
Mouse movement and wheel inputs cannot change it. In the explicit `--free-camera`
stage, right-drag orbits and wheel changes distance, clamped to third person.

## Controller interface

```python
from flybrain.arena import Action, ArenaConfig
from flybrain.arena.app import ArenaApp

env = ArenaApp(ArenaConfig(camera_locked=True), offscreen=True)
try:
    observation, info = env.reset(seed=42)
    observation, reward, terminated, truncated, info = env.step(
        Action(forward=1, strafe=0, jump=False, dip=True), frames=4
    )
finally:
    env.destroy()
```

- Observation: `(400, 640, 3)` RGB `uint8`, one third-person camera. The human HUD is
  excluded. There are no independent left-eye/right-eye inputs. Resize parameters
  affect the render window; `observation_size` controls returned pixel dimensions.
- Actions: forward/back and left/right strafe, normalized so diagonal movement is
  not faster; edge-triggered jump; held dip. Optional yaw/pitch/zoom deltas are
  ignored by the locked camera. Movement never directly manipulates limb joints.
  `convert=True` pulses E; `buy=1/2/3` buys the corresponding hive collectible.
- Time: fixed 1/60-second physics steps. `frames=4` holds buttons for four steps,
  with camera deltas applied once. Rendering is performed at observation boundaries.
- Reward: pollen actually collected, twice newly produced honey, plus 100 times
  successful purchase utility. No movement, idle, or repeated
  dry-swing reward is supplied. The neural controller additionally receives small
  empty-dip/blockage costs. Backpack capacity is 300; conversion at the hive requires
  an E toggle, then proceeds in timed simulated-bee batches at an average
  60 pollen/second, one honey per pollen. Releasing E does not cancel it; a second
  press, leaving the hive area, or empty inventory stops it. `app.step` pulses
  conversion input once per call; low-level `ArenaWorld` handles key rising edges.
  Purchases deduct honey, respect stock, and require the hive area. Stickers become
  more expensive with each purchase. See `docs/HONEY_SHOP.md` for neural reward scaling.
- Info: **privileged diagnostics**, including exact position, cooldown, and simulator
  state. Keep these out of the inputs to a camera-only policy.
- Episodes: truncate after 600 simulated seconds; there is no terminal death condition.
  A controller should reset when truncated. Manual play continues until reset or exit.
- Reset: an isolated seed recreates the flower layout and clears all transient state.

The app owns Panda3D's process-global renderer. Use one app per process and close it
with `destroy()`. It does not implement the Gymnasium package interface or vectorized
training yet. The CPU `ArenaWorld` can be stepped independently without graphics.

## Reproducible checks

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m flybrain.arena --offscreen --smoke-test --screenshot reports/arena-field.png
.\.venv\Scripts\python.exe -m flybrain.arena --offscreen --screenshot reports/arena-preview.png
```

The smoke check takes a scripted route from spawn to the blue flowers, confirms a
pixel change after movement, collects pollen, and verifies the cooldown. It writes
`reports/arena-smoke.json`. A scripted acceptance route is not learned competence.
`--software` offers an offscreen rendering fallback when hardware graphics are unavailable.

Unit tests cover repeatability, diagonal movement, collision, landing on obstacles,
gravity, dipper range/cooldown, camera locking, third-person limits, and episode timing.

## Next integration

An experimental connectome controller is available with `--brain`; see
`docs/NEURAL_CONTROL.md`. It includes an engineered pixel encoder, adaptive LIF
dynamics, and a reward-trained action readout. Useful learned competence remains
unproven. Compare it against random, untrained, and conventional-policy controls.
The earlier 1,000-neuron speed benchmark is not assumed to contain a complete
visual-to-motor circuit.

Physics constants, flower layout, dipper mechanics, and camera limits are prototype
choices rather than claims about Bee Swarm Simulator. No Roblox client is controlled.
