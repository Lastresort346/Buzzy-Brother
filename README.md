# FlyBrain

Buzzy-Brother: an open source AI project to create a modified version of the fruit fly brain that can play Roblox games as players would.

An early research project investigating whether fruit-fly-connectome-derived neural
systems can learn useful behavior and eventually support independently developing
agents. The long-term setting is Bee Swarm Simulator; the first milestone is data
acquisition, inspection, and a reproducible small neural benchmark.

**Current status:** data tooling, an experimental CPU rate-network benchmark, and a
local 3D motor-training arena. An experimental spiking connectome controller now
receives its camera pixels and drives movement, jumping, and the dipper. Its action
readout trains on collection, hive conversion, and honey-funded purchases, using
demonstrations followed by reinforcement fine-tuning. The visual recognizer
is measured on held-out synthetic views; navigation is evaluated in a fixed arena
curriculum, with no claim of broad generalization. There is no Roblox integration. Original
biological data are preserved; selections remain reversible.

The **640 × 400 third-person camera** feeds trained flower/hive recognition, a
24 × 16 spatial grid, directional motion, and avatar masking. The monitor includes
camera, recognition, and motion previews. See [docs/COLLECTION_TRAINING.md](docs/COLLECTION_TRAINING.md).

The arena has six fields and a path-integration **field memory** so the bee can be told
to go to a named field and fill its bag (`--goal "ROSE FIELD"`). See
[docs/FIELD_MEMORY.md](docs/FIELD_MEMORY.md).

## Neural control and live monitor

After installing the arena dependencies and acquiring/building the connectome:

```powershell
.\launch_brain.ps1
```

This opens the arena and a separate neural monitor. The new 3,304-neuron selection
preserves the original benchmark core and adds optic/visual-projection neurons.
The monitor shows measured spikes, a population firing trace, behavioral state,
a **happiness/reward proxy**, and **escape attempts** defined as sustained outward
wall pushes. It does not infer subjective experience or intentions.

Neural sessions default to the **collection/conversion cycle** with an empty
300-pollen backpack. Press E once at the hive to start automatic, step-by-step bee
conversion until the backpack is empty. Compatible trained
readouts load automatically; `--fresh` starts a new one. Separate `--task pollen`
and `--task convert` lessons support tool learning. The monitor shows backpack fill,
honey, purchases, recognition, and prediction errors. At the hive, keys **1/2/3** buy
collectibles with honey: stickers get more expensive, snacks and charms have limited
stock. Successful purchases give additional reward. See [docs/HONEY_SHOP.md](docs/HONEY_SHOP.md)
for the economy, training procedure, and measured results. Live runs use deterministic
trained actions; exploration and weight updates happen in the offline trainer.

**Tab** toggles neural/manual control; **Esc** pauses both simulation and learning.
Each session writes configuration, telemetry, and an exit-time readout checkpoint to
`runs/arena-brain/`. See [docs/NEURAL_CONTROL.md](docs/NEURAL_CONTROL.md) for what the
controller does, which parts are engineered, and how it is checked.

## Play the 3D training arena

Install the optional arena dependency into the same environment after setup:

```powershell
python -m pip --python .\.venv\Scripts\python.exe install -r requirements-arena.txt
.\launch_arena.ps1
```

Use **WASD / arrow keys** to move, **Space** to jump, and **hold left mouse** to
use the dipper. **E** toggles hive conversion; **1/2/3** buy at the hive. **R** resets, **Esc** pauses, and **F12** saves a screenshot. Close the
window to exit. Camera angle and third-person zoom are locked by default. The arena
is 48 by 48 simulation units, with three flower patches, walls, and jumpable obstacles.
There is no flight or limb-control action. The dipper has a 0.8-second cooldown and
collects local pollen up to backpack capacity; flowers regrow over 12 seconds.
Five simulated bee workers take turns converting 15 pollen every 0.25 seconds by
default (60 pollen/second), producing one honey per pollen. Press E again to stop;
leaving the hive area also cancels the process. These rates are prototype settings.

Optional later stage: `python -m flybrain.arena --free-camera` allows right-mouse orbit
and wheel zoom, always between distances 8 and 20; first person is unavailable.

See [docs/ARENA.md](docs/ARENA.md) for the Python controller interface and reproducible
graphics check. These are original simplified mechanics, not a verified replica of
Roblox or Bee Swarm Simulator physics and tool values.

## Setup (Windows PowerShell)

Python 3.14 was used for initial validation. Create an isolated environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If environment creation cannot bootstrap pip but your main Python already has pip:

```powershell
python -m pip --python .\.venv\Scripts\python.exe install -r requirements.txt
```

## First analysis

Run from the repository root. Acquisition needs internet access and downloads about
900 MB. Allow several GB of disk space for originals, SQLite, and temporary import
storage. First import can take several minutes.

```powershell
.\.venv\Scripts\python.exe -m flybrain acquire
.\.venv\Scripts\python.exe -m flybrain verify
.\.venv\Scripts\python.exe -m flybrain build
.\.venv\Scripts\python.exe -m flybrain inspect
.\.venv\Scripts\python.exe -m flybrain search descending --limit 5
```

The report is written to `reports/inspection.md` with a detailed JSON companion.
Source bytes, hashes, connectivity version, and annotation revision are recorded in
`data/raw/flywire-783/manifest.json`. The database is `data/connectome.sqlite`.
Rerunning acquisition checks existing files; builds refuse to overwrite a database.
For an independent reproduction of the initial experiment, use
`acquire --annotation-commit a83b2776d60d5764cef36b927f5f9679c16c47a2`
instead of plain `acquire`. Source identity excludes download timestamps, so identical
source bytes and annotation revisions work with shared schema-v2 selections.
Failed downloads remain `.part` files and restart on retry. Failed database builds
remain `.building` files; choose another `build --output` for a fresh attempt.

## Reversible selection and benchmark

Choose a real root ID from the report or search output:

```powershell
.\.venv\Scripts\python.exe -m flybrain select --seed ROOT_ID --limit 1000 --output selections/experiment-001.json
.\.venv\Scripts\python.exe -m flybrain benchmark --selection selections/experiment-001.json --steps 500 --repeats 3 --output reports/benchmark-001.json
```

Replace `ROOT_ID` with the numeric ID. Selection saves explicit IDs and source identity;
all internal connections are retained. It never deletes data. To undo a selection,
stop using that selection file. Create a new file for each variation.

The deterministic traversal explores incoming and outgoing neighbors, strongest first.
Its minimum synapse count affects traversal only; the induced benchmark graph retains
all source edges between selected neurons. It is an engineering subset, not a validated
sensory-motor circuit.

The portable reference selection is `selections/experiment-002.json`, with results in
`reports/benchmark-002.json`. Experiment 001 is retained as a local historical run;
its earlier schema ties it to the complete acquisition manifest, including its timestamp.

The benchmark uses a simple leaky tanh rate model, with incoming-normalized positive
weights and a seeded input stimulus. It records timing, array storage, final activity,
configuration, source identity, software, and platform. Timing is measured after warm-up
over repeated runs. Array storage excludes process peak RAM. No GPU acceleration or
biological time scale is claimed.

## Validation and next steps

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Next: review candidate circuits against primary literature, characterize dynamics,
measure equivalent CPU/GPU implementations, and define a tiny navigation learning task
with explicit observations, actions, baseline, and held-out evaluations. Demonstrate
one learning system before expanding to the five-sibling experiment.

Contributions are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).
Project code and project-authored documentation use [GNU GPL version 3 or later](LICENSE)
(`GPL-3.0-or-later`). Copyright (c) 2026 FlyBrain contributors.

You may redistribute and modify this program under the GNU General Public License
as published by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version. It is distributed without any warranty;
see the license for details.

GPLv3 accommodates combinations with Apache-2.0 components, including PyArrow.
Third-party software keeps its own notices and license.
FlyWire datasets retain their own terms; see [DATA_SOURCES.md](DATA_SOURCES.md)
for provenance, citations, and scientific limits.
