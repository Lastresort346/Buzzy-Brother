# Fields, field memory, and fill-the-bag quests

## Zones

The arena has six flower fields (`ArenaWorld.patches`): the original BLUE, WHITE and
ROSE fields plus EAST, SOUTH and NORTH-WEST. New fields reuse the three recognized
flower colors, so a field can only be identified by where it is. Perception was not
retrained for the new layout; check its held-out scores before relying on it.

## Memory nodes (`flybrain/arena/memory.py`)

An engineered module beside the connectome, not part of it.

- **Position estimate:** integrates the distance the body actually walked (odometry),
  so being stuck against an obstacle adds no drift. It starts from the known start
  position at each reset; there is no landmark re-anchoring yet, so real noise would
  accumulate. The policy never receives world coordinates.
- **Nodes:** each successful collection writes or updates a node at the current
  estimate; collections within 6 units merge. During training a tutor supplies the field
  name, only while the bee stands inside that field (`app.teach`). At run time nothing is named.
- **Recall:** `--goal "ROSE FIELD"` (or `app.goal`) recalls that node and gives a
  4-value cue: unit direction x/y, distance/30, arrived. An unknown field gives zeros.
- **Use:** `NeuralController` adds `nav_weights @ cue` to the 9 movement logits. These
  weights are separate from the main readout, so older checkpoints still load.
  Nodes are saved to `models/field-memory.json` and loaded with a checkpoint.

## Discovery nodes

- `--goal DISCOVER` points the bee at the nearest unexplored area. The arena is a grid of
  8-unit cells; a cell counts as explored once the bee's estimate passes within 5 units.
  Coverage resets each episode; nodes persist.
- Collecting somewhere with no nearby node creates an **unnamed node**: a discovery.
  `memory.discoveries()` lists them and `memory.name_here('LAVA FIELD')` names the nearest
  one, after which that name works as a goal.
- The goal-steering weights only read the cue's direction, so exploring reuses the quest
  weights; no extra training stage is needed. Whether the bee dips on arrival depends on its
  existing collection policy.
- Check: `python -m flybrain.arena.train_quest --discover-only` hides each field in turn
  from memory, explores from the hive, and records whether the field is found, then
  named and recalled (`reports/discovery-v1.json`). Needs the quest checkpoint and memory.

## Stage 1 quest training

```powershell
.\.venv\Scripts\python.exe -m flybrain.arena.train_quest
```

1. **Tour:** a training-only teacher visits all six fields so the memory forms and the
   tutor names each node.
2. **Demonstrations:** from the hive, the teacher goes to a named field and fills a
   300-pollen bag; the episode ends when the bag is full. The teacher reads true field
   centers; the student sees RGB, public status and the recalled cue.
3. **Fit:** only the goal-steering weights are fitted (one aggregation round with the
   student driving). Dipping, conversion and shop behavior are untouched.
4. **Evaluation:** no teacher. Each quest is also run with the goal cue removed.
   Results go to `reports/quest-fill-v1.json`.

Limits: imitation only (no reinforcement fine-tuning yet), fixed six-field layout,
hive start, noise-free odometry. Success is "bag full while nearest to the commanded field".

## Live window

In the interactive window the neural decision runs on a worker thread, so the display
keeps redrawing while the network computes. The action is applied when the decision
finishes, so live control has some latency that offscreen training does not.
