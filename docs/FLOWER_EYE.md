# Flower eye, wants, and the thinking center

How the bee recognizes flowers and how it decides which ones to go for. Everything here is
engineered scaffolding around the fixed connectome; none of it is a claim about feeling.

## Data and training

| Step | Command | Output |
|---|---|---|
| Cut the sprite sheet | `python -m flybrain.arena.flower_sheet sheet.png` | `data/bss-flowers/{tiles,sprites}/` (git-ignored game art; keep local) |
| Preview generated views | `python -m flybrain.arena.flower_scenes --preview out.png` | montage; red = flower label, yellow = negative object |
| Train the detector | `python -m flybrain.arena.train_flowers --context --filter --resolution full ...` | `models/perception-bss-v*.npz`, `reports/flower-training-v*.json` |

Generated scenes are axis-aligned lattices of upright sprites (straight rows and columns, constant pitch,
about 4% harvested cells), rendered through a pinhole camera at random pitch/yaw/distance, in bright and
dim lighting, with negatives that must not be called flowers: tokens, fake flowers, recolored sprites,
boxes, and screen-space clutter (big petals, fences, health bars, icons).

`flower_filter.clean` post-processes the mask: it removes outlines thinner than 3 px, big solid regions
(row fill > 0.94), and lone small blobs with fewer than two lattice neighbours. Dense merged fields keep
their gaps and survive.

## Measured so far

Synthetic held-out scenes (harder dim set), flower precision / recall:

| Model | raw | after cleanup | tokens called flower (raw / cleaned) |
|---|---|---|---|
| v3, 3x3 patches | 0.71 / 0.79 | 0.81 / 0.63 | 19.8% / 7.4% |
| v4, wide-context features | 0.69 / 0.85 | 0.78 / 0.70 | 18.5% / 8.0% |
| old arena model | 0.59 / 0.19 | n/a | 25.2% |

On four real screenshots there are no hand labels, so only overlays exist (`data/real-detect/`). The new
models find the lattice in all four, including dark frames; the arena model finds almost nothing there.
In the arena, none of the sprout-token pixels are called flowers (the arena model calls all of them
flowers), but the BSS detector finds only about 56% (33% after cleanup) of the arena's own flowers.
Not yet measured on labeled real frames.

Speed: about 1.3 s per frame at full resolution for v4 on this CPU (under load). Too slow for a 250 ms
decision loop; see Limits.

## Wiring into the neurons

```
RGB frame -> FlowerEye (detector + clean + hive model + color) -> ThirdPersonVision flower grid
          -> 2048 optic neurons -> connectome LIF -> readout
```

- `FlowerEye` answers the same `segment(image)` call as the old recognizer (class 1 flower, class 2 hive), so
  `NeuralController(..., perception=eye)` is the only switch. Enable with `--eye bss`; the checkpoint loader
  needs `allow_perception_change=True`, because the readout was trained on the old flower channel.
- Flower color (white / red / blue) comes from the pixels of the detected flowers.
- **Attention:** the thinking center passes a gain per color into `ThirdPersonVision.encode`. The wanted
  color drives the optic neurons at gain 1, other flower pixels at 0.4. This is top-down modulation of the
  neural input. The steering cue below does not pass through the connectome.

## Thinking center (`drives.py`)

Wants (set by the trainer): `--want-field "ROSE FIELD"` or `--want-color blue`. Needs come from the public
backpack and hive status: a full or converting bag, or standing in the hive, means deliver and every want is
paused so the base policy makes the hive trip.

Goal choice while collecting: a wanted field; else the field where it has collected the most of the wanted
color (`FieldMemory.best_field_for`, from per-color totals recorded at each collection); else DISCOVER if it
has never collected that color; else the nearest known field.

Steering cue priority: sprout tokens (when ordered) > nearest visible flower of the wanted color
(from the color masks, projected to the ground) > the memory cue for the chosen field.

The bee says what it wants through the communication center ("I want blue flowers", with the best known
field as evidence).

## Limits

- Imitation-trained steering only; no reinforcement fine-tuning of wants.
- Memory colors are learned at collection from the flower the dipper touches (contact sensing in the
  simulator), then recalled; there is no vision-only color learning yet.
- The detector was trained on synthetic Roblox-style scenes plus no arena frames; arena recall is low.
- Not yet fast enough for a 4 Hz loop at full resolution.
