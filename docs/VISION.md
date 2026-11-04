# Third-person camera adapter

Current v3 specifications, supervised recognition results, and collection/conversion
training are documented in [COLLECTION_TRAINING.md](COLLECTION_TRAINING.md).
The details below describe the earlier v2 adapter for reproducibility; its 12 × 8 grid
and 512 inputs have been superseded by a 24 × 16 grid and 2,048 optic inputs.

The simulated brain never received two eye images: its previous encoder sampled one
RGB frame and randomly assigned channels to optic neurons. That discarded useful
spatial structure. The new adapter still uses **one camera**, but preserves screen
regions and separates visual signals before driving the connectome.

## Input and mapping

- Raw observation remains 160 × 100 RGB, rendered from the third-person camera's
  actual pose and lens. Human HUD elements are excluded from this camera buffer.
- Area pooling preserves a 12 × 8 screen grid; it does not sample just one pixel per
  tile. There are luminance, red/green, blue/yellow, spatial-edge, temporal ON/OFF,
  and right/left/down/up local correlation channels.
- A conservative ellipse excludes the known on-screen avatar region. Masked pixels
  are excluded from pooling rather than treated as black objects. Low-confidence
  cells and edge/motion signals at the exclusion boundary are suppressed. This mask
  is a camera/body calibration, not target or flower segmentation.
- Mask size scales with known third-person camera distance and horizontal FOV.
  Its center/shape are tuned to the present avatar and camera. A changed avatar or
  substantially different pitch requires checking that calibration again.
- Left optic neurons receive columns 0–7; right optic neurons receive columns 4–11.
  Both groups share central columns. Neurons within each group have deterministic
  spatial receptive fields and four local response profiles: brightness/color,
  edge/ON/OFF, right/down motion, and left/up motion. All 512 inputs are mapped once.
- Source `side` annotations identify the two groups. **Screen-to-side routing and
  receptive fields are engineered choices**, not inferred biological retinotopy.
  Root-ID order does not establish where a neuron looks in the real fly.
- No synthetic second view, binocular disparity, metric depth, target locations,
  flower coordinates, or privileged environment segmentation are supplied.

Directional motion uses delayed neighbor correlations. It is a simple motion cue,
not a validated T4/T5 model or a dense optical-flow/depth estimator. Stationary scenes
yield zero temporal motion. Uniform brightness changes affect ON/OFF channels without
creating a preferred motion direction. Known rotation/zoom changes reset temporal
history so the transition is not interpreted as local object motion. Translation of
the camera-following character still produces scene flow; self-motion compensation
and learned perspective/depth are future work.

## Inspect and reproduce

The neural monitor displays the actual camera with a display-only amber avatar mask,
coarse spatial edges, and motion (right red, left blue; vertical motion green).
Raw camera pixels are not painted amber before feature extraction. The raster still
uses measured neural spikes, and selecting a row gives its exact source neuron ID.

Session configurations and checkpoints include encoder version
`third-person-spatial-v2`, calibration, and the full root-ID-to-receptive-field mapping.
Telemetry includes directional energy and optic-side counts. Earlier random-encoder
checkpoints are not interchangeable with this representation without retraining;
launches currently initialize a fresh controller as documented.

Run the unit suite and the real graph check:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m flybrain.arena --offscreen --brain-smoke-test --no-monitor
```

Tests cover spatial locality, one-camera invariants, deterministic mapping, avatar
exclusion, edges, ON/OFF responses, motion direction, camera-change resets, zoom-aware
masking, and finite inputs. The real-graph smoke check verifies pixel sensitivity and
recurrent-edge effects downstream in motor features. `reports/vision-upgrade-smoke.json`
records an additional equal-brightness left/right stimulus check and monitor fit.
These tests do not establish improved navigation, biological visual fidelity, or
generalization; those need controlled behavioral comparisons.

## Biological reference

The distinction between spatially arranged visual processing and motion pathways is
motivated by primary fly-vision work, including the
[2024 optic-lobe wiring study](https://www.nature.com/articles/s41586-024-07981-1)
and [connectome-constrained visual-system models](https://www.nature.com/articles/s41586-024-07939-3).
Our adapter does not reproduce their circuits or validated retinotopic maps. The
anatomical graph and its source data have not been rewritten to implement this change.
