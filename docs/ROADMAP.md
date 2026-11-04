# Research roadmap

## Current milestone: acquire, inspect, and measure

Acceptance criteria:

- Untouched upstream downloads with provenance, pinned versions, and checked hashes.
- A searchable local graph with exact neuron IDs and annotation reconciliation.
- Reports distinguishing connection-region rows, neuron pairs, and synapse counts.
- Explicit, reversible selections tied to the exact source manifest.
- A deterministic neural propagation benchmark with measured timing and stated limits.
- Small synthetic correctness tests that run without biological downloads.

## Next milestone: one learning experiment

1. Select candidate sensory, integration, and output populations using annotations
   and primary biological literature. Record evidence and unresolved assumptions.
2. Define observations and actions in a small navigation environment with controlled
   resets, finite episodes, and an observable goal.
3. Establish random and conventional learning baselines.
4. Specify the neural dynamics, input encoding, output decoding, reward mechanism,
   and which parameters may learn. Keep neural and policy state reproducible.
5. Evaluate across multiple seeds and unseen layouts. Include an untrained control
   and shuffled-connectivity ablation to test what the anatomical graph contributes.
6. Save checkpoints and compare learning curves, success rates, and compute cost.

No learning claim is established by activity propagating through a graph. A successful
experiment must improve through experience and hold up against its baselines.

## Later goals

- Integrate reliable motor control and increasingly complex game tasks.
- Give five instances independent memories, learning histories, and evolving preferences.
- Introduce measured social interaction and sentence selection.
- Consolidate validated capabilities into the sixth sibling, Buzzing Brother.

These are proposed research stages, not implemented features. Expansion to multiple
instances follows evidence from one learning system and measured resource budgets.
