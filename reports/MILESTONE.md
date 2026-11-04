# First connectome milestone

Completed November 3, 2026.

- Downloaded original FlyWire FAFB v783 proofread connectivity and neuron IDs.
- Retrieved annotations at Git revision `a83b2776d60d5764cef36b927f5f9679c16c47a2`.
- Verified published connectivity MD5 checksums and preserved SHA-256 source hashes.
- Built a read-only-query SQLite graph with 139,255 neurons, 15,091,983 directed
  neuron pairs, 16,847,997 connection-region rows, and 54,492,922 anatomical synapses.
- Found 616 neurons without connections in this graph. Missing annotation fields
  and candidate populations are detailed in `inspection.json` and `inspection.md`.
- Saved a portable, reversible 1,000-neuron selection in `selections/experiment-002.json`.
- Ran a CPU propagation benchmark with 35,135 directed connections representing
  290,236 anatomical synapses. Median time was 0.105 seconds for 500 updates across
  three repetitions after warm-up. Core-array storage was 586,160 bytes; that is
  not peak process RAM. See `benchmark-002.json` for measurements and configuration.
- Seven synthetic correctness tests pass, including source identity portability,
  tamper detection, exact integer IDs, graph counts, and the neural update equation.

System memory measured during setup: about 23.79 GiB usable physical RAM and 6.14 GiB
available at that moment. CPU identification and software versions are in the
benchmark report. GPU identity and performance have not been verified.

The benchmark is an unsigned normalized rate network on real anatomical connectivity.
It demonstrates reproducible signal propagation and measured runtime. Learning,
biological fidelity, game control, and sibling development remain future work.

The next research step is a literature-backed circuit selection and a tiny learning
environment with controls and held-out evaluation. See `docs/ROADMAP.md`.
