# Contributing to FlyBrain

FlyBrain is an early research project investigating connectome-derived learning systems.
The current deliverable is a reproducible acquisition, inspection, and benchmark tool.
There is no trained game-playing brain yet.

Useful contributions include:

- Improve dataset validation and annotation reconciliation, with small synthetic fixtures.
- Document candidate circuits using primary literature and exact annotation queries.
- Measure CPU or GPU implementations against an equivalent numerical reference.
- Build a small deterministic navigation environment with observable learning metrics.
- Investigate neural dynamics and plasticity, clearly separating assumptions from measurements.

Before proposing a large architectural change, open an issue describing the question,
evidence, expected outcome, and a small experiment that could disprove the hypothesis.

## Local checks

Install `requirements.txt` in a Python virtual environment, then run:

```powershell
python -m unittest discover -s tests -v
python -m flybrain --help
```

Tests use tiny synthetic graphs and do not download biological data. A change to
data import, selection, or neural dynamics should include a focused correctness check.
State your Python version and hardware when reporting performance. Include seeds,
configuration, source versions, and results; distinguish core-array storage from peak RAM.

## Research and data practices

Never edit or commit downloaded originals, local databases, account credentials, or
large checkpoints. Preserve source hashes and separate derived artifacts. Selections
must be reversible and retain neuron IDs without float conversion.

Annotation matches identify candidates, not proven functions. Missing labels are not
grounds for biological pruning. Transmitter predictions do not by themselves establish
synaptic sign, reward coding, subjective experience, or consciousness.

Keep experiments and pull requests small enough to review. Explain the problem,
behavioral change, validation, and remaining uncertainty in each pull request.

Contributions to project code and project-authored documentation are accepted under
GNU GPL version 3 or later (`GPL-3.0-or-later`), the same terms as the project.
Only submit work you have permission to contribute under those terms. External
datasets retain their own terms and attribution requirements; see `DATA_SOURCES.md`.
