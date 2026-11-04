# Data sources and scientific scope

## Connectivity

- FlyWire Consortium, *FlyWire Whole-brain Connectome Connectivity Data*, version 783.0.
- DOI: https://doi.org/10.5281/zenodo.10676866
- Dataset documentation: https://zenodo.org/records/10676866
- Dorkenwald et al. (2024), *Neuronal wiring diagram of an adult brain*:
  https://doi.org/10.1038/s41586-024-07558-y

The acquisition command downloads the proofread root IDs and connection table. The
connection table has one row per presynaptic neuron, postsynaptic neuron, and neuropil,
with a synapse count and average neurotransmitter probabilities. It is not the full
individual-synapse table. Its proofread graph also differs from a thresholded Codex export.
Published MD5 hashes are checked before promoting downloads, and SHA-256 hashes are
recorded in the local manifest. The command deliberately leaves the 9.5 GB synapse
coordinate file and neuropil summaries for a future experiment.

## Annotations

- Maintainer repository: https://github.com/flyconnectome/flywire_annotations
- `supplemental_files/Supplemental_file1_neuron_annotations.tsv`
- Schlegel et al. (2024), *Whole-brain annotation and multi-connectome cell typing of Drosophila*:
  https://doi.org/10.1038/s41586-024-07686-5

Annotations evolve independently of the connectivity snapshot. Acquisition resolves an
exact Git commit, saves it in an acquisition lock, and retrieves that revision. The
manifest records the commit and source hash. The repository README explains additional
citations required for newer annotation versions. Counts may differ from Codex because
its consolidated annotations have different sources. Annotation IDs outside the
proofread list are counted explicitly instead of silently adding neurons.

## Dataset terms

The project's GPL license does not relicense external data. FlyWire's public-release
guidelines currently specify CC BY-NC 4.0:
https://join.flywire.ai/guidelines

Keep dataset terms, source attribution, and the annotation repository's revision-specific
citation requirements with any use or redistribution. Raw data are ignored by Git and
downloaded from their maintainers rather than bundled in this project.

## Interpretation

An anatomical graph does not specify neuron dynamics, receptor-specific effects,
time constants, plasticity, sensory encoding, or motor decoding. The first CPU benchmark
uses unsigned normalized weights and a leaky rate update strictly as a reproducible
propagation workload. It does not demonstrate biological fidelity or learning.

Candidate populations in the inspection report are explicit annotation predicates.
They require literature review and targeted functional tests before circuit claims or
task-oriented pruning. The initial graph selection is an engineering sample only.
