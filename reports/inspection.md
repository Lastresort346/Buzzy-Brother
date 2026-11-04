# FlyBrain connectome inspection

- Proofread neurons: 139,255
- Directed neuron pairs: 15,091,983
- Connection-region rows: 16,847,997
- Anatomical synapses represented: 54,492,922
- Isolated neurons: 616

## Candidate populations

- sensory_candidates: 102,479 (annotation match only; requires circuit literature review)
- motor_output_candidates: 1,413 (annotation match only; requires circuit literature review)
- predicted_dopaminergic_candidates: 5,909 (annotation match only; requires circuit literature review)
- known_nt_exact_dopamine_candidates: 206 (annotation match only; requires circuit literature review)
- mushroom_body_candidates: 5,604 (annotation match only; requires circuit literature review)

## Missing annotations

| Field | Missing neurons |
|---|---:|
| cell_class | 31,744 |
| cell_sub_class | 113,430 |
| cell_type | 1,541 |
| dimorphism | 14 |
| fbbt_id | 110,353 |
| flow | 14 |
| fru_dsx | 135,947 |
| hartenstein_hemilineage | 104,452 |
| hemibrain_type | 105,987 |
| ito_lee_hemilineage | 101,713 |
| known_nt | 51,424 |
| known_nt_source | 51,424 |
| matching_notes | 138,881 |
| nerve | 129,611 |
| nucleus_id | 32,178 |
| pos_x | 14 |
| pos_y | 14 |
| pos_z | 14 |
| side | 44 |
| soma_x | 21,155 |
| soma_y | 21,155 |
| soma_z | 21,155 |
| status | 138,597 |
| super_class | 14 |
| supertype | 105,410 |
| supervoxel_id | 14 |
| synonyms | 136,792 |
| top_nt | 614 |
| top_nt_conf | 614 |
| vfb_id | 15 |

## Interpretation limits

- Connectivity is anatomical, not a validated dynamic brain model.
- Multiple neuropil rows can describe one neuron pair; rows are not individual synapses.
- Transmitter argmax is a prediction, not a validated synaptic sign or reward function.
- Annotation absence does not establish biological irrelevance. No circuits have been deleted.

## Sources

- Connectivity: https://zenodo.org/records/10676866
- Annotations: https://github.com/flyconnectome/flywire_annotations
- Exact annotation commit: `a83b2776d60d5764cef36b927f5f9679c16c47a2`

See the adjacent JSON report for source hashes, distributions, and import reconciliation.
