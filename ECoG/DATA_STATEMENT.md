# Data Statement

This export includes de-identified SMS and voice benchmark data required by the
released ablation scripts:

- `data/sms/evidence/keep`
- `data/voice/evidence/keep`

The evidence `keep` files contain the supervised evidence and rationale fields
used by the `label_evidence`, `label_rationale`, reconstruction, and last-label
ablations. The released runners read all train, validation, test, and
challenging splits from these two `evidence/keep` directories.

Model checkpoints, raw historical benchmark outputs, TensorBoard logs, and
local archives are not included.
