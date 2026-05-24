# Data Statement

This export includes de-identified SMS and voice benchmark data required by the
released ablation scripts:

- `data/sms/in_domain`
- `data/sms/ood`
- `data/sms/evidence/keep`
- `data/voice/in_domain`
- `data/voice/ood`
- `data/voice/evidence/keep`

The evidence `keep` files contain the supervised evidence and rationale fields
used by the `label_evidence`, `label_rationale`, reconstruction, and last-label
ablations.

Model checkpoints, raw historical benchmark outputs, TensorBoard logs, and
local archives are not included.
