# Export Manifest

This package is scoped to the seed-10 rationale/evidence ablation suite.

## Included

- `README.md`
- `DATA_STATEMENT.md`
- `pyproject.toml`
- `configs/README.md`
- `configs/benchmarks/main_ablation.yaml`
- `data/sms/in_domain`
- `data/sms/ood`
- `data/sms/evidence/keep`
- `data/voice/in_domain`
- `data/voice/ood`
- `data/voice/evidence/keep`
- `src/phishdec`
- selected benchmark execution scripts under `scripts/`
- selected parsing, metric, and aggregation scripts under `scripts/`
- `results/README.md`

## Excluded

- trained model checkpoints
- historical raw benchmark outputs
- TensorBoard logs
- generated configs from previous runs
- local archives
- unrelated ML/DL, encoder, zeroshot, queue, Slack, and notebook helpers

## Checkpoint Policy

The export does not ship model weights or fine-tuned checkpoints. Reproduction
runs create fresh checkpoints under `outputs/runs/...`.
