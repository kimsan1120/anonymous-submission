## Setup

Create and activate a Conda environment:

```bash
conda create -n ecog311 python=3.11 -y
conda activate ecog311
````

Move to the project directory:

```bash
cd ECoG
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Running Specific Ablations

### Label + Evidence Ablation

```bash
python scripts/run_ablation.py --method label_evidence --seed 10 --run
```

### Label + Rationale Ablation

```bash
python scripts/run_ablation.py --method label_rationale --seed 10 --run
```

### Label + Evidence + Rationale Ablation: ECoG

Runs ECoG with consistency weights from `0.0` to `0.2`.

```bash
python scripts/run_ablation.py --method label_rationale_rec --rec-lambdas 0,0.05,0.1,0.2 --seed 10 --run
```

### ECoG + Last Pooling

Runs ECoG with last-token pooling and `lambda = 0.1`.

```bash
python scripts/run_ablation.py --method label_rationale_rec_last_pooling --rec-lambda 0.1 --seed 10 --run
```

### Evidence + Rationale + Label: Label-Last Ablation

Runs the label-last target-order ablation.

```bash
python scripts/run_ablation.py --method last_label_ablation --seed 10 --run
```




### Per-fold OOD Macro-F1 breakdowns
| Model | Method | SMS ID-Test | SMS OOD-Test | SMS ID-Chall | SMS OOD-Chall | Voice ID-Test | Voice OOD-Test | Voice ID-Chall | Voice OOD-Chall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF+Linear SVM | Feature-based Supervised Learning | 97.29 | 74.92 | 66.92 | 40.13 | 99.13 | 80.33 | 88.15 | 65.95 |
| TF-IDF+Logistic Regression | Feature-based Supervised Learning | 96.29 | 73.74 | 61.02 | 39.05 | 98.58 | 78.72 | 83.31 | 63.89 |
| LightGBM | Feature-based Supervised Learning | 96.07 | 78.65 | 61.34 | 45.63 | 98.45 | 81.09 | 86.28 | 67.07 |
| BiLSTM | Supervised Training | 98.36 | 86.07 | 80.17 | 58.19 | 98.15 | 85.68 | 88.97 | 74.81 |
| CNN-BiLSTM | Supervised Training | 98.46 | 86.46 | 79.34 | 61.04 | 98.72 | 84.49 | 92.67 | 78.55 |
| CNN-BiLSTM + Attention | Supervised Training | 98.60 | 91.81 | 81.48 | 70.32 | 99.13 | 86.40 | 90.11 | 77.72 |
| KoBERT | SFT | 99.35 | 90.57 | 97.71 | 70.98 | 99.25 | 83.88 | 87.78 | 69.99 |
| DistilKoBERT | SFT | 98.92 | 94.43 | 89.80 | 85.50 | 99.46 | 83.27 | 91.10 | 68.93 |
| mBERT | SFT | 99.02 | 82.94 | 84.22 | 59.33 | 99.56 | 56.03 | 91.36 | 58.45 |
| DistilBERT | SFT | 97.01 | 82.09 | 62.63 | 61.03 | 94.43 | 62.17 | 67.92 | 43.27 |
| KcBERT | SFT | 99.17 | 88.16 | 87.15 | 63.06 | 99.46 | 87.39 | 93.94 | 72.91 |
| KoBERT | TAPT+LoRA | 95.70 | 89.15 | 74.96 | 72.42 | 77.77 | 45.57 | 60.64 | 44.46 |
| Qwen3-0.6B | ST-SFT | 98.02 | 90.87 | 89.24 | 81.01 | 98.57 | 85.44 | 91.71 | 71.25 |
| HyperCLOVAX-0.5B | ST-SFT | 98.52 | 91.00 | 92.09 | 84.98 | 99.10 | 91.62 | 93.84 | 81.62 |
| Polyglot-1B | Label + Rationale [QLoRA] | 96.15 | 90.72 | 87.40 | 81.95 | 99.08 | 89.49 | 88.84 | 72.04 |
| Polyglot-5B | Label + Rationale [QLoRA] | 98.42 | 88.30 | 88.83 | 73.98 | 99.59 | 84.18 | 96.21 | 78.59 |
| KULLM-5B | Label + Rationale [QLoRA] | 97.82 | 92.39 | 95.55 | 86.88 | 99.25 | 90.95 | 89.75 | 76.08 |
| HyperCLOVAX-0.5B | Label + Rationale [QLoRA] | 98.48 | 91.22 | 92.86 | 80.26 | 99.25 | 87.27 | 92.76 | 76.42 |
| Qwen3-0.6B | Label + Rationale [QLoRA] | 98.17 | 93.30 | 87.62 | 76.07 | 97.71 | 62.84 | 86.08 | 59.90 |
| HyperCLOVAX-0.5B | Sequential Training | 98.73 | 94.97 | 96.05 | 86.87 | 93.97 | 82.92 | 84.42 | 77.05 |
| HyperCLOVAX-0.5B | Label + Evidence | 99.52 | 92.84 | 97.01 | 85.45 | 100.00 | 96.59 | 98.19 | 87.70 |
| HyperCLOVAX-0.5B | Label + Rationale | 97.18 | 90.75 | 89.93 | 80.82 | 98.82 | 86.61 | 90.74 | 73.83 |
| HyperCLOVAX-0.5B | Label + Rationale + Cons ($\lambda=0.1$) | 99.50 | 94.03 | 97.64 | 86.35 | 100.00 | 97.47 | 93.81 | 86.24 |
| HyperCLOVAX-0.5B | Label + Evidence + Rationale + Cons ($\lambda=0$) | 99.37 | 96.00 | 96.83 | 88.62 | 99.80 | 96.48 | 98.24 | 86.63 |
| HyperCLOVAX-0.5B | Label + Evidence + Rationale + Cons ($\lambda=0.05$) | 99.44 | 95.04 | 96.14 | 89.16 | 99.90 | 96.38 | 96.81 | 87.60 |
| HyperCLOVAX-0.5B | Label + Evidence + Rationale + Cons ($\lambda=0.1$) | 99.65 | 94.97 | 96.80 | 92.71 | 100.00 | 94.56 | 97.36 | 88.98 |
| HyperCLOVAX-0.5B | Label + Evidence + Rationale + Cons ($\lambda=0.2$) | 99.54 | 95.15 | 95.68 | 87.20 | 99.90 | 94.84 | 98.35 | 87.49 |
| HyperCLOVAX-1.5B | Label + Evidence + Rationale + Cons ($\lambda=0$) | 99.65 | 94.77 | 98.62 | 89.31 | 99.56 | 96.78 | 97.03 | 85.29 |
| HyperCLOVAX-1.5B | Label + Evidence + Rationale + Cons ($\lambda=0.05$) | 99.52 | 96.05 | 97.96 | 92.68 | 99.66 | 94.95 | 96.70 | 84.12 |
| HyperCLOVAX-1.5B | Label + Evidence + Rationale + Cons ($\lambda=0.1$) | 99.62 | 95.04 | 96.18 | 93.01 | 99.66 | 95.15 | 97.47 | 87.44 |
| HyperCLOVAX-1.5B | Label + Evidence + Rationale + Cons ($\lambda=0.2$) | 99.61 | 94.34 | 97.94 | 90.89 | 99.46 | 95.65 | 96.43 | 86.90 |
| Qwen3-0.6B | Label + Evidence + Rationale + Cons ($\lambda=0$) | 99.53 | 94.78 | 97.20 | 88.90 | 99.90 | 93.56 | 96.92 | 87.94 |
| Qwen3-0.6B | Label + Evidence + Rationale + Cons ($\lambda=0.05$) | 99.51 | 95.28 | 97.13 | 88.90 | 99.90 | 93.67 | 96.92 | 87.78 |
| Qwen3-0.6B | Label + Evidence + Rationale + Cons ($\lambda=0.1$) | 99.40 | 94.93 | 97.88 | 89.75 | 99.66 | 92.05 | 95.66 | 84.78 |
| Qwen3-0.6B | Label + Evidence + Rationale + Cons ($\lambda=0.2$) | 99.49 | 97.64 | 97.46 | 96.46 | 99.90 | 92.39 | 96.87 | 87.66 |
| Qwen3-1.7B | Label + Evidence + Rationale + Cons ($\lambda=0$) | 99.41 | 94.04 | 97.60 | 85.84 | 99.80 | 93.03 | 96.87 | 86.54 |
| Qwen3-1.7B | Label + Evidence + Rationale + Cons ($\lambda=0.05$) | 99.46 | 93.67 | 96.43 | 84.03 | 99.80 | 94.57 | 95.53 | 88.47 |
| Qwen3-1.7B | Label + Evidence + Rationale + Cons ($\lambda=0.1$) | 99.46 | 95.80 | 96.34 | 91.70 | 99.80 | 93.51 | 97.91 | 87.57 |
| Qwen3-1.7B | Label + Evidence + Rationale + Cons ($\lambda=0.2$) | 99.43 | 94.66 | 97.14 | 81.69 | 99.80 | 92.60 | 97.03 | 86.00 |
| HyperCLOVAX-0.5B | Evidence + Rationale + Label | 93.14 | 77.21 | 82.65 | 61.43 | 95.29 | 78.91 | 90.41 | 70.26 |
| HyperCLOVAX-0.5B | Evidence + Rationale + Label + Cons ($\lambda=0.1$) | 93.06 | 77.90 | 82.20 | 62.05 | 96.73 | 81.76 | 91.39 | 74.08 |
| HyperCLOVAX-0.5B | Label + Evidence + Rationale + Cons ($\lambda=0.0$) + Last Pooling | 99.61 | 94.98 | 96.48 | 87.23 | 99.66 | 97.63 | 96.54 | 89.42 |
| HyperCLOVAX-0.5B | Label + Evidence + Rationale + Cons ($\lambda=0.1$) + Last Pooling | 99.52 | 96.18 | 97.41 | 93.38 | 99.80 | 97.54 | 96.76 | 89.86 |
| Gemini-3.1-Flash-lite | Zeroshot | 83.19 | 74.50 | 39.97 | 42.02 | 93.73 | 85.84 | 92.32 | 82.11 |
| GPT-5.4 | Zeroshot | 81.43 | 80.67 | 41.74 | 33.74 | 82.55 | 82.42 | 90.60 | 90.42 |

### Annotation Quality and Privacy
```
Before model training and evaluation, we de-identified the dataset to remove personally identifiable information, including names, addresses, phone numbers, and account numbers. We used GPT-4o-mini with a fixed redaction prompt to detect and redact PII. A manual audit of a randomly sampled 10% subset found no remaining sensitive PII. The main failure mode was conservative over-redaction, which affected 0.874% of spans and typically involved non-identifying fields such as timestamps and dates.
```
```
We also conducted a separate human audit to assess whether the evidence and rationale annotations were justified by the input and label. Five annotators rated each audited instance as **Pass**, **Acceptable**, or **Invalid**. **Pass** indicates that the evidence and rationale are directly supported by the input and label; **Acceptable** indicates minor incompleteness or ambiguity; and **Invalid** indicates unsupported or label-inconsistent annotations. Across 1,916 audited instances from SMS and voice challenging cases, 69.87% were rated as Pass, 27.00% as Acceptable, and 3.13% as Invalid. These results support using the annotations as reference evidence and rationales for supervision and evaluation.
```

## Training Configuration

### Common Settings

- Seed: `10`
- dtype / eval dtype: `bf16`
- Epochs: `7`
- Learning rate: `3e-5`
- Max sequence length:
  - SMS: `1000`
  - Voice: `2200`
- Batch setting:
  - Default GPUs: `0,1,2`
  - Gradient accumulation steps: `5`
  - Effective batch size: `15`
- Full-generation evaluation: enabled
- Evaluation `max_new_tokens`: `200`
- Temperature: `0.0`
- Repetition penalty: `1.1`

### Ablation Settings

| Ablation | Target Format | Evidence | Rationale | Reconstruction / Consistency Setting |
|---|---|---:|---:|---|
| `label_evidence` | `label_span` | yes | no | `0` |
| `label_rationale` | `label_first_explanation` | no | yes | `0` |
| `label_rationale_cons_0.0` | `label_first_explanation` | no | yes | lambda `0.0`, pooling `mean`, scope `explanation` |
| `label_rationale_cons_0.05` | `label_first_explanation` | no | yes | lambda `0.05`, pooling `mean`, scope `explanation` |
| `label_rationale_cons_0.1` | `label_first_explanation` | no | yes | lambda `0.1`, pooling `mean`, scope `explanation` |
| `label_rationale_cons_0.2` | `label_first_explanation` | no | yes | lambda `0.2`, pooling `mean`, scope `explanation` |
| `label_rationale_cons_last_pooling` | `label_first_explanation` | no | yes | lambda `0.1`, pooling `last`, scope `explanation` |
| `last_label_ablation` | `span_explanation_label` | yes | yes | `0`, label placed last |

## ECoG Hyperparameters

The `label_evidence` and `last_label_ablation` settings use `--joint-stage12`.

- Classification loss weight: `0.5`
- Evidence loss weight: `1.0`
- Evidence alpha: `1.0`
- Evidence beta: `1.0`
- Negative downsampling ratio: `8`

```
