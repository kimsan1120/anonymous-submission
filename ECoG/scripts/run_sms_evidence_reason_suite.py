#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import run_noevidence_reason_suite as reason_suite
import run_voice_evidence_reason_suite as voice_suite


ROOT = Path(__file__).resolve().parents[1]
PREPARED_DATA_ROOT = ROOT / "data" / "sms" / "evidence"
EVIDENCE_VARIANT = "keep"
SMS_PROMPT = reason_suite.SMS_PROMPT
SUITE_NAME = "sms_evidence_reason_suite"


def _default_bench_suffix() -> str:
    return "sms_evidence_reason_suite"


def _stage2_logging_dir() -> str:
    return "outputs/runs/tb_logs/sms/evidence_reason_suite/stage2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "SMS-only evidence+reason suite. Runs Stage 1 evidence-aware "
            "decoder classification, then Stage 2 label-first reason SFT, then label eval."
        )
    )
    parser.add_argument("--model", action="append", default=[], help="Format: alias=model_ref or model_ref")
    parser.add_argument("--scenario-letters", default="ABCD", help="SMS scenarios only: A, B, C, D")
    parser.add_argument(
        "--eval-target-ids",
        default=None,
        help="Optional comma/space-separated eval ids to run, e.g. D2 or D2,C1. Training scenarios are kept as needed.",
    )
    parser.add_argument("--bench-name", default=None)
    parser.add_argument(
        "--suite-name",
        default=SUITE_NAME,
        help=(
            "Benchmark namespace under configs/generated, outputs/runs/benchmarks, and outputs/analysis. "
            "Use evidence_reason_suite to store SMS and Voice runs under one benchmark root."
        ),
    )
    parser.add_argument(
        "--benchmark-layout",
        choices=("nested", "flat"),
        default="nested",
        help=(
            "Directory layout for benchmark outputs. nested keeps "
            "outputs/runs/benchmarks/{suite}/{bench}/{model}; flat uses "
            "outputs/runs/benchmarks/{suite}/{model} for category roots such as label_evidence."
        ),
    )
    parser.add_argument("--run", action="store_true", help="Execute configs through scripts/run_decode.sh")
    parser.add_argument("--skip-prepare-data", action="store_true")
    parser.add_argument("--stages", choices=("both", "stage1", "stage2"), default="both")
    parser.add_argument(
        "--peft",
        choices=("none", "lora", "dora"),
        default="none",
        help="Fine-tuning mode for both stages. Default is full SFT/full fine-tune.",
    )
    parser.add_argument(
        "--stage1-peft",
        choices=("none", "lora", "dora"),
        default=None,
        help="Override Stage 1 fine-tuning mode. Defaults to --peft.",
    )
    parser.add_argument(
        "--stage2-peft",
        choices=("none", "lora", "dora"),
        default=None,
        help="Override Stage 2 fine-tuning mode. Defaults to --peft.",
    )
    parser.add_argument(
        "--skip-length-profiles",
        action="store_true",
        help="Use conservative static batch planning instead of noevidence reason length profiles.",
    )

    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--report-to", default="tensorboard")
    parser.add_argument("--backend", default="hf")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--eval-dtype", default="bf16")

    parser.add_argument("--voice-epochs", type=float, default=7.0)
    parser.add_argument("--voice-lr", type=float, default=3e-5, help="Stage 2 reason SFT LR")
    parser.add_argument("--voice-batch-size", type=int, default=None)
    parser.add_argument("--voice-grad-accum", type=int, default=None)
    parser.add_argument("--voice-eval-batch-size", type=int, default=None)
    parser.add_argument("--voice-hf-batch-size", type=int, default=None)
    parser.add_argument("--voice-max-length", type=int, default=2200)

    parser.add_argument("--sms-epochs", type=float, default=7.0)
    parser.add_argument("--sms-lr", type=float, default=3e-5)
    parser.add_argument("--sms-batch-size", type=int, default=None)
    parser.add_argument("--sms-grad-accum", type=int, default=None)
    parser.add_argument("--sms-eval-batch-size", type=int, default=None)
    parser.add_argument("--sms-hf-batch-size", type=int, default=None)
    parser.add_argument("--sms-max-length", type=int, default=1000)

    parser.add_argument("--stage1-epochs", type=float, default=None)
    parser.add_argument(
        "--stage1-lr",
        type=float,
        default=None,
        help="Stage 1 LR. Default: 2e-4 for LoRA/Dora, 2e-5 for full fine-tune.",
    )
    parser.add_argument("--stage1-batch-size", type=int, default=None)
    parser.add_argument("--stage1-grad-accum", type=int, default=None)
    parser.add_argument("--stage1-eval-batch-size", type=int, default=None)
    parser.add_argument("--stage1-lora-r", type=int, default=16)
    parser.add_argument("--stage1-lora-alpha", type=int, default=32)
    parser.add_argument("--stage1-label-loss-weight", type=float, default=0.5)
    parser.add_argument("--stage1-evidence-lambda", type=float, default=1.0)
    parser.add_argument("--stage1-evidence-warmup-epochs", type=int, default=0)
    parser.add_argument("--stage1-evidence-threshold", type=float, default=0.7)
    parser.add_argument("--stage1-negative-downsample-ratio", type=int, default=8)
    parser.add_argument("--stage1-evidence-beta", type=float, default=1.0)
    parser.add_argument("--stage1-evidence-metric-weight", type=float, default=0.5)
    parser.add_argument(
        "--stage1-metric-for-best-model",
        default="evidence_aware_score",
        help="Metric used by train_decoder_classifier for best_checkpoint.",
    )
    parser.add_argument("--stage2-label-loss-weight", type=float, default=1.0)
    parser.add_argument("--stage2-explanation-loss-weight", type=float, default=0.5)
    parser.add_argument(
        "--stage2-reconstruction-loss-weight",
        type=float,
        default=0.0,
        help="Evidence-free Stage 2 label+explanation consistency reconstruction loss weight.",
    )
    parser.add_argument(
        "--stage2-reconstruction-pooling",
        choices=("mean", "last"),
        default="mean",
        help="Pooling over generated explanation token hidden states for evidence-free Stage 2 reconstruction.",
    )
    parser.add_argument(
        "--stage2-reconstruction-scope",
        choices=("all", "explanation"),
        default="explanation",
        help="Token scope used by the evidence-free Stage 2 reconstruction head.",
    )
    parser.add_argument(
        "--joint-stage12",
        action="store_true",
        help="Train one Stage 2-style run from the base model with label, evidence, first-token, and explanation losses.",
    )
    parser.add_argument("--joint-classification-loss-weight", type=float, default=0.5)
    parser.add_argument("--joint-evidence-loss-weight", type=float, default=1.0)
    parser.add_argument("--joint-evidence-alpha", type=float, default=1.0)
    parser.add_argument("--joint-evidence-beta", type=float, default=None)
    parser.add_argument("--joint-negative-downsample-ratio", type=int, default=None)
    parser.add_argument(
        "--joint-reconstruction-loss-weight",
        type=float,
        default=0.1,
        help=(
            "Weak auxiliary explanation-to-label reconstruction loss weight for --joint-stage12. "
            "Use 0 to reproduce the base joint objective."
        ),
    )
    parser.add_argument(
        "--joint-reconstruction-pooling",
        choices=("mean", "last"),
        default="mean",
        help="Pooling over generated explanation token hidden states for reconstruction.",
    )
    parser.add_argument(
        "--joint-reconstruction-scope",
        choices=("all", "explanation"),
        default="all",
        help=(
            "Token scope used by the reconstruction head. all keeps the existing label-after-target path; "
            "explanation uses only the explanation body after the '설명:' marker."
        ),
    )
    parser.add_argument(
        "--stage2-target-format",
        choices=("label_first_explanation", "label_span", "span_explanation_label"),
        default="label_first_explanation",
        help=(
            "Generation target format for Stage 2. "
            "label_span emits label+evidence spans only; span_explanation_label emits "
            "evidence+explanation+trailing label."
        ),
    )
    parser.add_argument(
        "--stage2-from-base",
        action="store_true",
        help=(
            "Ablation mode: train Stage 2 directly from the base model instead of "
            "warm-starting from a completed Stage 1 checkpoint. Use with --stages stage2 "
            "for a strict one-stage evidence/reason run."
        ),
    )
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--save-total-limit", type=int, default=1)
    parser.add_argument("--train-eval-steps", type=int, default=0)
    parser.add_argument("--train-save-steps", type=int, default=0)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--early-stopping-threshold", type=float, default=0.0)
    parser.add_argument("--early-stopping-min-epochs", type=float, default=3.0)
    parser.add_argument("--early-stopping-min-steps", type=int, default=0)
    parser.add_argument("--logging-steps-ratio", type=float, default=0.05)
    parser.add_argument("--patience-ratio", type=float, default=0.10)
    parser.add_argument("--batch-scale", type=float, default=1.0)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--no-length-bucket", dest="length_bucket", action="store_false")
    parser.add_argument("--use-validation", dest="use_validation", action="store_true")
    parser.add_argument("--no-validation", dest="use_validation", action="store_false")

    parser.add_argument("--decode-max-input-tokens", default="auto_p95")
    parser.add_argument("--decode-max-input-tokens-quantile", type=float, default=0.99)
    parser.add_argument("--decode-max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--eval-full-generation",
        action="store_true",
        help="Use --decode-max-new-tokens for eval generation instead of label-only max_new_tokens=1.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument(
        "--eval-data-parallel",
        action="store_true",
        help=(
            "Run eval targets for each trained scenario in parallel, with one single-GPU "
            "model copy per eval process instead of HF device_map auto-sharding."
        ),
    )
    parser.add_argument(
        "--eval-data-parallel-devices",
        default=None,
        help="Comma-separated GPU ids used for --eval-data-parallel. Defaults to --cuda-visible-devices or 0.",
    )
    parser.add_argument(
        "--force-rerun-eval",
        action="store_true",
        help="Ignore completed eval runs and create fresh eval runs while still reusing completed training runs.",
    )
    parser.add_argument("--profile-env", default="decoder311")
    parser.add_argument("--refresh-length-profiles", action="store_true")
    parser.add_argument("--autobatch-conservative-steps", type=int, default=0)
    parser.add_argument(
        "--validation-subset-dir",
        default=str(ROOT / "data" / "arc" / "validation_subsets"),
    )
    parser.set_defaults(length_bucket=True, use_validation=True)
    return parser.parse_args()


def make_dirs(bench_name: str, benchmark_layout: str = "nested") -> dict[str, Path]:
    suite_name = str(SUITE_NAME or "sms_evidence_reason_suite").strip() or "sms_evidence_reason_suite"
    layout = str(benchmark_layout or "nested").strip().lower()
    if layout == "flat":
        return {
            "config_root": ROOT / "configs" / "generated" / suite_name,
            "output_root": ROOT / "outputs" / "runs" / "benchmarks" / suite_name,
            "analysis_root": ROOT / "outputs" / "analysis" / suite_name,
        }
    return {
        "config_root": ROOT / "configs" / "generated" / suite_name / bench_name,
        "output_root": ROOT / "outputs" / "runs" / "benchmarks" / suite_name / bench_name,
        "analysis_root": ROOT / "outputs" / "analysis" / suite_name / bench_name,
    }


def _select_scenarios(raw_value: str, variant_root: Path) -> tuple[reason_suite.Scenario, ...]:
    raw = str(raw_value or "ABCD").strip()
    parts = [part.strip().upper() for part in raw.replace(",", " ").split() if part.strip()]
    if len(parts) == 1 and parts[0].isalpha() and len(parts[0]) > 1:
        parts = list(parts[0])
    allowed = {"A", "B", "C", "D"}
    if not parts:
        parts = ["A", "B", "C", "D"]
    bad = [part for part in parts if part not in allowed]
    if bad:
        raise ValueError(f"SMS evidence+reason runner supports only A/B/C/D, got: {bad}")

    base_by_letter = {scenario.letter: scenario for scenario in reason_suite.SCENARIOS}
    scenarios: list[reason_suite.Scenario] = []
    seen: set[str] = set()
    for letter in parts:
        if letter in seen:
            continue
        base = base_by_letter[letter]
        scenarios.append(
            reason_suite.Scenario(
                letter=letter,
                name=base.name.replace("_reason", "_evidence_reason"),
                modality="sms",
                setting_group=base.setting_group,
                train_csv=reason_suite.repo_rel(variant_root / f"{letter}_train.csv"),
                train_eval_csv=reason_suite.repo_rel(variant_root / f"{letter}_validation.csv"),
                prompt_instruction_path=SMS_PROMPT,
                train_text_col="text",
                eval_targets=base.eval_targets,
            )
        )
        seen.add(letter)
    return tuple(scenarios)


def ensure_prepared_data(
    args: argparse.Namespace,
    scenarios: tuple[reason_suite.Scenario, ...],
    variant_root: Path,
) -> None:
    expected = [variant_root / f"{scenario.letter}_train.csv" for scenario in scenarios]
    expected += [variant_root / f"{scenario.letter}_validation.csv" for scenario in scenarios]
    if args.skip_prepare_data and all(path.exists() for path in expected):
        return
    cmd = [
        sys.executable,
        "scripts/prepare_sms_evidence_reason_suite_data.py",
        "--variant",
        EVIDENCE_VARIANT,
        "--out-root",
        reason_suite.repo_rel(PREPARED_DATA_ROOT),
    ]
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def _fallback_batch_plan(args: argparse.Namespace) -> dict[str, Any]:
    world_size = reason_suite._cuda_world_size(args.cuda_visible_devices)
    micro = args.sms_batch_size if args.sms_batch_size is not None else 1
    micro = max(1, int(math.floor(micro * float(args.batch_scale))))
    grad_accum = args.sms_grad_accum if args.sms_grad_accum is not None else max(1, 16 // max(1, world_size))
    eval_batch = args.sms_eval_batch_size if args.sms_eval_batch_size is not None else 1
    eval_batch = max(1, int(math.floor(eval_batch * float(args.batch_scale))))
    return {
        "batch_size": int(micro),
        "grad_accum": int(grad_accum),
        "eval_batch_size": int(eval_batch),
        "world_size": int(world_size),
        "per_device_effective_batch_size": int(micro * grad_accum),
        "effective_batch_size": int(micro * grad_accum * world_size),
        "planning_train_length": int(args.sms_max_length),
        "used_mean_length": 0.0,
        "estimated_tokens_per_micro_step": int(micro * args.sms_max_length),
        "estimated_tokens_per_update": int(micro * grad_accum * args.sms_max_length * world_size),
        "target_effective_batch_size": int(16 * world_size),
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "source": "static_fallback",
    }


def _fallback_eval_batch_plan(args: argparse.Namespace) -> dict[str, Any]:
    batch = args.sms_hf_batch_size if args.sms_hf_batch_size is not None else 1
    batch = max(1, int(math.floor(batch * float(args.batch_scale))))
    return {
        "hf_batch_size": int(batch),
        "planning_eval_length": int(args.sms_max_length + 1),
        "estimated_eval_tokens_per_step": int(batch * (args.sms_max_length + 1)),
        "source": "static_fallback",
    }


def _stage1_available(scenario: reason_suite.Scenario) -> bool:
    return scenario.letter in {"A", "B", "C", "D"}


def _stage1_test_csv(scenario_letter: str) -> str:
    if scenario_letter == "A":
        return "data/sms/in_domain/challenging.csv"
    if scenario_letter == "B":
        return "data/sms/ood/challenging/credit_challenging.csv"
    if scenario_letter == "C":
        return "data/sms/ood/challenging/finance_challenging.csv"
    if scenario_letter == "D":
        return "data/sms/ood/challenging/parcel_challenging.csv"
    raise ValueError(f"No stage1 test CSV for {scenario_letter}")


def build_stage1_cfg(
    args: argparse.Namespace,
    dirs: dict[str, Path],
    model: dict[str, str],
    scenario: reason_suite.Scenario,
    batch_plan: dict[str, Any],
) -> dict[str, Any]:
    out_root = dirs["output_root"] / model["alias"] / "stage1_train" / scenario.letter
    peft_mode = voice_suite._effective_stage_peft(args, stage=1)
    stage1_epochs = args.stage1_epochs if args.stage1_epochs is not None else args.sms_epochs
    stage1_lr = float(args.stage1_lr) if args.stage1_lr is not None else voice_suite._default_stage1_lr(peft_mode)
    batch_size = args.stage1_batch_size if args.stage1_batch_size is not None else int(batch_plan["batch_size"])
    grad_accum = args.stage1_grad_accum if args.stage1_grad_accum is not None else int(batch_plan["grad_accum"])
    eval_batch_size = (
        args.stage1_eval_batch_size if args.stage1_eval_batch_size is not None else int(batch_plan["eval_batch_size"])
    )
    patience = int(args.early_stopping_patience) if int(args.early_stopping_patience) > 0 else 2
    cfg = {
        "exp_name": voice_suite._pipeline_exp_name(model["alias"], scenario, "stage1_evidence"),
        "task": "train_decoder_classifier",
        "model": {
            "model_type": "Decoder",
            "model_name": model["model_ref"],
            "dtype": args.dtype,
        },
        "data": {
            "train_csv": scenario.train_csv,
            **({"eval_csv": scenario.train_eval_csv} if args.use_validation else {}),
            "test_csv": _stage1_test_csv(scenario.letter),
            "text_col": "text",
            "label_col": "label",
            "category_col": "category",
            "evidence_spans_col": "spans",
            "evidences_col": "evidences",
            "use_evidence_loss_col": "use_evidence_loss",
        },
        "train": {
            "loss_mode": "multitask_evidence",
            "binary_classifier": True,
            "decision_threshold": 0.5,
            "peft": peft_mode,
            "label_loss_weight": float(args.stage1_label_loss_weight),
            "lora_r": int(args.stage1_lora_r),
            "lora_alpha": int(args.stage1_lora_alpha),
            "lora_dropout": 0.05,
            "lora_bias": "none",
            "lora_target_modules": reason_suite._lora_target_modules(model),
            "evidence": {
                "enabled": True,
                "lambda": float(args.stage1_evidence_lambda),
                "warmup_epochs": int(args.stage1_evidence_warmup_epochs),
                "alpha": 1.0,
                "beta": float(args.stage1_evidence_beta),
                "negative_downsample_ratio": int(args.stage1_negative_downsample_ratio),
                "threshold": float(args.stage1_evidence_threshold),
                "max_pred_spans": 3,
                "metric_weight": float(args.stage1_evidence_metric_weight),
            },
            "epochs": int(math.ceil(float(stage1_epochs))),
            "lr": float(stage1_lr),
            "batch_size": int(batch_size),
            "grad_accum": int(grad_accum),
            "eval_batch_size": int(eval_batch_size),
            "max_length": int(args.sms_max_length),
            "weight_decay": 0.01,
            "warmup_ratio": 0.03,
            "max_grad_norm": 1.0,
            "gradient_checkpointing": bool(args.gradient_checkpointing),
            "dataloader_num_workers": int(args.dataloader_num_workers),
            "ddp_find_unused_parameters": True,
            "use_balanced_batch_sampler": True,
            "logging_steps": 20,
            "metric_for_best_model": str(args.stage1_metric_for_best_model),
            "greater_is_better": True,
            "early_stopping_patience": patience,
            "early_stopping_threshold": float(args.early_stopping_threshold),
            "min_epoch_for_best_model": 1,
            "min_epoch_for_early_stopping": 1,
            "report_to": args.report_to,
            "logging_dir": "outputs/runs/tb_logs/sms/evidence_reason_suite/stage1",
        },
        "run": {
            "seed": int(args.seed),
            "out_root": str(out_root),
            "use_running_dir": True,
            "running_root": "outputs/runs/running",
            "running_tb_root": "outputs/runs/tb_logs/running",
            **({"cuda_visible_devices": args.cuda_visible_devices} if args.cuda_visible_devices else {}),
        },
    }
    if peft_mode not in {"lora", "dora"}:
        voice_suite._strip_lora_keys(cfg["train"])
    return cfg


def _install_overrides() -> None:
    voice_suite.PREPARED_DATA_ROOT = PREPARED_DATA_ROOT
    voice_suite.parse_args = parse_args
    voice_suite.make_dirs = make_dirs
    voice_suite._default_bench_suffix = _default_bench_suffix
    voice_suite._stage2_logging_dir = _stage2_logging_dir
    voice_suite._select_scenarios = _select_scenarios
    voice_suite.ensure_prepared_data = ensure_prepared_data
    voice_suite._fallback_batch_plan = _fallback_batch_plan
    voice_suite._fallback_eval_batch_plan = _fallback_eval_batch_plan
    voice_suite._stage1_available = _stage1_available
    voice_suite._stage1_test_csv = _stage1_test_csv
    voice_suite.build_stage1_cfg = build_stage1_cfg


def main() -> int:
    _install_overrides()
    args = parse_args()
    global SUITE_NAME
    SUITE_NAME = str(args.suite_name or SUITE_NAME).strip() or SUITE_NAME
    voice_suite.parse_args = lambda: args
    return voice_suite.main()


if __name__ == "__main__":
    os.chdir(ROOT)
    raise SystemExit(main())
