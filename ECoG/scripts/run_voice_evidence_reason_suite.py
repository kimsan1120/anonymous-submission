#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import sys
import threading
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

import run_noevidence_reason_suite as reason_suite
from phishdec.metrics.binary import compute_binary_metrics


ROOT = Path(__file__).resolve().parents[1]
PREPARED_DATA_ROOT = ROOT / "data" / "voice" / "evidence"
EVIDENCE_VARIANT = "keep"
INCLUDE_EVIDENCE_SPAN_OFFSETS = False
VOICE_PROMPT = reason_suite.VOICE_PROMPT
SMS_SPAN_PROMPT = "src/phishdec/prompts/instructions/korsmishing_explainer/kor_label_span_sms_sys.txt"
VOICE_SPAN_PROMPT = "src/phishdec/prompts/instructions/korsmishing_explainer/kor_label_span_voice_sys.txt"
SMS_SPAN_ONLY_PROMPT = "src/phishdec/prompts/instructions/korsmishing_explainer/kor_label_span_only_sms_sys.txt"
VOICE_SPAN_ONLY_PROMPT = "src/phishdec/prompts/instructions/korsmishing_explainer/kor_label_span_only_voice_sys.txt"
SMS_LABEL_LAST_PROMPT = "src/phishdec/prompts/instructions/korsmishing_explainer/kor_span_explanation_label_sms_sys.txt"
VOICE_LABEL_LAST_PROMPT = "src/phishdec/prompts/instructions/korsmishing_explainer/kor_span_explanation_label_voice_sys.txt"
SPAN_TARGET_FORMATS = {
    "label_span",
    "span_explanation_label",
}

DEFAULT_MODEL = "hyperclovax0p5b=naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B"
PIPELINE_TAG = "jointv1"
SUITE_NAME = "voice_evidence_reason_suite"


def _default_bench_suffix() -> str:
    return "voice_evidence_reason_suite"


def _stage2_logging_dir() -> str:
    return "outputs/runs/tb_logs/voice/evidence_reason_suite/stage2"


def _safe_float_suffix(value: float) -> str:
    text = f"{float(value):.4g}"
    return text.replace("-", "m").replace(".", "p")


def _joint_reconstruction_suffix(args: argparse.Namespace) -> str:
    weight = float(getattr(args, "joint_reconstruction_loss_weight", 0.0) or 0.0)
    if not bool(getattr(args, "joint_stage12", False)) or weight <= 0.0:
        return ""
    suffix = f"_rec{_safe_float_suffix(weight)}"
    scope = str(getattr(args, "joint_reconstruction_scope", "all") or "all").strip().lower()
    if scope == "explanation":
        suffix = f"{suffix}_recexp"
    pooling = str(getattr(args, "joint_reconstruction_pooling", "mean") or "mean").strip().lower()
    if pooling != "mean":
        suffix = f"{suffix}_{pooling}"
    return suffix


def _stage2_reconstruction_suffix(args: argparse.Namespace) -> str:
    weight = float(getattr(args, "stage2_reconstruction_loss_weight", 0.0) or 0.0)
    if weight <= 0.0:
        return ""
    suffix = f"_rec{_safe_float_suffix(weight)}"
    scope = str(getattr(args, "stage2_reconstruction_scope", "all") or "all").strip().lower()
    if scope == "explanation":
        suffix = f"{suffix}_recexp"
    pooling = str(getattr(args, "stage2_reconstruction_pooling", "mean") or "mean").strip().lower()
    if pooling != "mean":
        suffix = f"{suffix}_{pooling}"
    return suffix


def _label_span_prompt_path(scenario: reason_suite.Scenario) -> str:
    return SMS_SPAN_PROMPT if scenario.modality == "sms" else VOICE_SPAN_PROMPT


def _stage2_target_uses_span(target_format: str) -> bool:
    return str(target_format or "").strip() in SPAN_TARGET_FORMATS


def _stage2_target_uses_explanation(target_format: str) -> bool:
    return str(target_format or "").strip() in {
        "label_first_explanation",
        "span_explanation_label",
    }


def _stage2_target_suffix(target_format: str) -> str:
    fmt = str(target_format or "").strip()
    if fmt == "label_span":
        return "_spanonly1"
    if fmt == "span_explanation_label":
        return "_spanout1_labellast1"
    return ""


def _stage2_target_instruction_path(scenario: reason_suite.Scenario, target_format: str) -> str:
    fmt = str(target_format or "").strip()
    if fmt == "label_span":
        return SMS_SPAN_ONLY_PROMPT if scenario.modality == "sms" else VOICE_SPAN_ONLY_PROMPT
    if fmt == "span_explanation_label":
        return SMS_LABEL_LAST_PROMPT if scenario.modality == "sms" else VOICE_LABEL_LAST_PROMPT
    return _label_span_prompt_path(scenario)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Voice-only evidence+reason suite. v0 runs Stage 1 evidence-aware "
            "decoder classification, then Stage 2 label-first reason SFT, then label eval."
        )
    )
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--scenario-letters", default="EFG")
    parser.add_argument(
        "--eval-target-ids",
        default=None,

    )
    parser.add_argument("--bench-name", default=None)
    parser.add_argument(
        "--suite-name",
        default=SUITE_NAME,
        help=(

        ),
    )
    parser.add_argument(
        "--benchmark-layout",
        choices=("nested", "flat"),
        default="nested",
        help=(

        ),
    )
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--skip-prepare-data", action="store_true")
    parser.add_argument("--stages", choices=("both", "stage1", "stage2"), default="both")
    parser.add_argument(
        "--peft",
        choices=("none", "lora", "dora"),
        default="none",

    )
    parser.add_argument(
        "--stage1-peft",
        choices=("none", "lora", "dora"),
        default=None,

    )
    parser.add_argument(
        "--stage2-peft",
        choices=("none", "lora", "dora"),
        default=None,

    )
    parser.add_argument(
        "--skip-length-profiles",
        action="store_true",

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

    )
    parser.add_argument(
        "--joint-reconstruction-scope",
        choices=("all", "explanation"),
        default="all",
        help=(
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

        ),
    )
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--save-total-limit", type=int, default=1)
    parser.add_argument("--train-eval-steps", type=int, default=0)
    parser.add_argument("--train-save-steps", type=int, default=0)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,

    )
    parser.add_argument("--early-stopping-threshold", type=float, default=0.0)
    parser.add_argument("--early-stopping-min-epochs", type=float, default=3.0)
    parser.add_argument("--early-stopping-min-steps", type=int, default=0)
    parser.add_argument("--logging-steps-ratio", type=float, default=0.05)
    parser.add_argument(
        "--patience-ratio",
        type=float,
        default=0.10,

    )
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

        ),
    )
    parser.add_argument(
        "--eval-data-parallel-devices",
        default=None,

    )
    parser.add_argument(
        "--force-rerun-eval",
        action="store_true",

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


def repo_rel(path_like: str | Path) -> str:
    return reason_suite.repo_rel(path_like)


def write_yaml(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


def _strip_lora_keys(train_cfg: dict[str, Any]) -> None:
    for key in (
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "lora_bias",
        "lora_target_modules",
    ):
        train_cfg.pop(key, None)


def _effective_stage_peft(args: argparse.Namespace, stage: int) -> str:
    override = args.stage1_peft if int(stage) == 1 else args.stage2_peft
    return str(override if override is not None else args.peft)


def _default_stage1_lr(peft_mode: str) -> float:
    return 2e-4 if str(peft_mode) in {"lora", "dora"} else 2e-5


def _pipeline_exp_name(model_alias: str, scenario: reason_suite.Scenario, stage_name: str) -> str:
    return f"{model_alias}_{scenario.letter}_{scenario.name}_{stage_name}_{PIPELINE_TAG}"


def _is_hyperclovax_model(model: dict[str, str] | str) -> bool:
    if isinstance(model, dict):
        raw = f"{model.get('alias', '')} {model.get('model_ref', '')}"
    else:
        raw = str(model)
    return "hyperclovax" in raw.lower()


def weight_args(
    *,
    args: argparse.Namespace,
    model: dict[str, str],
    scenario_letter: str,
) -> float | None:
    if str(scenario_letter).upper() != "F":
        return None
    if _is_hyperclovax_model(model):
        return 1.0
    if bool(args.joint_stage12) and float(args.joint_reconstruction_loss_weight) > 0.0:
        return 0.5
    return 1.0


def make_dirs(bench_name: str, benchmark_layout: str = "nested") -> dict[str, Path]:
    suite_name = str(SUITE_NAME or "voice_evidence_reason_suite").strip() or "voice_evidence_reason_suite"
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


def load_model_specs(args: argparse.Namespace) -> list[dict[str, str]]:
    if not args.model:
        args.model = [DEFAULT_MODEL]
    return reason_suite.load_model_specs(args)


def _select_scenarios(raw_value: str, variant_root: Path) -> tuple[reason_suite.Scenario, ...]:
    raw = str(raw_value or "EFG").strip()
    parts = [part.strip().upper() for part in raw.replace(",", " ").split() if part.strip()]
    if len(parts) == 1 and parts[0].isalpha() and len(parts[0]) > 1:
        parts = list(parts[0])
    allowed = {"E", "F", "G"}
    if not parts:
        parts = ["E", "F", "G"]
    bad = [part for part in parts if part not in allowed]
    if bad:
        raise ValueError(f"Voice evidence+reason runner supports only E/F/G, got: {bad}")

    base_by_letter = {scenario.letter: scenario for scenario in reason_suite.SCENARIOS}
    scenarios: list[reason_suite.Scenario] = []
    seen: set[str] = set()
    for letter in parts:
        if letter in seen:
            continue
        base = base_by_letter[letter]
        eval_targets = (
            reason_suite.EvalTarget(
                f"{letter}1",
                "test",
                repo_rel(variant_root / f"{letter}_test.csv"),
                "text",
            ),
            reason_suite.EvalTarget(
                f"{letter}2",
                "challenging",
                repo_rel(variant_root / f"{letter}_challenge.csv"),
                "text",
            ),
        )
        scenarios.append(
            reason_suite.Scenario(
                letter=letter,
                name=base.name.replace("_reason", "_evidence_reason"),
                modality="voice",
                setting_group=base.setting_group,
                train_csv=repo_rel(variant_root / f"{letter}_train.csv"),
                train_eval_csv=repo_rel(variant_root / f"{letter}_validation.csv"),
                prompt_instruction_path=VOICE_PROMPT,
                train_text_col="text",
                eval_targets=eval_targets,
            )
        )
        seen.add(letter)
    return tuple(scenarios)


def _parse_eval_target_ids(raw_value: str | None) -> tuple[str, ...] | None:
    raw = str(raw_value or "").strip()
    if not raw or raw.upper() == "ALL":
        return None
    parts = [part.strip().upper() for part in raw.replace(",", " ").split() if part.strip()]
    selected: list[str] = []
    seen: set[str] = set()
    for part in parts:
        if part in seen:
            continue
        selected.append(part)
        seen.add(part)
    return tuple(selected)


def _filter_scenarios_eval_targets(
    scenarios: tuple[reason_suite.Scenario, ...],
    raw_eval_target_ids: str | None,
) -> tuple[reason_suite.Scenario, ...]:
    requested = _parse_eval_target_ids(raw_eval_target_ids)
    if requested is None:
        return scenarios

    wanted = set(requested)
    found: set[str] = set()
    filtered: list[reason_suite.Scenario] = []
    for scenario in scenarios:
        targets = tuple(target for target in scenario.eval_targets if target.eval_id.upper() in wanted)
        if not targets:
            continue
        found.update(target.eval_id.upper() for target in targets)
        filtered.append(
            reason_suite.Scenario(
                letter=scenario.letter,
                name=scenario.name,
                modality=scenario.modality,
                setting_group=scenario.setting_group,
                train_csv=scenario.train_csv,
                train_eval_csv=scenario.train_eval_csv,
                prompt_instruction_path=scenario.prompt_instruction_path,
                train_text_col=scenario.train_text_col,
                eval_targets=targets,
            )
        )

    missing = [eval_id for eval_id in requested if eval_id not in found]
    if missing:
        available = sorted({target.eval_id for scenario in scenarios for target in scenario.eval_targets})
        raise ValueError(
            f"Unknown or unselected eval target ids: {missing}. "
            f"Available for selected scenarios: {available}"
        )
    return tuple(filtered)


def ensure_prepared_data(args: argparse.Namespace, scenarios: tuple[reason_suite.Scenario, ...], variant_root: Path) -> None:
    expected = [variant_root / f"{scenario.letter}_train.csv" for scenario in scenarios]
    expected += [variant_root / f"{scenario.letter}_validation.csv" for scenario in scenarios]
    expected += [ROOT / target.csv_path for scenario in scenarios for target in scenario.eval_targets]
    missing = [path for path in expected if not path.exists()]
    if not missing:
        return
    missing_rel = [repo_rel(path) for path in missing]
    raise FileNotFoundError(
        "Missing prepared voice evidence/keep splits. The public artifact expects "
        f"prebuilt files under {repo_rel(variant_root)}; missing: {missing_rel}"
    )


def _fallback_batch_plan(args: argparse.Namespace) -> dict[str, Any]:
    world_size = reason_suite._cuda_world_size(args.cuda_visible_devices)
    micro = args.voice_batch_size if args.voice_batch_size is not None else 1
    micro = max(1, int(math.floor(micro * float(args.batch_scale))))
    grad_accum = args.voice_grad_accum if args.voice_grad_accum is not None else max(1, 16 // max(1, world_size))
    eval_batch = args.voice_eval_batch_size if args.voice_eval_batch_size is not None else 1
    eval_batch = max(1, int(math.floor(eval_batch * float(args.batch_scale))))
    return {
        "batch_size": int(micro),
        "grad_accum": int(grad_accum),
        "eval_batch_size": int(eval_batch),
        "world_size": int(world_size),
        "per_device_effective_batch_size": int(micro * grad_accum),
        "effective_batch_size": int(micro * grad_accum * world_size),
        "planning_train_length": int(args.voice_max_length),
        "used_mean_length": 0.0,
        "estimated_tokens_per_micro_step": int(micro * args.voice_max_length),
        "estimated_tokens_per_update": int(micro * grad_accum * args.voice_max_length * world_size),
        "target_effective_batch_size": int(16 * world_size),
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "source": "static_fallback",
    }


def _fallback_eval_batch_plan(args: argparse.Namespace) -> dict[str, Any]:
    if args.voice_hf_batch_size is not None:
        batch = args.voice_hf_batch_size
    else:
        batch = 2 if bool(args.eval_full_generation) else 1
    batch = max(1, int(math.floor(batch * float(args.batch_scale))))
    return {
        "hf_batch_size": int(batch),
        "planning_eval_length": int(args.voice_max_length + 1),
        "estimated_eval_tokens_per_step": int(batch * (args.voice_max_length + 1)),
        "source": "static_fallback",
    }


def _stage1_available(scenario: reason_suite.Scenario) -> bool:
    return scenario.letter in {"E", "F", "G"}


def _stage1_test_csv(scenario_letter: str) -> str:
    letter = str(scenario_letter).strip().upper()
    if letter in {"E", "F", "G"}:
        return f"data/voice/evidence/keep/{letter}_challenge.csv"
    raise ValueError(f"No stage1 test CSV for {scenario_letter}")


def build_stage1_cfg(
    args: argparse.Namespace,
    dirs: dict[str, Path],
    model: dict[str, str],
    scenario: reason_suite.Scenario,
    batch_plan: dict[str, Any],
) -> dict[str, Any]:
    out_root = dirs["output_root"] / model["alias"] / "stage1_train" / scenario.letter
    peft_mode = _effective_stage_peft(args, stage=1)
    stage1_epochs = args.stage1_epochs if args.stage1_epochs is not None else args.voice_epochs
    stage1_lr = float(args.stage1_lr) if args.stage1_lr is not None else _default_stage1_lr(peft_mode)
    batch_size = args.stage1_batch_size if args.stage1_batch_size is not None else int(batch_plan["batch_size"])
    grad_accum = args.stage1_grad_accum if args.stage1_grad_accum is not None else int(batch_plan["grad_accum"])
    eval_batch_size = (
        args.stage1_eval_batch_size if args.stage1_eval_batch_size is not None else int(batch_plan["eval_batch_size"])
    )
    cfg = {
        "exp_name": _pipeline_exp_name(model["alias"], scenario, "stage1_evidence"),
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
            "max_length": int(args.voice_max_length),
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
            "min_epoch_for_best_model": 1,
            "report_to": args.report_to,
            "logging_dir": "outputs/runs/tb_logs/voice/evidence_reason_suite/stage1",
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
        _strip_lora_keys(cfg["train"])
    return cfg


def _latest_completed_stage1(out_root: Path, exp_name: str) -> Path | None:
    for run_dir in reason_suite._find_existing_run_dirs(out_root, exp_name):
        if (run_dir / "summary.json").exists() and (run_dir / "best_checkpoint" / "decoder_heads.pt").exists():
            return run_dir
    return None


def _stage1_checkpoint_dir(stage1_run_dir: str | Path) -> Path:
    run_dir = Path(stage1_run_dir)
    direct_best = run_dir / "best_checkpoint"
    if direct_best.is_dir() and (direct_best / "decoder_heads.pt").exists():
        return direct_best
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8")) or {}
        checkpoint_dir = summary.get("checkpoint_dir")
        if checkpoint_dir:
            candidate = Path(str(checkpoint_dir))
            if candidate.is_dir() and (candidate / "decoder_heads.pt").exists():
                return candidate
    raise FileNotFoundError(f"Could not resolve stage1 best checkpoint from {run_dir}")


def _stage2_warmstart_model_cfg(
    *,
    stage1_run_dir: str | None,
    base_model_ref: str,
    from_base: bool = False,
) -> dict[str, Any]:
    if from_base:
        return {
            "model_name": str(base_model_ref),
            "adapter_path": None,
            "merge_adapter": False,
        }
    if (not stage1_run_dir) or str(stage1_run_dir).startswith("__"):
        return {
            "model_name": "__STAGE1_BEST_CHECKPOINT__",
            "adapter_path": None,
            "merge_adapter": False,
        }

    checkpoint_dir = _stage1_checkpoint_dir(stage1_run_dir)
    metadata_path = checkpoint_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    peft_mode = str(metadata.get("peft_mode", "none")).lower()
    base_model_name = str(metadata.get("base_model_name", base_model_ref))
    if peft_mode in {"lora", "dora"}:
        return {
            "model_name": base_model_name,
            "adapter_path": str(checkpoint_dir),
            "merge_adapter": True,
        }
    return {
        "model_name": str(checkpoint_dir),
        "adapter_path": None,
        "merge_adapter": False,
    }


def _run_or_reuse_stage1(
    *,
    args: argparse.Namespace,
    cfg_path: Path,
    cfg: dict[str, Any],
    orchestrator_log: Path,
    dirs: dict[str, Path],
    model_alias: str,
    scenario_letter: str,
) -> str | None:
    if not args.run:
        return None
    out_root = dirs["output_root"] / model_alias / "stage1_train" / scenario_letter
    existing = _latest_completed_stage1(out_root, str(cfg["exp_name"]))
    if existing is not None:
        print(f"\n=== {model_alias} :: {scenario_letter} stage1 evidence ===\n[resume] {repo_rel(existing)}")
        return str(existing)
    print(f"\n=== {model_alias} :: {scenario_letter} stage1 evidence ===")
    return reason_suite.run_config(cfg_path, orchestrator_log)


def _eval_parallel_devices(args: argparse.Namespace) -> list[str]:
    raw = str(
        getattr(args, "eval_data_parallel_devices", None)
        or getattr(args, "cuda_visible_devices", None)
        or "0"
    )
    devices: list[str] = []
    seen = set()
    for token in raw.split(","):
        device = token.strip()
        if not device or device in seen:
            continue
        seen.add(device)
        devices.append(device)
    return devices or ["0"]


def _run_eval_configs_parallel(jobs: list[dict[str, Any]], orchestrator_log: Path) -> list[str]:
    if not jobs:
        return []

    env = os.environ.copy()
    python_bin_dir = str(Path(sys.executable).resolve().parent)
    env["PATH"] = python_bin_dir + os.pathsep + env.get("PATH", "")
    lock = threading.Lock()
    states: list[dict[str, Any]] = []

    with orchestrator_log.open("a", encoding="utf-8") as logf:
        for job in jobs:
            cfg_path = Path(job["cfg_path"])
            label = str(job["label"])
            cmd = ["bash", "scripts/run_decode.sh", str(cfg_path)]
            logf.write(f"\n$ {' '.join(cmd)}\n")
            print(f"\n--- {label} eval ---\n[parallel] {repo_rel(cfg_path)}")
            proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            state: dict[str, Any] = {"label": label, "proc": proc, "run_dir": None, "cmd": cmd}

            def _reader(proc_state: dict[str, Any]) -> None:
                stream = proc_state["proc"].stdout
                if stream is None:
                    return
                prefix = f"[{proc_state['label']}] "
                for line in stream:
                    raw = line.rstrip("\n")
                    if "[done] run_dir=" in raw:
                        proc_state["run_dir"] = raw.split("[done] run_dir=", 1)[1].strip()
                    elif "[run.sh]" in raw and "run_dir=" in raw:
                        proc_state["run_dir"] = raw.split("run_dir=", 1)[1].strip()
                    with lock:
                        sys.stdout.write(prefix + line)
                        logf.write(prefix + line)
                        logf.flush()

            thread = threading.Thread(target=_reader, args=(state,), daemon=True)
            state["thread"] = thread
            states.append(state)
            thread.start()

        failures: list[tuple[str, int, list[str]]] = []
        for state in states:
            rc = int(state["proc"].wait())
            state["thread"].join()
            if rc != 0:
                failures.append((str(state["label"]), rc, list(state["cmd"])))
            if not state.get("run_dir"):
                failures.append((str(state["label"]), -1, list(state["cmd"])))

    if failures:
        label, rc, cmd = failures[0]
        raise subprocess.CalledProcessError(rc, cmd, output=f"parallel eval failed: {label}")
    return [str(state["run_dir"]) for state in states]


def _unique_timestamped_run_dir(out_root: Path, exp_name: str) -> Path:
    out_root.mkdir(parents=True, exist_ok=True)
    base = out_root / f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}_{exp_name}"
    if not base.exists():
        base.mkdir(parents=True)
        return base
    for idx in range(1, 1000):
        candidate = out_root / f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}_{exp_name}_{idx}"
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
    raise RuntimeError(f"Could not allocate run directory under {out_root}")


def _result_csv_path(run_dir: str | Path) -> Path:
    run_path = Path(run_dir)
    candidates = sorted(run_path.glob("results_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No results_*.csv found under shard run: {run_path}")
    return candidates[0]


def _merged_eval_metrics(result_df: pd.DataFrame, shard_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    if "label" not in result_df.columns or "pred" not in result_df.columns:
        return {
            "n": int(len(result_df)),
            "invalid_parse_count": int(sum(int(m.get("invalid_parse_count", 0)) for m in shard_metrics)),
        }
    y_true = pd.to_numeric(result_df["label"], errors="coerce").fillna(0).astype(int).tolist()
    y_pred = pd.to_numeric(result_df["pred"], errors="coerce").fillna(0).astype(int).tolist()
    metrics = compute_binary_metrics(y_true, y_pred)
    by_category = {}
    if "category" in result_df.columns:
        for cat, group in result_df.groupby("category", sort=False):
            yt = pd.to_numeric(group["label"], errors="coerce").fillna(0).astype(int).tolist()
            yp = pd.to_numeric(group["pred"], errors="coerce").fillna(0).astype(int).tolist()
            by_category[str(cat)] = compute_binary_metrics(yt, yp)
    by_subset = {}
    if "subset_label" in result_df.columns:
        for sub, group in result_df.groupby("subset_label", sort=False):
            yt = pd.to_numeric(group["label"], errors="coerce").fillna(0).astype(int).tolist()
            yp = pd.to_numeric(group["pred"], errors="coerce").fillna(0).astype(int).tolist()
            by_subset[str(sub)] = compute_binary_metrics(yt, yp)
    by_length_sml = {}
    if "length_sml" in result_df.columns:
        for length_sml, group in result_df.groupby("length_sml", sort=False):
            yt = pd.to_numeric(group["label"], errors="coerce").fillna(0).astype(int).tolist()
            yp = pd.to_numeric(group["pred"], errors="coerce").fillna(0).astype(int).tolist()
            by_length_sml[str(length_sml)] = compute_binary_metrics(yt, yp)
    metrics["by_category"] = by_category
    metrics["by_subset"] = by_subset
    metrics["by_length_sml"] = by_length_sml
    metrics["invalid_parse_count"] = int(sum(int(m.get("invalid_parse_count", 0)) for m in shard_metrics))
    metrics["eval_data_parallel_shards"] = int(len(shard_metrics))
    return metrics


def _run_eval_config_data_parallel(
    *,
    args: argparse.Namespace,
    cfg: dict[str, Any],
    cfg_path: Path,
    eval_out_root: Path,
    devices: list[str],
    dirs: dict[str, Path],
    model_alias: str,
    scenario_letter: str,
    eval_id: str,
    orchestrator_log: Path,
) -> str:
    eval_csv = Path(str(cfg.get("data", {}).get("eval_csv", "")))
    if not eval_csv.exists():
        raise FileNotFoundError(f"Eval CSV not found for data-parallel eval: {eval_csv}")
    source_df = pd.read_csv(eval_csv, encoding="utf-8-sig")
    if source_df.empty:
        raise ValueError(f"Eval CSV is empty: {eval_csv}")

    n_shards = min(len(devices), len(source_df))
    active_devices = devices[:n_shards]
    shard_root = (
        dirs["analysis_root"]
        / "eval_data_parallel_shards"
        / model_alias
        / scenario_letter
        / eval_id
    )
    cfg_root = (
        dirs["config_root"]
        / model_alias
        / "eval_dp"
        / scenario_letter
        / eval_id
    )
    shard_jobs: list[dict[str, Any]] = []
    for shard_idx, device in enumerate(active_devices):
        shard_df = source_df.iloc[shard_idx::n_shards].copy()
        if shard_df.empty:
            continue
        shard_df["_dp_original_index"] = list(range(shard_idx, len(source_df), n_shards))
        shard_csv = shard_root / f"shard{shard_idx}_of{n_shards}_gpu{device}.csv"
        shard_csv.parent.mkdir(parents=True, exist_ok=True)
        shard_df.to_csv(shard_csv, index=False, encoding="utf-8-sig")

        shard_cfg = copy.deepcopy(cfg)
        shard_cfg["exp_name"] = f"{cfg['exp_name']}_shard{shard_idx}of{n_shards}_gpu{device}"
        shard_cfg.setdefault("data", {})["eval_csv"] = str(shard_csv)
        shard_cfg.setdefault("run", {})["cuda_visible_devices"] = str(device)
        shard_cfg["run"]["out_root"] = str(eval_out_root / "_dp_shards" / eval_id / f"shard{shard_idx}_gpu{device}")
        shard_cfg_path = cfg_root / f"shard{shard_idx}_of{n_shards}_gpu{device}.yaml"
        write_yaml(shard_cfg_path, shard_cfg)
        shard_jobs.append(
            {
                "label": f"{model_alias}::{eval_id}::shard{shard_idx}/gpu{device}",
                "cfg_path": shard_cfg_path,
            }
        )

    if not shard_jobs:
        raise RuntimeError(f"No shard jobs were created for {eval_id}")

    shard_run_dirs = _run_eval_configs_parallel(shard_jobs, orchestrator_log)
    shard_frames = []
    shard_metrics = []
    for shard_run_dir in shard_run_dirs:
        shard_result = pd.read_csv(_result_csv_path(shard_run_dir), encoding="utf-8-sig")
        shard_frames.append(shard_result)
        metrics_path = Path(shard_run_dir) / "metrics.json"
        shard_metrics.append(reason_suite.load_json(metrics_path) if metrics_path.exists() else {})

    merged_df = pd.concat(shard_frames, ignore_index=True)
    if "_dp_original_index" not in merged_df.columns:
        raise ValueError("Shard result CSVs are missing _dp_original_index; cannot restore row order.")
    merged_df["_dp_original_index"] = pd.to_numeric(
        merged_df["_dp_original_index"],
        errors="coerce",
    ).fillna(-1).astype(int)
    merged_df = merged_df.sort_values("_dp_original_index").reset_index(drop=True)

    run_dir = _unique_timestamped_run_dir(eval_out_root, str(cfg["exp_name"]))
    write_yaml(run_dir / "config.yaml", cfg)
    result_path = run_dir / "results_data_parallel_merged.csv"
    merged_df.to_csv(result_path, index=False, encoding="utf-8-sig")
    metrics = _merged_eval_metrics(merged_df, shard_metrics)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "logs.txt").write_text(
        "\n".join(
            [
                "[eval-data-parallel] merged shard eval",
                f"config: {repo_rel(cfg_path)}",
                f"devices: {','.join(active_devices)}",
                f"rows: {len(merged_df)}",
                "shards:",
                *[f"- {repo_rel(Path(path))}" for path in shard_run_dirs],
                f"results: {result_path.name}",
                "metrics: metrics.json",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[eval-data-parallel] merged {eval_id}: {repo_rel(run_dir)}")
    return str(run_dir)


def main() -> int:
    global SUITE_NAME
    args = parse_args()
    SUITE_NAME = str(args.suite_name or SUITE_NAME).strip() or SUITE_NAME
    args.evidence_variant = EVIDENCE_VARIANT
    if args.voice_hf_batch_size is None and bool(args.eval_full_generation):
        args.voice_hf_batch_size = 2
    if float(args.joint_reconstruction_loss_weight) < 0.0:
        raise ValueError("--joint-reconstruction-loss-weight must be >= 0")
    if float(args.stage2_reconstruction_loss_weight) < 0.0:
        raise ValueError("--stage2-reconstruction-loss-weight must be >= 0")
    if str(args.joint_reconstruction_scope) == "explanation" and not _stage2_target_uses_explanation(
        args.stage2_target_format
    ):
        raise ValueError("--joint-reconstruction-scope explanation requires an explanation target format")
    if float(args.stage2_reconstruction_loss_weight) > 0.0 and not _stage2_target_uses_explanation(
        args.stage2_target_format
    ):
        raise ValueError("--stage2-reconstruction-loss-weight requires an explanation target format")
    if args.joint_stage12:
        if args.stages == "stage1":
            raise ValueError("--joint-stage12 trains the combined objective in the Stage 2 runner; use --stages stage2 or both.")
        args.stage2_from_base = True
        if args.stages == "both":
            args.stages = "stage2"
    models = load_model_specs(args)
    variant_root = PREPARED_DATA_ROOT / args.evidence_variant
    scenarios = _select_scenarios(args.scenario_letters, variant_root)
    scenarios = _filter_scenarios_eval_targets(scenarios, getattr(args, "eval_target_ids", None))
    ensure_prepared_data(args, scenarios, variant_root)

    length_profiles: dict[str, Any] | None = None
    if not args.skip_length_profiles:
        reason_suite.ensure_length_profiles(args, models, scenarios)
        length_profiles = reason_suite.load_json(reason_suite.LENGTH_PROFILE_JSON)

    bench_name = args.bench_name or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_default_bench_suffix()}"
    dirs = make_dirs(bench_name, getattr(args, "benchmark_layout", "nested"))
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    orchestrator_log = dirs["analysis_root"] / "orchestrator.log"
    manifest: dict[str, Any] = {
        "bench_name": bench_name,
        "generated_at": datetime.now().isoformat(),
        "run_enabled": bool(args.run),
        "evidence_variant": args.evidence_variant,
        "stages": args.stages,
        "peft": args.peft,
        "stage1_peft": _effective_stage_peft(args, stage=1),
        "stage2_peft": _effective_stage_peft(args, stage=2),
        "joint_stage12": bool(args.joint_stage12),
        "stage2_target_format": str(args.stage2_target_format),
        "eval_full_generation": bool(args.eval_full_generation),
        "models": models,
        "scenarios": [asdict(s) | {"eval_targets": [asdict(et) for et in s.eval_targets]} for s in scenarios],
        "prepared_data_root": repo_rel(variant_root),
        "prepared_data_manifest": repo_rel(variant_root / "manifest.json"),
        "v0_note": (
            "Joint Stage1+Stage2 ablation: one Stage 2-style run starts from the base model "
            "and trains label classification, evidence span, first-token label, and explanation losses together."
            if args.joint_stage12
            else (
                "Stage 1 and Stage 2 are orchestrated sequentially. "
                "Stage 2 warm-starts from the Stage 1 best checkpoint."
            )
        ),
    }

    eval_rows: list[dict[str, Any]] = []

    for model in models:
        for scenario in scenarios:
            stage1_run_dir: str | None = None
            stage1_cfg: dict[str, Any] | None = None
            if length_profiles is None:
                batch_plan = _fallback_batch_plan(args)
            else:
                batch_plan = reason_suite.build_batch_plan(
                    args,
                    model=model,
                    scenario=scenario,
                    length_profiles=length_profiles,
                )
            train_schedule = reason_suite.estimate_train_schedule(
                args,
                scenario=scenario,
                batch_plan=batch_plan,
            )
            manifest.setdefault("auto_batch_plans", []).append(
                {
                    "model_alias": model["alias"],
                    "scenario_letter": scenario.letter,
                    "scenario_name": scenario.name,
                    **batch_plan,
                }
            )
            manifest.setdefault("auto_train_schedules", []).append(
                {
                    "model_alias": model["alias"],
                    "scenario_letter": scenario.letter,
                    "scenario_name": scenario.name,
                    **train_schedule,
                }
            )

            if args.stages in {"both", "stage1"}:
                if _stage1_available(scenario):
                    stage1_cfg = build_stage1_cfg(args, dirs, model, scenario, batch_plan)
                    stage1_cfg_path = (
                        dirs["config_root"] / model["alias"] / "stage1" / f"{scenario.letter}_stage1_evidence.yaml"
                    )
                    write_yaml(stage1_cfg_path, stage1_cfg)
                    manifest.setdefault("generated_stage1_configs", []).append(repo_rel(stage1_cfg_path))
                    stage1_run_dir = _run_or_reuse_stage1(
                        args=args,
                        cfg_path=stage1_cfg_path,
                        cfg=stage1_cfg,
                        orchestrator_log=orchestrator_log,
                        dirs=dirs,
                        model_alias=model["alias"],
                        scenario_letter=scenario.letter,
                    )
                    manifest.setdefault("stage1_runs", []).append(
                        {
                            "model_alias": model["alias"],
                            "scenario_letter": scenario.letter,
                            "config": repo_rel(stage1_cfg_path),
                            "run_dir": stage1_run_dir,
                        }
                    )
                else:
                    manifest.setdefault("stage1_skipped", []).append(
                        {
                            "model_alias": model["alias"],
                            "scenario_letter": scenario.letter,
                            "reason": "No functional evidence source is available for this scenario.",
                        }
                    )

            if args.stages == "stage1":
                continue

            if (not args.stage2_from_base) and _stage1_available(scenario) and stage1_run_dir is None:
                if args.run:
                    existing_stage1 = _latest_completed_stage1(
                        dirs["output_root"] / model["alias"] / "stage1_train" / scenario.letter,
                        _pipeline_exp_name(model["alias"], scenario, "stage1_evidence"),
                    )
                    if existing_stage1 is None:
                        raise FileNotFoundError(
                            f"Stage 2 requires a completed Stage 1 run for {model['alias']} {scenario.letter}."
                        )
                    stage1_run_dir = str(existing_stage1)
                else:
                    stage1_run_dir = "__STAGE1_RUN_DIR__"

            train_cfg = reason_suite.build_train_cfg(
                args,
                dirs,
                model,
                scenario,
                batch_plan=batch_plan,
                train_schedule=train_schedule,
                train_eval_csv=scenario.train_eval_csv,
            )
            train_cfg["exp_name"] = _pipeline_exp_name(model["alias"], scenario, "stage2_reason")
            if args.stage2_from_base:
                train_cfg["exp_name"] = f"{train_cfg['exp_name']}_1stage1"
            if args.joint_stage12:
                train_cfg["exp_name"] = f"{train_cfg['exp_name']}_joint12v1"
                train_cfg["exp_name"] = f"{train_cfg['exp_name']}{_joint_reconstruction_suffix(args)}"
            else:
                train_cfg["exp_name"] = f"{train_cfg['exp_name']}{_stage2_reconstruction_suffix(args)}"
            if _stage2_target_uses_span(args.stage2_target_format):
                train_cfg["exp_name"] = f"{train_cfg['exp_name']}{_stage2_target_suffix(args.stage2_target_format)}"
                train_cfg.setdefault("prompt", {})["instruction_path"] = _stage2_target_instruction_path(
                    scenario,
                    args.stage2_target_format,
                )
                if not INCLUDE_EVIDENCE_SPAN_OFFSETS:
                    train_cfg["exp_name"] = f"{train_cfg['exp_name']}_joint12v1"
            train_cfg["run"]["out_root"] = str(dirs["output_root"] / model["alias"] / "stage2_train" / scenario.letter)
            train_cfg["train"]["logging_dir"] = _stage2_logging_dir()
            train_cfg["train"]["label_loss_weight"] = float(args.stage2_label_loss_weight)
            train_cfg["train"]["explanation_loss_weight"] = float(args.stage2_explanation_loss_weight)
            if not args.joint_stage12 and float(args.stage2_reconstruction_loss_weight) > 0.0:
                train_cfg["train"]["reconstruction_loss_weight"] = float(args.stage2_reconstruction_loss_weight)
                train_cfg["train"]["reconstruction_pooling"] = str(args.stage2_reconstruction_pooling)
                train_cfg["train"]["reconstruction_scope"] = str(args.stage2_reconstruction_scope)
            train_cfg["data"]["label_reason_target_format"] = str(args.stage2_target_format)
            if args.joint_stage12 or _stage2_target_uses_span(args.stage2_target_format):
                train_cfg["data"]["evidence_spans_col"] = "spans"
                train_cfg["data"]["evidences_col"] = "evidences"
                train_cfg["data"]["use_evidence_loss_col"] = "use_evidence_loss"
                train_cfg["data"]["evidence_supervision_mode"] = "column"
                train_cfg["data"]["evidence_supervise_empty_negatives"] = False
                train_cfg["data"]["max_evidence_items"] = 3
                train_cfg["data"]["include_evidence_span_offsets"] = INCLUDE_EVIDENCE_SPAN_OFFSETS
            if args.joint_stage12:
                train_cfg["train"]["joint_evidence_explanation"] = True
                train_cfg["train"]["evidence_supervision_mode"] = "column"
                train_cfg["train"]["evidence_supervise_empty_negatives"] = False
                train_cfg["train"]["classification_loss_weight"] = float(args.joint_classification_loss_weight)
                train_cfg["train"]["evidence_loss_weight"] = float(args.joint_evidence_loss_weight)
                train_cfg["train"]["reconstruction_loss_weight"] = float(args.joint_reconstruction_loss_weight)
                train_cfg["train"]["reconstruction_pooling"] = str(args.joint_reconstruction_pooling)
                train_cfg["train"]["reconstruction_scope"] = str(args.joint_reconstruction_scope)
                train_cfg["train"]["evidence_alpha"] = float(args.joint_evidence_alpha)
                train_cfg["train"]["evidence_beta"] = float(
                    args.stage1_evidence_beta if args.joint_evidence_beta is None else args.joint_evidence_beta
                )
                train_cfg["train"]["evidence_negative_downsample_ratio"] = int(
                    args.stage1_negative_downsample_ratio
                    if args.joint_negative_downsample_ratio is None
                    else args.joint_negative_downsample_ratio
                )
                train_cfg["train"]["evidence_negative_only_loss"] = False
                train_cfg["train"]["evidence_negative_only_max_tokens"] = 128
            stage2_peft = _effective_stage_peft(args, stage=2)
            train_cfg["train"]["peft"] = stage2_peft
            warmstart_model_cfg = _stage2_warmstart_model_cfg(
                stage1_run_dir=stage1_run_dir,
                base_model_ref=model["model_ref"],
                from_base=bool(args.stage2_from_base),
            )
            train_cfg["model"]["model_name"] = str(warmstart_model_cfg["model_name"])
            if warmstart_model_cfg.get("adapter_path"):
                train_cfg["model"]["adapter_path"] = str(warmstart_model_cfg["adapter_path"])
                train_cfg["model"]["merge_adapter"] = bool(warmstart_model_cfg.get("merge_adapter", False))
            else:
                train_cfg["model"].pop("adapter_path", None)
                train_cfg["model"]["merge_adapter"] = False
            if stage2_peft not in {"lora", "dora"}:
                train_cfg["model"]["load_in_4bit"] = False
                train_cfg["model"].pop("bnb_4bit_quant_type", None)
                train_cfg["model"].pop("bnb_4bit_compute_dtype", None)
                train_cfg["model"].pop("bnb_4bit_use_double_quant", None)
                _strip_lora_keys(train_cfg["train"])
            stage2_cfg_path = dirs["config_root"] / model["alias"] / "stage2" / f"{scenario.letter}_stage2_reason.yaml"
            write_yaml(stage2_cfg_path, train_cfg)
            manifest.setdefault("generated_stage2_configs", []).append(repo_rel(stage2_cfg_path))

            train_run_dir: str | None
            selected_eval_model_name: str | None = None
            selected_eval_adapter_path: str | None = None
            if args.run:
                train_out_root = dirs["output_root"] / model["alias"] / "stage2_train" / scenario.letter
                existing_train_run = reason_suite._latest_completed_train_run(train_out_root, str(train_cfg["exp_name"]))
                if existing_train_run is not None:
                    train_run_dir = str(existing_train_run)
                    print(f"\n=== {model['alias']} :: {scenario.letter} stage2 reason ===\n[resume] {repo_rel(existing_train_run)}")
                else:
                    print(f"\n=== {model['alias']} :: {scenario.letter} stage2 reason ===")
                    train_run_dir = reason_suite.run_config(stage2_cfg_path, orchestrator_log)
                if str(train_cfg["train"].get("peft", "none")).lower() in {"lora", "dora"}:
                    selected_eval_model_name = model["model_ref"]
                    selected_eval_adapter_path = train_run_dir
                else:
                    selected_eval_model_name = train_run_dir
                    selected_eval_adapter_path = None
            else:
                train_run_dir = "__STAGE2_TRAIN_RUN_DIR__"
                if str(train_cfg["train"].get("peft", "none")).lower() in {"lora", "dora"}:
                    selected_eval_model_name = model["model_ref"]
                    selected_eval_adapter_path = train_run_dir
                else:
                    selected_eval_model_name = train_run_dir
                    selected_eval_adapter_path = None

            eval_parallel_devices = _eval_parallel_devices(args)

            for eval_target in scenario.eval_targets:
                if length_profiles is None:
                    eval_batch_plan = _fallback_eval_batch_plan(args)
                else:
                    eval_batch_plan = reason_suite.build_eval_batch_plan(
                        args,
                        model=model,
                        scenario=scenario,
                        eval_target=eval_target,
                        length_profiles=length_profiles,
                    )
                eval_cfg = reason_suite.build_eval_cfg(
                    args,
                    dirs,
                    model,
                    scenario,
                    eval_target,
                    adapter_path=selected_eval_adapter_path,
                    eval_batch_plan=eval_batch_plan,
                    model_name_override=selected_eval_model_name,
                )
                eval_cfg["exp_name"] = f"{model['alias']}_{eval_target.eval_id}_{scenario.name}_stage2_reason_{PIPELINE_TAG}"
                if args.stage2_from_base:
                    eval_cfg["exp_name"] = f"{eval_cfg['exp_name']}_1stage1"
                if args.joint_stage12:
                    eval_cfg["exp_name"] = f"{eval_cfg['exp_name']}_joint12v1"
                    eval_cfg["exp_name"] = f"{eval_cfg['exp_name']}{_joint_reconstruction_suffix(args)}"
                else:
                    eval_cfg["exp_name"] = f"{eval_cfg['exp_name']}{_stage2_reconstruction_suffix(args)}"
                if _stage2_target_uses_span(args.stage2_target_format):
                    eval_cfg["exp_name"] = f"{eval_cfg['exp_name']}{_stage2_target_suffix(args.stage2_target_format)}"
                    eval_cfg.setdefault("prompt", {})["instruction_path"] = _stage2_target_instruction_path(
                        scenario,
                        args.stage2_target_format,
                    )
                    if not INCLUDE_EVIDENCE_SPAN_OFFSETS:
                        eval_cfg["exp_name"] = f"{eval_cfg['exp_name']}_joint12v1"
                if str(args.stage2_target_format).strip() == "span_explanation_label":
                    eval_cfg.setdefault("decode", {})["prediction_format"] = "trailing_binary"
                    eval_cfg.setdefault("decode", {})["constrain_binary_output"] = False
                    eval_cfg.setdefault("decode", {})["constrain_trailing_binary_output"] = True
                    eval_cfg.setdefault("decode", {})["trailing_binary_marker"] = "정답:\n"
                if args.eval_full_generation:
                    eval_cfg["decode"]["max_new_tokens"] = int(args.decode_max_new_tokens)
                    eval_cfg["exp_name"] = f"{eval_cfg['exp_name']}_fullgen{int(args.decode_max_new_tokens)}"
                ruleF = weight_args(
                    args=args,
                    model=model,
                    scenario_letter=scenario.letter,
                )
                label_last_target = str(args.stage2_target_format).strip() == "span_explanation_label"
                if ruleF is not None and not label_last_target:
                    eval_cfg["decode"]["score_adjust"] = {
                        "enabled": True,
                        "method": "learned_logreg_v1",
                        "source_csv": scenario.train_eval_csv,
                        "text_col": scenario.train_text_col,
                        "label_col": "label",
                        "strength": float(ruleF),
                    }
                    eval_cfg["exp_name"] = f"{eval_cfg['exp_name']}_fscore{_safe_float_suffix(ruleF)}"
                eval_cfg_path = dirs["config_root"] / model["alias"] / "eval" / f"{eval_target.eval_id}_eval.yaml"
                write_yaml(eval_cfg_path, eval_cfg)
                manifest.setdefault("generated_eval_configs", []).append(repo_rel(eval_cfg_path))

                if not args.run:
                    continue

                eval_out_root = dirs["output_root"] / model["alias"] / "eval" / scenario.letter / eval_target.eval_id
                run_eval_cfg_path = eval_cfg_path
                run_eval_cfg = eval_cfg
                if bool(getattr(args, "eval_data_parallel", False)):
                    run_eval_cfg = copy.deepcopy(eval_cfg)
                    run_eval_cfg["exp_name"] = f"{run_eval_cfg['exp_name']}_dp{len(eval_parallel_devices)}"
                    run_eval_cfg_path = (
                        dirs["config_root"]
                        / model["alias"]
                        / "eval_dp"
                        / scenario.letter
                        / eval_target.eval_id
                        / f"{eval_target.eval_id}_eval_dp{len(eval_parallel_devices)}.yaml"
                    )
                    write_yaml(run_eval_cfg_path, run_eval_cfg)
                    manifest.setdefault("generated_eval_dp_configs", []).append(repo_rel(run_eval_cfg_path))

                existing_eval_run = reason_suite._latest_completed_eval_run(
                    eval_out_root,
                    str(run_eval_cfg["exp_name"]),
                )
                if existing_eval_run is not None and not bool(getattr(args, "force_rerun_eval", False)):
                    eval_run_dir = str(existing_eval_run)
                    print(f"\n--- {model['alias']} :: {eval_target.eval_id} eval ---\n[resume] {repo_rel(existing_eval_run)}")
                elif bool(getattr(args, "eval_data_parallel", False)):
                    print(
                        f"\n--- {model['alias']} :: {eval_target.eval_id} eval ---\n"
                        f"[data-parallel] shard rows across devices {','.join(eval_parallel_devices)}"
                    )
                    eval_run_dir = _run_eval_config_data_parallel(
                        args=args,
                        cfg=run_eval_cfg,
                        cfg_path=run_eval_cfg_path,
                        eval_out_root=eval_out_root,
                        devices=eval_parallel_devices,
                        dirs=dirs,
                        model_alias=model["alias"],
                        scenario_letter=scenario.letter,
                        eval_id=eval_target.eval_id,
                        orchestrator_log=orchestrator_log,
                    )
                else:
                    print(f"\n--- {model['alias']} :: {eval_target.eval_id} eval ---")
                    eval_run_dir = reason_suite.run_config(run_eval_cfg_path, orchestrator_log)

                metrics_path = Path(eval_run_dir) / "metrics.json"
                if not metrics_path.exists():
                    raise FileNotFoundError(f"Missing eval metrics: {metrics_path}")
                metrics = reason_suite.load_json(metrics_path)
                row_base = {
                    "model_alias": model["alias"],
                    "model_ref": model["model_ref"],
                    "scenario_letter": scenario.letter,
                    "scenario_name": scenario.name,
                    "setting_group": scenario.setting_group,
                    "modality": scenario.modality,
                    "train_csv": scenario.train_csv,
                    "train_config": repo_rel(stage2_cfg_path),
                    "train_run_dir": train_run_dir,
                    "train_eval_adapter_path": str(train_run_dir),
                    "train_summary": repo_rel(Path(train_run_dir) / "summary.json") if train_run_dir else "",
                    "train_checkpoint_alias_manifest": "",
                    "eval_id": eval_target.eval_id,
                    "eval_split_name": eval_target.split_name,
                    "eval_csv": eval_target.csv_path,
                    "eval_config": repo_rel(run_eval_cfg_path),
                    "eval_run_dir": eval_run_dir,
                    "metrics_path": repo_rel(metrics_path),
                }
                eval_rows.append(reason_suite.flatten_eval_record(row_base, metrics))

    manifest["config_root"] = repo_rel(dirs["config_root"])
    manifest["output_root"] = repo_rel(dirs["output_root"])
    manifest["analysis_root"] = repo_rel(dirs["analysis_root"])

    manifest_path = dirs["analysis_root"] / "manifest.json"
    eval_summary_json = dirs["analysis_root"] / "benchmark_eval_summary.json"
    eval_summary_csv = dirs["analysis_root"] / "benchmark_eval_summary.csv"
    model_summary_json = dirs["analysis_root"] / "benchmark_model_summary.json"
    model_summary_csv = dirs["analysis_root"] / "benchmark_model_summary.csv"

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    eval_summary_json.write_text(json.dumps(eval_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    reason_suite.save_csv(eval_summary_csv, eval_rows)
    model_rows = reason_suite.model_summary(eval_rows)
    model_summary_json.write_text(json.dumps(model_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    reason_suite.save_csv(model_summary_csv, model_rows)

    print(f"\nSaved manifest: {repo_rel(manifest_path)}")
    print(f"Saved eval summary: {repo_rel(eval_summary_json)}")
    print(f"Saved model summary: {repo_rel(model_summary_json)}")
    return 0


if __name__ == "__main__":
    os.chdir(ROOT)
    raise SystemExit(main())
