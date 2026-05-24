#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
PREPARED_DATA_ROOT = ROOT / "data"
LENGTH_PROFILE_JSON = PREPARED_DATA_ROOT / "length_profiles.json"
SMS_PROMPT = "src/phishdec/prompts/instructions/korsmishing_explainer/kor_label_first_sms_sys.txt"
VOICE_PROMPT = "src/phishdec/prompts/instructions/korsmishing_explainer/kor_label_first_voice_sys.txt"
LENGTH_PROFILE_TARGET_FORMAT_VERSION = "label_first_explanation_v1"

DEFAULT_MODELS: tuple[str, ...] = (
    "polyglot1b=EleutherAI/polyglot-ko-1.3b",
    "kullm5b=nlpai-lab/kullm-polyglot-5.8b-v2",
    "polyglot5b=EleutherAI/polyglot-ko-5.8b",
)


@dataclass(frozen=True)
class EvalTarget:
    eval_id: str
    split_name: str
    csv_path: str
    text_col: str


@dataclass(frozen=True)
class Scenario:
    letter: str
    name: str
    modality: str
    setting_group: str
    train_csv: str
    train_eval_csv: str
    prompt_instruction_path: str
    train_text_col: str
    eval_targets: tuple[EvalTarget, ...]


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        letter="A",
        name="sms_in_domain_reason",
        modality="sms",
        setting_group="sms_in_domain",
        train_csv="data/sms/evidence/keep/A_train.csv",
        train_eval_csv="data/sms/evidence/keep/A_validation.csv",
        prompt_instruction_path=SMS_PROMPT,
        train_text_col="text",
        eval_targets=(
            EvalTarget("A1", "test", "data/sms/evidence/keep/A_test.csv", "text"),
            EvalTarget("A2", "challenging", "data/sms/evidence/keep/A_challenge.csv", "text"),
        ),
    ),
    Scenario(
        letter="B",
        name="sms_ood_credit_reason",
        modality="sms",
        setting_group="sms_ood",
        train_csv="data/sms/evidence/keep/B_train.csv",
        train_eval_csv="data/sms/evidence/keep/B_validation.csv",
        prompt_instruction_path=SMS_PROMPT,
        train_text_col="text",
        eval_targets=(
            EvalTarget("B1", "test", "data/sms/evidence/keep/B_test.csv", "text"),
            EvalTarget("B2", "challenging", "data/sms/evidence/keep/B_challenge.csv", "text"),
        ),
    ),
    Scenario(
        letter="C",
        name="sms_ood_finance_reason",
        modality="sms",
        setting_group="sms_ood",
        train_csv="data/sms/evidence/keep/C_train.csv",
        train_eval_csv="data/sms/evidence/keep/C_validation.csv",
        prompt_instruction_path=SMS_PROMPT,
        train_text_col="text",
        eval_targets=(
            EvalTarget("C1", "test", "data/sms/evidence/keep/C_test.csv", "text"),
            EvalTarget("C2", "challenging", "data/sms/evidence/keep/C_challenge.csv", "text"),
        ),
    ),
    Scenario(
        letter="D",
        name="sms_ood_parcel_reason",
        modality="sms",
        setting_group="sms_ood",
        train_csv="data/sms/evidence/keep/D_train.csv",
        train_eval_csv="data/sms/evidence/keep/D_validation.csv",
        prompt_instruction_path=SMS_PROMPT,
        train_text_col="text",
        eval_targets=(
            EvalTarget("D1", "test", "data/sms/evidence/keep/D_test.csv", "text"),
            EvalTarget("D2", "challenging", "data/sms/evidence/keep/D_challenge.csv", "text"),
        ),
    ),
    Scenario(
        letter="E",
        name="voice_in_domain_reason",
        modality="voice",
        setting_group="voice_in_domain",
        train_csv="data/voice/evidence/keep/E_train.csv",
        train_eval_csv="data/voice/evidence/keep/E_validation.csv",
        prompt_instruction_path=VOICE_PROMPT,
        train_text_col="text",
        eval_targets=(
            EvalTarget("E1", "test", "data/voice/evidence/keep/E_test.csv", "text"),
            EvalTarget("E2", "challenging", "data/voice/evidence/keep/E_challenge.csv", "text"),
        ),
    ),
    Scenario(
        letter="F",
        name="voice_ood_finance_reason",
        modality="voice",
        setting_group="voice_ood",
        train_csv="data/voice/evidence/keep/F_train.csv",
        train_eval_csv="data/voice/evidence/keep/F_validation.csv",
        prompt_instruction_path=VOICE_PROMPT,
        train_text_col="text",
        eval_targets=(
            EvalTarget("F1", "test", "data/voice/evidence/keep/F_test.csv", "text"),
            EvalTarget("F2", "challenging", "data/voice/evidence/keep/F_challenge.csv", "text"),
        ),
    ),
    Scenario(
        letter="G",
        name="voice_ood_government_reason",
        modality="voice",
        setting_group="voice_ood",
        train_csv="data/voice/evidence/keep/G_train.csv",
        train_eval_csv="data/voice/evidence/keep/G_validation.csv",
        prompt_instruction_path=VOICE_PROMPT,
        train_text_col="text",
        eval_targets=(
            EvalTarget("G1", "test", "data/voice/evidence/keep/G_test.csv", "text"),
            EvalTarget("G2", "challenging", "data/voice/evidence/keep/G_challenge.csv", "text"),
        ),
    ),
)




SCENARIO_MAX_NEW_TOKENS: dict[str, int] = {
    "A": 172,
    "B": 172,
    "C": 172,
    "D": 162,
    "E": 32,
    "F": 32,
    "G": 32,
}


def _scenario_max_new_tokens(letter: str) -> int:
    return int(SCENARIO_MAX_NEW_TOKENS.get(str(letter).upper(), 128))


def _eval_label_max_new_tokens() -> int:
    return 1


def _eval_decode_max_new_tokens(args: argparse.Namespace) -> int:
    if bool(getattr(args, "eval_full_generation", False)):
        return int(getattr(args, "decode_max_new_tokens", 128))
    return _eval_label_max_new_tokens()


def _eval_top_p(temperature: float, top_p: float) -> float:
    return 1.0 if float(temperature) <= 0.0 else float(top_p)


def _scenario_max_input_margin(scenario: Scenario) -> int:
    
    return 16 if scenario.modality == "voice" else 64


def _select_scenarios(raw_value: str) -> tuple[Scenario, ...]:
    available = {scenario.letter.upper(): scenario for scenario in SCENARIOS}
    raw = str(raw_value or "").strip()
    if not raw or raw.upper() == "ALL":
        return SCENARIOS

    parts = [part.strip().upper() for part in raw.replace(",", " ").split() if part.strip()]
    if len(parts) == 1 and parts[0].isalpha() and len(parts[0]) > 1:
        parts = list(parts[0])

    selected: list[Scenario] = []
    seen: set[str] = set()
    for part in parts:
        if part not in available:
            choices = ", ".join(sorted(available))
            raise ValueError(f"Unknown scenario letter: {part}. Choose from: {choices}")
        if part in seen:
            continue
        selected.append(available[part])
        seen.add(part)
    return tuple(selected)


def _length_profiles_cover(
    length_profiles: dict[str, Any],
    *,
    models: list[dict[str, str]],
    scenarios: tuple[Scenario, ...],
) -> bool:
    model_profiles = length_profiles.get("models", {}) or {}
    needed_train = {scenario.letter for scenario in scenarios}
    needed_eval = {target.eval_id for scenario in scenarios for target in scenario.eval_targets}
    for model in models:
        alias = model["alias"]
        profile = model_profiles.get(alias, {}) or {}
        train_profiles = profile.get("train", {}) or {}
        eval_profiles = profile.get("eval", {}) or {}
        if not needed_train.issubset(set(train_profiles)):
            return False
        if not needed_eval.issubset(set(eval_profiles)):
            return False
    return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate and optionally run A~G decoder-reason QLoRA suite."
    )
    p.add_argument("--model", action="append", default=[])
    p.add_argument(
        "--scenario-letters",
        default="ALL",

    )
    p.add_argument("--bench-name", default=None)
    p.add_argument(
        "--suite-name",
        default="noevidence_reason_suite",

    )
    p.add_argument(
        "--benchmark-layout",
        choices=("legacy", "nested", "flat"),
        default="legacy",
        help=(

        ),
    )
    p.add_argument("--run", action="store_true")
    p.add_argument("--skip-prepare-data", action="store_true")
    p.add_argument("--cuda-visible-devices", default=None)
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--report-to", default="tensorboard")
    p.add_argument("--backend", default="hf")
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--eval-dtype", default="bf16")
    p.add_argument("--sms-epochs", type=float, default=10.0)
    p.add_argument("--sms-lr", type=float, default=3e-5)
    p.add_argument("--sms-batch-size", type=int, default=None)
    p.add_argument("--sms-grad-accum", type=int, default=None)
    p.add_argument("--sms-eval-batch-size", type=int, default=None)
    p.add_argument("--sms-max-length", type=int, default=1000)
    p.add_argument("--voice-epochs", type=float, default=10.0)
    p.add_argument("--voice-lr", type=float, default=3e-5)
    p.add_argument("--voice-batch-size", type=int, default=None)
    p.add_argument("--voice-grad-accum", type=int, default=None)
    p.add_argument("--voice-eval-batch-size", type=int, default=None)
    p.add_argument("--voice-max-length", type=int, default=2200)
    p.add_argument(
        "--stage2-reconstruction-loss-weight",
        "--label-explanation-reconstruction-loss-weight",
        dest="stage2_reconstruction_loss_weight",
        type=float,
        default=0.0,
        help="Evidence-free label+explanation consistency reconstruction loss weight.",
    )
    p.add_argument(
        "--stage2-reconstruction-pooling",
        "--label-explanation-reconstruction-pooling",
        dest="stage2_reconstruction_pooling",
        choices=("mean", "last"),
        default="mean",
        help="Pooling over generated explanation token hidden states for reconstruction.",
    )
    p.add_argument(
        "--stage2-reconstruction-scope",
        "--label-explanation-reconstruction-scope",
        dest="stage2_reconstruction_scope",
        choices=("all", "explanation"),
        default="explanation",
        help="Token scope used by the reconstruction head.",
    )
    p.add_argument("--sms-hf-batch-size", type=int, default=None)
    p.add_argument("--voice-hf-batch-size", type=int, default=None)
    p.add_argument("--dataloader-num-workers", type=int, default=2)
    p.add_argument("--save-total-limit", type=int, default=1)
    p.add_argument("--train-eval-steps", type=int, default=0)
    p.add_argument("--train-save-steps", type=int, default=0)
    p.add_argument(
        "--primary-eval-epoch",
        type=float,
        default=7.0,

    )
    p.add_argument(
        "--best-epoch-min",
        type=float,
        default=7.0,

    )
    p.add_argument(
        "--best-epoch-max",
        type=float,
        default=10.0,

    )
    p.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,

    )
    p.add_argument("--early-stopping-threshold", type=float, default=0.0)
    p.add_argument(
        "--early-stopping-min-epochs",
        type=float,
        default=3.0,

    )
    p.add_argument(
        "--early-stopping-min-steps",
        type=int,
        default=0,

    )
    p.add_argument("--metric-for-best-model", default=None)
    p.add_argument("--logging-steps-ratio", type=float, default=0.05)
    p.add_argument(
        "--patience-ratio",
        type=float,
        default=0.10,

    )
    p.add_argument("--load-best-model-at-end", dest="load_best_model_at_end", action="store_true")
    p.add_argument("--no-load-best-model-at-end", dest="load_best_model_at_end", action="store_false")
    p.add_argument("--greater-is-better", dest="greater_is_better", action="store_true")
    p.add_argument("--not-greater-is-better", dest="greater_is_better", action="store_false")
    p.add_argument("--use-validation", dest="use_validation", action="store_true")
    p.add_argument("--no-validation", dest="use_validation", action="store_false")
    p.add_argument("--decode-max-input-tokens", default="auto_p95")
    p.add_argument("--decode-max-input-tokens-quantile", type=float, default=0.99)
    p.add_argument("--decode-max-new-tokens", type=int, default=128)
    p.add_argument(
        "--eval-full-generation",
        action="store_true",
        help="Use --decode-max-new-tokens for eval generation instead of label-only max_new_tokens=1.",
    )
    p.add_argument(
        "--force-rerun-eval",
        action="store_true",

    )
    p.add_argument(
        "--batch-scale",
        type=float,
        default=1.0,
        help="Scale auto-generated train/eval batch sizes (e.g., 0.8 -> 80%%).",
    )
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--repetition-penalty", type=float, default=1.1)
    p.add_argument("--profile-env", default="decoder311")
    p.add_argument("--refresh-length-profiles", action="store_true")
    p.add_argument(
        "--autobatch-conservative-steps",
        type=int,
        default=0,

    )
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--no-length-bucket", dest="length_bucket", action="store_false")
    p.add_argument("--val-frac-a", type=float, default=1.00)
    p.add_argument("--val-frac-b", type=float, default=1.00)
    p.add_argument("--val-frac-c", type=float, default=1.00)
    p.add_argument("--val-frac-d", type=float, default=1.00)
    p.add_argument("--val-frac-e", type=float, default=1.00)
    p.add_argument("--val-frac-f", type=float, default=1.00)
    p.add_argument("--val-frac-g", type=float, default=1.00)
    p.add_argument(
        "--validation-subset-dir",
        default=str(ROOT / "data" / "arc" / "validation_subsets"),

    )
    p.set_defaults(length_bucket=True, load_best_model_at_end=False, greater_is_better=True, use_validation=True)
    return p.parse_args()


def safe_name(value: str) -> str:
    keep = []
    for ch in str(value):
        if ch.isalnum() or ch in "._-":
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep).strip("_.-")
    return out or "model"


def _safe_float_suffix(value: float) -> str:
    text = f"{float(value):.4g}"
    return text.replace("-", "m").replace(".", "p")


def _label_explanation_reconstruction_suffix(args: argparse.Namespace) -> str:
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


def load_model_specs(args: argparse.Namespace) -> list[dict[str, str]]:
    specs = list(args.model or []) or list(DEFAULT_MODELS)
    parsed: list[dict[str, str]] = []
    for item in specs:
        if "=" in item:
            alias, model_ref = item.split("=", 1)
            alias = alias.strip()
            model_ref = model_ref.strip()
        else:
            model_ref = item.strip()
            alias = safe_name(model_ref.split("/")[-1])
        if not alias or not model_ref:
            raise ValueError(f"Invalid model spec: {item}")
        parsed.append({"alias": safe_name(alias), "model_ref": model_ref})
    return parsed


def make_dirs(
    bench_name: str,
    suite_name: str = "noevidence_reason_suite",
    benchmark_layout: str = "legacy",
) -> dict[str, Path]:
    suite = str(suite_name or "noevidence_reason_suite").strip() or "noevidence_reason_suite"
    layout = str(benchmark_layout or "legacy").strip().lower()
    if layout == "flat":
        return {
            "config_root": ROOT / "configs" / "generated" / suite,
            "output_root": ROOT / "outputs" / "runs" / "benchmarks" / suite,
            "analysis_root": ROOT / "outputs" / "analysis" / suite,
        }
    if layout == "nested":
        return {
            "config_root": ROOT / "configs" / "generated" / suite / bench_name,
            "output_root": ROOT / "outputs" / "runs" / "benchmarks" / suite / bench_name,
            "analysis_root": ROOT / "outputs" / "analysis" / suite / bench_name,
        }
    return {
        "config_root": ROOT / "configs" / "generated" / suite,
        "output_root": ROOT / "outputs" / "runs" / "benchmarks" / suite / bench_name,
        "analysis_root": ROOT / "outputs" / "analysis" / suite / bench_name,
    }


_CSV_ROWS_CACHE: dict[str, int] = {}


def _csv_num_rows(csv_path: str) -> int:
    key = str(csv_path)
    if key in _CSV_ROWS_CACHE:
        return int(_CSV_ROWS_CACHE[key])
    path = ROOT / csv_path
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV for schedule estimation: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        
        n_all = sum(1 for _ in reader)
    n_rows = max(0, n_all - 1)
    _CSV_ROWS_CACHE[key] = int(n_rows)
    return int(n_rows)


def _scenario_val_fraction(args: argparse.Namespace, letter: str) -> float:
    key = f"val_frac_{str(letter).lower()}"
    frac = float(getattr(args, key, 1.0))
    if not (0.0 < frac <= 1.0):
        raise ValueError(f"{key} must be in (0, 1], got: {frac}")
    return frac


def _materialize_validation_subset_csv(
    *,
    scenario_letter: str,
    src_csv: str,
    fraction: float,
    seed: int,
    subset_dir: Path,
) -> str:
    if float(fraction) >= 0.999999:
        return str(src_csv)

    src_path = ROOT / src_csv
    if not src_path.exists():
        raise FileNotFoundError(f"Validation CSV not found: {src_path}")
    subset_dir.mkdir(parents=True, exist_ok=True)
    pct = int(round(float(fraction) * 100))
    dst_path = subset_dir / f"{scenario_letter}_validation_frac{pct}_seed{int(seed)}.csv"

    with src_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError(f"Empty CSV: {src_path}")
    header = rows[0]
    body = rows[1:]
    if not body:
        with dst_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
        return repo_rel(dst_path)

    keep_n = max(1, int(math.ceil(len(body) * float(fraction))))
    if keep_n >= len(body):
        chosen = body
    else:
        rng = random.Random(int(seed) + int(ord(str(scenario_letter)[0])))
        indices = sorted(rng.sample(range(len(body)), keep_n))
        chosen = [body[i] for i in indices]

    with dst_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(chosen)
    return repo_rel(dst_path)


def estimate_train_schedule(
    args: argparse.Namespace,
    *,
    scenario: Scenario,
    batch_plan: dict[str, Any],
) -> dict[str, Any]:
    is_sms = scenario.modality == "sms"
    epochs = float(args.sms_epochs if is_sms else args.voice_epochs)
    num_rows = int(_csv_num_rows(scenario.train_csv))
    effective_batch = max(1, int(batch_plan.get("effective_batch_size", 1)))
    steps_per_epoch = max(1, int(math.ceil(num_rows / effective_batch)))
    total_steps = max(1, int(math.ceil(steps_per_epoch * max(0.0, epochs))))

    ratio = float(max(1e-4, args.logging_steps_ratio))
    logging_steps = max(1, int(math.ceil(total_steps * ratio)))
    eval_steps = int(args.train_eval_steps) if int(args.train_eval_steps) > 0 else int(logging_steps)
    save_steps = int(args.train_save_steps) if int(args.train_save_steps) > 0 else int(eval_steps)

    return {
        "num_rows": int(num_rows),
        "effective_batch_size": int(effective_batch),
        "steps_per_epoch": int(steps_per_epoch),
        "total_steps": int(total_steps),
        "logging_steps": int(logging_steps),
        "eval_steps": int(eval_steps),
        "save_steps": int(save_steps),
    }


def prompt_cfg(scenario: Scenario) -> dict[str, Any]:
    return {
        "use_instruction": True,
        "instruction_path": scenario.prompt_instruction_path,
        "format": "plain",
        "user_prefix": "문자:" if scenario.modality == "sms" else "통화 내용:",
        "answer_prefix": "",
    }


def _lora_target_modules(model: dict[str, str]) -> list[str]:
    value = f"{model.get('alias', '')} {model.get('model_ref', '')}".lower()
    if "polyglot" in value or "kullm" in value:
        return ["query_key_value"]
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _model_capacity_profile(model_alias: str) -> dict[str, int]:
    alias = str(model_alias).lower()
    if "1p3b" in alias or "polyglot1b" in alias:
        return {
            "sms_train_token_budget": 49152,
            "voice_train_token_budget": 32768,
            "sms_eval_token_budget": 65536,
            "voice_eval_token_budget": 32768,
            "sms_target_effective_batch": 128,
            "voice_target_effective_batch": 32,
            "sms_max_train_batch": 64,
            "voice_max_train_batch": 16,
            "sms_max_eval_batch": 64,
            "voice_max_eval_batch": 16,
        }
    if "kullm" in alias:
        return {
            "sms_train_token_budget": 32768,
            "voice_train_token_budget": 20480,
            "sms_eval_token_budget": 49152,
            "voice_eval_token_budget": 20480,
            "sms_target_effective_batch": 64,
            "voice_target_effective_batch": 16,
            "sms_max_train_batch": 32,
            "voice_max_train_batch": 10,
            "sms_max_eval_batch": 48,
            "voice_max_eval_batch": 8,
        }
    return {
        "sms_train_token_budget": 32768,
        "voice_train_token_budget": 20480,
        "sms_eval_token_budget": 49152,
        "voice_eval_token_budget": 20480,
        "sms_target_effective_batch": 64,
        "voice_target_effective_batch": 16,
        "sms_max_train_batch": 32,
        "voice_max_train_batch": 10,
        "sms_max_eval_batch": 48,
        "voice_max_eval_batch": 8,
    }


def _floor_nice_batch(raw_value: int) -> int:
    value = max(1, int(raw_value))
    if value >= 32:
        rounded = value - (value % 8)
        return rounded if rounded > 0 else value
    if value >= 16:
        rounded = value - (value % 4)
        return rounded if rounded > 0 else value
    if value >= 8:
        rounded = value - (value % 2)
        return rounded if rounded > 0 else value
    return value


_BATCH_LADDER: tuple[int, ...] = (
    1,
    2,
    3,
    4,
    5,
    6,
    8,
    10,
    12,
    14,
    16,
    18,
    20,
    24,
    28,
    32,
    40,
    48,
    56,
    64,
)


def _conservative_step_down(value: int, steps: int) -> int:
    v = max(1, int(value))
    s = max(0, int(steps))
    if s == 0:
        return v
    ladder = [x for x in _BATCH_LADDER if x <= v]
    if not ladder:
        return 1
    idx = len(ladder) - 1
    idx = max(0, idx - s)
    return int(ladder[idx])


def _train_token_budget(profile: dict[str, int], *, is_sms: bool, gradient_checkpointing: bool) -> int:
    base = int(profile["sms_train_token_budget"] if is_sms else profile["voice_train_token_budget"])
    if gradient_checkpointing:
        return base
    scale = 0.85 if is_sms else 0.60
    return max(1, int(base * scale))


def _scaled_batch(value: int, scale: float) -> int:
    s = float(scale)
    if s <= 0:
        raise ValueError("--batch-scale must be > 0")
    if s == 1.0:
        return max(1, int(value))
    return max(1, int(math.floor(int(value) * s)))


def _cuda_world_size(cuda_visible_devices: Any) -> int:
    raw = cuda_visible_devices
    if raw is None:
        return 1
    if isinstance(raw, (list, tuple)):
        tokens = [str(item).strip() for item in raw if str(item).strip()]
        return max(1, len(tokens))
    tokens = [tok.strip() for tok in str(raw).split(",") if tok.strip()]
    return max(1, len(tokens)) if tokens else 1


def ensure_length_profiles(args: argparse.Namespace, models: list[dict[str, str]], scenarios: tuple[Scenario, ...]) -> None:
    if not args.refresh_length_profiles and LENGTH_PROFILE_JSON.exists():
        try:
            doc = load_json(LENGTH_PROFILE_JSON)
            profile_args = doc.get("profile_args", {}) or {}
            if (
                int(profile_args.get("sms_max_length", -1)) == int(args.sms_max_length)
                and int(profile_args.get("voice_max_length", -1)) == int(args.voice_max_length)
                and int(profile_args.get("decode_max_new_tokens", -1)) == int(args.decode_max_new_tokens)
                and str(profile_args.get("target_format_version", "")) == LENGTH_PROFILE_TARGET_FORMAT_VERSION
                and _length_profiles_cover(doc, models=models, scenarios=scenarios)
            ):
                return
        except Exception:
            pass

    cmd = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        args.profile_env,
        "python",
        "scripts/profile_noevidence_reason_suite_lengths.py",
        "--sms-max-length",
        str(args.sms_max_length),
        "--voice-max-length",
        str(args.voice_max_length),
        "--decode-max-new-tokens",
        str(args.decode_max_new_tokens),
        "--scenario-letters",
        "".join(scenario.letter for scenario in scenarios),
        "--output-json",
        str(LENGTH_PROFILE_JSON),
        "--output-csv",
        str(PREPARED_DATA_ROOT / "length_profiles.csv"),
    ]
    for model in models:
        cmd.extend(["--model", f"{model['alias']}={model['model_ref']}"])
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def _lookup_train_profile(
    length_profiles: dict[str, Any],
    *,
    model_alias: str,
    scenario_letter: str,
) -> dict[str, Any]:
    return ((length_profiles.get("models", {}) or {}).get(model_alias, {}) or {}).get("train", {}).get(scenario_letter, {})


def _lookup_eval_profile(
    length_profiles: dict[str, Any],
    *,
    model_alias: str,
    eval_id: str,
) -> dict[str, Any]:
    return ((length_profiles.get("models", {}) or {}).get(model_alias, {}) or {}).get("eval", {}).get(eval_id, {})


def build_batch_plan(
    args: argparse.Namespace,
    *,
    model: dict[str, str],
    scenario: Scenario,
    length_profiles: dict[str, Any],
) -> dict[str, Any]:
    is_sms = scenario.modality == "sms"
    world_size = _cuda_world_size(args.cuda_visible_devices)
    profile = _model_capacity_profile(model["alias"])
    train_profile = _lookup_train_profile(length_profiles, model_alias=model["alias"], scenario_letter=scenario.letter)
    used_p99 = int(math.ceil(float((((train_profile.get("used", {}) or {}).get("p99", 0.0)) or 0.0))))
    used_mean = float((((train_profile.get("used", {}) or {}).get("mean", 0.0)) or 0.0))
    planning_length = max(1, int(train_profile.get("planning_train_length_p99", used_p99 or 1)))

    if is_sms:
        micro_budget = _train_token_budget(profile, is_sms=True, gradient_checkpointing=bool(args.gradient_checkpointing))
        max_train_batch = int(profile["sms_max_train_batch"])
        max_eval_batch = int(profile["sms_max_eval_batch"])
        target_effective_batch = int(profile["sms_target_effective_batch"]) * max(1, world_size)
        manual_batch = args.sms_batch_size
        manual_grad_accum = args.sms_grad_accum
        manual_eval_batch = args.sms_eval_batch_size
    else:
        micro_budget = _train_token_budget(profile, is_sms=False, gradient_checkpointing=bool(args.gradient_checkpointing))
        max_train_batch = int(profile["voice_max_train_batch"])
        max_eval_batch = int(profile["voice_max_eval_batch"])
        target_effective_batch = int(profile["voice_target_effective_batch"]) * max(1, world_size)
        manual_batch = args.voice_batch_size
        manual_grad_accum = args.voice_grad_accum
        manual_eval_batch = args.voice_eval_batch_size

    auto_micro_batch = min(max_train_batch, max(1, micro_budget // max(1, planning_length)))
    auto_micro_batch = _floor_nice_batch(auto_micro_batch)
    if manual_batch is not None:
        micro_batch = int(manual_batch)
    else:
        auto_micro_batch = max(1, int(math.floor(auto_micro_batch / max(1, world_size))))
        micro_batch = _conservative_step_down(auto_micro_batch, int(args.autobatch_conservative_steps))
    micro_batch = _scaled_batch(micro_batch, float(args.batch_scale))
    grad_accum = (
        int(manual_grad_accum)
        if manual_grad_accum is not None
        else max(1, int(math.ceil(target_effective_batch / max(1, micro_batch * world_size))))
    )
    if manual_eval_batch is not None:
        eval_batch = int(manual_eval_batch)
    else:
        eval_batch = _conservative_step_down(min(max_eval_batch, max(1, micro_batch)), int(args.autobatch_conservative_steps))
    eval_batch = _scaled_batch(eval_batch, float(args.batch_scale))
    per_device_effective_batch = int(micro_batch * grad_accum)
    effective_batch = int(per_device_effective_batch * world_size)

    return {
        "batch_size": int(micro_batch),
        "grad_accum": int(grad_accum),
        "eval_batch_size": int(eval_batch),
        "world_size": int(world_size),
        "per_device_effective_batch_size": int(per_device_effective_batch),
        "effective_batch_size": int(effective_batch),
        "planning_train_length": int(planning_length),
        "used_mean_length": float(used_mean),
        "estimated_tokens_per_micro_step": int(micro_batch * planning_length),
        "estimated_tokens_per_update": int(micro_batch * grad_accum * planning_length * world_size),
        "target_effective_batch_size": int(target_effective_batch),
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "source": "length_profile_auto",
    }


def build_eval_batch_plan(
    args: argparse.Namespace,
    *,
    model: dict[str, str],
    scenario: Scenario,
    eval_target: EvalTarget,
    length_profiles: dict[str, Any],
) -> dict[str, Any]:
    is_sms = scenario.modality == "sms"
    profile = _model_capacity_profile(model["alias"])
    eval_profile = _lookup_eval_profile(length_profiles, model_alias=model["alias"], eval_id=eval_target.eval_id)
    prompt_p99 = float((((eval_profile.get("prompt", {}) or {}).get("p99", 0.0)) or 0.0))
    decode_new_tokens = _eval_decode_max_new_tokens(args)
    planning_length = max(1, int(math.ceil(prompt_p99)) + int(decode_new_tokens))

    if is_sms:
        eval_budget = int(profile["sms_eval_token_budget"])
        max_eval_batch = int(profile["sms_max_eval_batch"])
        manual_hf_batch = args.sms_hf_batch_size
    else:
        eval_budget = int(profile["voice_eval_token_budget"])
        max_eval_batch = int(profile["voice_max_eval_batch"])
        manual_hf_batch = args.voice_hf_batch_size

    auto_hf_batch = min(max_eval_batch, max(1, eval_budget // max(1, planning_length)))
    auto_hf_batch = _floor_nice_batch(auto_hf_batch)
    if manual_hf_batch is not None:
        hf_batch_size = int(manual_hf_batch)
    else:
        hf_batch_size = _conservative_step_down(auto_hf_batch, int(args.autobatch_conservative_steps))
    hf_batch_size = _scaled_batch(hf_batch_size, float(args.batch_scale))
    return {
        "hf_batch_size": int(hf_batch_size),
        "planning_eval_length": int(planning_length),
        "estimated_eval_tokens_per_step": int(hf_batch_size * planning_length),
        "source": "length_profile_auto",
    }


def build_train_cfg(
    args: argparse.Namespace,
    dirs: dict[str, Path],
    model: dict[str, str],
    scenario: Scenario,
    batch_plan: dict[str, Any],
    train_schedule: dict[str, Any],
    train_eval_csv: str,
) -> dict[str, Any]:
    is_sms = scenario.modality == "sms"
    scenario_new_tokens = _scenario_max_new_tokens(scenario.letter)
    out_root = dirs["output_root"] / model["alias"] / "train" / scenario.letter
    exp_suffix = _label_explanation_reconstruction_suffix(args)
    return {
        "exp_name": f"{model['alias']}_{scenario.letter}_{scenario.name}{exp_suffix}",
        "task": "train_sft",
        "model": {
            "model_type": "Decoder",
            "model_name": model["model_ref"],
            "backend": args.backend,
            "dtype": args.dtype,
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": args.dtype,
            "bnb_4bit_use_double_quant": True,
        },
        "data": {
            "train_csv": scenario.train_csv,
            **({"eval_csv": str(train_eval_csv)} if args.use_validation else {}),
            "text_col": scenario.train_text_col,
            "label_col": "label",
            "metric_label_col": "label",
            "reason_col": "reason_value",
            "compose_label_reason_target": True,
            "eval_compose_label_reason_target": True,
            "label_reason_target_format": "label_first_explanation",
        },
        "prompt": prompt_cfg(scenario),
        "train": {
            "trainer": "hf",
            "label_explanation_multitask": True,
            "explanation_loss_weight": 0.1,
            **(
                {
                    "reconstruction_loss_weight": float(args.stage2_reconstruction_loss_weight),
                    "reconstruction_pooling": str(args.stage2_reconstruction_pooling),
                    "reconstruction_scope": str(args.stage2_reconstruction_scope),
                }
                if float(args.stage2_reconstruction_loss_weight) > 0.0
                else {}
            ),
            "eval_generate_metrics": False,
            "epochs": args.sms_epochs if is_sms else args.voice_epochs,
            "lr": args.sms_lr if is_sms else args.voice_lr,
            "batch_size": int(batch_plan["batch_size"]),
            "grad_accum": int(batch_plan["grad_accum"]),
            "eval_batch_size": int(batch_plan["eval_batch_size"]),
            "eval_generate_batch_size": int(batch_plan["eval_batch_size"]),
            "eval_max_input_tokens": args.sms_max_length if is_sms else args.voice_max_length,
            "eval_max_new_tokens": int(scenario_new_tokens),
            "max_length": args.sms_max_length if is_sms else args.voice_max_length,
            "warmup_ratio": 0.03,
            "gradient_checkpointing": bool(args.gradient_checkpointing),
            "length_bucket": bool(args.length_bucket),
            "train_sampling_strategy": "group_by_length" if args.length_bucket else "random",
            "length_column_name": "length",
            "peft": "lora",
            "lora_r": 8,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "lora_bias": "none",
            "lora_target_modules": _lora_target_modules(model),
            "save_only_model": True,
            "save_strategy": "epoch",
            "save_steps": int(train_schedule["save_steps"]),
            "eval_strategy": "epoch" if args.use_validation else "no",
            "eval_steps": int(train_schedule["eval_steps"]),
            "logging_steps": int(train_schedule["logging_steps"]),
            "save_total_limit": args.save_total_limit,
            "dataloader_num_workers": args.dataloader_num_workers,
            "report_to": args.report_to,
            "logging_dir": f"outputs/runs/tb_logs/{scenario.modality}/reason_suite",
            "ddp_find_unused_parameters": False,
            **(
                {
                    "load_best_model_at_end": True,
                    "metric_for_best_model": "eval_label_loss",
                    "greater_is_better": False,
                }
                if args.use_validation
                else {}
            ),
        },
        "run": {
            "seed": args.seed,
            "out_root": str(out_root),
            "use_running_dir": True,
            "running_root": "outputs/runs/running",
            "running_tb_root": "outputs/runs/tb_logs/running",
            **({"cuda_visible_devices": args.cuda_visible_devices} if args.cuda_visible_devices else {}),
        },
    }


def build_eval_cfg(
    args: argparse.Namespace,
    dirs: dict[str, Path],
    model: dict[str, str],
    scenario: Scenario,
    eval_target: EvalTarget,
    adapter_path: str | None,
    eval_batch_plan: dict[str, Any],
    model_name_override: str | None = None,
) -> dict[str, Any]:
    out_root = dirs["output_root"] / model["alias"] / "eval" / scenario.letter / eval_target.eval_id
    scenario_new_tokens = _eval_decode_max_new_tokens(args)
    model_name = str(model_name_override) if model_name_override else model["model_ref"]
    exp_suffix = _label_explanation_reconstruction_suffix(args)
    fullgen_suffix = f"_fullgen{int(args.decode_max_new_tokens)}" if bool(args.eval_full_generation) else ""
    model_cfg = {
        "model_type": "Finetuned",
        "model_name": model_name,
        "backend": "hf",
        "dtype": args.eval_dtype,
        "merge_adapter": False,
    }
    if adapter_path:
        model_cfg["adapter_path"] = adapter_path
    return {
        "exp_name": f"{model['alias']}_{eval_target.eval_id}_{scenario.name}{exp_suffix}{fullgen_suffix}",
        "task": "eval_decode",
        "model": model_cfg,
        "data": {
            "eval_csv": eval_target.csv_path,
            "text_col": eval_target.text_col,
            "label_col": "label",
        },
        "prompt": prompt_cfg(scenario),
        "decode": {
            "max_input_tokens": args.decode_max_input_tokens,
            "max_input_tokens_quantile": args.decode_max_input_tokens_quantile,
            "max_input_tokens_margin": int(_scenario_max_input_margin(scenario)),
            "max_new_tokens": int(scenario_new_tokens),
            "temperature": args.temperature,
            "top_p": _eval_top_p(args.temperature, args.top_p),
            "repetition_penalty": args.repetition_penalty,
            "hf_batch_size": int(eval_batch_plan["hf_batch_size"]),
            "allow_hf_fallback": True,
            "prediction_format": "leading_binary",
            "constrain_binary_output": True,
            "constrain_binary_choices": ["0", "1"],
            "compute_metrics": True,
        },
        "run": {
            "seed": args.seed,
            "out_root": str(out_root),
            **({"cuda_visible_devices": args.cuda_visible_devices} if args.cuda_visible_devices else {}),
        },
    }


def write_yaml(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


def run_config(cfg_path: Path, orchestrator_log: Path) -> str:
    cmd = ["bash", "scripts/run_decode.sh", str(cfg_path)]
    env = os.environ.copy()
    python_bin_dir = str(Path(sys.executable).resolve().parent)
    env["PATH"] = python_bin_dir + os.pathsep + env.get("PATH", "")
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    run_dir: str | None = None
    with orchestrator_log.open("a", encoding="utf-8") as logf:
        logf.write(f"\n$ {' '.join(cmd)}\n")
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            logf.write(line)
            if "[done] run_dir=" in line:
                run_dir = line.split("[done] run_dir=", 1)[1].strip()
            elif "[run.sh]" in line and "run_dir=" in line:
                
                run_dir = line.split("run_dir=", 1)[1].strip()
    rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)
    if not run_dir:
        raise RuntimeError(f"Could not determine run_dir for config: {cfg_path}")
    return run_dir


def repo_rel(path_like: str | Path) -> str:
    p = Path(path_like)
    if not p.is_absolute():
        return str(p)
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def _find_existing_run_dirs(out_root: Path, exp_name: str) -> list[Path]:
    if not out_root.exists():
        return []
    suffix = f"_{exp_name}"
    runs = [p for p in out_root.iterdir() if p.is_dir() and p.name.endswith(suffix)]
    return sorted(runs, key=lambda p: p.name, reverse=True)


def _latest_completed_train_run(out_root: Path, exp_name: str) -> Path | None:
    required_any = (
        "adapter_model.safetensors",
        "pytorch_model.bin",
        "model.safetensors",
    )
    for run_dir in _find_existing_run_dirs(out_root, exp_name):
        if any((run_dir / name).exists() for name in required_any):
            return run_dir
    return None


def _latest_completed_eval_run(out_root: Path, exp_name: str) -> Path | None:
    for run_dir in _find_existing_run_dirs(out_root, exp_name):
        if (run_dir / "metrics.json").exists():
            return run_dir
    return None


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _discover_epoch_checkpoints(run_dir: Path) -> list[dict[str, Any]]:
    checkpoints: list[dict[str, Any]] = []
    for ckpt_dir in sorted(run_dir.glob("checkpoint-*")):
        state_path = ckpt_dir / "trainer_state.json"
        if not state_path.exists():
            continue
        try:
            state = load_json(state_path)
        except Exception:
            continue
        epoch = _float_or_none(state.get("epoch"))
        step = state.get("global_step")
        if epoch is None:
            continue
        checkpoints.append(
            {
                "path": ckpt_dir,
                "epoch": float(epoch),
                "global_step": int(step) if step is not None else None,
            }
        )
    return sorted(checkpoints, key=lambda row: (row["epoch"], str(row["path"])))


def _select_checkpoint_by_epoch(checkpoints: list[dict[str, Any]], target_epoch: float) -> dict[str, Any] | None:
    if not checkpoints:
        return None
    target = float(target_epoch)
    exact = [row for row in checkpoints if abs(float(row["epoch"]) - target) < 1e-6]
    if exact:
        return sorted(exact, key=lambda row: str(row["path"]))[-1]
    return min(
        checkpoints,
        key=lambda row: (abs(float(row["epoch"]) - target), -float(row["epoch"])),
    )


def _select_best_epoch_record(
    metrics_rows: list[dict[str, Any]],
    *,
    min_epoch: float,
    max_epoch: float,
) -> dict[str, Any] | None:
    lo = float(min_epoch)
    hi = float(max_epoch)
    candidates = []
    for row in metrics_rows:
        epoch = _float_or_none(row.get("epoch"))
        macro_f1 = _float_or_none(row.get("macro_f1"))
        acc = _float_or_none(row.get("accuracy"))
        invalid_parse = row.get("invalid_parse_count")
        if epoch is None or epoch < lo or epoch > hi:
            continue
        candidates.append(
            (
                -(macro_f1 if macro_f1 is not None else -1.0),
                -(acc if acc is not None else -1.0),
                int(invalid_parse) if invalid_parse is not None else 10**9,
                float(epoch),
                row,
            )
        )
    if not candidates:
        return None
    return min(candidates)[-1]


def _replace_dir_alias(src_dir: Path, dst_dir: Path) -> None:
    src_dir = src_dir.resolve()
    if dst_dir.is_symlink() or dst_dir.exists():
        if dst_dir.is_symlink() or dst_dir.is_file():
            dst_dir.unlink()
        else:
            shutil.rmtree(dst_dir)
    dst_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(str(src_dir), str(dst_dir), target_is_directory=True)
    except OSError:
        shutil.copytree(src_dir, dst_dir)


def _prepare_train_checkpoint_aliases(
    *,
    run_dir: Path,
    primary_eval_epoch: float,
    best_epoch_min: float,
    best_epoch_max: float,
) -> dict[str, Any]:
    primary_epoch_tag = f"{int(primary_eval_epoch)}" if float(primary_eval_epoch).is_integer() else str(primary_eval_epoch).replace(".", "p")
    best_min_tag = f"{int(best_epoch_min)}" if float(best_epoch_min).is_integer() else str(best_epoch_min).replace(".", "p")
    best_max_tag = f"{int(best_epoch_max)}" if float(best_epoch_max).is_integer() else str(best_epoch_max).replace(".", "p")
    checkpoints = _discover_epoch_checkpoints(run_dir)
    metrics_rows = _load_jsonl(run_dir / "eval_validation_generate_metrics.jsonl")
    primary_ckpt = _select_checkpoint_by_epoch(checkpoints, primary_eval_epoch)
    best_row = _select_best_epoch_record(
        metrics_rows,
        min_epoch=best_epoch_min,
        max_epoch=best_epoch_max,
    )
    best_ckpt = (
        _select_checkpoint_by_epoch(checkpoints, float(best_row["epoch"]))
        if best_row is not None
        else None
    )

    aliases: dict[str, Any] = {
        "run_dir": str(run_dir),
        "checkpoints": [
            {
                "epoch": float(row["epoch"]),
                "global_step": row["global_step"],
                "path": str(row["path"]),
            }
            for row in checkpoints
        ],
    }

    if primary_ckpt is not None:
        primary_alias = run_dir / f"selected_epoch{primary_epoch_tag}"
        _replace_dir_alias(Path(primary_ckpt["path"]), primary_alias)
        aliases["primary_eval"] = {
            "epoch": float(primary_ckpt["epoch"]),
            "global_step": primary_ckpt.get("global_step"),
            "path": str(primary_alias),
            "source_checkpoint": str(primary_ckpt["path"]),
        }

    if best_ckpt is not None:
        best_alias = run_dir / f"selected_best_epoch{best_min_tag}_to_{best_max_tag}"
        _replace_dir_alias(Path(best_ckpt["path"]), best_alias)
        aliases["best_epoch7_to_10"] = {
            "epoch": float(best_ckpt["epoch"]),
            "global_step": best_ckpt.get("global_step"),
            "path": str(best_alias),
            "source_checkpoint": str(best_ckpt["path"]),
            "metric_row": best_row,
        }

    alias_path = run_dir / "checkpoint_aliases.json"
    alias_path.write_text(json.dumps(aliases, ensure_ascii=False, indent=2), encoding="utf-8")
    aliases["alias_manifest"] = str(alias_path)
    return aliases


def flatten_eval_record(base: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    confusion = metrics.get("confusion", {}) or {}
    return {
        **base,
        "n": metrics.get("n"),
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "positive_recall": (metrics.get("class_1", {}) or {}).get("recall"),
        "tp": confusion.get("TP"),
        "tn": confusion.get("TN"),
        "fp": confusion.get("FP"),
        "fn": confusion.get("FN"),
        "invalid_parse_count": metrics.get("invalid_parse_count"),
        "by_category_json": json.dumps(metrics.get("by_category", {}), ensure_ascii=False),
        "by_subset_json": json.dumps(metrics.get("by_subset", {}), ensure_ascii=False),
    }


def mean(values: list[float]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def model_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        by_model.setdefault(row["model_alias"], []).append(row)
    out: list[dict[str, Any]] = []
    for model_alias, rows in by_model.items():
        macro = [r["macro_f1"] for r in rows if r.get("macro_f1") is not None]
        out.append(
            {
                "model_alias": model_alias,
                "model_ref": rows[0]["model_ref"],
                "num_evals": len(rows),
                "overall_mean_macro_f1": mean(macro),
                "overall_min_macro_f1": min(macro) if macro else None,
                "sms_in_domain_mean_macro_f1": mean([r["macro_f1"] for r in rows if r["setting_group"] == "sms_in_domain"]),
                "sms_ood_mean_macro_f1": mean([r["macro_f1"] for r in rows if r["setting_group"] == "sms_ood"]),
                "voice_in_domain_mean_macro_f1": mean([r["macro_f1"] for r in rows if r["setting_group"] == "voice_in_domain"]),
                "voice_ood_mean_macro_f1": mean([r["macro_f1"] for r in rows if r["setting_group"] == "voice_ood"]),
            }
        )
    return sorted(
        out,
        key=lambda x: (
            x["overall_min_macro_f1"] is None,
            -(x["overall_min_macro_f1"] or -1e9),
            -(x["overall_mean_macro_f1"] or -1e9),
        ),
    )


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def ensure_prepared_data(skip_prepare: bool, scenarios: tuple[Scenario, ...]) -> None:
    expected = [ROOT / scenario.train_csv for scenario in scenarios]
    expected += [ROOT / scenario.train_eval_csv for scenario in scenarios]
    expected += [ROOT / target.csv_path for scenario in scenarios for target in scenario.eval_targets]
    missing = [path for path in expected if not path.exists()]
    if not missing:
        return
    missing_rel = [repo_rel(path) for path in missing]
    raise FileNotFoundError(
        "Missing prepared evidence/keep splits. The public artifact expects "
        f"prebuilt files under data/sms/evidence/keep and data/voice/evidence/keep; missing: {missing_rel}"
    )


def main() -> int:
    args = parse_args()
    if float(args.stage2_reconstruction_loss_weight) < 0.0:
        raise ValueError("--stage2-reconstruction-loss-weight must be >= 0")
    models = load_model_specs(args)
    scenarios = _select_scenarios(args.scenario_letters)
    ensure_prepared_data(skip_prepare=args.skip_prepare_data, scenarios=scenarios)
    ensure_length_profiles(args, models, scenarios)
    length_profiles = load_json(LENGTH_PROFILE_JSON)

    bench_name = args.bench_name or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_noevidence_reason_suite"
    dirs = make_dirs(
        bench_name,
        suite_name=str(args.suite_name),
        benchmark_layout=str(args.benchmark_layout),
    )
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    orchestrator_log = dirs["analysis_root"] / "orchestrator.log"
    manifest: dict[str, Any] = {
        "bench_name": bench_name,
        "models": models,
        "scenarios": [s.__dict__ | {"eval_targets": [et.__dict__ for et in s.eval_targets]} for s in scenarios],
        "scenario_letters": [scenario.letter for scenario in scenarios],
        "suite_name": str(args.suite_name),
        "benchmark_layout": str(args.benchmark_layout),
        "generated_at": datetime.now().isoformat(),
        "run_enabled": bool(args.run),
        "prepared_data_root": repo_rel(PREPARED_DATA_ROOT),
        "prepared_data_manifest": repo_rel(PREPARED_DATA_ROOT / "reason_manifest.json"),
        "length_profiles_json": repo_rel(LENGTH_PROFILE_JSON),
    }

    scenario_train_eval_csv_map: dict[str, str] = {}
    subset_root = Path(args.validation_subset_dir)
    for scenario in scenarios:
        frac = _scenario_val_fraction(args, scenario.letter)
        selected_eval_csv = str(scenario.train_eval_csv)
        if args.use_validation:
            selected_eval_csv = _materialize_validation_subset_csv(
                scenario_letter=scenario.letter,
                src_csv=str(scenario.train_eval_csv),
                fraction=float(frac),
                seed=int(args.seed),
                subset_dir=subset_root,
            )
        scenario_train_eval_csv_map[scenario.letter] = str(selected_eval_csv)
        manifest.setdefault("validation_sampling", []).append(
            {
                "scenario_letter": scenario.letter,
                "fraction": float(frac),
                "source_eval_csv": str(scenario.train_eval_csv),
                "selected_eval_csv": str(selected_eval_csv),
                "selected_rows": (
                    int(_csv_num_rows(str(selected_eval_csv)))
                    if args.use_validation
                    else None
                ),
            }
        )

    eval_rows: list[dict[str, Any]] = []

    for model in models:
        for scenario in scenarios:
            batch_plan = build_batch_plan(
                args,
                model=model,
                scenario=scenario,
                length_profiles=length_profiles,
            )
            train_schedule = estimate_train_schedule(
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
            train_cfg = build_train_cfg(
                args,
                dirs,
                model,
                scenario,
                batch_plan=batch_plan,
                train_schedule=train_schedule,
                train_eval_csv=scenario_train_eval_csv_map[scenario.letter],
            )
            train_cfg_path = dirs["config_root"] / model["alias"] / f"{scenario.letter}_train.yaml"
            write_yaml(train_cfg_path, train_cfg)
            manifest.setdefault("generated_train_configs", []).append(repo_rel(train_cfg_path))

            scenario_eval_batch_plans: dict[str, dict[str, Any]] = {}
            for eval_target in scenario.eval_targets:
                eval_batch_plan = build_eval_batch_plan(
                    args,
                    model=model,
                    scenario=scenario,
                    eval_target=eval_target,
                    length_profiles=length_profiles,
                )
                scenario_eval_batch_plans[eval_target.eval_id] = eval_batch_plan
                manifest.setdefault("auto_eval_batch_plans", []).append(
                    {
                        "model_alias": model["alias"],
                        "scenario_letter": scenario.letter,
                        "eval_id": eval_target.eval_id,
                        "eval_split_name": eval_target.split_name,
                        **eval_batch_plan,
                    }
                )

            train_run_dir: str | None = None
            train_summary_path: Path | None = None
            selected_eval_model_name: str | None = None
            selected_eval_adapter_path: str | None = None
            if args.run:
                train_out_root = dirs["output_root"] / model["alias"] / "train" / scenario.letter
                existing_train_run = _latest_completed_train_run(train_out_root, str(train_cfg["exp_name"]))
                if existing_train_run is not None:
                    train_run_dir = str(existing_train_run)
                    train_summary_path = existing_train_run / "summary.json"
                    print(
                        f"\n=== {model['alias']} :: {scenario.letter} {scenario.name} train ===\n"
                        f"[resume] reusing existing train run: {repo_rel(existing_train_run)}"
                    )
                else:
                    print(f"\n=== {model['alias']} :: {scenario.letter} {scenario.name} train ===")
                    train_run_dir = run_config(train_cfg_path, orchestrator_log)
                    train_summary_path = Path(train_run_dir) / "summary.json"
                if str(train_cfg["train"].get("peft", "none")).lower() in {"lora", "dora"}:
                    selected_eval_model_name = model["model_ref"]
                    selected_eval_adapter_path = train_run_dir
                else:
                    selected_eval_model_name = train_run_dir
                    selected_eval_adapter_path = None
            else:
                train_run_dir = "__TRAIN_RUN_DIR__"
                if str(train_cfg["train"].get("peft", "none")).lower() in {"lora", "dora"}:
                    selected_eval_model_name = model["model_ref"]
                    selected_eval_adapter_path = train_run_dir
                else:
                    selected_eval_model_name = train_run_dir
                    selected_eval_adapter_path = None

            for eval_target in scenario.eval_targets:
                eval_batch_plan = scenario_eval_batch_plans[eval_target.eval_id]
                eval_cfg = build_eval_cfg(
                    args,
                    dirs,
                    model,
                    scenario,
                    eval_target,
                    adapter_path=selected_eval_adapter_path,
                    eval_batch_plan=eval_batch_plan,
                    model_name_override=selected_eval_model_name,
                )
                eval_cfg_path = dirs["config_root"] / model["alias"] / f"{eval_target.eval_id}_eval.yaml"
                write_yaml(eval_cfg_path, eval_cfg)
                manifest.setdefault("generated_eval_configs", []).append(repo_rel(eval_cfg_path))

                if not args.run:
                    continue

                eval_out_root = dirs["output_root"] / model["alias"] / "eval" / scenario.letter / eval_target.eval_id
                existing_eval_run = _latest_completed_eval_run(eval_out_root, str(eval_cfg["exp_name"]))
                if existing_eval_run is not None and not bool(args.force_rerun_eval):
                    eval_run_dir = str(existing_eval_run)
                    print(
                        f"\n--- {model['alias']} :: {eval_target.eval_id} {eval_target.split_name} eval ---\n"
                        f"[resume] skipping completed eval: {repo_rel(existing_eval_run)}"
                    )
                else:
                    print(f"\n--- {model['alias']} :: {eval_target.eval_id} {eval_target.split_name} eval ---")
                    eval_run_dir = run_config(eval_cfg_path, orchestrator_log)
                metrics_path = Path(eval_run_dir) / "metrics.json"
                if not metrics_path.exists():
                    raise FileNotFoundError(f"Missing eval metrics: {metrics_path}")
                metrics = load_json(metrics_path)
                row_base = {
                    "model_alias": model["alias"],
                    "model_ref": model["model_ref"],
                    "scenario_letter": scenario.letter,
                    "scenario_name": scenario.name,
                    "setting_group": scenario.setting_group,
                    "modality": scenario.modality,
                    "train_csv": scenario.train_csv,
                    "train_config": repo_rel(train_cfg_path),
                    "train_run_dir": train_run_dir,
                    "train_eval_adapter_path": str(selected_eval_adapter_path),
                    "train_summary": repo_rel(train_summary_path) if train_summary_path and train_summary_path.exists() else "",
                    "train_checkpoint_alias_manifest": "",
                    "eval_id": eval_target.eval_id,
                    "eval_split_name": eval_target.split_name,
                    "eval_csv": eval_target.csv_path,
                    "eval_config": repo_rel(eval_cfg_path),
                    "eval_run_dir": eval_run_dir,
                    "metrics_path": repo_rel(metrics_path),
                }
                eval_rows.append(flatten_eval_record(row_base, metrics))

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
    save_csv(eval_summary_csv, eval_rows)

    model_rows = model_summary(eval_rows)
    model_summary_json.write_text(json.dumps(model_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    save_csv(model_summary_csv, model_rows)

    print(f"\nSaved manifest: {repo_rel(manifest_path)}")
    print(f"Saved eval summary: {repo_rel(eval_summary_json)}")
    print(f"Saved model summary: {repo_rel(model_summary_json)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
