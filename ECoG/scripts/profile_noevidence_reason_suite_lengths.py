#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
PREPARED_DATA_ROOT = ROOT / "data"
SMS_PROMPT = ROOT / "src" / "phishdec" / "prompts" / "instructions" / "korsmishing_explainer" / "kor_label_first_sms_sys.txt"
VOICE_PROMPT = ROOT / "src" / "phishdec" / "prompts" / "instructions" / "korsmishing_explainer" / "kor_label_first_voice_sys.txt"
TARGET_FORMAT_VERSION = "label_first_explanation_v1"

DEFAULT_MODELS: tuple[str, ...] = (
    "kullm5b=nlpai-lab/kullm-polyglot-5.8b-v2",
    "polyglot1b=EleutherAI/polyglot-ko-1.3b",
    "polyglot5b=EleutherAI/polyglot-ko-5.8b",
)

SCENARIOS: tuple[dict[str, Any], ...] = (
    {
        "letter": "A",
        "modality": "sms",
        "train_csv": "data/sms/reason/A_train.csv",
        "prompt_instruction_path": str(SMS_PROMPT),
        "train_text_col": "text",
        "eval_targets": (
            {"eval_id": "A1", "split_name": "test", "csv_path": "data/sms/in_domain/test.csv", "text_col": "text"},
            {
                "eval_id": "A2",
                "split_name": "challenging",
                "csv_path": "data/sms/in_domain/challenging.csv",
                "text_col": "text",
            },
        ),
    },
    {
        "letter": "B",
        "modality": "sms",
        "train_csv": "data/sms/reason/B_train.csv",
        "prompt_instruction_path": str(SMS_PROMPT),
        "train_text_col": "text",
        "eval_targets": (
            {
                "eval_id": "B1",
                "split_name": "test",
                "csv_path": "data/sms/ood/test/credit_test.csv",
                "text_col": "text",
            },
            {
                "eval_id": "B2",
                "split_name": "challenging",
                "csv_path": "data/sms/ood/challenging/credit_challenging.csv",
                "text_col": "text",
            },
        ),
    },
    {
        "letter": "C",
        "modality": "sms",
        "train_csv": "data/sms/reason/C_train.csv",
        "prompt_instruction_path": str(SMS_PROMPT),
        "train_text_col": "text",
        "eval_targets": (
            {
                "eval_id": "C1",
                "split_name": "test",
                "csv_path": "data/sms/ood/test/finance_test.csv",
                "text_col": "text",
            },
            {
                "eval_id": "C2",
                "split_name": "challenging",
                "csv_path": "data/sms/ood/challenging/finance_challenging.csv",
                "text_col": "text",
            },
        ),
    },
    {
        "letter": "D",
        "modality": "sms",
        "train_csv": "data/sms/reason/D_train.csv",
        "prompt_instruction_path": str(SMS_PROMPT),
        "train_text_col": "text",
        "eval_targets": (
            {
                "eval_id": "D1",
                "split_name": "test",
                "csv_path": "data/sms/ood/test/parcel_test.csv",
                "text_col": "text",
            },
            {
                "eval_id": "D2",
                "split_name": "challenging",
                "csv_path": "data/sms/ood/challenging/parcel_challenging.csv",
                "text_col": "text",
            },
        ),
    },
    {
        "letter": "E",
        "modality": "voice",
        "train_csv": "data/voice/reason/E_train.csv",
        "prompt_instruction_path": str(VOICE_PROMPT),
        "train_text_col": "text",
        "eval_targets": (
            {"eval_id": "E1", "split_name": "test", "csv_path": "data/voice/in_domain/test.csv", "text_col": "text"},
            {
                "eval_id": "E2",
                "split_name": "challenging",
                "csv_path": "data/voice/in_domain/challenge.csv",
                "text_col": "text",
            },
        ),
    },
    {
        "letter": "F",
        "modality": "voice",
        "train_csv": "data/voice/reason/F_train.csv",
        "prompt_instruction_path": str(VOICE_PROMPT),
        "train_text_col": "text",
        "eval_targets": (
            {
                "eval_id": "F1",
                "split_name": "test",
                "csv_path": "data/voice/ood/test/ood_test_finance.csv",
                "text_col": "text",
            },
            {
                "eval_id": "F2",
                "split_name": "challenging",
                "csv_path": "data/voice/ood/challenging/finance_ood_challenge.csv",
                "text_col": "text",
            },
        ),
    },
    {
        "letter": "G",
        "modality": "voice",
        "train_csv": "data/voice/reason/G_train.csv",
        "prompt_instruction_path": str(VOICE_PROMPT),
        "train_text_col": "text",
        "eval_targets": (
            {
                "eval_id": "G1",
                "split_name": "test",
                "csv_path": "data/voice/ood/test/ood_test_government.csv",
                "text_col": "text",
            },
            {
                "eval_id": "G2",
                "split_name": "challenging",
                "csv_path": "data/voice/ood/challenging/government_ood_challenge.csv",
                "text_col": "text",
            },
        ),
    },
)


def _select_scenarios(raw_value: str) -> tuple[dict[str, Any], ...]:
    available = {str(scenario["letter"]).upper(): scenario for scenario in SCENARIOS}
    raw = str(raw_value or "").strip()
    if not raw or raw.upper() == "ALL":
        return SCENARIOS

    parts = [part.strip().upper() for part in raw.replace(",", " ").split() if part.strip()]
    if len(parts) == 1 and parts[0].isalpha() and len(parts[0]) > 1:
        parts = list(parts[0])

    selected: list[dict[str, Any]] = []
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile token lengths for no-evidence reason suite A~G data.")
    p.add_argument("--model", action="append", default=[], help="Format: alias=model_ref or model_ref")
    p.add_argument(
        "--scenario-letters",
        default="ALL",
        help="Scenario letters to profile, e.g. BCDFG or B,C,D,F,G. Default: ALL",
    )
    p.add_argument("--sms-max-length", type=int, default=1000)
    p.add_argument("--voice-max-length", type=int, default=2200)
    p.add_argument("--decode-max-new-tokens", type=int, default=128)
    p.add_argument(
        "--output-json",
        default=str(PREPARED_DATA_ROOT / "length_profiles.json"),
    )
    p.add_argument(
        "--output-csv",
        default=str(PREPARED_DATA_ROOT / "length_profiles.csv"),
    )
    p.add_argument("--batch-size", type=int, default=256)
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


def _read_instruction(path: str) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def _render_instruction_with_text(instruction: str, text: str) -> tuple[str, bool]:
    if not instruction:
        return "", False
    rendered = instruction
    text_value = str(text)
    for placeholder in ("{text}", "{{text}}", "{{transcript}}", "{{input_text}}"):
        rendered = rendered.replace(placeholder, text_value)
    return rendered, rendered != instruction


def _build_train_text(*, instruction: str, modality: str, text: str, target_text: str) -> str:
    instr_text, embeds_text = _render_instruction_with_text(instruction, text)
    user_prefix = "문자:" if modality == "sms" else "통화 내용:"
    base = "" if embeds_text else f"{user_prefix}{text}"
    prompt_text = f"{instr_text}\n{base}".strip() if instr_text else base
    target_value = str(target_text).strip()
    return f"{prompt_text}\n{target_value}" if prompt_text else target_value


def _compose_target_text(label_value: Any, reason_value: Any) -> str:
    try:
        label_int = int(label_value)
    except Exception:
        label_int = 0
    reason = " ".join(str("" if reason_value is None else reason_value).strip().split())
    return f"{label_int} {reason}".strip() if reason else str(label_int)


def _build_eval_prompt(*, instruction: str, modality: str, text: str) -> str:
    instr_text, embeds_text = _render_instruction_with_text(instruction, text)
    user_prefix = "문자:" if modality == "sms" else "통화 내용:"
    base = "" if embeds_text else f"{user_prefix}{text}"
    return f"{instr_text}\n{base}".strip() if instr_text else base


def _quantile(arr: np.ndarray, q: float) -> float:
    if arr.size == 0:
        return 0.0
    return float(np.quantile(arr, q))


def _summarize_lengths(lengths: list[int]) -> dict[str, Any]:
    arr = np.asarray(lengths, dtype=np.int32)
    if arr.size == 0:
        return {
            "n": 0,
            "mean": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0,
        }
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "p50": _quantile(arr, 0.50),
        "p90": _quantile(arr, 0.90),
        "p95": _quantile(arr, 0.95),
        "p99": _quantile(arr, 0.99),
        "max": int(arr.max()),
    }


def _profile_train_scenario(
    *,
    tokenizer,
    scenario: dict[str, Any],
    batch_size: int,
    max_length: int,
) -> dict[str, Any]:
    csv_path = ROOT / scenario["train_csv"]
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    instruction = _read_instruction(scenario["prompt_instruction_path"])
    text_col = scenario["train_text_col"]
    texts = df[text_col].fillna("").astype(str).tolist()
    if "target_text" in df.columns:
        targets = df["target_text"].fillna("").astype(str).tolist()
    elif {"label", "reason_value"}.issubset(set(df.columns)):
        targets = [
            _compose_target_text(label_value=lbl, reason_value=reason)
            for lbl, reason in zip(df["label"], df["reason_value"])
        ]
    else:
        raise ValueError(
            f"{csv_path} must include either target_text or (label, reason_value) columns."
        )

    full_lengths: list[int] = []
    prompt_lengths: list[int] = []
    for start in range(0, len(df), max(1, batch_size)):
        batch_texts = texts[start : start + batch_size]
        batch_targets = targets[start : start + batch_size]
        train_texts = [
            _build_train_text(
                instruction=instruction,
                modality=scenario["modality"],
                text=text,
                target_text=target_text,
            )
            for text, target_text in zip(batch_texts, batch_targets)
        ]
        prompts = [
            _build_eval_prompt(
                instruction=instruction,
                modality=scenario["modality"],
                text=text,
            )
            for text in batch_texts
        ]
        enc = tokenizer(train_texts, add_special_tokens=True, truncation=False)
        prompt_enc = tokenizer(prompts, add_special_tokens=True, truncation=False)
        full_lengths.extend(len(ids) for ids in enc["input_ids"])
        prompt_lengths.extend(len(ids) for ids in prompt_enc["input_ids"])

    used_lengths = [min(int(max_length), int(v)) for v in full_lengths]
    full_stats = _summarize_lengths(full_lengths)
    used_stats = _summarize_lengths(used_lengths)
    prompt_stats = _summarize_lengths(prompt_lengths)
    truncated_frac = (
        float(sum(1 for v in full_lengths if int(v) > int(max_length)) / len(full_lengths))
        if full_lengths
        else 0.0
    )

    return {
        "csv_path": scenario["train_csv"],
        "rows": int(len(df)),
        "max_length": int(max_length),
        "truncated_frac": truncated_frac,
        "full": full_stats,
        "used": used_stats,
        "prompt": prompt_stats,
        "planning_train_length_p99": int(math.ceil(float(used_stats["p99"]))),
    }


def _profile_eval_target(
    *,
    tokenizer,
    scenario: dict[str, Any],
    eval_target: dict[str, Any],
    batch_size: int,
    decode_max_new_tokens: int,
) -> dict[str, Any]:
    csv_path = ROOT / eval_target["csv_path"]
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    instruction = _read_instruction(scenario["prompt_instruction_path"])
    text_col = eval_target["text_col"]
    texts = df[text_col].fillna("").astype(str).tolist()

    prompt_lengths: list[int] = []
    for start in range(0, len(df), max(1, batch_size)):
        batch_texts = texts[start : start + batch_size]
        prompts = [
            _build_eval_prompt(
                instruction=instruction,
                modality=scenario["modality"],
                text=text,
            )
            for text in batch_texts
        ]
        prompt_enc = tokenizer(prompts, add_special_tokens=True, truncation=False)
        prompt_lengths.extend(len(ids) for ids in prompt_enc["input_ids"])

    prompt_stats = _summarize_lengths(prompt_lengths)
    planning_eval_length = int(math.ceil(float(prompt_stats["p99"]))) + int(decode_max_new_tokens)
    return {
        "csv_path": eval_target["csv_path"],
        "split_name": eval_target["split_name"],
        "rows": int(len(df)),
        "decode_max_new_tokens": int(decode_max_new_tokens),
        "prompt": prompt_stats,
        "planning_eval_length_p99_plus_new_tokens": int(planning_eval_length),
    }


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    models = load_model_specs(args)
    scenarios = _select_scenarios(args.scenario_letters)
    PREPARED_DATA_ROOT.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "profile_args": {
            "scenario_letters": [str(scenario["letter"]) for scenario in scenarios],
            "sms_max_length": int(args.sms_max_length),
            "voice_max_length": int(args.voice_max_length),
            "decode_max_new_tokens": int(args.decode_max_new_tokens),
            "batch_size": int(args.batch_size),
            "target_format_version": TARGET_FORMAT_VERSION,
            "models": models,
        },
        "models": {},
    }
    csv_rows: list[dict[str, Any]] = []

    for model in models:
        alias = model["alias"]
        model_ref = model["model_ref"]
        print(f"[profile] loading tokenizer: {alias} -> {model_ref}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=True)
        model_result = {
            "model_ref": model_ref,
            "tokenizer_name_or_path": str(getattr(tokenizer, "name_or_path", model_ref)),
            "tokenizer_class": tokenizer.__class__.__name__,
            "vocab_size": int(getattr(tokenizer, "vocab_size", 0) or 0),
            "pad_token_id": getattr(tokenizer, "pad_token_id", None),
            "eos_token_id": getattr(tokenizer, "eos_token_id", None),
            "train": {},
            "eval": {},
        }

        for scenario in scenarios:
            max_length = int(args.sms_max_length if scenario["modality"] == "sms" else args.voice_max_length)
            train_profile = _profile_train_scenario(
                tokenizer=tokenizer,
                scenario=scenario,
                batch_size=args.batch_size,
                max_length=max_length,
            )
            model_result["train"][scenario["letter"]] = train_profile
            csv_rows.append(
                {
                    "model_alias": alias,
                    "model_ref": model_ref,
                    "profile_type": "train",
                    "scenario_letter": scenario["letter"],
                    "eval_id": "",
                    "modality": scenario["modality"],
                    "rows": train_profile["rows"],
                    "max_length": train_profile["max_length"],
                    "planning_length": train_profile["planning_train_length_p99"],
                    "truncated_frac": train_profile["truncated_frac"],
                    "prompt_mean": train_profile["prompt"]["mean"],
                    "prompt_p95": train_profile["prompt"]["p95"],
                    "prompt_p99": train_profile["prompt"]["p99"],
                    "full_mean": train_profile["full"]["mean"],
                    "full_p95": train_profile["full"]["p95"],
                    "full_p99": train_profile["full"]["p99"],
                    "used_mean": train_profile["used"]["mean"],
                    "used_p95": train_profile["used"]["p95"],
                    "used_p99": train_profile["used"]["p99"],
                    "used_max": train_profile["used"]["max"],
                }
            )
            print(
                f"[profile] {alias} {scenario['letter']} train "
                f"used_p99={train_profile['used']['p99']:.2f} "
                f"planning={train_profile['planning_train_length_p99']}",
                flush=True,
            )

            for eval_target in scenario["eval_targets"]:
                eval_profile = _profile_eval_target(
                    tokenizer=tokenizer,
                    scenario=scenario,
                    eval_target=eval_target,
                    batch_size=args.batch_size,
                    decode_max_new_tokens=args.decode_max_new_tokens,
                )
                model_result["eval"][eval_target["eval_id"]] = eval_profile
                csv_rows.append(
                    {
                        "model_alias": alias,
                        "model_ref": model_ref,
                        "profile_type": "eval",
                        "scenario_letter": scenario["letter"],
                        "eval_id": eval_target["eval_id"],
                        "modality": scenario["modality"],
                        "rows": eval_profile["rows"],
                        "max_length": "",
                        "planning_length": eval_profile["planning_eval_length_p99_plus_new_tokens"],
                        "truncated_frac": "",
                        "prompt_mean": eval_profile["prompt"]["mean"],
                        "prompt_p95": eval_profile["prompt"]["p95"],
                        "prompt_p99": eval_profile["prompt"]["p99"],
                        "full_mean": "",
                        "full_p95": "",
                        "full_p99": "",
                        "used_mean": "",
                        "used_p95": "",
                        "used_p99": "",
                        "used_max": "",
                    }
                )
                print(
                    f"[profile] {alias} {eval_target['eval_id']} eval "
                    f"prompt_p99={eval_profile['prompt']['p99']:.2f} "
                    f"planning={eval_profile['planning_eval_length_p99_plus_new_tokens']}",
                    flush=True,
                )

        result["models"][alias] = model_result

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    save_csv(output_csv, csv_rows)
    print(f"[saved] {output_json}", flush=True)
    print(f"[saved] {output_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
