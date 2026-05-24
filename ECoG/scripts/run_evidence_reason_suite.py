#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SMS_SCRIPT = ROOT / "scripts" / "run_sms_evidence_reason_suite.py"
VOICE_SCRIPT = ROOT / "scripts" / "run_voice_evidence_reason_suite.py"
PERFORMANCE_REPORT_SCRIPT = ROOT / "scripts" / "write_benchmark_performance.py"
SMS_LETTERS = {"A", "B", "C", "D"}
VOICE_LETTERS = {"E", "F", "G"}
DEFAULT_SLACK_SETTINGS = ROOT / "settings.json"
DEFAULT_MODEL_SPEC = "hyperclovax0p5b=naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B"
BENCHMARK_TYPES = {
    "label_decoder",
    "label_dlml",
    "label_encoder",
    "label_evidence",
    "label_evidence_explanation",
    "label_explanation",
}
DEFAULT_PER_ALPHABET_ORDER = (
    "D2",
    "D1",
    "C2",
    "F2",
    "B2",
    "A2",
    "C1",
    "B1",
    "F1",
    "E2",
    "G2",
    "A1",
    "E1",
    "G1",
)


@dataclass(frozen=True)
class BenchmarkPerformanceReportConfig:
    bench_name: str
    benchmark_layout: str
    benchmark_type: str | None
    run_enabled: bool
    eval_full_generation: bool
    skip: bool
    performance_out_root: str
    skip_performance: bool
    skip_performance_bertscore: bool
    decode_max_new_tokens: int
    model_aliases: tuple[str, ...]


def _parse_letters(raw_value: str) -> list[str]:
    raw = str(raw_value or "ABCDEFG").strip()
    if not raw or raw.upper() == "ALL":
        return list("ABCDEFG")
    parts = [part.strip().upper() for part in raw.replace(",", " ").split() if part.strip()]
    if len(parts) == 1 and parts[0].isalpha() and len(parts[0]) > 1:
        parts = list(parts[0])
    letters: list[str] = []
    seen: set[str] = set()
    for part in parts:
        if part in seen:
            continue
        if part not in SMS_LETTERS | VOICE_LETTERS:
            raise ValueError(f"Unknown scenario letter: {part}. Choose from A, B, C, D, E, F, G.")
        letters.append(part)
        seen.add(part)
    return letters


def _strip_arg(argv: list[str], option: str) -> list[str]:
    out: list[str] = []
    skip_next = False
    for idx, token in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if token == option:
            if idx + 1 < len(argv):
                skip_next = True
            continue
        if token.startswith(f"{option}="):
            continue
        out.append(token)
    return out


def _strip_optional_bool_arg(argv: list[str], option: str) -> list[str]:
    out: list[str] = []
    skip_next = False
    bool_values = {"1", "0", "true", "false", "yes", "no", "y", "n", "on", "off"}
    for idx, token in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if token == option:
            if idx + 1 < len(argv):
                nxt = str(argv[idx + 1]).strip().lower()
                if nxt and not nxt.startswith("-") and nxt in bool_values:
                    skip_next = True
            continue
        if token.startswith(f"{option}="):
            continue
        out.append(token)
    return out


def _replace_arg(argv: list[str], option: str, value: str | None) -> list[str]:
    out = _strip_arg(list(argv), option)
    if value is not None and str(value).strip():
        out.extend([option, str(value).strip()])
    return out


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"", "0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Expected boolean value, got: {value}")


def _parse_eval_order(raw_value: str | None) -> list[str]:
    raw = str(raw_value or "").strip()
    if not raw:
        return list(DEFAULT_PER_ALPHABET_ORDER)
    parts = [part.strip().upper() for part in raw.replace(",", " ").split() if part.strip()]
    order: list[str] = []
    seen: set[str] = set()
    for part in parts:
        if len(part) < 2 or part[0] not in SMS_LETTERS | VOICE_LETTERS:
            raise ValueError(f"Invalid eval target id in --per-alphabet-order: {part}")
        if part in seen:
            continue
        order.append(part)
        seen.add(part)
    return order


def _safe_name(value: str) -> str:
    keep: list[str] = []
    for ch in str(value):
        if ch.isalnum() or ch in "._-":
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep).strip("_.-")
    return out or "model"


def _parse_model_aliases(raw_models: list[str]) -> tuple[str, ...]:
    specs = list(raw_models or []) or [DEFAULT_MODEL_SPEC]
    aliases: list[str] = []
    seen: set[str] = set()
    for item in specs:
        if "=" in item:
            alias, _model_ref = item.split("=", 1)
            alias = alias.strip()
        else:
            alias = Path(str(item).strip()).name
        alias = _safe_name(alias)
        if alias and alias not in seen:
            aliases.append(alias)
            seen.add(alias)
    return tuple(aliases)


def _infer_benchmark_type(
    *,
    suite_name: str,
    explicit: str | None,
    stage2_target_format: str | None,
) -> str | None:
    explicit = str(explicit or "").strip()
    if explicit:
        return explicit
    suite_name = str(suite_name or "").strip()
    if suite_name in BENCHMARK_TYPES:
        return suite_name

    fmt = str(stage2_target_format or "").strip().lower()
    if not fmt:
        return None
    uses_span = "span" in fmt
    uses_explanation = "explanation" in fmt
    if uses_span and uses_explanation:
        return "label_evidence_explanation"
    if uses_span:
        return "label_evidence"
    if uses_explanation:
        return "label_explanation"
    return None


def parse_args(
    argv: list[str],
) -> tuple[
    list[str],
    list[str],
    list[str],
    bool,
    str | None,
    str,
    str | None,
    str | None,
    bool,
    list[str],
    BenchmarkPerformanceReportConfig,
]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--scenario-letters", default="ABCDEFG")
    parser.add_argument("--bench-name", default=None)
    parser.add_argument("--benchmark-layout", choices=("nested", "flat"), default="nested")
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--eval-full-generation", action="store_true")
    parser.add_argument("--decode-max-new-tokens", type=int, default=128)
    parser.add_argument("--stage2-target-format", default=None)
    parser.add_argument("--benchmark-type", choices=sorted(BENCHMARK_TYPES), default=None)
    parser.add_argument(
        "--suite-name",
        default="evidence_reason_suite",
        help=(

        ),
    )
    parser.add_argument(
        "--per-alphabet",
        nargs="?",
        const="true",
        default="false",
        help=(

        ),
    )
    parser.add_argument(
        "--per-alphabet-order",
        default=",".join(DEFAULT_PER_ALPHABET_ORDER),

    )
    parser.add_argument("--slack-alert", "--slack_alert", action="store_true")
    parser.add_argument("--slack-settings", default=None)
    parser.add_argument(
        "--sms-cuda-visible-devices",
        default=None,
        help="Top-level override for SMS child runs; rewrites --cuda-visible-devices only for SMS.",
    )
    parser.add_argument(
        "--voice-cuda-visible-devices",
        default=None,
        help="Top-level override for Voice child runs; rewrites --cuda-visible-devices only for Voice.",
    )
    parser.add_argument(
        "--skip-fullgen-quality-report",
        action="store_true",

    )
    parser.add_argument(
        "--fullgen-quality-report-out-dir",
        default="reports/evidence_explanation_metrics",

    )
    parser.add_argument(
        "--skip-benchmark-performance-report",
        action="store_true",

    )
    parser.add_argument(
        "--benchmark-performance-out-root",
        default="outputs/runs/benchmark_performance",

    )
    parser.add_argument(
        "--skip-benchmark-performance-bertscore",
        action="store_true",

    )
    args, _ = parser.parse_known_args(argv)
    letters = _parse_letters(args.scenario_letters)
    suite_name = str(args.suite_name or "evidence_reason_suite").strip() or "evidence_reason_suite"
    bench_name = str(args.bench_name or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{suite_name}").strip()
    shared_argv = _strip_arg(list(argv), "--scenario-letters")
    shared_argv = _strip_arg(shared_argv, "--suite-name")
    shared_argv = _strip_optional_bool_arg(shared_argv, "--per-alphabet")
    shared_argv = _strip_arg(shared_argv, "--per-alphabet-order")
    shared_argv = _strip_arg(shared_argv, "--slack-settings")
    shared_argv = _strip_arg(shared_argv, "--sms-cuda-visible-devices")
    shared_argv = _strip_arg(shared_argv, "--voice-cuda-visible-devices")
    shared_argv = _strip_arg(shared_argv, "--fullgen-quality-report-out-dir")
    shared_argv = _strip_arg(shared_argv, "--benchmark-type")
    shared_argv = _strip_arg(shared_argv, "--benchmark-performance-out-root")
    shared_argv = _replace_arg(shared_argv, "--bench-name", bench_name)
    shared_argv = [token for token in shared_argv if token not in {"--slack-alert", "--slack_alert"}]
    shared_argv = [token for token in shared_argv if token != "--skip-fullgen-quality-report"]
    shared_argv = [token for token in shared_argv if token != "--skip-benchmark-performance-report"]
    shared_argv = [token for token in shared_argv if token != "--skip-benchmark-performance-bertscore"]
    sms = [letter for letter in letters if letter in SMS_LETTERS]
    voice = [letter for letter in letters if letter in VOICE_LETTERS]
    report_config = BenchmarkPerformanceReportConfig(
        bench_name=bench_name,
        benchmark_layout=str(args.benchmark_layout or "nested"),
        benchmark_type=_infer_benchmark_type(
            suite_name=suite_name,
            explicit=args.benchmark_type,
            stage2_target_format=args.stage2_target_format,
        ),
        run_enabled=bool(args.run),
        eval_full_generation=bool(args.eval_full_generation),
        skip=bool(args.skip_fullgen_quality_report),
        performance_out_root=str(args.benchmark_performance_out_root or "outputs/runs/benchmark_performance"),
        skip_performance=bool(args.skip_benchmark_performance_report),
        skip_performance_bertscore=bool(args.skip_benchmark_performance_bertscore),
        decode_max_new_tokens=int(args.decode_max_new_tokens),
        model_aliases=_parse_model_aliases(list(args.model or [])),
    )
    return (
        shared_argv,
        sms,
        voice,
        bool(args.slack_alert),
        args.slack_settings,
        suite_name,
        args.sms_cuda_visible_devices,
        args.voice_cuda_visible_devices,
        _parse_bool(args.per_alphabet),
        _parse_eval_order(args.per_alphabet_order),
        report_config,
    )


def _run(
    script_path: Path,
    argv: list[str],
    letters: list[str],
    alert=None,
    suite_name: str | None = None,
    cuda_visible_devices: str | None = None,
    eval_target_id: str | None = None,
) -> None:
    if not letters:
        return
    child_argv = list(argv)
    if cuda_visible_devices is not None:
        child_argv = _replace_arg(child_argv, "--cuda-visible-devices", cuda_visible_devices)
    child_argv = _replace_arg(child_argv, "--suite-name", suite_name)
    if eval_target_id is not None:
        child_argv = _replace_arg(child_argv, "--eval-target-ids", eval_target_id)
    cmd = [
        sys.executable,
        os.fspath(script_path),
        *child_argv,
        "--scenario-letters",
        "".join(letters),
    ]
    if alert is not None:
        label = f"{script_path.name}::{eval_target_id or ''.join(letters)}"
        alert.post_step(label=label, command=cmd)
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def _run_per_alphabet(
    *,
    shared_argv: list[str],
    selected_letters: list[str],
    eval_order: list[str],
    alert=None,
    suite_name: str | None = None,
    sms_cuda_visible_devices: str | None = None,
    voice_cuda_visible_devices: str | None = None,
) -> None:
    selected = set(selected_letters)
    for eval_target_id in eval_order:
        letter = eval_target_id[0]
        if letter not in selected:
            continue
        if letter in SMS_LETTERS:
            _run(
                SMS_SCRIPT,
                shared_argv,
                [letter],
                alert=alert,
                suite_name=suite_name,
                cuda_visible_devices=sms_cuda_visible_devices,
                eval_target_id=eval_target_id,
            )
        elif letter in VOICE_LETTERS:
            _run(
                VOICE_SCRIPT,
                shared_argv,
                [letter],
                alert=alert,
                suite_name=suite_name,
                cuda_visible_devices=voice_cuda_visible_devices,
                eval_target_id=eval_target_id,
            )


def _run_selected(
    *,
    shared_argv: list[str],
    sms_letters: list[str],
    voice_letters: list[str],
    selected_letters: list[str],
    per_alphabet: bool,
    per_alphabet_order: list[str],
    alert=None,
    suite_name: str | None = None,
    sms_cuda_visible_devices: str | None = None,
    voice_cuda_visible_devices: str | None = None,
) -> None:
    if per_alphabet:
        _run_per_alphabet(
            shared_argv=shared_argv,
            selected_letters=selected_letters,
            eval_order=per_alphabet_order,
            alert=alert,
            suite_name=suite_name,
            sms_cuda_visible_devices=sms_cuda_visible_devices,
            voice_cuda_visible_devices=voice_cuda_visible_devices,
        )
        return

    _run(
        SMS_SCRIPT,
        shared_argv,
        sms_letters,
        alert=alert,
        suite_name=suite_name,
        cuda_visible_devices=sms_cuda_visible_devices,
    )
    _run(
        VOICE_SCRIPT,
        shared_argv,
        voice_letters,
        alert=alert,
        suite_name=suite_name,
        cuda_visible_devices=voice_cuda_visible_devices,
    )


def _run_benchmark_performance_reports(config: BenchmarkPerformanceReportConfig, suite_name: str) -> None:
    if config.skip_performance or not config.run_enabled:
        return
    if not config.benchmark_type:
        print(
            "[warn] benchmark performance report skipped: could not infer benchmark type. "
            "Pass --benchmark-type label_evidence_explanation, label_evidence, "
            "label_explanation, label_decoder, label_encoder, or label_dlml.",
            flush=True,
        )
        return

    run_name_contains = f"fullgen{config.decode_max_new_tokens}"
    for model_alias in config.model_aliases:
        cmd = [
            sys.executable,
            os.fspath(PERFORMANCE_REPORT_SCRIPT),
            "--benchmark-type",
            config.benchmark_type,
            "--suite-name",
            suite_name,
            "--bench-name",
            config.bench_name,
            "--benchmark-layout",
            config.benchmark_layout,
            "--model-alias",
            model_alias,
            "--run-name-contains",
            run_name_contains,
            "--decode-max-new-tokens",
            str(config.decode_max_new_tokens),
            "--out-root",
            config.performance_out_root,
            "--allow-missing",
        ]
        if config.skip or not config.eval_full_generation:
            cmd.append("--skip-quality")
        if config.skip_performance_bertscore:
            cmd.append("--skip-bertscore")
        print("\n=== benchmark performance report ===")
        print("[run]", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=str(ROOT), check=True)


def main(argv: list[str] | None = None) -> int:
    (
        shared_argv,
        sms_letters,
        voice_letters,
        slack_alert,
        slack_settings,
        suite_name,
        sms_cuda_visible_devices,
        voice_cuda_visible_devices,
        per_alphabet,
        per_alphabet_order,
        report_config,
    ) = parse_args(
        list(sys.argv[1:] if argv is None else argv)
    )
    selected_letters = sms_letters + voice_letters
    if slack_alert:
        if slack_settings is None and DEFAULT_SLACK_SETTINGS.exists():
            slack_settings = os.fspath(DEFAULT_SLACK_SETTINGS)
        from slack_alert import SlackRunAlert

        task = f"run_evidence_reason_suite scenarios={''.join(selected_letters)}"
        with SlackRunAlert(task=task, enabled=True, settings_path=slack_settings) as alert:
            _run_selected(
                shared_argv=shared_argv,
                sms_letters=sms_letters,
                voice_letters=voice_letters,
                selected_letters=selected_letters,
                per_alphabet=per_alphabet,
                per_alphabet_order=per_alphabet_order,
                alert=alert,
                suite_name=suite_name,
                sms_cuda_visible_devices=sms_cuda_visible_devices,
                voice_cuda_visible_devices=voice_cuda_visible_devices,
            )
            _run_benchmark_performance_reports(report_config, suite_name)
        return 0

    _run_selected(
        shared_argv=shared_argv,
        sms_letters=sms_letters,
        voice_letters=voice_letters,
        selected_letters=selected_letters,
        per_alphabet=per_alphabet,
        per_alphabet_order=per_alphabet_order,
        suite_name=suite_name,
        sms_cuda_visible_devices=sms_cuda_visible_devices,
        voice_cuda_visible_devices=voice_cuda_visible_devices,
    )
    _run_benchmark_performance_reports(report_config, suite_name)
    return 0


if __name__ == "__main__":
    os.chdir(ROOT)
    raise SystemExit(main())
