#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


BENIGN_PATTERNS = [
    r"정상\s*(?:문자|메시지|대화|상담|안내|거래|절차|서비스|마케팅|프로모션|광고|사례|흐름|성격|범주)",
    r"정상적인\s*(?:문자|메시지|대화|상담|안내|거래|절차|서비스|마케팅|프로모션|광고|사례|흐름|성격|범주)",
    r"(?:정상|안전|비사기)\s*(?:으로|라고)?\s*(?:판단|분류|해석|보임|보입니다|볼 수)",
    r"(?:스미싱|보이스피싱|피싱|사기)\s*(?:으로)?\s*(?:분류|판단)\s*되지\s*않",
    r"(?:스미싱|보이스피싱|피싱|사기)\s*(?:문자|대화|메시지)?\s*(?:가|이|은|는)?\s*아니",
    r"(?:스미싱|보이스피싱|피싱|사기)\s*(?:요소|징후|정황|신호|특징|위험|의도|가능성)\s*(?:가|이|은|는)?\s*(?:없|부족|낮|보이지|드러나지|나타나지)",
    r"(?:의심스러운|악성)\s*(?:링크|URL|앱|어플|설치|개인정보|개인 정보|금융정보|금융 정보|송금|이체|입금|결제|인증)\s*(?:요구|유도)?\s*(?:가|이|은|는)?\s*(?:없|보이지|나타나지)",
    r"(?:긴급성|압박|사칭|금전\s*요구|개인정보\s*요구|개인\s*정보\s*요구)\s*(?:이|가|은|는)?\s*(?:없|보이지|나타나지)",
    r"(?:공식|합법|일반적인|통상적인|신뢰할 수 있는)\s*(?:안내|상담|절차|서비스|거래|마케팅|광고|기관|브랜드)",
]

MALICIOUS_PATTERNS = [
    r"(?:스미싱|보이스피싱|피싱|사기)\s*(?:문자|메시지|대화|상담|사례)?\s*(?:로|으로|라고)?\s*(?:판단|분류|해석|보임|보입니다|의심|볼 수)",
    r"(?:스미싱|보이스피싱|피싱|사기)\s*(?:의도|위험|위험성|가능성|정황|요소|징후|신호|특징)\s*(?:가|이|은|는)?\s*(?:있|높|드러나|나타나|명확)",
    r"(?:전형적인|대표적인)\s*(?:스미싱|보이스피싱|피싱|사기)",
    r"(?:사기범|범죄자|공격자)\s*(?:이|가|은|는)?",
    r"(?:사칭|불안감|긴급성|압박|금전\s*요구|개인정보\s*요구|개인\s*정보\s*요구|금융정보\s*요구|금융\s*정보\s*요구)\s*(?:이|가|은|는)?\s*(?:드러나|나타나|보이|포함|존재|있)",
    r"(?:악성|의심스러운)\s*(?:링크|URL|앱|어플|설치|접속|인증|로그인)",
    r"(?:송금|이체|입금|결제|수수료|보증금|선입금|가상계좌|계좌)\s*(?:요구|유도|압박)",
    r"(?:개인정보|개인\s*정보|금융정보|금융\s*정보|인증번호|비밀번호|OTP|신분증|계좌번호)\s*(?:요구|입력|제공|확인|탈취|유도)",
    r"(?:주의|경계|위험)\s*(?:가|이)?\s*(?:필요|요구)",
]

NEGATION_PATTERNS = [
    r"없",
    r"아니",
    r"않",
    r"부족",
    r"낮",
    r"어렵",
    r"분류\s*되지",
    r"판단\s*하기\s*어렵",
    r"보이지",
    r"나타나지",
    r"드러나지",
    r"전혀",
]

MALICIOUS_TO_BENIGN_PATTERNS = [
    r"(?:스미싱|보이스피싱|피싱|사기).*?(?:없|아니|않|부족|낮|어렵|분류\s*되지|보이지|나타나지|드러나지)",
    r"(?:의심스러운|악성).*?(?:링크|URL|앱|어플|설치|개인정보|개인 정보|금융정보|금융 정보).*?(?:없|보이지|나타나지)",
    r"(?:긴급성|압박|사칭|금전\s*요구|개인정보\s*요구|개인\s*정보\s*요구).*?(?:없|보이지|나타나지)",
]

BENIGN_TO_MALICIOUS_PATTERNS = [
    r"정상.*?(?:아니|않|어렵|보기\s*어렵)",
    r"공식.*?(?:아니|않|어렵|보기\s*어렵)",
]


@dataclass
class RowConsistency:
    benign_cues: list[str]
    malicious_cues: list[str]
    inferred_direction: str
    strict_inconsistency: bool
    broad_opposite_cue: bool
    mixed_benign_and_malicious: bool
    no_direction_cue: bool
    pred_label_wrong: bool | None


def compile_patterns(patterns: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(pattern, flags=re.IGNORECASE | re.DOTALL) for pattern in patterns]


BENIGN_RE = compile_patterns(BENIGN_PATTERNS)
MALICIOUS_RE = compile_patterns(MALICIOUS_PATTERNS)
NEGATION_RE = compile_patterns(NEGATION_PATTERNS)
MALICIOUS_TO_BENIGN_RE = compile_patterns(MALICIOUS_TO_BENIGN_PATTERNS)
BENIGN_TO_MALICIOUS_RE = compile_patterns(BENIGN_TO_MALICIOUS_PATTERNS)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"[ \t]+", " ", text).strip()


def extract_explanation(generated: Any, scope: str) -> str:
    text = normalize_text(generated)
    if scope == "gen":
        return text
    marker = "설명:"
    idx = text.find(marker)
    if idx >= 0:
        start = idx + len(marker)
        answer_idx = text.find("정답:", start)
        end = answer_idx if answer_idx >= 0 else len(text)
        return text[start:end].strip()
    return text


def split_units(text: str) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    rough = re.split(r"[.!?。！？]\s+|\n+", text)
    units: list[str] = []
    for chunk in rough:
        chunk = chunk.strip()
        if not chunk:
            continue
        units.append(chunk.strip(" \t\n.。"))
    return units or [text]


def _matches(patterns: list[re.Pattern[str]], text: str) -> list[str]:
    out: list[str] = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            cue = re.sub(r"\s+", " ", match.group(0)).strip()
            if cue and cue not in out:
                out.append(cue)
    return out


def _has_negation(text: str) -> bool:
    return any(pattern.search(text) for pattern in NEGATION_RE)


def infer_explanation_direction(explanation: str) -> tuple[list[str], list[str], str]:
    benign_cues: list[str] = []
    malicious_cues: list[str] = []

    for unit in split_units(explanation):
        explicit_benign = _matches(BENIGN_RE, unit)
        explicit_malicious = _matches(MALICIOUS_RE, unit)
        negated_malicious = _matches(MALICIOUS_TO_BENIGN_RE, unit)
        negated_benign = _matches(BENIGN_TO_MALICIOUS_RE, unit)

        for cue in explicit_benign + negated_malicious:
            if cue not in benign_cues:
                benign_cues.append(cue)

        for cue in explicit_malicious:
            if cue in negated_malicious or _has_negation(unit):
                continue
            if cue not in malicious_cues:
                malicious_cues.append(cue)

        for cue in negated_benign:
            if cue not in malicious_cues:
                malicious_cues.append(cue)

    if benign_cues and malicious_cues:
        direction = "mixed"
    elif benign_cues:
        direction = "benign_only"
    elif malicious_cues:
        direction = "malicious_only"
    else:
        direction = "none"
    return benign_cues, malicious_cues, direction


def parse_binary(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value).strip()
    if text in {"0", "0.0"}:
        return 0
    if text in {"1", "1.0"}:
        return 1
    match = re.search(r"[01]", text)
    if match:
        return int(match.group(0))
    return None


def judge_row(pred: int | None, label: int | None, explanation: str) -> RowConsistency:
    benign_cues, malicious_cues, direction = infer_explanation_direction(explanation)
    strict = bool((pred == 1 and direction == "benign_only") or (pred == 0 and direction == "malicious_only"))
    broad = bool((pred == 1 and benign_cues) or (pred == 0 and malicious_cues))
    pred_wrong = None if pred is None or label is None else bool(pred != label)
    return RowConsistency(
        benign_cues=benign_cues,
        malicious_cues=malicious_cues,
        inferred_direction=direction,
        strict_inconsistency=strict,
        broad_opposite_cue=broad,
        mixed_benign_and_malicious=direction == "mixed",
        no_direction_cue=direction == "none",
        pred_label_wrong=pred_wrong,
    )


def rate(num: int | float, den: int | float) -> float:
    return 0.0 if not den else float(num) / float(den)


def summarize_details(details: pd.DataFrame) -> dict[str, Any]:
    n = int(len(details))
    strict = int(details["strict_inconsistency"].sum()) if n else 0
    broad = int(details["broad_opposite_cue"].sum()) if n else 0
    mixed = int(details["mixed_benign_and_malicious"].sum()) if n else 0
    no_cue = int(details["no_direction_cue"].sum()) if n else 0
    pred1_benign = int(((details["pred"] == 1) & (details["inferred_direction"] == "benign_only")).sum()) if n else 0
    pred0_malicious = int(((details["pred"] == 0) & (details["inferred_direction"] == "malicious_only")).sum()) if n else 0

    summary: dict[str, Any] = {
        "n": n,
        "strict_inconsistency": strict,
        "strict_inconsistency_rate": rate(strict, n),
        "strict_consistency_rate": 1.0 - rate(strict, n),
        "broad_opposite_cue": broad,
        "broad_opposite_cue_rate": rate(broad, n),
        "mixed_benign_and_malicious": mixed,
        "mixed_benign_and_malicious_rate": rate(mixed, n),
        "no_direction_cue": no_cue,
        "no_direction_cue_rate": rate(no_cue, n),
        "pred1_benign_only": pred1_benign,
        "pred1_benign_only_rate": rate(pred1_benign, n),
        "pred0_malicious_only": pred0_malicious,
        "pred0_malicious_only_rate": rate(pred0_malicious, n),
    }

    if "label" in details.columns and details["label"].notna().any():
        valid = details["pred"].notna() & details["label"].notna()
        valid_n = int(valid.sum())
        correct = valid & (details["pred"] == details["label"])
        summary["accuracy_from_pred_label"] = rate(int(correct.sum()), valid_n)
        strict_mask = details["strict_inconsistency"].astype(bool)
        summary["strict_pred_correct"] = int((strict_mask & correct).sum())
        summary["strict_pred_wrong"] = int((strict_mask & valid & ~correct).sum())

    if "category" in details.columns:
        by_category: dict[str, Any] = {}
        for category, group in details.groupby("category", dropna=False):
            if pd.isna(category):
                key = "__missing__"
            else:
                key = str(category)
            by_category[key] = summarize_details(group.drop(columns=["category"], errors="ignore"))
        summary["by_category"] = by_category

    return summary


def find_results_csv(run_dir: Path) -> Path:
    if run_dir.is_file():
        return run_dir
    candidates = sorted(run_dir.glob("results*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No results*.csv found under: {run_dir}")
    return candidates[0]


def collect_inputs(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    for raw in args.results_csv:
        paths.append(Path(raw))
    for raw in args.run_dir:
        paths.append(find_results_csv(Path(raw)))
    for pattern in args.glob:
        for matched in glob.glob(pattern, recursive=True):
            path = Path(matched)
            if path.is_dir():
                try:
                    paths.append(find_results_csv(path))
                except FileNotFoundError:
                    continue
            elif path.is_file():
                paths.append(path)
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    if not unique:
        raise ValueError("No input result CSVs were found. Use --results-csv, --run-dir, or --glob.")
    return unique


def evaluate_csv(path: Path, args: argparse.Namespace) -> tuple[dict[str, Any], pd.DataFrame]:
    df = pd.read_csv(path)
    if args.text_col not in df.columns:
        raise KeyError(f"{path}: missing text column: {args.text_col}")
    if args.pred_col not in df.columns:
        raise KeyError(f"{path}: missing pred column: {args.pred_col}")

    records: list[dict[str, Any]] = []
    has_label = args.label_col in df.columns
    has_category = args.category_col in df.columns

    for row_idx, row in df.iterrows():
        pred = parse_binary(row.get(args.pred_col))
        label = parse_binary(row.get(args.label_col)) if has_label else None
        explanation = extract_explanation(row.get(args.text_col), args.text_scope)
        judged = judge_row(pred=pred, label=label, explanation=explanation)
        record: dict[str, Any] = {
            "row_idx": int(row_idx),
            "source_csv": str(path),
            "run_dir": str(path.parent),
            "pred": pred,
            "explanation_text": explanation,
            **asdict(judged),
        }
        record["benign_cues"] = " | ".join(judged.benign_cues)
        record["malicious_cues"] = " | ".join(judged.malicious_cues)
        if has_label:
            record["label"] = label
        if has_category:
            record["category"] = row.get(args.category_col)
        if args.input_col in df.columns:
            record["input"] = row.get(args.input_col)
        records.append(record)

    details = pd.DataFrame.from_records(records)
    summary = summarize_details(details)
    summary["source_csv"] = str(path)
    summary["run_dir"] = str(path.parent)
    summary["metric_version"] = "label_explanation_consistency_v1"
    summary["text_scope"] = args.text_scope
    summary["pred_col"] = args.pred_col
    summary["text_col"] = args.text_col
    if has_label:
        summary["label_col"] = args.label_col
    if has_category:
        summary["category_col"] = args.category_col
    return summary, details


def default_output_base(inputs: list[Path]) -> Path:
    if len(inputs) == 1:
        return inputs[0].parent / "consistency_metrics"
    return Path("reports") / "consistency_metrics" / "consistency_metrics"


def write_outputs(
    summaries: list[dict[str, Any]],
    details_frames: list[pd.DataFrame],
    args: argparse.Namespace,
    inputs: list[Path],
) -> None:
    base = Path(args.out_prefix) if args.out_prefix else default_output_base(inputs)
    base.parent.mkdir(parents=True, exist_ok=True)

    details = pd.concat(details_frames, ignore_index=True) if details_frames else pd.DataFrame()
    summary_rows = []
    for summary in summaries:
        flat = {k: v for k, v in summary.items() if not isinstance(v, dict)}
        summary_rows.append(flat)
    summary_df = pd.DataFrame(summary_rows)

    json_path = Path(args.out_json) if args.out_json else base.with_suffix(".json")
    details_path = Path(args.details_csv) if args.details_csv else base.with_name(base.name + "_details.csv")
    summary_path = Path(args.summary_csv) if args.summary_csv else base.with_name(base.name + "_summary.csv")

    payload: dict[str, Any]
    if len(summaries) == 1:
        payload = summaries[0]
    else:
        total_details = pd.concat(details_frames, ignore_index=True)
        payload = {
            "metric_version": "label_explanation_consistency_v1",
            "n_runs": len(summaries),
            "aggregate": summarize_details(total_details),
            "runs": summaries,
        }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    details_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    details.to_csv(details_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"[write] {json_path}")
    print(f"[write] {summary_path}")
    print(f"[write] {details_path}")


def update_metrics_json(summary: dict[str, Any], run_dir: Path, key: str) -> None:
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    else:
        metrics = {}
    metrics[key] = summary
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[update] {metrics_path} <- {key}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute rule-based label-explanation consistency metrics from eval_decode result CSVs. "
            "Strict inconsistency means pred=1 with benign-only explanation cues, or pred=0 with "
            "malicious-only explanation cues."
        )
    )
    parser.add_argument("--results-csv", action="append", default=[])
    parser.add_argument("--run-dir", action="append", default=[])
    parser.add_argument("--glob", action="append", default=[])
    parser.add_argument("--text-col", default="gen")
    parser.add_argument("--pred-col", default="pred")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--category-col", default="category")
    parser.add_argument("--input-col", default="input")
    parser.add_argument(
        "--text-scope",
        choices=("explanation", "gen"),
        default="explanation",
  )
    parser.add_argument("--out-prefix", default=None)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--summary-csv", default=None)
    parser.add_argument("--details-csv", default=None)
    parser.add_argument(
        "--update-metrics",
        action="store_true",

    )
    parser.add_argument("--metrics-key", default="label_explanation_consistency")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inputs = collect_inputs(args)
    summaries: list[dict[str, Any]] = []
    detail_frames: list[pd.DataFrame] = []

    for csv_path in inputs:
        summary, details = evaluate_csv(csv_path, args)
        summaries.append(summary)
        detail_frames.append(details)
        if args.update_metrics:
            update_metrics_json(summary, csv_path.parent, args.metrics_key)

    write_outputs(summaries, detail_frames, args, inputs)

    for summary in summaries:
        print(
            "[summary] "
            f"{summary['run_dir']} "
            f"n={summary['n']} "
            f"strict={summary['strict_inconsistency']} "
            f"strict_rate={summary['strict_inconsistency_rate']:.4f} "
            f"broad={summary['broad_opposite_cue']} "
            f"broad_rate={summary['broad_opposite_cue_rate']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
