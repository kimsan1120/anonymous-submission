#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import glob
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None


SMS_LETTERS = set("ABCD")
VOICE_LETTERS = set("EFG")
EVAL_IDS = {f"{letter}{idx}" for letter in "ABCDEFG" for idx in (1, 2)}


@dataclass
class EvidenceItem:
    text: str
    span: tuple[int, int] | None = None


def normalize_space(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\s+", " ", text).strip()


def parse_jsonish_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text:
        return []
    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(text)
        except Exception:
            continue
        if isinstance(parsed, list):
            return parsed
    return []


def parse_span(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        start = int(value[0])
        end = int(value[1])
    except Exception:
        return None
    if end < start:
        start, end = end, start
    return (max(0, start), max(0, end))


def safe_substr(text: str, span: tuple[int, int] | None) -> str:
    if span is None:
        return ""
    start, end = span
    return str(text)[max(0, start) : max(0, end)]


def parse_gold_evidence(row: pd.Series, text_col: str, spans_col: str, evidences_col: str) -> list[EvidenceItem]:
    text = "" if text_col not in row else str(row.get(text_col, ""))
    raw_spans = parse_jsonish_list(row.get(spans_col)) if spans_col in row.index else []
    raw_evidences = parse_jsonish_list(row.get(evidences_col)) if evidences_col in row.index else []
    spans = [span for span in (parse_span(item) for item in raw_spans) if span is not None]
    evidences = [normalize_space(item) for item in raw_evidences if normalize_space(item)]

    n = max(len(spans), len(evidences))
    items: list[EvidenceItem] = []
    for idx in range(n):
        span = spans[idx] if idx < len(spans) else None
        ev_text = evidences[idx] if idx < len(evidences) else safe_substr(text, span)
        ev_text = normalize_space(ev_text)
        if span is not None or ev_text:
            items.append(EvidenceItem(text=ev_text, span=span))
    return items


def strip_leading_label(text: str) -> str:
    return re.sub(r"^\s*[01]\s*(?:\n|$)", "", text, count=1)


def extract_generated_sections(generated: Any) -> tuple[str, str]:
    text = strip_leading_label(str("" if generated is None else generated).strip())
    evidence_marker = "근거 스팬:"
    explanation_marker = "설명:"
    answer_marker = "정답:"

    evidence = ""
    explanation = ""
    evidence_idx = text.find(evidence_marker)
    explanation_idx = text.find(explanation_marker)

    if evidence_idx >= 0:
        start = evidence_idx + len(evidence_marker)
        end = explanation_idx if explanation_idx >= 0 and explanation_idx > start else len(text)
        evidence = text[start:end].strip()
    if explanation_idx >= 0:
        explanation_start = explanation_idx + len(explanation_marker)
        answer_idx = text.find(answer_marker, explanation_start)
        explanation_end = answer_idx if answer_idx >= 0 else len(text)
        explanation = text[explanation_start:explanation_end].strip()
    elif evidence_idx < 0:
        explanation = text.strip()
    return evidence, explanation


def parse_predicted_evidence(evidence_section: str) -> list[EvidenceItem]:
    section = str(evidence_section or "").strip()
    if not section:
        return []

    pattern = re.compile(
        r"(?:^|\n)\s*(?:\d+\s*[\).]|[-*])\s*(.*?)(?=(?:\n\s*(?:\d+\s*[\).]|[-*])\s*)|\Z)",
        flags=re.DOTALL,
    )
    raw_items = [m.group(1).strip() for m in pattern.finditer(section)]
    if not raw_items:
        raw_items = [line.strip() for line in section.splitlines() if line.strip()]

    items: list[EvidenceItem] = []
    for raw in raw_items:
        raw = normalize_space(raw)
        if not raw:
            continue
        if raw in {"없음", "없습니다", "N/A", "NA", "none", "None"}:
            continue
        span = None
        match = re.match(r"^\[\s*(\d+)\s*,\s*(\d+)\s*\]\s*(.*)$", raw)
        if match:
            span = (int(match.group(1)), int(match.group(2)))
            raw = normalize_space(match.group(3))
        items.append(EvidenceItem(text=raw, span=span))
    return items


def span_tokens(text: str, mode: str) -> list[str]:
    text = normalize_space(text)
    if not text:
        return []
    if mode == "word":
        return re.findall(r"[가-힣A-Za-z0-9#_]+|[^\s]", text)
    if mode == "char":
        return [ch for ch in text if not ch.isspace()]
    raise ValueError(f"Unknown span tokenizer: {mode}")


def rouge_tokens(text: str, mode: str) -> list[str]:
    return span_tokens(text, mode)


def multiset_overlap(a: list[str], b: list[str]) -> int:
    counts: dict[str, int] = {}
    for item in a:
        counts[item] = counts.get(item, 0) + 1
    overlap = 0
    for item in b:
        cur = counts.get(item, 0)
        if cur <= 0:
            continue
        overlap += 1
        counts[item] = cur - 1
    return overlap


def token_f1(a_text: str, b_text: str, mode: str) -> float:
    a = span_tokens(a_text, mode)
    b = span_tokens(b_text, mode)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    overlap = multiset_overlap(a, b)
    if overlap <= 0:
        return 0.0
    precision = overlap / len(b)
    recall = overlap / len(a)
    return 2.0 * precision * recall / (precision + recall)


def interval_f1(gold: tuple[int, int] | None, pred: tuple[int, int] | None) -> float | None:
    if gold is None or pred is None:
        return None
    gs, ge = gold
    ps, pe = pred
    glen = max(0, ge - gs)
    plen = max(0, pe - ps)
    if glen == 0 and plen == 0:
        return 1.0
    if glen == 0 or plen == 0:
        return 0.0
    overlap = max(0, min(ge, pe) - max(gs, ps))
    if overlap <= 0:
        return 0.0
    precision = overlap / plen
    recall = overlap / glen
    return 2.0 * precision * recall / (precision + recall)


def evidence_pair_score(gold: EvidenceItem, pred: EvidenceItem, mode: str, prefer_offsets: bool) -> float:
    if prefer_offsets:
        score = interval_f1(gold.span, pred.span)
        if score is not None:
            return score
    return token_f1(gold.text, pred.text, mode)


def match_evidence(
    gold_items: list[EvidenceItem],
    pred_items: list[EvidenceItem],
    threshold: float,
    tokenizer: str,
    prefer_offsets: bool,
) -> tuple[int, list[float], list[tuple[int, int, float]]]:
    if not gold_items or not pred_items:
        return 0, [], []
    scores = [
        [evidence_pair_score(gold, pred, tokenizer, prefer_offsets) for pred in pred_items]
        for gold in gold_items
    ]
    matched: list[tuple[int, int, float]] = []
    if linear_sum_assignment is not None:
        cost = [[-score for score in row] for row in scores]
        row_ids, col_ids = linear_sum_assignment(cost)
        for gi, pi in zip(row_ids, col_ids):
            score = float(scores[int(gi)][int(pi)])
            if score >= threshold:
                matched.append((int(gi), int(pi), score))
    else:
        pairs = sorted(
            ((gi, pi, scores[gi][pi]) for gi in range(len(gold_items)) for pi in range(len(pred_items))),
            key=lambda item: item[2],
            reverse=True,
        )
        used_g: set[int] = set()
        used_p: set[int] = set()
        for gi, pi, score in pairs:
            if score < threshold or gi in used_g or pi in used_p:
                continue
            used_g.add(gi)
            used_p.add(pi)
            matched.append((gi, pi, float(score)))
    return len(matched), [score for _, _, score in matched], matched


def normalized_set(items: list[EvidenceItem]) -> set[str]:
    return {normalize_space(item.text) for item in items if normalize_space(item.text)}


def normalized_span_set(items: list[EvidenceItem]) -> set[tuple[int, int]]:
    return {item.span for item in items if item.span is not None}


def lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for item_a in a:
        cur = [0]
        for j, item_b in enumerate(b, start=1):
            if item_a == item_b:
                cur.append(prev[j - 1] + 1)
            else:
                cur.append(max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_l(candidate: str, reference: str, tokenizer: str) -> tuple[float, float, float]:
    cand = rouge_tokens(candidate, tokenizer)
    ref = rouge_tokens(reference, tokenizer)
    if not cand and not ref:
        return 1.0, 1.0, 1.0
    if not cand or not ref:
        return 0.0, 0.0, 0.0
    lcs = lcs_len(cand, ref)
    precision = lcs / len(cand)
    recall = lcs / len(ref)
    f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1


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
    return int(match.group(0)) if match else None


def infer_eval_id_from_path(path: Path) -> str | None:
    for part in path.parts:
        if part in EVAL_IDS:
            return part
    match = re.search(r"\b([A-G][12])\b", str(path))
    return match.group(1) if match else None


def gold_split_for_eval_id(eval_id: str) -> str:
    if eval_id.endswith("1"):
        return "test"
    if eval_id.endswith("2"):
        return "challenge"
    raise ValueError(f"Cannot infer split for eval id: {eval_id}")


def modality_for_eval_id(eval_id: str) -> str:
    letter = eval_id[0]
    if letter in SMS_LETTERS:
        return "sms"
    if letter in VOICE_LETTERS:
        return "voice"
    raise ValueError(f"Cannot infer modality for eval id: {eval_id}")


def find_results_csv(run_dir: Path) -> Path:
    if run_dir.is_file():
        return run_dir
    candidates = sorted(run_dir.glob("results*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No results*.csv found under: {run_dir}")
    return candidates[0]


def load_config_for_results(results_csv: Path) -> dict[str, Any]:
    config_path = results_csv.parent / "config.yaml"
    if not config_path.exists():
        config_path = results_csv.parent / "config_used.yaml"
    if not config_path.exists():
        return {}
    return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


def infer_gold_csv(results_csv: Path, args: argparse.Namespace, cfg: dict[str, Any]) -> Path:
    if args.gold_csv:
        return Path(args.gold_csv)

    cfg_eval_csv = Path(str(((cfg.get("data", {}) or {}).get("eval_csv", ""))))
    if cfg_eval_csv.exists():
        try:
            cols = set(pd.read_csv(cfg_eval_csv, nrows=0).columns)
        except Exception:
            cols = set()
        if {args.gold_reason_col, args.gold_spans_col, args.gold_evidences_col}.issubset(cols):
            return cfg_eval_csv

    eval_id = infer_eval_id_from_path(results_csv)
    if eval_id is None:
        raise ValueError(
            f"Cannot infer eval id from path: {results_csv}. Pass --gold-csv explicitly."
        )
    modality = modality_for_eval_id(eval_id)
    split = gold_split_for_eval_id(eval_id)
    letter = eval_id[0]
    candidate = Path("data") / modality / "evidence" / args.evidence_variant / f"{letter}_{split}.csv"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Gold CSV not found: {candidate}. Pass --gold-csv explicitly.")


def align_gold(results_df: pd.DataFrame, gold_df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if len(results_df) == len(gold_df):
        result_texts = results_df[args.result_input_col].map(normalize_space) if args.result_input_col in results_df else None
        gold_texts = gold_df[args.gold_text_col].map(normalize_space) if args.gold_text_col in gold_df else None
        if result_texts is None or gold_texts is None or result_texts.equals(gold_texts):
            return gold_df.reset_index(drop=True)

    if args.result_input_col not in results_df.columns or args.gold_text_col not in gold_df.columns:
        raise ValueError("Cannot align by position and missing text columns for merge alignment.")

    result_keys = pd.DataFrame({"text_key": results_df[args.result_input_col].map(normalize_space)})
    gold_keys = pd.DataFrame({"text_key": gold_df[args.gold_text_col].map(normalize_space)})
    for result_col, gold_col in ((args.result_label_col, args.gold_label_col), (args.category_col, args.category_col)):
        if result_col in results_df.columns and gold_col in gold_df.columns:
            result_keys[result_col] = results_df[result_col].astype(str)
            gold_keys[result_col] = gold_df[gold_col].astype(str)
    key_cols = list(result_keys.columns)
    result_keys["_occ"] = result_keys.groupby(key_cols).cumcount()
    gold_keys["_occ"] = gold_keys.groupby(key_cols).cumcount()

    gold_indexed = gold_df.reset_index(drop=True).copy()
    gold_indexed["_gold_row_pos"] = gold_indexed.index
    merged = result_keys.reset_index(names="_result_row_pos").merge(
        pd.concat([gold_keys, gold_indexed[["_gold_row_pos"]]], axis=1),
        on=[*key_cols, "_occ"],
        how="left",
    )
    if merged["_gold_row_pos"].isna().any():
        missing = int(merged["_gold_row_pos"].isna().sum())
        raise ValueError(f"Gold alignment failed for {missing} rows.")
    return gold_df.iloc[merged["_gold_row_pos"].astype(int).tolist()].reset_index(drop=True)


def mean(values: list[float]) -> float | None:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return None if not vals else float(sum(vals) / len(vals))


def compute_bertscore(
    candidates: list[str],
    references: list[str],
    model_name: str,
    batch_size: int,
    max_length: int,
    device: str,
) -> tuple[list[float], list[float], list[float]]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    model.eval()

    special_ids = set(int(x) for x in tokenizer.all_special_ids)
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []

    def encode(batch_texts: list[str]):
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            hidden = model(**enc).last_hidden_state
        hidden = torch.nn.functional.normalize(hidden.float(), p=2, dim=-1)
        mask = enc["attention_mask"].bool()
        for special_id in special_ids:
            mask &= enc["input_ids"] != int(special_id)
        return hidden, mask

    for start in range(0, len(candidates), batch_size):
        cand_batch = candidates[start : start + batch_size]
        ref_batch = references[start : start + batch_size]
        cand_hidden, cand_mask = encode(cand_batch)
        ref_hidden, ref_mask = encode(ref_batch)

        for idx in range(len(cand_batch)):
            c = cand_hidden[idx][cand_mask[idx]]
            r = ref_hidden[idx][ref_mask[idx]]
            if c.numel() == 0 and r.numel() == 0:
                p = rec = f1 = 1.0
            elif c.numel() == 0 or r.numel() == 0:
                p = rec = f1 = 0.0
            else:
                sim = c @ r.T
                p = float(sim.max(dim=1).values.mean().item())
                rec = float(sim.max(dim=0).values.mean().item())
                f1 = 0.0 if p + rec == 0 else 2.0 * p * rec / (p + rec)
            precisions.append(p)
            recalls.append(rec)
            f1s.append(f1)
    return precisions, recalls, f1s


def evaluate_one(results_csv: Path, args: argparse.Namespace) -> tuple[dict[str, Any], pd.DataFrame]:
    cfg = load_config_for_results(results_csv)
    gold_csv = infer_gold_csv(results_csv, args, cfg)
    results_df = pd.read_csv(results_csv)
    gold_df_raw = pd.read_csv(gold_csv)
    gold_df = align_gold(results_df, gold_df_raw, args)

    if args.generated_col not in results_df.columns:
        raise KeyError(f"{results_csv}: missing generated column {args.generated_col}")
    for col in (args.gold_reason_col, args.gold_spans_col, args.gold_evidences_col):
        if col not in gold_df.columns:
            raise KeyError(f"{gold_csv}: missing gold column {col}")

    rows: list[dict[str, Any]] = []
    total_gold = 0
    total_pred = 0
    total_matched = 0
    matched_scores_all: list[float] = []

    for idx, (res_row, gold_row) in enumerate(zip(results_df.to_dict("records"), gold_df.to_dict("records"))):
        generated = res_row.get(args.generated_col, "")
        evidence_section, explanation = extract_generated_sections(generated)
        pred_evidence = parse_predicted_evidence(evidence_section)
        gold_evidence = parse_gold_evidence(
            pd.Series(gold_row),
            text_col=args.gold_text_col,
            spans_col=args.gold_spans_col,
            evidences_col=args.gold_evidences_col,
        )
        matched, matched_scores, matched_pairs = match_evidence(
            gold_evidence,
            pred_evidence,
            threshold=float(args.span_match_threshold),
            tokenizer=args.span_tokenizer,
            prefer_offsets=bool(args.prefer_offsets),
        )
        total_gold += len(gold_evidence)
        total_pred += len(pred_evidence)
        total_matched += matched
        matched_scores_all.extend(matched_scores)

        if len(gold_evidence) == 0 and len(pred_evidence) == 0:
            sample_precision = sample_recall = sample_f1 = 1.0
        else:
            sample_precision = 0.0 if len(pred_evidence) == 0 else matched / len(pred_evidence)
            sample_recall = 0.0 if len(gold_evidence) == 0 else matched / len(gold_evidence)
            sample_f1 = (
                0.0
                if sample_precision + sample_recall == 0
                else 2.0 * sample_precision * sample_recall / (sample_precision + sample_recall)
            )

        ref_explanation = normalize_space(gold_row.get(args.gold_reason_col, ""))
        pred_explanation = normalize_space(explanation)
        rouge_p, rouge_r, rouge_f1 = rouge_l(
            pred_explanation,
            ref_explanation,
            tokenizer=args.rouge_tokenizer,
        )

        pred_set = normalized_set(pred_evidence)
        gold_set = normalized_set(gold_evidence)
        offset_exact: bool | None = None
        if pred_evidence and all(item.span is not None for item in pred_evidence):
            offset_exact = normalized_span_set(pred_evidence) == normalized_span_set(gold_evidence)

        record: dict[str, Any] = {
            "row_idx": idx,
            "source_csv": str(results_csv),
            "gold_csv": str(gold_csv),
            "run_dir": str(results_csv.parent),
            "category": gold_row.get(args.category_col, res_row.get(args.category_col, "")),
            "label": parse_binary(gold_row.get(args.gold_label_col)),
            "pred": parse_binary(res_row.get(args.pred_label_col)),
            "gold_evidence_count": len(gold_evidence),
            "pred_evidence_count": len(pred_evidence),
            "matched_evidence_count": matched,
            "evidence_precision": sample_precision,
            "evidence_recall": sample_recall,
            "evidence_f1": sample_f1,
            "evidence_exact_match": pred_set == gold_set,
            "evidence_offset_exact_match": offset_exact,
            "matched_evidence_scores": json.dumps(matched_scores, ensure_ascii=False),
            "matched_evidence_pairs": json.dumps(matched_pairs, ensure_ascii=False),
            "gold_evidences": json.dumps([item.text for item in gold_evidence], ensure_ascii=False),
            "pred_evidences": json.dumps([item.text for item in pred_evidence], ensure_ascii=False),
            "gold_reason": ref_explanation,
            "pred_explanation": pred_explanation,
            "rouge_l_precision": rouge_p,
            "rouge_l_recall": rouge_r,
            "rouge_l_f1": rouge_f1,
        }
        if args.result_input_col in res_row:
            record["input"] = res_row.get(args.result_input_col)
        rows.append(record)

    details = pd.DataFrame(rows)

    if args.compute_bertscore:
        p, r, f1 = compute_bertscore(
            candidates=details["pred_explanation"].fillna("").astype(str).tolist(),
            references=details["gold_reason"].fillna("").astype(str).tolist(),
            model_name=args.bertscore_model,
            batch_size=int(args.bertscore_batch_size),
            max_length=int(args.bertscore_max_length),
            device=args.bertscore_device,
        )
        details["bertscore_precision"] = p
        details["bertscore_recall"] = r
        details["bertscore_f1"] = f1

    span_precision = 0.0 if total_pred == 0 else total_matched / total_pred
    span_recall = 0.0 if total_gold == 0 else total_matched / total_gold
    span_f1 = 0.0 if span_precision + span_recall == 0 else 2 * span_precision * span_recall / (span_precision + span_recall)
    summary: dict[str, Any] = {
        "metric_version": "evidence_explanation_v1",
        "source_csv": str(results_csv),
        "gold_csv": str(gold_csv),
        "run_dir": str(results_csv.parent),
        "n": int(len(details)),
        "evidence_total_gold_spans": int(total_gold),
        "evidence_total_pred_spans": int(total_pred),
        "evidence_total_matched_spans": int(total_matched),
        "evidence_span_precision": span_precision,
        "evidence_span_recall": span_recall,
        "evidence_span_f1": span_f1,
        "evidence_match_score_mean": mean(matched_scores_all),
        "evidence_sample_f1_mean": mean(details["evidence_f1"].tolist()),
        "evidence_exact_match_rate": mean(details["evidence_exact_match"].astype(float).tolist()),
        "rouge_l_precision": mean(details["rouge_l_precision"].tolist()),
        "rouge_l_recall": mean(details["rouge_l_recall"].tolist()),
        "rouge_l_f1": mean(details["rouge_l_f1"].tolist()),
        "span_match_threshold": float(args.span_match_threshold),
        "span_tokenizer": args.span_tokenizer,
        "rouge_tokenizer": args.rouge_tokenizer,
        "bertscore_model": args.bertscore_model if args.compute_bertscore else "",
    }
    if "bertscore_f1" in details.columns:
        summary["bertscore_precision"] = mean(details["bertscore_precision"].tolist())
        summary["bertscore_recall"] = mean(details["bertscore_recall"].tolist())
        summary["bertscore_f1"] = mean(details["bertscore_f1"].tolist())

    if args.category_col in details.columns:
        by_category: dict[str, Any] = {}
        for category, group in details.groupby(args.category_col, dropna=False):
            key = "__missing__" if pd.isna(category) else str(category)
            gp = int(group["pred_evidence_count"].sum())
            gg = int(group["gold_evidence_count"].sum())
            gm = int(group["matched_evidence_count"].sum())
            p_val = 0.0 if gp == 0 else gm / gp
            r_val = 0.0 if gg == 0 else gm / gg
            f_val = 0.0 if p_val + r_val == 0 else 2 * p_val * r_val / (p_val + r_val)
            by_category[key] = {
                "n": int(len(group)),
                "evidence_span_precision": p_val,
                "evidence_span_recall": r_val,
                "evidence_span_f1": f_val,
                "evidence_exact_match_rate": mean(group["evidence_exact_match"].astype(float).tolist()),
                "rouge_l_f1": mean(group["rouge_l_f1"].tolist()),
                **(
                    {"bertscore_f1": mean(group["bertscore_f1"].tolist())}
                    if "bertscore_f1" in group.columns
                    else {}
                ),
            }
        summary["by_category"] = by_category
    return summary, details


def evaluate_one_bertscore_only(results_csv: Path, args: argparse.Namespace) -> tuple[dict[str, Any], pd.DataFrame]:
    if not args.compute_bertscore:
        raise ValueError("--bertscore-only requires --compute-bertscore")

    cfg = load_config_for_results(results_csv)
    gold_csv = infer_gold_csv(results_csv, args, cfg)
    results_df = pd.read_csv(results_csv)
    gold_df_raw = pd.read_csv(gold_csv)
    gold_df = align_gold(results_df, gold_df_raw, args)

    if args.generated_col not in results_df.columns:
        raise KeyError(f"{results_csv}: missing generated column {args.generated_col}")
    if args.gold_reason_col not in gold_df.columns:
        raise KeyError(f"{gold_csv}: missing gold column {args.gold_reason_col}")

    rows: list[dict[str, Any]] = []
    for idx, (res_row, gold_row) in enumerate(zip(results_df.to_dict("records"), gold_df.to_dict("records"))):
        generated = res_row.get(args.generated_col, "")
        _evidence_section, explanation = extract_generated_sections(generated)
        record: dict[str, Any] = {
            "row_idx": idx,
            "source_csv": str(results_csv),
            "gold_csv": str(gold_csv),
            "run_dir": str(results_csv.parent),
            "category": gold_row.get(args.category_col, res_row.get(args.category_col, "")),
            "gold_reason": normalize_space(gold_row.get(args.gold_reason_col, "")),
            "pred_explanation": normalize_space(explanation),
        }
        if args.result_input_col in res_row:
            record["input"] = res_row.get(args.result_input_col)
        rows.append(record)

    details = pd.DataFrame(rows)
    p, r, f1 = compute_bertscore(
        candidates=details["pred_explanation"].fillna("").astype(str).tolist(),
        references=details["gold_reason"].fillna("").astype(str).tolist(),
        model_name=args.bertscore_model,
        batch_size=int(args.bertscore_batch_size),
        max_length=int(args.bertscore_max_length),
        device=args.bertscore_device,
    )
    details["bertscore_precision"] = p
    details["bertscore_recall"] = r
    details["bertscore_f1"] = f1

    summary: dict[str, Any] = {
        "metric_version": "bertscore_only_v1",
        "source_csv": str(results_csv),
        "gold_csv": str(gold_csv),
        "run_dir": str(results_csv.parent),
        "n": int(len(details)),
        "bertscore_model": args.bertscore_model,
        "bertscore_precision": mean(details["bertscore_precision"].tolist()),
        "bertscore_recall": mean(details["bertscore_recall"].tolist()),
        "bertscore_f1": mean(details["bertscore_f1"].tolist()),
    }
    if args.category_col in details.columns:
        summary["by_category"] = {
            "__missing__" if pd.isna(category) else str(category): {
                "n": int(len(group)),
                "bertscore_f1": mean(group["bertscore_f1"].tolist()),
            }
            for category, group in details.groupby(args.category_col, dropna=False)
        }
    return summary, details


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
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    if not unique:
        raise ValueError("No inputs found. Use --results-csv, --run-dir, or --glob.")
    return unique


def flatten_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in summary.items() if not isinstance(value, dict)}


def aggregate_summaries(details: pd.DataFrame, summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if "evidence_f1" not in details.columns:
        out: dict[str, Any] = {
            "metric_version": "bertscore_only_v1",
            "n_runs": len(summaries),
            "n": int(len(details)),
        }
        if "bertscore_f1" in details.columns:
            out["bertscore_f1"] = mean(details["bertscore_f1"].tolist())
        return out

    gp = int(details["pred_evidence_count"].sum())
    gg = int(details["gold_evidence_count"].sum())
    gm = int(details["matched_evidence_count"].sum())
    p_val = 0.0 if gp == 0 else gm / gp
    r_val = 0.0 if gg == 0 else gm / gg
    f_val = 0.0 if p_val + r_val == 0 else 2 * p_val * r_val / (p_val + r_val)
    out: dict[str, Any] = {
        "metric_version": "evidence_explanation_v1",
        "n_runs": len(summaries),
        "n": int(len(details)),
        "evidence_total_gold_spans": gg,
        "evidence_total_pred_spans": gp,
        "evidence_total_matched_spans": gm,
        "evidence_span_precision": p_val,
        "evidence_span_recall": r_val,
        "evidence_span_f1": f_val,
        "evidence_sample_f1_mean": mean(details["evidence_f1"].tolist()),
        "evidence_exact_match_rate": mean(details["evidence_exact_match"].astype(float).tolist()),
        "rouge_l_f1": mean(details["rouge_l_f1"].tolist()),
    }
    if "bertscore_f1" in details.columns:
        out["bertscore_f1"] = mean(details["bertscore_f1"].tolist())
    return out


def update_metrics_json(summary: dict[str, Any], run_dir: Path, key: str) -> None:
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    metrics[key] = summary
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[update] {metrics_path} <- {key}")


def write_outputs(summaries: list[dict[str, Any]], details_frames: list[pd.DataFrame], args: argparse.Namespace) -> None:
    base = Path(args.out_prefix) if args.out_prefix else Path("reports/evidence_explanation_metrics/evidence_explanation_metrics")
    base.parent.mkdir(parents=True, exist_ok=True)
    details = pd.concat(details_frames, ignore_index=True)
    summary_df = pd.DataFrame([flatten_summary(summary) for summary in summaries])

    payload: dict[str, Any]
    if len(summaries) == 1:
        payload = summaries[0]
    else:
        payload = {
            "metric_version": "evidence_explanation_v1",
            "aggregate": aggregate_summaries(details, summaries),
            "runs": summaries,
        }

    json_path = base.with_suffix(".json")
    summary_path = base.with_name(base.name + "_summary.csv")
    details_path = base.with_name(base.name + "_details.csv")

    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_df.to_csv(summary_path, index=False)
    details.to_csv(details_path, index=False)
    print(f"[write] {json_path}")
    print(f"[write] {summary_path}")
    print(f"[write] {details_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute generated evidence/explanation quality metrics from label_span_explanation outputs. "
            "Generated evidence is parsed from '근거 스팬:' and explanation from '설명:'."
        )
    )
    parser.add_argument("--results-csv", action="append", default=[])
    parser.add_argument("--run-dir", action="append", default=[])
    parser.add_argument("--glob", action="append", default=[])
    parser.add_argument("--gold-csv", default=None, help="Optional explicit gold CSV for all inputs.")
    parser.add_argument("--evidence-variant", choices=("keep", "top1"), default="keep")
    parser.add_argument("--generated-col", default="gen")
    parser.add_argument("--result-input-col", default="input")
    parser.add_argument("--result-label-col", default="label")
    parser.add_argument("--pred-label-col", default="pred")
    parser.add_argument("--gold-text-col", default="text")
    parser.add_argument("--gold-label-col", default="label")
    parser.add_argument("--gold-reason-col", default="reason_value")
    parser.add_argument("--gold-spans-col", default="spans")
    parser.add_argument("--gold-evidences-col", default="evidences")
    parser.add_argument("--category-col", default="category")
    parser.add_argument("--span-match-threshold", type=float, default=0.5)
    parser.add_argument("--span-tokenizer", choices=("char", "word"), default="char")
    parser.add_argument("--rouge-tokenizer", choices=("char", "word"), default="word")
    parser.add_argument("--prefer-offsets", action="store_true", help="Use [start,end] overlap when generated offsets exist.")
    parser.add_argument("--compute-bertscore", action="store_true")
    parser.add_argument(
        "--bertscore-only",
        action="store_true",
        help="Compute only BERTScore from generated/gold explanations; skip span and ROUGE calculations.",
    )
    parser.add_argument("--bertscore-model", default="klue/roberta-base")
    parser.add_argument("--bertscore-batch-size", type=int, default=16)
    parser.add_argument("--bertscore-max-length", type=int, default=512)
    parser.add_argument("--bertscore-device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--out-prefix", default=None)
    parser.add_argument("--update-metrics", action="store_true")
    parser.add_argument("--metrics-key", default="evidence_explanation")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inputs = collect_inputs(args)
    summaries: list[dict[str, Any]] = []
    details_frames: list[pd.DataFrame] = []
    for path in inputs:
        if args.bertscore_only:
            summary, details = evaluate_one_bertscore_only(path, args)
        else:
            summary, details = evaluate_one(path, args)
        summaries.append(summary)
        details_frames.append(details)
        if args.update_metrics:
            update_metrics_json(summary, path.parent, args.metrics_key)
        if args.bertscore_only:
            print(
                "[summary] "
                f"{path.parent} "
                f"bertscore_f1={summary['bertscore_f1']:.4f}"
            )
        else:
            print(
                "[summary] "
                f"{path.parent} "
                f"span_f1={summary['evidence_span_f1']:.4f} "
                f"rouge_l_f1={summary['rouge_l_f1']:.4f}"
                + (
                    f" bertscore_f1={summary['bertscore_f1']:.4f}"
                    if "bertscore_f1" in summary and summary["bertscore_f1"] is not None
                    else ""
                )
            )
    write_outputs(summaries, details_frames, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
