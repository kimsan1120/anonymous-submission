import os
import ast
import json
import datetime
import inspect
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional dependency
    SummaryWriter = None
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer

try:
    from transformers import BitsAndBytesConfig
except Exception:  # pragma: no cover - optional dependency
    BitsAndBytesConfig = None

try:
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
except Exception:  # pragma: no cover - optional dependency
    LoraConfig = None
    PeftModel = None
    TaskType = None
    get_peft_model = None
    prepare_model_for_kbit_training = None

from phishdec.utils.model_registry import resolve_model_name
from phishdec.decoding.parser import parse_pred
from phishdec.eval.eval_classification import compute_binary_classification_metrics
from phishdec.train.decoder import (
    _classification_loss as _decoder_classification_loss,
    _compute_class_weights as _decoder_compute_class_weights,
    _evidence_token_loss as _decoder_evidence_token_loss,
    _normalize_spans as _decoder_normalize_spans,
)


def _get_cfg(cfg: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur = cfg
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def load_yaml_config(path: str) -> Dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def timestamp_run_dir(exp_name: str, out_root: str) -> str:
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return os.path.join(out_root, f"{exp_name}_{ts}")


def _read_instruction(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _render_instruction_with_text(instruction: str, text: str) -> Tuple[str, bool]:
    if not instruction:
        return "", False

    text_value = str(text)
    rendered = instruction
    for placeholder in ("{text}", "{{text}}", "{{transcript}}", "{{input_text}}"):
        rendered = rendered.replace(placeholder, text_value)
    return rendered, rendered != instruction


def _append_jsonl(path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _uses_tensorboard(report_to: Any) -> bool:
    if report_to is None:
        return True
    if isinstance(report_to, str):
        normalized = report_to.strip().lower()
        if normalized in {"", "none"}:
            return False
        return "tensorboard" in {part.strip() for part in normalized.split(",")}
    if isinstance(report_to, (list, tuple, set)):
        return any(str(item).strip().lower() == "tensorboard" for item in report_to)
    return False


def _hmean(a: float, b: float) -> float:
    a = float(a)
    b = float(b)
    if a <= 0.0 or b <= 0.0:
        return 0.0
    return float((2.0 * a * b) / max(1e-12, a + b))


def _log_classification_scalars(writer, split: str, metrics: Dict[str, Any], step: int) -> None:
    if writer is None or not metrics:
        return

    accuracy = float(metrics.get("accuracy", 0.0))
    macro_f1 = float(metrics.get("macro_f1", 0.0))
    positive_precision = float(metrics.get("positive_precision", 0.0))
    positive_recall = float(metrics.get("positive_recall", 0.0))

    writer.add_scalars(
        f"classification/{split}_core",
        {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "positive_precision": positive_precision,
            "positive_recall": positive_recall,
        },
        step,
    )
    writer.add_scalar(
        f"classification/{split}_hmean_accuracy_macro_f1",
        _hmean(accuracy, macro_f1),
        step,
    )


def build_prompt_builder(prompt_cfg: Dict[str, Any]) -> Callable[[str, Optional[Any]], str]:
    use_instruction = bool(prompt_cfg.get("use_instruction", False))
    instruction_path = prompt_cfg.get("instruction_path", "")
    fmt = prompt_cfg.get("format", "plain")
    user_prefix = prompt_cfg.get("user_prefix", "문장: ")
    answer_prefix_raw = prompt_cfg.get("answer_prefix", None)
    if answer_prefix_raw is None:
        answer_prefix = "정답:"
    else:
        answer_prefix = str(answer_prefix_raw).rstrip()
    answer_with_space = f"{answer_prefix} " if answer_prefix else ""

    instr = _read_instruction(instruction_path) if use_instruction and instruction_path else ""

    def _formatter(text: str, label: Optional[Any]) -> str:
        instr_text, embeds_text = _render_instruction_with_text(instr, text)
        base = "" if embeds_text else f"{user_prefix}{text}"
        prompt_text = f"{instr_text}\n{base}".strip() if instr_text else base
        if label is None:
            return prompt_text
        lbl = str(label).strip()
        if fmt == "chat":
            if answer_with_space:
                return f"{prompt_text}\n{answer_with_space}{lbl}" if prompt_text else f"{answer_with_space}{lbl}"
            return f"{prompt_text}\n{lbl}" if prompt_text else lbl
        if answer_with_space:
            return f"{prompt_text}\n{answer_with_space}{lbl}" if prompt_text else f"{answer_with_space}{lbl}"
        return f"{prompt_text}\n{lbl}" if prompt_text else lbl

    return _formatter


def build_inference_prompt_builder(prompt_cfg: Dict[str, Any]) -> Callable[[str], str]:
    use_instruction = bool(prompt_cfg.get("use_instruction", False))
    instruction_path = prompt_cfg.get("instruction_path", "")
    user_prefix = prompt_cfg.get("user_prefix", "문장: ")
    answer_prefix_raw = prompt_cfg.get("answer_prefix", None)
    if answer_prefix_raw is None:
        answer_prefix = "정답:"
    else:
        answer_prefix = str(answer_prefix_raw).rstrip()
    answer_with_space = f"{answer_prefix} " if answer_prefix else ""

    instr = _read_instruction(instruction_path) if use_instruction and instruction_path else ""

    def _formatter(text: str) -> str:
        instr_text, embeds_text = _render_instruction_with_text(instr, text)
        base = "" if embeds_text else f"{user_prefix}{text}"
        prompt_text = f"{instr_text}\n{base}".strip() if instr_text else base
        if answer_with_space:
            return f"{prompt_text}\n{answer_with_space}" if prompt_text else answer_with_space
        return f"{prompt_text}\n" if prompt_text else ""

    return _formatter


def _metrics_by_category(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    categories: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, Tuple[List[int], List[int]]] = {}
    for yt, yp, cat in zip(y_true, y_pred, categories):
        key = str(cat)
        if key not in grouped:
            grouped[key] = ([], [])
        grouped[key][0].append(int(yt))
        grouped[key][1].append(int(yp))
    return {
        cat: compute_binary_classification_metrics(yt, yp)
        for cat, (yt, yp) in grouped.items()
    }


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def _read_csv_normalized(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = [str(col).replace("\ufeff", "").strip() for col in df.columns]
    return df


def _normalize_reason_text(reason_value: Any) -> str:
    return re.sub(r"\s+", " ", str("" if reason_value is None else reason_value)).strip()


def _safe_list_literal(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, float) and np.isnan(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return []
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, tuple):
        return list(parsed)
    return []


def _evidence_items_for_generation(
    *,
    source_text: Optional[Any],
    spans_raw: Optional[Any],
    evidences_raw: Optional[Any],
    max_items: int = 3,
) -> List[Tuple[Optional[int], Optional[int], str]]:
    text = str("" if source_text is None else source_text)
    evidence_texts = [str(v).strip() for v in _safe_list_literal(evidences_raw) if str(v).strip()]
    spans = _decoder_normalize_spans(spans_raw=spans_raw, text=text, evidences_raw=evidences_raw)

    items: List[Tuple[Optional[int], Optional[int], str]] = []
    for idx, (st, ed) in enumerate(spans):
        snippet = ""
        if idx < len(evidence_texts):
            snippet = evidence_texts[idx]
        if not snippet and text:
            snippet = text[max(0, int(st)) : max(0, int(ed))].strip()
        if snippet:
            items.append((int(st), int(ed), re.sub(r"\s+", " ", snippet).strip()))

    if not items:
        for ev in evidence_texts:
            items.append((None, None, re.sub(r"\s+", " ", ev).strip()))

    if max_items > 0:
        items = items[: int(max_items)]
    return items


def _compose_label_reason_target(
    label_value: Any,
    reason_value: Any,
    *,
    target_format: str = "anchored_braced_reason",
    source_text: Optional[Any] = None,
    evidence_spans_value: Optional[Any] = None,
    evidences_value: Optional[Any] = None,
    max_evidence_items: int = 3,
    include_evidence_span_offsets: bool = True,
    explanation_label_anchor: bool = False,
    explanation_label_anchor_modality: str = "sms",
) -> str:
    try:
        label_int = int(label_value)
    except Exception:
        label_int = 0
    reason = _normalize_reason_text(reason_value)
    if explanation_label_anchor:
        reason = _prepend_label_anchor(
            reason,
            label_int=label_int,
            modality=explanation_label_anchor_modality,
        )
    fmt = str(target_format)
    if fmt == "label_first_explanation":
        return f"{label_int} {reason}".strip() if reason else str(label_int)
    if fmt in {
        "label_span",
        "label_evidence",
        "label_span_explanation",
        "label_evidence_explanation",
        "span_explanation_label",
        "evidence_explanation_label",
    }:
        evidence_items = _evidence_items_for_generation(
            source_text=source_text,
            spans_raw=evidence_spans_value,
            evidences_raw=evidences_value,
            max_items=max_evidence_items,
        )
        if evidence_items:
            evidence_lines = []
            for idx, (st, ed, snippet) in enumerate(evidence_items, start=1):
                span_prefix = (
                    f"[{st}, {ed}] "
                    if include_evidence_span_offsets and st is not None and ed is not None
                    else ""
                )
                evidence_lines.append(f"{idx}) {span_prefix}{snippet}")
            evidence_block = "\n".join(evidence_lines)
        else:
            evidence_block = "없음"
        if fmt in {"label_span", "label_evidence"}:
            return f"{label_int}\n근거 스팬:\n{evidence_block}"
        if fmt in {"span_explanation_label", "evidence_explanation_label"}:
            if reason:
                return f"근거 스팬:\n{evidence_block}\n설명:\n{reason}\n정답:\n{label_int}"
            return f"근거 스팬:\n{evidence_block}\n정답:\n{label_int}"
        if reason:
            return f"{label_int}\n근거 스팬:\n{evidence_block}\n설명:\n{reason}"
        return f"{label_int}\n근거 스팬:\n{evidence_block}"
    safe_reason = reason.replace("{", "(").replace("}", ")")
    if str(target_format) == "anchored_plain_reason":
        return f"정답: {label_int}\n설명: {reason}"
    return f"정답: {{{label_int}}}\n설명: {{{safe_reason}}}"


def _label_anchor_text(*, label_int: int, modality: str) -> str:
    mode = str(modality or "sms").strip().lower()
    is_voice = mode == "voice"
    if int(label_int) == 1:
        return "이 통화는 보이스피싱으로 판단된다." if is_voice else "이 메시지는 스미싱으로 판단된다."
    return "이 통화는 정상 상담으로 판단된다." if is_voice else "이 메시지는 정상 메시지로 판단된다."


def _prepend_label_anchor(reason: str, *, label_int: int, modality: str) -> str:
    anchor = _label_anchor_text(label_int=label_int, modality=modality)
    body = _normalize_reason_text(reason)
    if not body:
        return anchor
    if body.startswith(anchor):
        return body
    return f"{anchor} {body}"


def _explanation_body_char_start(
    *,
    full_text: str,
    target_text: str,
    label_value: int,
    target_format: str,
) -> Optional[int]:
    target_start = full_text.rfind(target_text)
    if target_start < 0:
        return None
    fmt = str(target_format or "").strip()
    if fmt == "label_first_explanation":
        start = target_start + len(str(int(label_value)))
    elif fmt in {
        "label_span_explanation",
        "label_evidence_explanation",
        "span_explanation_label",
        "evidence_explanation_label",
    }:
        marker = "\n설명:\n"
        marker_idx = target_text.find(marker)
        marker_len = len(marker)
        if marker_idx < 0:
            marker = "설명:"
            marker_idx = target_text.find(marker)
            marker_len = len(marker)
        if marker_idx < 0:
            return None
        start = target_start + marker_idx + marker_len
    else:
        return None
    while start < len(full_text) and full_text[start].isspace():
        start += 1
    return start if start < len(full_text) else None


def _explanation_body_char_end(*, full_text: str, target_text: str, target_format: str) -> int:
    target_start = full_text.rfind(target_text)
    if target_start < 0:
        return len(full_text)
    fmt = str(target_format or "").strip()
    if fmt in {"span_explanation_label", "evidence_explanation_label"}:
        for marker in ("\n정답:\n", "\n정답:", "정답:\n", "정답:"):
            marker_idx = target_text.rfind(marker)
            if marker_idx >= 0:
                return max(target_start, target_start + marker_idx)
    return len(full_text)


def _label_last_char_start(*, full_text: str, target_text: str) -> Optional[int]:
    target_start = full_text.rfind(target_text)
    if target_start < 0:
        return None
    for marker in ("\n정답:\n", "\n정답:", "정답:\n", "정답:"):
        marker_idx = target_text.rfind(marker)
        if marker_idx < 0:
            continue
        start = target_start + marker_idx + len(marker)
        while start < len(full_text) and full_text[start].isspace():
            start += 1
        return start if start < len(full_text) else None
    return None


def _token_index_for_char(
    *,
    offset_mapping: Sequence[Any],
    input_ids: Sequence[int],
    char_start: Optional[int],
    expected_token_id: int,
) -> Optional[int]:
    if char_start is None:
        return None
    for token_idx, off in enumerate(offset_mapping):
        if token_idx >= len(input_ids):
            break
        if not isinstance(off, (list, tuple)) or len(off) != 2:
            continue
        st, ed = int(off[0]), int(off[1])
        if ed <= st:
            continue
        if st <= int(char_start) < ed and int(input_ids[token_idx]) == int(expected_token_id):
            return int(token_idx)
    return None


def _char_span_token_mask(
    *,
    offset_mapping: Sequence[Any],
    start_char: Optional[int],
    end_char: int,
    min_token_idx: int,
    length: int,
) -> List[float]:
    mask = [0.0] * int(length)
    if start_char is None:
        return mask
    start = int(start_char)
    end = max(start, int(end_char))
    for token_idx, off in enumerate(offset_mapping):
        if token_idx >= len(mask):
            break
        if token_idx < int(min_token_idx):
            continue
        if not isinstance(off, (list, tuple)) or len(off) != 2:
            continue
        st, ed = int(off[0]), int(off[1])
        if ed <= st:
            continue
        if not (ed <= start or st >= end):
            mask[token_idx] = 1.0
    return mask


def _load_metrics_frame(csv_path: str, text_col: str, label_col: str) -> pd.DataFrame:
    df = _read_csv_normalized(csv_path)
    missing = [c for c in (text_col, label_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Columns {missing} not found in {csv_path}")
    return df.copy()


def _evaluate_generation_classifier(
    model: torch.nn.Module,
    tokenizer,
    df: pd.DataFrame,
    *,
    prompt_builder: Callable[[str], str],
    text_col: str,
    label_col: str,
    batch_size: int,
    max_input_tokens: int,
    max_new_tokens: int,
) -> Dict[str, Any]:
    model = _unwrap_model(model)
    model.eval()
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    dist_enabled = torch.distributed.is_available() and torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size() if dist_enabled else 1
    rank = torch.distributed.get_rank() if dist_enabled else 0

    texts = df[text_col].fillna("").astype(str).tolist()
    prompts = [prompt_builder(t) for t in texts]
    y_true = df[label_col].astype(int).tolist()
    categories = df["category"].fillna("").astype(str).tolist() if "category" in df.columns else [""] * len(df)
    preds: List[int] = []
    invalid_parse_count = 0
    eval_indices = list(range(rank, len(prompts), world_size)) if world_size > 1 else list(range(len(prompts)))
    shard_prompts = [prompts[i] for i in eval_indices]
    shard_y_true = [y_true[i] for i in eval_indices]
    shard_categories = [categories[i] for i in eval_indices]

    old_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    model_max_len = getattr(tokenizer, "model_max_length", max_input_tokens)
    try:
        model_max_len = int(model_max_len)
    except Exception:
        model_max_len = max_input_tokens
    safe_len = max(16, min(model_max_len, max_input_tokens))

    autocast_enabled = torch.cuda.is_available() and model_dtype in (torch.float16, torch.bfloat16)
    try:
        for start in range(0, len(shard_prompts), max(1, batch_size)):
            batch_prompts = shard_prompts[start : start + batch_size]
            enc = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=safe_len,
            )
            enc.pop("token_type_ids", None)
            enc = {k: v.to(device) for k, v in enc.items()}
            prompt_seq_len = enc["input_ids"].shape[1]
            with torch.inference_mode():
                if autocast_enabled:
                    with torch.autocast(device_type="cuda", dtype=model_dtype):
                        gen_out = model.generate(
                            **enc,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                        )
                else:
                    gen_out = model.generate(
                        **enc,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
            for i in range(len(batch_prompts)):
                gen_tokens = gen_out[i, int(prompt_seq_len) :]
                txt = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                y_true_i = int(shard_y_true[start + i])
                try:
                    parsed = parse_pred(txt, strict=True)
                except Exception:
                    parsed = 1 - y_true_i
                    invalid_parse_count += 1
                preds.append(int(parsed))
            del enc, gen_out
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        tokenizer.padding_side = old_padding_side

    if dist_enabled and world_size > 1:
        gathered_preds: List[List[int]] = [None] * world_size
        gathered_y_true: List[List[int]] = [None] * world_size
        gathered_categories: List[List[str]] = [None] * world_size
        gathered_invalid_counts: List[int] = [0] * world_size
        torch.distributed.all_gather_object(gathered_preds, preds)
        torch.distributed.all_gather_object(gathered_y_true, shard_y_true)
        torch.distributed.all_gather_object(gathered_categories, shard_categories)
        torch.distributed.all_gather_object(gathered_invalid_counts, int(invalid_parse_count))

        if rank != 0:
            return {}

        flat_preds = [int(v) for chunk in gathered_preds for v in (chunk or [])]
        flat_y_true = [int(v) for chunk in gathered_y_true for v in (chunk or [])]
        flat_categories = [str(v) for chunk in gathered_categories for v in (chunk or [])]
        metrics = compute_binary_classification_metrics(flat_y_true, flat_preds)
        metrics["n"] = int(len(flat_y_true))
        metrics["by_category"] = _metrics_by_category(flat_y_true, flat_preds, flat_categories)
        metrics["invalid_parse_count"] = int(sum(int(v) for v in gathered_invalid_counts))
        return metrics

    metrics = compute_binary_classification_metrics(shard_y_true, preds)
    metrics["n"] = int(len(shard_y_true))
    metrics["by_category"] = _metrics_by_category(shard_y_true, preds, shard_categories)
    metrics["invalid_parse_count"] = int(invalid_parse_count)
    return metrics


class StrictGenerationValidationCallback(TrainerCallback):
    def __init__(
        self,
        *,
        cfg: Dict[str, Any],
        run_dir: str,
        tokenizer,
        text_col: str,
        metric_label_col: str,
        prompt_cfg: Dict[str, Any],
    ) -> None:
        self.cfg = cfg
        self.run_dir = run_dir
        self.tokenizer = tokenizer
        self.text_col = text_col
        self.metric_label_col = metric_label_col
        self.prompt_builder = build_inference_prompt_builder(prompt_cfg)
        self.eval_csv = _get_cfg(cfg, "data.eval_csv")
        self.model_name = _get_cfg(cfg, "model.model_name")
        self.eval_batch_size = int(
            _get_cfg(cfg, "train.eval_generate_batch_size", _get_cfg(cfg, "train.eval_batch_size", 8))
        )
        self.max_input_tokens = int(_get_cfg(cfg, "train.eval_max_input_tokens", _get_cfg(cfg, "train.max_length", 1024)))
        self.max_new_tokens = int(_get_cfg(cfg, "train.eval_max_new_tokens", 32))
        self.log_path = os.path.join(run_dir, "eval_validation_generate_metrics.jsonl")
        self._tb_writer = None

    def _is_main(self, state) -> bool:
        return bool(getattr(state, "is_world_process_zero", True))

    def _barrier(self) -> None:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

    def _use_distributed_eval(self) -> bool:
        return (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        )

    def _ensure_tb_writer(self, args) -> None:
        if self._tb_writer is not None:
            return
        if SummaryWriter is None:
            return
        if not _uses_tensorboard(getattr(args, "report_to", None)):
            return
        log_dir = getattr(args, "logging_dir", None)
        if not log_dir:
            return
        os.makedirs(log_dir, exist_ok=True)
        self._tb_writer = SummaryWriter(log_dir=log_dir)

    def _evaluate_split(self, model) -> Optional[Dict[str, Any]]:
        if not self.eval_csv:
            return None
        df = _load_metrics_frame(self.eval_csv, self.text_col, self.metric_label_col)
        return _evaluate_generation_classifier(
            model=model,
            tokenizer=self.tokenizer,
            df=df,
            prompt_builder=self.prompt_builder,
            text_col=self.text_col,
            label_col=self.metric_label_col,
            batch_size=self.eval_batch_size,
            max_input_tokens=self.max_input_tokens,
            max_new_tokens=self.max_new_tokens,
        )

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        if not self.eval_csv or model is None:
            return
        if metrics is None:
            metrics = {}

        is_main = self._is_main(state)
        dist_eval = self._use_distributed_eval()

        if is_main:
            self._ensure_tb_writer(args)

        self._barrier()
        val_metrics = None
        if dist_eval:
            val_metrics = self._evaluate_split(model)
            payload = [val_metrics if val_metrics else None]
            torch.distributed.broadcast_object_list(payload, src=0)
            val_metrics = payload[0] if payload[0] else {}
        elif is_main:
            val_metrics = self._evaluate_split(model)
        self._barrier()

        if not val_metrics:
            return

        n = int(val_metrics.get("n", 0))
        invalid_count = int(val_metrics.get("invalid_parse_count", 0))
        invalid_rate = float(invalid_count / max(1, n))
        metrics["eval_accuracy_strict"] = float(val_metrics.get("accuracy", 0.0))
        metrics["eval_macro_f1_strict"] = float(val_metrics.get("macro_f1", 0.0))
        metrics["eval_positive_recall_strict"] = float(val_metrics.get("positive_recall", 0.0))
        metrics["eval_invalid_parse_count"] = int(invalid_count)
        metrics["eval_invalid_parse_rate"] = invalid_rate

        if not is_main:
            return

        row = {
            "epoch": float(state.epoch or 0.0),
            "global_step": int(state.global_step),
            "n": n,
            "accuracy": float(metrics["eval_accuracy_strict"]),
            "macro_f1": float(metrics["eval_macro_f1_strict"]),
            "positive_recall": float(metrics["eval_positive_recall_strict"]),
            "invalid_parse_count": int(invalid_count),
            "invalid_parse_rate": float(invalid_rate),
        }
        _append_jsonl(self.log_path, row)

        if self._tb_writer is not None:
            step = int(state.global_step)
            self._tb_writer.add_scalar("classification/eval_accuracy_strict", float(metrics["eval_accuracy_strict"]), step)
            self._tb_writer.add_scalar("classification/eval_macro_f1_strict", float(metrics["eval_macro_f1_strict"]), step)
            self._tb_writer.add_scalar(
                "classification/eval_positive_recall_strict",
                float(metrics["eval_positive_recall_strict"]),
                step,
            )
            self._tb_writer.add_scalar("classification/eval_invalid_parse_rate", float(invalid_rate), step)

    def on_train_end(self, args, state, control, **kwargs):
        if not self._is_main(state):
            return
        if self._tb_writer is not None:
            self._tb_writer.close()
            self._tb_writer = None


class WarmupEarlyStoppingCallback(EarlyStoppingCallback):
    def __init__(
        self,
        *,
        early_stopping_patience: int,
        early_stopping_threshold: float = 0.0,
        min_epochs: float = 0.0,
        min_steps: int = 0,
    ) -> None:
        super().__init__(
            early_stopping_patience=int(early_stopping_patience),
            early_stopping_threshold=float(early_stopping_threshold),
        )
        self.min_epochs = float(min_epochs or 0.0)
        self.min_steps = int(min_steps or 0)

    def _warmup_ready(self, state) -> bool:
        if int(getattr(state, "global_step", 0) or 0) < self.min_steps:
            return False
        if self.min_epochs <= 0.0:
            return True
        epoch = getattr(state, "epoch", None)
        if epoch is None:
            return False
        return float(epoch) >= self.min_epochs

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if not self._warmup_ready(state):
            return control
        return super().on_evaluate(args, state, control, metrics, **kwargs)


class EpochClassificationEvalCallback(TrainerCallback):
    def __init__(
        self,
        *,
        cfg: Dict[str, Any],
        run_dir: str,
        tokenizer,
        text_col: str,
        metric_label_col: str,
        prompt_cfg: Dict[str, Any],
    ) -> None:
        self.cfg = cfg
        self.run_dir = run_dir
        self.tokenizer = tokenizer
        self.text_col = text_col
        self.metric_label_col = metric_label_col
        self.prompt_builder = build_inference_prompt_builder(prompt_cfg)
        self.train_metrics_csv = _get_cfg(cfg, "data.train_metrics_csv")
        self.eval_csv = _get_cfg(cfg, "data.eval_csv")
        self.test_csv = _get_cfg(cfg, "data.test_csv")
        self.model_name = _get_cfg(cfg, "model.model_name")
        self.metric_for_best = str(
            _get_cfg(
                cfg,
                "train.classification_metric_for_best_model",
                _get_cfg(cfg, "train.metric_for_best_model", "accuracy"),
            )
        )
        self.greater_is_better = bool(
            _get_cfg(
                cfg,
                "train.classification_greater_is_better",
                _get_cfg(cfg, "train.greater_is_better", True),
            )
        )
        self.eval_batch_size = int(
            _get_cfg(cfg, "train.eval_generate_batch_size", _get_cfg(cfg, "train.eval_batch_size", 8))
        )
        self.max_input_tokens = int(_get_cfg(cfg, "train.eval_max_input_tokens", _get_cfg(cfg, "train.max_length", 1024)))
        self.max_new_tokens = int(_get_cfg(cfg, "train.eval_max_new_tokens", 3))
        self.train_log_path = os.path.join(run_dir, "train_epoch_metrics.jsonl")
        self.val_log_path = os.path.join(run_dir, "eval_validation_epoch_metrics.jsonl")
        self.test_log_path = os.path.join(run_dir, "eval_test_epoch_metrics.jsonl")
        self.best_epoch = -1
        self.best_score = -float("inf")
        self.best_val_row: Optional[Dict[str, Any]] = None
        self.best_test_row: Optional[Dict[str, Any]] = None
        self._last_log_history_idx = 0
        self._tb_writer = None

    def _is_main(self, state) -> bool:
        return bool(getattr(state, "is_world_process_zero", True))

    def _barrier(self) -> None:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

    def _use_distributed_eval(self) -> bool:
        return (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        )

    def _ensure_tb_writer(self, args) -> None:
        if self._tb_writer is not None:
            return
        if SummaryWriter is None:
            return
        if not _uses_tensorboard(getattr(args, "report_to", None)):
            return
        log_dir = getattr(args, "logging_dir", None)
        if not log_dir:
            return
        os.makedirs(log_dir, exist_ok=True)
        self._tb_writer = SummaryWriter(log_dir=log_dir)

    def _consume_train_logs(self, state) -> Dict[str, float]:
        new_logs = state.log_history[self._last_log_history_idx :]
        self._last_log_history_idx = len(state.log_history)
        loss_logs = [row for row in new_logs if "loss" in row]
        if not loss_logs:
            return {"train_loss": 0.0, "learning_rate": 0.0}
        train_loss = float(sum(float(row["loss"]) for row in loss_logs) / max(1, len(loss_logs)))
        learning_rate = float(loss_logs[-1].get("learning_rate", 0.0))
        return {"train_loss": train_loss, "learning_rate": learning_rate}

    def _evaluate_split(self, model, csv_path: Optional[str]) -> Optional[Dict[str, Any]]:
        if not csv_path:
            return None
        df = _load_metrics_frame(csv_path, self.text_col, self.metric_label_col)
        return _evaluate_generation_classifier(
            model=model,
            tokenizer=self.tokenizer,
            df=df,
            prompt_builder=self.prompt_builder,
            text_col=self.text_col,
            label_col=self.metric_label_col,
            batch_size=self.eval_batch_size,
            max_input_tokens=self.max_input_tokens,
            max_new_tokens=self.max_new_tokens,
        )

    def _same_csv(self, left: Optional[str], right: Optional[str]) -> bool:
        if not left or not right:
            return False
        try:
            return os.path.abspath(str(left)) == os.path.abspath(str(right))
        except Exception:
            return str(left) == str(right)

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        epoch = int(round(float(state.epoch or 0.0)))
        train_meta = self._consume_train_logs(state)
        if self._is_main(state):
            self._ensure_tb_writer(args)

        self._barrier()
        train_metrics = val_metrics = test_metrics = None
        if self._use_distributed_eval():
            train_metrics = self._evaluate_split(model, self.train_metrics_csv) if self.train_metrics_csv else None
            val_metrics = self._evaluate_split(model, self.eval_csv) if self.eval_csv else None
            if self.test_csv:
                if self._same_csv(self.eval_csv, self.test_csv):
                    test_metrics = val_metrics
                else:
                    test_metrics = self._evaluate_split(model, self.test_csv)
        elif self._is_main(state):
            train_metrics = self._evaluate_split(model, self.train_metrics_csv) if self.train_metrics_csv else None
            val_metrics = self._evaluate_split(model, self.eval_csv) if self.eval_csv else None
            if self.test_csv:
                if self._same_csv(self.eval_csv, self.test_csv):
                    test_metrics = val_metrics
                else:
                    test_metrics = self._evaluate_split(model, self.test_csv)
        self._barrier()

        if not self._is_main(state):
            return

        train_row: Dict[str, Any] = {
            "epoch": epoch,
            "global_step": int(state.global_step),
            "train_loss": train_meta["train_loss"],
            "learning_rate": train_meta["learning_rate"],
        }
        if train_metrics is not None:
            train_row.update(train_metrics)
        _append_jsonl(self.train_log_path, train_row)
        if self._tb_writer is not None:
            self._tb_writer.add_scalar("classification/train_loss", float(train_meta["train_loss"]), epoch)
            self._tb_writer.add_scalar("classification/train_lr", float(train_meta["learning_rate"]), epoch)
            if train_metrics is not None:
                _log_classification_scalars(self._tb_writer, "train", train_metrics, epoch)

        if val_metrics is not None:
            val_row = {"epoch": epoch, "split": "validation", **val_metrics}
            _append_jsonl(self.val_log_path, val_row)
            if self._tb_writer is not None:
                _log_classification_scalars(self._tb_writer, "eval", val_metrics, epoch)
            sign = 1.0 if self.greater_is_better else -1.0
            score = sign * float(val_metrics.get(self.metric_for_best, -1e9))
            if score > self.best_score:
                self.best_score = score
                self.best_epoch = epoch
                self.best_val_row = val_row
        else:
            val_row = None

        if test_metrics is not None:
            test_row = {"epoch": epoch, "split": "test", **test_metrics}
            _append_jsonl(self.test_log_path, test_row)
            if self._tb_writer is not None:
                _log_classification_scalars(self._tb_writer, "test", test_metrics, epoch)
            if self.best_epoch == epoch:
                self.best_test_row = test_row

    def on_train_end(self, args, state, control, **kwargs):
        if not self._is_main(state):
            return
        summary = {
            "model_name": self.model_name,
            "train_csv": _get_cfg(self.cfg, "data.train_csv"),
            "train_metrics_csv": self.train_metrics_csv,
            "eval_csv": self.eval_csv,
            "test_csv": self.test_csv,
            "batch_size": int(_get_cfg(self.cfg, "train.batch_size", 8)),
            "eval_batch_size": int(_get_cfg(self.cfg, "train.eval_batch_size", 8)),
            "peft": str(_get_cfg(self.cfg, "train.peft", "none")),
            "lora_r": _get_cfg(self.cfg, "train.lora_r"),
            "lora_alpha": _get_cfg(self.cfg, "train.lora_alpha"),
            "lora_dropout": _get_cfg(self.cfg, "train.lora_dropout"),
            "lora_target_modules": _get_cfg(self.cfg, "train.lora_target_modules"),
            "best_epoch": self.best_epoch,
            "metric_for_best_model": self.metric_for_best,
            "best_val": self.best_val_row,
            "best_test": self.best_test_row,
        }
        with open(os.path.join(self.run_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        if self._tb_writer is not None:
            self._tb_writer.close()
            self._tb_writer = None


def build_sft_datasets(
    train_csv: str,
    text_col: str,
    label_col: str,
    prompt_builder: Callable[[str, Optional[Any]], str],
    eval_csv: Optional[str] = None,
    *,
    reason_col: Optional[str] = None,
    compose_label_reason_target: bool = False,
    eval_compose_label_reason_target: Optional[bool] = None,
    label_reason_target_format: str = "anchored_braced_reason",
    evidence_spans_col: str = "spans",
    evidences_col: str = "evidences",
    max_evidence_items: int = 3,
    include_evidence_span_offsets: bool = True,
    explanation_label_anchor: bool = False,
    explanation_label_anchor_modality: str = "sms",
) -> Tuple[Dataset, Optional[Dataset]]:
    def _to_dataset(csv_path: str, use_label_reason_target: bool) -> Dataset:
        df = _read_csv_normalized(csv_path)
        required = [text_col, label_col]
        if use_label_reason_target:
            if not reason_col:
                raise ValueError("compose_label_reason_target=true requires data.reason_col")
            required.append(str(reason_col))
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Columns {missing} not found in {csv_path}")
        df = df.copy()
        if use_label_reason_target:
            spans_values = df[evidence_spans_col].tolist() if evidence_spans_col in df.columns else [None] * len(df)
            evidences_values = df[evidences_col].tolist() if evidences_col in df.columns else [None] * len(df)
            targets = [
                _compose_label_reason_target(
                    label_value=lbl,
                    reason_value=rsn,
                    target_format=label_reason_target_format,
                    source_text=text_value,
                    evidence_spans_value=spans_raw,
                    evidences_value=evidences_raw,
                    max_evidence_items=max_evidence_items,
                    include_evidence_span_offsets=include_evidence_span_offsets,
                    explanation_label_anchor=explanation_label_anchor,
                    explanation_label_anchor_modality=explanation_label_anchor_modality,
                )
                for lbl, rsn, text_value, spans_raw, evidences_raw in zip(
                    df[label_col],
                    df[str(reason_col)],
                    df[text_col].fillna("").astype(str),
                    spans_values,
                    evidences_values,
                )
            ]
        else:
            targets = df[label_col].tolist()
        df["text"] = [
            prompt_builder(str(t), targets[i])
            for i, t in enumerate(df[text_col].astype(str).tolist())
        ]
        return Dataset.from_pandas(df[["text"]], preserve_index=False)

    train_ds = _to_dataset(train_csv, bool(compose_label_reason_target))
    eval_use_label_reason = (
        bool(compose_label_reason_target)
        if eval_compose_label_reason_target is None
        else bool(eval_compose_label_reason_target)
    )
    eval_ds = _to_dataset(eval_csv, eval_use_label_reason) if eval_csv else None
    return train_ds, eval_ds


def build_text_only_datasets(
    train_csv: str,
    text_col: str,
    eval_csv: Optional[str] = None,
) -> Tuple[Dataset, Optional[Dataset]]:
    def _to_dataset(csv_path: str) -> Dataset:
        df = _read_csv_normalized(csv_path)
        if text_col not in df.columns:
            raise ValueError(f"Column {text_col} not found in {csv_path}")
        texts = df[text_col].fillna("").astype(str)
        rows = [{"text": text} for text in texts.tolist() if text.strip()]
        if not rows:
            raise ValueError(f"No non-empty texts found in {csv_path}")
        return Dataset.from_list(rows)

    train_ds = _to_dataset(train_csv)
    eval_ds = _to_dataset(eval_csv) if eval_csv else None
    return train_ds, eval_ds


def build_label_only_tokenized_datasets(
    train_csv: str,
    text_col: str,
    label_col: str,
    *,
    prompt_builder: Callable[[str, Optional[Any]], str],
    inference_prompt_builder: Callable[[str], str],
    tokenizer,
    max_length: int,
    eval_csv: Optional[str] = None,
    reason_col: Optional[str] = None,
    compose_label_reason_target: bool = False,
    eval_compose_label_reason_target: Optional[bool] = None,
    label_reason_target_format: str = "anchored_braced_reason",
    evidence_spans_col: str = "spans",
    evidences_col: str = "evidences",
    max_evidence_items: int = 3,
    include_evidence_span_offsets: bool = True,
    explanation_label_anchor: bool = False,
    explanation_label_anchor_modality: str = "sms",
) -> Tuple[Dataset, Optional[Dataset], Dict[str, int]]:
    stats = {
        "train_total": 0,
        "train_dropped_no_supervised_token": 0,
        "eval_total": 0,
        "eval_dropped_no_supervised_token": 0,
    }

    def _longest_common_prefix_len(a: List[int], b: List[int]) -> int:
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return i

    def _to_dataset(csv_path: str, split: str, use_label_reason_target: bool) -> Dataset:
        df = _read_csv_normalized(csv_path)
        required = [text_col, label_col]
        if use_label_reason_target:
            if not reason_col:
                raise ValueError("compose_label_reason_target=true requires data.reason_col")
            required.append(str(reason_col))
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Columns {missing} not found in {csv_path}")

        rows: List[Dict[str, Any]] = []
        total = 0
        dropped = 0

        if use_label_reason_target:
            spans_values = df[evidence_spans_col].tolist() if evidence_spans_col in df.columns else [None] * len(df)
            evidences_values = df[evidences_col].tolist() if evidences_col in df.columns else [None] * len(df)
            supervised_targets = [
                _compose_label_reason_target(
                    label_value=lbl,
                    reason_value=rsn,
                    target_format=label_reason_target_format,
                    source_text=text_value,
                    evidence_spans_value=spans_raw,
                    evidences_value=evidences_raw,
                    max_evidence_items=max_evidence_items,
                    include_evidence_span_offsets=include_evidence_span_offsets,
                    explanation_label_anchor=explanation_label_anchor,
                    explanation_label_anchor_modality=explanation_label_anchor_modality,
                )
                for lbl, rsn, text_value, spans_raw, evidences_raw in zip(
                    df[label_col],
                    df[str(reason_col)],
                    df[text_col].fillna("").astype(str),
                    spans_values,
                    evidences_values,
                )
            ]
        else:
            supervised_targets = df[label_col].astype(str).tolist()

        for text_value, label_value in zip(df[text_col].fillna("").astype(str), supervised_targets):
            total += 1
            full_text = prompt_builder(text_value, label_value)
            prefix_text = inference_prompt_builder(text_value)

            full_enc = tokenizer(
                full_text,
                truncation=True,
                max_length=max_length,
                add_special_tokens=True,
            )
            prefix_enc = tokenizer(
                prefix_text,
                truncation=True,
                max_length=max_length,
                add_special_tokens=True,
            )

            input_ids = list(full_enc["input_ids"])
            attention_mask = list(full_enc.get("attention_mask", [1] * len(input_ids)))
            prefix_ids = list(prefix_enc["input_ids"])

            # Length-based masking can fail when tokenizer keeps the same length
            # but mutates tail tokens after appending the label (e.g. "정답: " vs "정답: 0").
            prefix_len = _longest_common_prefix_len(prefix_ids, input_ids)

            labels = list(input_ids)
            for i in range(prefix_len):
                labels[i] = -100

            if all(v == -100 for v in labels):
                dropped += 1
                continue

            rows.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                    "length": len(input_ids),
                }
            )

        stats[f"{split}_total"] = total
        stats[f"{split}_dropped_no_supervised_token"] = dropped

        if not rows:
            raise ValueError(
                f"No usable rows in {csv_path}. "
                "All rows were truncated before the supervised label segment."
            )
        return Dataset.from_list(rows)

    train_ds = _to_dataset(train_csv, "train", bool(compose_label_reason_target))
    eval_use_label_reason = (
        bool(compose_label_reason_target)
        if eval_compose_label_reason_target is None
        else bool(eval_compose_label_reason_target)
    )
    eval_ds = _to_dataset(eval_csv, "eval", eval_use_label_reason) if eval_csv else None
    return train_ds, eval_ds, stats


def build_label_explanation_tokenized_datasets(
    train_csv: str,
    text_col: str,
    label_col: str,
    *,
    prompt_builder: Callable[[str, Optional[Any]], str],
    inference_prompt_builder: Callable[[str], str],
    tokenizer,
    max_length: int,
    eval_csv: Optional[str] = None,
    reason_col: Optional[str] = None,
    label_reason_target_format: str = "label_first_explanation",
    evidence_spans_col: str = "spans",
    evidences_col: str = "evidences",
    max_evidence_items: int = 3,
    include_evidence_span_offsets: bool = True,
    include_reconstruction_mask: bool = False,
    explanation_label_anchor: bool = False,
    explanation_label_anchor_modality: str = "sms",
) -> Tuple[Dataset, Optional[Dataset], Dict[str, int]]:
    stats = {
        "train_total": 0,
        "train_dropped_no_supervised_token": 0,
        "train_dropped_bad_label_token": 0,
        "eval_total": 0,
        "eval_dropped_no_supervised_token": 0,
        "eval_dropped_bad_label_token": 0,
    }

    zero_ids = tokenizer.encode("0", add_special_tokens=False)
    one_ids = tokenizer.encode("1", add_special_tokens=False)
    if len(zero_ids) != 1 or len(one_ids) != 1:
        raise ValueError(
            "2-way label CE requires tokenizer to encode '0' and '1' as single tokens. "
            f"got zero_ids={zero_ids}, one_ids={one_ids}"
        )
    zero_token_id = int(zero_ids[0])
    one_token_id = int(one_ids[0])

    def _longest_common_prefix_len(a: List[int], b: List[int]) -> int:
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return i

    def _to_dataset(csv_path: str, split: str) -> Dataset:
        df = _read_csv_normalized(csv_path)
        required = [text_col, label_col]
        if reason_col:
            required.append(str(reason_col))
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Columns {missing} not found in {csv_path}")

        rows: List[Dict[str, Any]] = []
        total = 0
        dropped_no_supervised = 0
        dropped_bad_label = 0

        reasons = (
            df[str(reason_col)].tolist()
            if reason_col and str(reason_col) in df.columns
            else [""] * len(df)
        )
        spans_values = df[evidence_spans_col].tolist() if evidence_spans_col in df.columns else [None] * len(df)
        evidences_values = df[evidences_col].tolist() if evidences_col in df.columns else [None] * len(df)

        for text_value, label_value, reason_value, spans_raw, evidences_raw in zip(
            df[text_col].fillna("").astype(str),
            df[label_col].tolist(),
            reasons,
            spans_values,
            evidences_values,
        ):
            total += 1
            try:
                label_int = int(label_value)
            except Exception:
                label_int = 0
            if label_int not in (0, 1):
                label_int = 0
            target_text = _compose_label_reason_target(
                label_value=label_int,
                reason_value=reason_value,
                target_format=label_reason_target_format,
                source_text=text_value,
                evidence_spans_value=spans_raw,
                evidences_value=evidences_raw,
                max_evidence_items=max_evidence_items,
                include_evidence_span_offsets=include_evidence_span_offsets,
                explanation_label_anchor=explanation_label_anchor,
                explanation_label_anchor_modality=explanation_label_anchor_modality,
            )
            full_text = prompt_builder(text_value, target_text)
            prefix_text = inference_prompt_builder(text_value)

            full_enc_kwargs: Dict[str, Any] = {
                "truncation": True,
                "max_length": max_length,
                "add_special_tokens": True,
            }
            if include_reconstruction_mask:
                full_enc_kwargs["return_offsets_mapping"] = True
            full_enc = tokenizer(full_text, **full_enc_kwargs)
            prefix_enc = tokenizer(
                prefix_text,
                truncation=True,
                max_length=max_length,
                add_special_tokens=True,
            )

            input_ids = list(full_enc["input_ids"])
            attention_mask = list(full_enc.get("attention_mask", [1] * len(input_ids)))
            offset_mapping = list(full_enc.get("offset_mapping", []))
            prefix_ids = list(prefix_enc["input_ids"])
            target_start_position = _longest_common_prefix_len(prefix_ids, input_ids)

            if target_start_position <= 0 or target_start_position >= len(input_ids):
                dropped_no_supervised += 1
                continue

            expected_label_token_id = zero_token_id if label_int == 0 else one_token_id
            fmt = str(label_reason_target_format or "").strip()
            if fmt in {"span_explanation_label", "evidence_explanation_label"}:
                label_position = _token_index_for_char(
                    offset_mapping=offset_mapping,
                    input_ids=input_ids,
                    char_start=_label_last_char_start(full_text=full_text, target_text=target_text),
                    expected_token_id=expected_label_token_id,
                )
                if label_position is None:
                    dropped_bad_label += 1
                    continue
            else:
                label_position = target_start_position
            if int(input_ids[label_position]) != expected_label_token_id:
                dropped_bad_label += 1
                continue

            labels = list(input_ids)
            for i in range(target_start_position):
                labels[i] = -100
            row = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "label_position": int(label_position),
                "target_start_position": int(target_start_position),
                "label_class": int(label_int),
                "length": len(input_ids),
            }
            if include_reconstruction_mask:
                explanation_start = _explanation_body_char_start(
                    full_text=full_text,
                    target_text=target_text,
                    label_value=label_int,
                    target_format=label_reason_target_format,
                )
                row["reconstruction_mask"] = _char_span_token_mask(
                    offset_mapping=offset_mapping,
                    start_char=explanation_start,
                    end_char=_explanation_body_char_end(
                        full_text=full_text,
                        target_text=target_text,
                        target_format=label_reason_target_format,
                    ),
                    min_token_idx=target_start_position,
                    length=len(input_ids),
                )
            rows.append(row)

        stats[f"{split}_total"] = total
        stats[f"{split}_dropped_no_supervised_token"] = dropped_no_supervised
        stats[f"{split}_dropped_bad_label_token"] = dropped_bad_label

        if not rows:
            raise ValueError(
                f"No usable rows in {csv_path}. "
                "All rows were truncated before the label token or violated the label-first token contract."
            )
        return Dataset.from_list(rows)

    train_ds = _to_dataset(train_csv, "train")
    eval_ds = _to_dataset(eval_csv, "eval") if eval_csv else None
    return train_ds, eval_ds, stats


def _infer_hidden_size_from_model(model: torch.nn.Module) -> int:
    config = getattr(model, "config", None)
    if config is None and hasattr(model, "base_model"):
        config = getattr(getattr(model, "base_model"), "config", None)
    for attr in ("hidden_size", "n_embd", "d_model"):
        value = getattr(config, attr, None)
        if isinstance(value, int) and value > 0:
            return int(value)
    raise ValueError("Could not infer hidden size from model config for joint evidence trainer")


def _mean_pool_hidden(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return summed / denom


def _attach_joint_evidence_heads(
    model: torch.nn.Module,
    *,
    enable_span_reranker: bool = False,
    enable_evidence_fusion: bool = False,
    enable_label_rationale_adapter: bool = False,
) -> torch.nn.Module:
    hidden_size = _infer_hidden_size_from_model(model)
    ref_device = torch.device("cpu")
    ref_dtype = torch.float32
    for param in model.parameters():
        ref_device = param.device
        if param.is_floating_point():
            ref_dtype = param.dtype
            break
    if not hasattr(model, "joint_binary_classifier"):
        model.add_module("joint_binary_classifier", torch.nn.Linear(hidden_size, 1))
    if not hasattr(model, "joint_evidence_classifier"):
        model.add_module("joint_evidence_classifier", torch.nn.Linear(hidden_size, 1))
    if not hasattr(model, "joint_reconstruction_classifier"):
        model.add_module("joint_reconstruction_classifier", torch.nn.Linear(hidden_size, 2))
    if enable_span_reranker and not hasattr(model, "joint_span_reranker"):
        model.add_module("joint_span_reranker", torch.nn.Linear(hidden_size, 1))
    if enable_evidence_fusion and not hasattr(model, "joint_evidence_fusion_proj"):
        fusion_proj = torch.nn.Linear(hidden_size, hidden_size)
        torch.nn.init.zeros_(fusion_proj.weight)
        torch.nn.init.zeros_(fusion_proj.bias)
        model.add_module("joint_evidence_fusion_proj", fusion_proj)
    if enable_evidence_fusion and not hasattr(model, "joint_evidence_fusion_gate"):
        fusion_gate = torch.nn.Linear(hidden_size, hidden_size)
        torch.nn.init.zeros_(fusion_gate.weight)
        torch.nn.init.constant_(fusion_gate.bias, -2.0)
        model.add_module("joint_evidence_fusion_gate", fusion_gate)
    if enable_label_rationale_adapter and not hasattr(model, "joint_label_rationale_embedding"):
        label_embedding = torch.nn.Embedding(2, hidden_size)
        torch.nn.init.normal_(label_embedding.weight, mean=0.0, std=0.02)
        model.add_module("joint_label_rationale_embedding", label_embedding)
    if enable_label_rationale_adapter and not hasattr(model, "joint_label_rationale_adapter"):
        label_adapter = torch.nn.Linear(hidden_size, hidden_size)
        torch.nn.init.zeros_(label_adapter.weight)
        torch.nn.init.zeros_(label_adapter.bias)
        model.add_module("joint_label_rationale_adapter", label_adapter)
    model.joint_binary_classifier.to(device=ref_device, dtype=ref_dtype)
    model.joint_evidence_classifier.to(device=ref_device, dtype=ref_dtype)
    model.joint_reconstruction_classifier.to(device=ref_device, dtype=ref_dtype)
    if enable_span_reranker:
        model.joint_span_reranker.to(device=ref_device, dtype=ref_dtype)
    if enable_evidence_fusion:
        model.joint_evidence_fusion_proj.to(device=ref_device, dtype=ref_dtype)
        model.joint_evidence_fusion_gate.to(device=ref_device, dtype=ref_dtype)
    if enable_label_rationale_adapter:
        model.joint_label_rationale_embedding.to(device=ref_device, dtype=ref_dtype)
        model.joint_label_rationale_adapter.to(device=ref_device, dtype=ref_dtype)
    return model


def _attach_label_explanation_reconstruction_head(model: torch.nn.Module) -> torch.nn.Module:
    hidden_size = _infer_hidden_size_from_model(model)
    ref_device = torch.device("cpu")
    ref_dtype = torch.float32
    for param in model.parameters():
        ref_device = param.device
        if param.is_floating_point():
            ref_dtype = param.dtype
            break
    if not hasattr(model, "label_explanation_reconstruction_classifier"):
        model.add_module(
            "label_explanation_reconstruction_classifier",
            torch.nn.Linear(hidden_size, 2),
        )
    model.label_explanation_reconstruction_classifier.to(device=ref_device, dtype=ref_dtype)
    return model


def _zero_from_module(module: torch.nn.Module, like: torch.Tensor) -> torch.Tensor:
    zero = like.sum() * 0.0
    for param in module.parameters():
        zero = zero + (param.sum() * 0.0)
    return zero


def _contiguous_true_segments(mask: torch.Tensor, *, max_segments: int = 0) -> List[Tuple[int, int]]:
    idx = torch.nonzero(mask, as_tuple=False).squeeze(-1).detach().cpu().tolist()
    if not idx:
        return []
    segments: List[Tuple[int, int]] = []
    start = int(idx[0])
    prev = int(idx[0])
    for value in idx[1:]:
        cur = int(value)
        if cur == prev + 1:
            prev = cur
            continue
        segments.append((start, prev + 1))
        if max_segments > 0 and len(segments) >= max_segments:
            return segments
        start = cur
        prev = cur
    segments.append((start, prev + 1))
    if max_segments > 0:
        segments = segments[:max_segments]
    return segments


def _span_overlaps_any(span: Tuple[int, int], spans: Sequence[Tuple[int, int]]) -> bool:
    st, ed = int(span[0]), int(span[1])
    for other_st, other_ed in spans:
        if not (ed <= int(other_st) or st >= int(other_ed)):
            return True
    return False


def _candidate_negative_spans(
    *,
    evidence_logits: torch.Tensor,
    prompt_mask: torch.Tensor,
    positive_mask: torch.Tensor,
    positive_spans: Sequence[Tuple[int, int]],
    widths: Sequence[int],
    count: int,
    max_width: int,
) -> List[Tuple[int, int]]:
    if count <= 0:
        return []
    valid_neg = (prompt_mask > 0) & (~positive_mask)
    neg_idx = torch.nonzero(valid_neg, as_tuple=False).squeeze(-1)
    if neg_idx.numel() == 0:
        return []

    width_values = [max(1, min(int(w), int(max_width))) for w in widths if int(w) > 0]
    if not width_values:
        width_values = [min(8, int(max_width))]

    masked_scores = evidence_logits.detach().float().masked_fill(~valid_neg, -1.0e9)
    k = min(int(count) * 3, int(neg_idx.numel()))
    top_idx = torch.topk(masked_scores, k=k).indices.detach().cpu().tolist()
    valid_positions = torch.nonzero(prompt_mask > 0, as_tuple=False).squeeze(-1)
    min_pos = int(valid_positions.min().item())
    max_pos_exclusive = int(valid_positions.max().item()) + 1

    out: List[Tuple[int, int]] = []
    seen = set()
    for rank, center_raw in enumerate(top_idx):
        center = int(center_raw)
        width = width_values[rank % len(width_values)]
        half = max(0, width // 2)
        st = max(min_pos, center - half)
        ed = min(max_pos_exclusive, st + width)
        st = max(min_pos, ed - width)
        if ed <= st:
            continue
        span = (int(st), int(ed))
        if span in seen or _span_overlaps_any(span, positive_spans):
            continue
        seen.add(span)
        out.append(span)
        if len(out) >= count:
            break
    return out


def _span_rerank_loss_and_summary(
    *,
    last_hidden: torch.Tensor,
    prompt_mask: torch.Tensor,
    evidence_mask: Optional[torch.Tensor],
    use_evidence_loss: Optional[torch.Tensor],
    evidence_logits: torch.Tensor,
    span_reranker: torch.nn.Module,
    margin: float,
    negatives_per_positive: int,
    max_positive_spans: int,
    max_negative_spans: int,
    max_width: int,
    detach_scores_for_summary: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden_size = int(last_hidden.shape[-1])
    summary = last_hidden.new_zeros((int(last_hidden.shape[0]), hidden_size))
    summary_active = last_hidden.new_zeros((int(last_hidden.shape[0]),), dtype=torch.bool)
    if evidence_mask is None or use_evidence_loss is None:
        return _zero_from_module(span_reranker, last_hidden), summary, summary_active

    total_loss = _zero_from_module(span_reranker, last_hidden)
    active = 0
    for row_idx in range(int(last_hidden.shape[0])):
        if int(use_evidence_loss[row_idx].item()) != 1:
            continue
        row_prompt = prompt_mask[row_idx] > 0
        row_pos = (evidence_mask[row_idx].to(row_prompt.device) > 0.5) & row_prompt
        positive_spans = _contiguous_true_segments(row_pos, max_segments=max_positive_spans)
        if not positive_spans:
            continue
        widths = [ed - st for st, ed in positive_spans]
        neg_count = min(
            int(max_negative_spans),
            max(1, int(negatives_per_positive) * len(positive_spans)),
        )
        negative_spans = _candidate_negative_spans(
            evidence_logits=evidence_logits[row_idx],
            prompt_mask=row_prompt,
            positive_mask=row_pos,
            positive_spans=positive_spans,
            widths=widths,
            count=neg_count,
            max_width=max_width,
        )
        if not negative_spans:
            continue

        candidate_spans = list(positive_spans) + list(negative_spans)
        span_reprs = []
        for st, ed in candidate_spans:
            span_reprs.append(last_hidden[row_idx, int(st) : int(ed)].mean(dim=0))
        span_repr = torch.stack(span_reprs, dim=0)
        weight = span_reranker.weight
        span_scores = span_reranker(span_repr.to(device=weight.device, dtype=weight.dtype)).squeeze(-1).float()
        pos_scores = span_scores[: len(positive_spans)]
        neg_scores = span_scores[len(positive_spans) :]
        pair_loss = F.relu(float(margin) - pos_scores.unsqueeze(1) + neg_scores.unsqueeze(0)).mean()
        total_loss = total_loss + pair_loss
        active += 1

        summary_scores = span_scores.detach() if detach_scores_for_summary else span_scores
        weights = torch.softmax(summary_scores, dim=0).to(span_repr.dtype)
        summary[row_idx] = (weights.unsqueeze(-1) * span_repr.to(summary.device)).sum(dim=0)
        summary_active[row_idx] = True

    if active == 0:
        return total_loss, summary, summary_active
    return total_loss / float(active), summary, summary_active


def _evidence_weighted_summary(
    *,
    last_hidden: torch.Tensor,
    prompt_mask: torch.Tensor,
    evidence_logits: torch.Tensor,
    detach_scores: bool,
) -> torch.Tensor:
    masked_logits = evidence_logits.float().masked_fill(prompt_mask <= 0, -1.0e9)
    if detach_scores:
        masked_logits = masked_logits.detach()
    weights = torch.softmax(masked_logits, dim=1).to(last_hidden.dtype)
    return (weights.unsqueeze(-1) * last_hidden).sum(dim=1)


def _evidence_confidence(
    *,
    prompt_mask: torch.Tensor,
    evidence_logits: torch.Tensor,
    detach_scores: bool,
) -> torch.Tensor:
    probs = torch.sigmoid(evidence_logits.float()).masked_fill(prompt_mask <= 0, 0.0)
    if detach_scores:
        probs = probs.detach()
    return probs.max(dim=1).values.clamp(0.0, 1.0)


def _apply_evidence_guided_fusion(
    *,
    last_hidden: torch.Tensor,
    evidence_summary: torch.Tensor,
    span_summary: Optional[torch.Tensor],
    span_summary_active: Optional[torch.Tensor],
    fusion_confidence: Optional[torch.Tensor],
    fusion_proj: torch.nn.Module,
    fusion_gate: torch.nn.Module,
    scale: float,
) -> torch.Tensor:
    summary = evidence_summary
    if span_summary is not None and span_summary_active is not None and bool(span_summary_active.any().item()):
        active = span_summary_active.to(summary.device).unsqueeze(-1)
        summary = torch.where(active, 0.5 * (summary + span_summary.to(summary.device, dtype=summary.dtype)), summary)
    proj_weight = fusion_proj.weight
    summary_delta = fusion_proj(summary.to(device=proj_weight.device, dtype=proj_weight.dtype))
    gate_weight = fusion_gate.weight
    gate = torch.sigmoid(fusion_gate(last_hidden.to(device=gate_weight.device, dtype=gate_weight.dtype)))
    delta = float(scale) * gate * summary_delta.unsqueeze(1)
    if fusion_confidence is not None:
        confidence = fusion_confidence.to(device=delta.device, dtype=delta.dtype).view(-1, 1, 1)
        delta = confidence * delta
    fused = last_hidden.to(device=summary_delta.device, dtype=summary_delta.dtype) + delta
    return fused.to(device=last_hidden.device, dtype=last_hidden.dtype)


def _apply_label_conditioned_rationale_adapter(
    *,
    last_hidden: torch.Tensor,
    label_classes: torch.Tensor,
    label_positions: torch.Tensor,
    label_embedding: torch.nn.Module,
    label_adapter: torch.nn.Module,
    scale: float,
) -> torch.Tensor:
    emb_weight = label_embedding.weight
    labels = label_classes.to(device=emb_weight.device).long().clamp(0, 1)
    label_delta = label_adapter(label_embedding(labels)).to(dtype=last_hidden.dtype)
    label_delta = label_delta.to(device=last_hidden.device)
    seq_len = int(last_hidden.shape[1])
    positions = torch.arange(seq_len, device=last_hidden.device).unsqueeze(0)
    rationale_mask = positions >= label_positions.to(device=last_hidden.device).unsqueeze(1)
    delta = float(scale) * label_delta.unsqueeze(1) * rationale_mask.unsqueeze(-1).to(last_hidden.dtype)
    return last_hidden + delta


def build_joint_evidence_explanation_tokenized_datasets(
    train_csv: str,
    text_col: str,
    label_col: str,
    *,
    prompt_builder: Callable[[str, Optional[Any]], str],
    inference_prompt_builder: Callable[[str], str],
    tokenizer,
    max_length: int,
    eval_csv: Optional[str] = None,
    reason_col: Optional[str] = None,
    label_reason_target_format: str = "label_first_explanation",
    evidence_spans_col: str = "spans",
    evidences_col: str = "evidences",
    use_evidence_loss_col: str = "use_evidence_loss",
    evidence_supervision_mode: str = "column",
    evidence_supervise_empty_negatives: bool = False,
    max_evidence_items: int = 3,
    include_evidence_span_offsets: bool = True,
    explanation_label_anchor: bool = False,
    explanation_label_anchor_modality: str = "sms",
) -> Tuple[Dataset, Optional[Dataset], Dict[str, int]]:
    supervision_mode = str(evidence_supervision_mode or "column").strip().lower()
    if supervision_mode in {"default", "use_column", "phishing_only", "phishing_only_or_column"}:
        supervision_mode = "column"
    if supervision_mode in {"all_span", "all_spans", "rationale_all_labels"}:
        supervision_mode = "allspan"
    if supervision_mode not in {"column", "allspan"}:
        raise ValueError("evidence_supervision_mode must be one of: column, allspan")
    stats = {
        "train_total": 0,
        "train_dropped_no_supervised_token": 0,
        "train_dropped_bad_label_token": 0,
        "train_evidence_active": 0,
        "eval_total": 0,
        "eval_dropped_no_supervised_token": 0,
        "eval_dropped_bad_label_token": 0,
        "eval_evidence_active": 0,
    }

    zero_ids = tokenizer.encode("0", add_special_tokens=False)
    one_ids = tokenizer.encode("1", add_special_tokens=False)
    if len(zero_ids) != 1 or len(one_ids) != 1:
        raise ValueError(
            "joint_evidence_explanation requires tokenizer to encode '0' and '1' as single tokens. "
            f"got zero_ids={zero_ids}, one_ids={one_ids}"
        )
    zero_token_id = int(zero_ids[0])
    one_token_id = int(one_ids[0])

    def _longest_common_prefix_len(a: List[int], b: List[int]) -> int:
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return i

    def _to_dataset(csv_path: str, split: str) -> Dataset:
        df = _read_csv_normalized(csv_path)
        required = [text_col, label_col]
        if reason_col:
            required.append(str(reason_col))
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Columns {missing} not found in {csv_path}")

        reasons = (
            df[str(reason_col)].tolist()
            if reason_col and str(reason_col) in df.columns
            else [""] * len(df)
        )
        spans_values = df[evidence_spans_col].tolist() if evidence_spans_col in df.columns else [None] * len(df)
        evidences_values = df[evidences_col].tolist() if evidences_col in df.columns else [None] * len(df)
        use_values = (
            df[use_evidence_loss_col].tolist()
            if use_evidence_loss_col in df.columns
            else [None] * len(df)
        )

        rows: List[Dict[str, Any]] = []
        total = 0
        dropped_no_supervised = 0
        dropped_bad_label = 0
        evidence_active = 0
        evidence_negative_only = 0

        for text_value, label_value, reason_value, spans_raw, evidences_raw, use_raw in zip(
            df[text_col].fillna("").astype(str),
            df[label_col].tolist(),
            reasons,
            spans_values,
            evidences_values,
            use_values,
        ):
            total += 1
            text_value = str(text_value)
            try:
                label_int = int(label_value)
            except Exception:
                label_int = 0
            if label_int not in (0, 1):
                label_int = 0
            target_text = _compose_label_reason_target(
                label_value=label_int,
                reason_value=reason_value,
                target_format=label_reason_target_format,
                source_text=text_value,
                evidence_spans_value=spans_raw,
                evidences_value=evidences_raw,
                max_evidence_items=max_evidence_items,
                include_evidence_span_offsets=include_evidence_span_offsets,
                explanation_label_anchor=explanation_label_anchor,
                explanation_label_anchor_modality=explanation_label_anchor_modality,
            )
            full_text = prompt_builder(text_value, target_text)
            prefix_text = inference_prompt_builder(text_value)

            full_enc = tokenizer(
                full_text,
                truncation=True,
                max_length=max_length,
                add_special_tokens=True,
                return_offsets_mapping=True,
            )
            prefix_enc = tokenizer(
                prefix_text,
                truncation=True,
                max_length=max_length,
                add_special_tokens=True,
            )

            input_ids = list(full_enc["input_ids"])
            attention_mask = list(full_enc.get("attention_mask", [1] * len(input_ids)))
            offset_mapping = list(full_enc.get("offset_mapping", []))
            prefix_ids = list(prefix_enc["input_ids"])
            target_start_position = _longest_common_prefix_len(prefix_ids, input_ids)

            if target_start_position <= 0 or target_start_position >= len(input_ids):
                dropped_no_supervised += 1
                continue

            expected_label_token_id = zero_token_id if label_int == 0 else one_token_id
            fmt = str(label_reason_target_format or "").strip()
            if fmt in {"span_explanation_label", "evidence_explanation_label"}:
                label_position = _token_index_for_char(
                    offset_mapping=offset_mapping,
                    input_ids=input_ids,
                    char_start=_label_last_char_start(full_text=full_text, target_text=target_text),
                    expected_token_id=expected_label_token_id,
                )
                if label_position is None:
                    dropped_bad_label += 1
                    continue
            else:
                label_position = target_start_position
            if int(input_ids[label_position]) != expected_label_token_id:
                dropped_bad_label += 1
                continue

            labels = list(input_ids)
            for i in range(target_start_position):
                labels[i] = -100
            explanation_start = _explanation_body_char_start(
                full_text=full_text,
                target_text=target_text,
                label_value=label_int,
                target_format=label_reason_target_format,
            )
            reconstruction_mask = _char_span_token_mask(
                offset_mapping=offset_mapping,
                start_char=explanation_start,
                end_char=_explanation_body_char_end(
                    full_text=full_text,
                    target_text=target_text,
                    target_format=label_reason_target_format,
                ),
                min_token_idx=target_start_position,
                length=len(input_ids),
            )

            spans = _decoder_normalize_spans(
                spans_raw=spans_raw,
                text=text_value,
                evidences_raw=evidences_raw,
            )
            evidence_mask = [0.0] * len(input_ids)
            use_evidence = 0
            if supervision_mode == "allspan":
                use_requested = bool(spans) or bool(evidence_supervise_empty_negatives)
            else:
                try:
                    use_requested = int(use_raw) == 1
                except Exception:
                    use_requested = False
            text_start = full_text.find(text_value)
            if use_requested and spans and text_start >= 0 and offset_mapping:
                shifted_spans = [(int(s) + text_start, int(e) + text_start) for s, e in spans]
                for token_idx, off in enumerate(offset_mapping):
                    if token_idx >= len(evidence_mask):
                        break
                    if not isinstance(off, (list, tuple)) or len(off) != 2:
                        continue
                    st, ed = int(off[0]), int(off[1])
                    if ed <= st:
                        continue
                    if token_idx >= target_start_position:
                        continue
                    for span_st, span_ed in shifted_spans:
                        if not (ed <= span_st or st >= span_ed):
                            evidence_mask[token_idx] = 1.0
                            break
                if any(v > 0.5 for v in evidence_mask):
                    use_evidence = 1
                    evidence_active += 1
            if use_requested and not spans and bool(evidence_supervise_empty_negatives):
                use_evidence = 1
                evidence_negative_only += 1

            rows.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                    "label_position": int(label_position),
                    "target_start_position": int(target_start_position),
                    "label_class": int(label_int),
                    "evidence_mask": evidence_mask,
                    "reconstruction_mask": reconstruction_mask,
                    "use_evidence_loss": int(use_evidence),
                    "length": len(input_ids),
                }
            )

        stats[f"{split}_total"] = total
        stats[f"{split}_dropped_no_supervised_token"] = dropped_no_supervised
        stats[f"{split}_dropped_bad_label_token"] = dropped_bad_label
        stats[f"{split}_evidence_active"] = evidence_active
        stats[f"{split}_evidence_negative_only"] = evidence_negative_only
        if not rows:
            raise ValueError(
                f"No usable rows in {csv_path}. "
                "All rows were truncated before the label token or violated the label-first token contract."
            )
        return Dataset.from_list(rows)

    train_ds = _to_dataset(train_csv, "train")
    eval_ds = _to_dataset(eval_csv, "eval") if eval_csv else None
    return train_ds, eval_ds, stats


class LabelOnlyDataCollator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        labels_list = [list(f["labels"]) for f in features]
        model_features = [
            {
                "input_ids": list(f["input_ids"]),
                "attention_mask": list(f.get("attention_mask", [1] * len(f["input_ids"]))),
            }
            for f in features
        ]

        batch = self.tokenizer.pad(
            model_features,
            padding=True,
            return_tensors="pt",
        )
        batch.pop("token_type_ids", None)
        max_len = int(batch["input_ids"].shape[1])

        padded_labels: List[List[int]] = []
        for labels in labels_list:
            pad_len = max_len - len(labels)
            if pad_len < 0:
                labels = labels[:max_len]
                pad_len = 0
            padded_labels.append(labels + ([-100] * pad_len))
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


class LabelExplanationDataCollator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        labels_list = [list(f["labels"]) for f in features]
        reconstruction_masks = [list(f.get("reconstruction_mask", [])) for f in features]
        label_positions = [int(f["label_position"]) for f in features]
        target_start_positions = [int(f.get("target_start_position", f["label_position"])) for f in features]
        label_classes = [int(f["label_class"]) for f in features]
        model_features = [
            {
                "input_ids": list(f["input_ids"]),
                "attention_mask": list(f.get("attention_mask", [1] * len(f["input_ids"]))),
            }
            for f in features
        ]

        batch = self.tokenizer.pad(
            model_features,
            padding=True,
            return_tensors="pt",
        )
        batch.pop("token_type_ids", None)
        max_len = int(batch["input_ids"].shape[1])

        padded_labels: List[List[int]] = []
        padded_reconstruction_masks: List[List[float]] = []
        for labels, reconstruction_mask in zip(labels_list, reconstruction_masks):
            if len(reconstruction_mask) != len(labels):
                reconstruction_mask = [0.0] * len(labels)
            pad_len = max_len - len(labels)
            if pad_len < 0:
                labels = labels[:max_len]
                reconstruction_mask = reconstruction_mask[:max_len]
                pad_len = 0
            padded_labels.append(labels + ([-100] * pad_len))
            padded_reconstruction_masks.append(reconstruction_mask + ([0.0] * pad_len))
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        batch["label_positions"] = torch.tensor(label_positions, dtype=torch.long)
        batch["target_start_positions"] = torch.tensor(target_start_positions, dtype=torch.long)
        batch["label_classes"] = torch.tensor(label_classes, dtype=torch.long)
        batch["reconstruction_mask"] = torch.tensor(padded_reconstruction_masks, dtype=torch.float32)
        return batch


class JointEvidenceExplanationDataCollator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        labels_list = [list(f["labels"]) for f in features]
        evidence_masks = [list(f.get("evidence_mask", [])) for f in features]
        reconstruction_masks = [list(f.get("reconstruction_mask", [])) for f in features]
        label_positions = [int(f["label_position"]) for f in features]
        target_start_positions = [int(f.get("target_start_position", f["label_position"])) for f in features]
        label_classes = [int(f["label_class"]) for f in features]
        use_evidence_loss = [int(f.get("use_evidence_loss", 0)) for f in features]
        model_features = [
            {
                "input_ids": list(f["input_ids"]),
                "attention_mask": list(f.get("attention_mask", [1] * len(f["input_ids"]))),
            }
            for f in features
        ]

        batch = self.tokenizer.pad(
            model_features,
            padding=True,
            return_tensors="pt",
        )
        batch.pop("token_type_ids", None)
        max_len = int(batch["input_ids"].shape[1])

        padded_labels: List[List[int]] = []
        padded_evidence_masks: List[List[float]] = []
        padded_reconstruction_masks: List[List[float]] = []
        for labels, evidence_mask, reconstruction_mask in zip(labels_list, evidence_masks, reconstruction_masks):
            if len(reconstruction_mask) != len(labels):
                reconstruction_mask = [0.0] * len(labels)
            pad_len = max_len - len(labels)
            if pad_len < 0:
                labels = labels[:max_len]
                evidence_mask = evidence_mask[:max_len]
                reconstruction_mask = reconstruction_mask[:max_len]
                pad_len = 0
            padded_labels.append(labels + ([-100] * pad_len))
            padded_evidence_masks.append(evidence_mask + ([0.0] * pad_len))
            padded_reconstruction_masks.append(reconstruction_mask + ([0.0] * pad_len))
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        batch["label_positions"] = torch.tensor(label_positions, dtype=torch.long)
        batch["target_start_positions"] = torch.tensor(target_start_positions, dtype=torch.long)
        batch["label_classes"] = torch.tensor(label_classes, dtype=torch.long)
        batch["evidence_mask"] = torch.tensor(padded_evidence_masks, dtype=torch.float32)
        batch["reconstruction_mask"] = torch.tensor(padded_reconstruction_masks, dtype=torch.float32)
        batch["use_evidence_loss"] = torch.tensor(use_evidence_loss, dtype=torch.float32)
        return batch


def _compute_label_eval_metrics(eval_pred) -> Dict[str, float]:
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    pred_arr = np.asarray(predictions)
    label_arr = np.asarray(labels)
    if pred_arr.ndim == 1:
        pred_classes = pred_arr.astype(np.int64)
    else:
        pred_classes = pred_arr.argmax(axis=-1).astype(np.int64)
    true_classes = label_arr.astype(np.int64)
    metrics = compute_binary_classification_metrics(
        true_classes.tolist(),
        pred_classes.tolist(),
    )
    return {
        "label_accuracy": float(metrics.get("accuracy", 0.0)),
        "label_macro_f1": float(metrics.get("macro_f1", 0.0)),
        "label_positive_recall": float(metrics.get("positive_recall", 0.0)),
    }


class LabelExplanationTrainer(Trainer):
    def __init__(
        self,
        *args,
        zero_token_id: int,
        one_token_id: int,
        label_loss_weight: float = 1.0,
        explanation_loss_weight: float = 0.1,
        reconstruction_loss_weight: float = 0.0,
        reconstruction_pooling: str = "mean",
        reconstruction_scope: str = "all",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.zero_token_id = int(zero_token_id)
        self.one_token_id = int(one_token_id)
        self.label_loss_weight = float(label_loss_weight)
        self.explanation_loss_weight = float(explanation_loss_weight)
        self.reconstruction_loss_weight = float(reconstruction_loss_weight)
        self.reconstruction_pooling = str(reconstruction_pooling or "mean").strip().lower()
        self.reconstruction_scope = str(reconstruction_scope or "all").strip().lower()
        if self.reconstruction_pooling not in {"mean", "last"}:
            raise ValueError("reconstruction_pooling must be one of: mean, last")
        if self.reconstruction_scope not in {"all", "explanation"}:
            raise ValueError("reconstruction_scope must be one of: all, explanation")
        if self.reconstruction_loss_weight > 0.0:
            _attach_label_explanation_reconstruction_head(self.model)

    def _forward_label_explanation(
        self,
        model,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        model_inputs = dict(inputs)
        labels = model_inputs.pop("labels")
        label_positions = model_inputs.pop("label_positions")
        target_start_positions = model_inputs.pop("target_start_positions", label_positions)
        label_classes = model_inputs.pop("label_classes")
        reconstruction_mask = model_inputs.pop("reconstruction_mask", None)

        if self.reconstruction_loss_weight > 0.0:
            model_inputs["output_hidden_states"] = True
        outputs = model(**model_inputs)
        hidden_states = getattr(outputs, "hidden_states", None)
        last_hidden = hidden_states[-1] if hidden_states else None
        logits = outputs.logits
        if logits.ndim != 3:
            raise ValueError(f"Expected 3D logits, got shape={tuple(logits.shape)}")

        pred_positions = label_positions - 1
        if torch.any(pred_positions < 0):
            raise ValueError("label_positions must be >= 1 for causal next-token label loss.")

        batch_indices = torch.arange(logits.shape[0], device=logits.device)
        label_step_logits = logits[batch_indices, pred_positions]
        label_logits = torch.stack(
            (
                label_step_logits[:, self.zero_token_id],
                label_step_logits[:, self.one_token_id],
            ),
            dim=-1,
        )
        label_loss = F.cross_entropy(label_logits.float(), label_classes.long())

        explanation_labels = labels.clone()
        explanation_labels[batch_indices, label_positions] = -100
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = explanation_labels[:, 1:].contiguous()
        valid_explanation = shift_labels.ne(-100)
        if bool(valid_explanation.any().item()):
            explanation_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)).float(),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        else:
            explanation_loss = label_loss.new_zeros(())

        seq_len = int(labels.shape[1])
        token_arange = torch.arange(seq_len, device=labels.device).unsqueeze(0)
        explanation_token_mask = labels.ne(-100) & (token_arange >= target_start_positions.unsqueeze(1))
        explanation_token_mask[batch_indices, label_positions] = False
        reconstruction_token_mask = explanation_token_mask
        if self.reconstruction_scope == "explanation" and reconstruction_mask is not None:
            reconstruction_token_mask = (
                reconstruction_mask.to(device=labels.device).bool()
                & labels.ne(-100)
                & (token_arange >= target_start_positions.unsqueeze(1))
            )
            reconstruction_token_mask[batch_indices, label_positions] = False

        reconstruction_loss = label_loss.new_zeros(())
        if self.reconstruction_loss_weight > 0.0:
            if last_hidden is None:
                raise RuntimeError("label explanation reconstruction requires model outputs.hidden_states")
            head_model = _unwrap_model(model)
            reconstruction_head = head_model.label_explanation_reconstruction_classifier
            reconstruction_loss = _zero_from_module(reconstruction_head, last_hidden)
            if bool(reconstruction_token_mask.any().item()):
                reconstruction_weight = reconstruction_head.weight
                if self.reconstruction_pooling == "last":
                    active_rows = reconstruction_token_mask.any(dim=1)
                    active_indices = torch.nonzero(active_rows, as_tuple=False).squeeze(-1)
                    masked_positions = reconstruction_token_mask.long() * token_arange
                    last_positions = masked_positions.max(dim=1).values
                    rec_hidden = last_hidden[active_indices, last_positions[active_indices]]
                    rec_labels = label_classes[active_indices]
                else:
                    rec_mask = reconstruction_token_mask.long()
                    active_rows = rec_mask.sum(dim=1) > 0
                    active_indices = torch.nonzero(active_rows, as_tuple=False).squeeze(-1)
                    rec_hidden_all = _mean_pool_hidden(last_hidden=last_hidden, attention_mask=rec_mask)
                    rec_hidden = rec_hidden_all[active_indices]
                    rec_labels = label_classes[active_indices]
                rec_logits = reconstruction_head(
                    rec_hidden.to(device=reconstruction_weight.device, dtype=reconstruction_weight.dtype)
                )
                reconstruction_loss = F.cross_entropy(rec_logits.float(), rec_labels.to(rec_logits.device).long())

        loss = (
            (self.label_loss_weight * label_loss)
            + (self.explanation_loss_weight * explanation_loss)
            + (self.reconstruction_loss_weight * reconstruction_loss)
        )
        return {
            "loss": loss,
            "label_loss": label_loss,
            "explanation_loss": explanation_loss,
            "reconstruction_loss": reconstruction_loss,
            "label_logits": label_logits,
            "label_classes": label_classes,
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        output = self._forward_label_explanation(model, inputs)
        if return_outputs:
            return output["loss"], {
                "logits": output["label_logits"],
                "label_loss": output["label_loss"],
                "explanation_loss": output["explanation_loss"],
                "reconstruction_loss": output["reconstruction_loss"],
            }
        return output["loss"]

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):
        has_labels = "labels" in inputs and "label_positions" in inputs and "label_classes" in inputs
        inputs = self._prepare_inputs(inputs)
        if not has_labels:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

        with torch.no_grad():
            with self.compute_loss_context_manager():
                output = self._forward_label_explanation(model, inputs)

        loss = output["label_loss"].mean().detach()
        if prediction_loss_only:
            return loss, None, None
        logits = output["label_logits"].detach()
        labels = output["label_classes"].detach()
        return loss, logits, labels

    def evaluation_loop(self, *args, **kwargs):
        metric_key_prefix = str(kwargs.get("metric_key_prefix", "eval"))
        output = super().evaluation_loop(*args, **kwargs)
        loss_key = f"{metric_key_prefix}_loss"
        if loss_key in output.metrics:
            output.metrics[f"{metric_key_prefix}_label_loss"] = float(output.metrics[loss_key])
        return output


class JointEvidenceExplanationTrainer(LabelExplanationTrainer):
    _COMPONENT_LOSS_NAMES = (
        "label",
        "explanation",
        "classification",
        "evidence",
        "reconstruction",
        "span_rerank",
    )

    def __init__(
        self,
        *args,
        classification_loss_weight: float = 0.5,
        evidence_loss_weight: float = 1.0,
        class_weights: Optional[Sequence[float]] = None,
        evidence_alpha: float = 1.0,
        evidence_beta: float = 1.0,
        evidence_negative_downsample_ratio: int = 8,
        evidence_negative_only_loss: bool = False,
        evidence_negative_only_max_tokens: int = 128,
        reconstruction_loss_weight: float = 0.0,
        reconstruction_pooling: str = "mean",
        reconstruction_scope: str = "all",
        reconstruction_adaptive: bool = False,
        reconstruction_adaptive_target_share: float = 0.02,
        reconstruction_adaptive_max_weight: float = 2.0,
        reconstruction_margin: float = 0.0,
        reconstruction_margin_weight: float = 0.0,
        span_rerank_loss_weight: float = 0.0,
        span_rerank_margin: float = 1.0,
        span_rerank_negatives_per_positive: int = 4,
        span_rerank_max_positive_spans: int = 3,
        span_rerank_max_negative_spans: int = 12,
        span_rerank_max_width: int = 32,
        evidence_guided_fusion: bool = False,
        evidence_guided_fusion_detach_scores: bool = True,
        evidence_guided_fusion_scale: float = 0.1,
        evidence_guided_fusion_confidence_gate: bool = False,
        label_conditioned_rationale: bool = False,
        label_conditioned_rationale_scale: float = 0.05,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.span_rerank_loss_weight = float(span_rerank_loss_weight)
        self.span_rerank_margin = float(span_rerank_margin)
        self.span_rerank_negatives_per_positive = int(span_rerank_negatives_per_positive)
        self.span_rerank_max_positive_spans = int(span_rerank_max_positive_spans)
        self.span_rerank_max_negative_spans = int(span_rerank_max_negative_spans)
        self.span_rerank_max_width = int(span_rerank_max_width)
        self.evidence_guided_fusion = bool(evidence_guided_fusion)
        self.evidence_guided_fusion_detach_scores = bool(evidence_guided_fusion_detach_scores)
        self.evidence_guided_fusion_scale = float(evidence_guided_fusion_scale)
        self.evidence_guided_fusion_confidence_gate = bool(evidence_guided_fusion_confidence_gate)
        self.label_conditioned_rationale = bool(label_conditioned_rationale)
        self.label_conditioned_rationale_scale = float(label_conditioned_rationale_scale)
        if self.label_conditioned_rationale and (
            self.span_rerank_loss_weight > 0.0 or self.evidence_guided_fusion
        ):
            raise ValueError(
                "label_conditioned_rationale is the active V1 path and should not be combined "
                "with archived span rerank/evidence fusion options."
            )
        head_model = _attach_joint_evidence_heads(
            self.model,
            enable_span_reranker=self.span_rerank_loss_weight > 0.0 or self.evidence_guided_fusion,
            enable_evidence_fusion=self.evidence_guided_fusion,
            enable_label_rationale_adapter=self.label_conditioned_rationale,
        )
        self.classification_loss_weight = float(classification_loss_weight)
        self.evidence_loss_weight = float(evidence_loss_weight)
        self.evidence_alpha = float(evidence_alpha)
        self.evidence_beta = float(evidence_beta)
        self.evidence_negative_downsample_ratio = int(evidence_negative_downsample_ratio)
        self.evidence_negative_only_loss = bool(evidence_negative_only_loss)
        self.evidence_negative_only_max_tokens = int(evidence_negative_only_max_tokens)
        self.reconstruction_loss_weight = float(reconstruction_loss_weight)
        self.reconstruction_pooling = str(reconstruction_pooling or "mean").strip().lower()
        self.reconstruction_scope = str(reconstruction_scope or "all").strip().lower()
        self.reconstruction_adaptive = bool(reconstruction_adaptive)
        self.reconstruction_adaptive_target_share = float(reconstruction_adaptive_target_share)
        self.reconstruction_adaptive_max_weight = float(reconstruction_adaptive_max_weight)
        self.reconstruction_margin = float(reconstruction_margin)
        self.reconstruction_margin_weight = float(reconstruction_margin_weight)
        if self.reconstruction_adaptive:
            if self.reconstruction_loss_weight <= 0.0:
                raise ValueError("reconstruction_adaptive requires reconstruction_loss_weight > 0")
            if self.reconstruction_adaptive_target_share <= 0.0:
                raise ValueError("reconstruction_adaptive_target_share must be > 0")
            if self.reconstruction_adaptive_max_weight < self.reconstruction_loss_weight:
                raise ValueError("reconstruction_adaptive_max_weight must be >= reconstruction_loss_weight")
        if self.reconstruction_margin < 0.0:
            raise ValueError("reconstruction_margin must be >= 0")
        if self.reconstruction_margin_weight < 0.0:
            raise ValueError("reconstruction_margin_weight must be >= 0")
        if self.reconstruction_margin_weight > 0.0 and self.reconstruction_loss_weight <= 0.0:
            raise ValueError("reconstruction_margin_weight requires reconstruction_loss_weight > 0")
        if self.reconstruction_loss_weight <= 0.0 and hasattr(head_model, "joint_reconstruction_classifier"):
            head_model.joint_reconstruction_classifier.requires_grad_(False)
        if self.reconstruction_pooling not in {"mean", "last"}:
            raise ValueError("reconstruction_pooling must be one of: mean, last")
        if self.reconstruction_scope not in {"all", "explanation"}:
            raise ValueError("reconstruction_scope must be one of: all, explanation")
        self.class_weights = (
            torch.tensor(class_weights, dtype=torch.float32)
            if class_weights is not None
            else None
        )
        self._component_loss_sums: Dict[str, float] = {}
        self._component_loss_count = 0

    def _component_loss_weights(self) -> Dict[str, float]:
        return {
            "label": float(self.label_loss_weight),
            "explanation": float(self.explanation_loss_weight),
            "classification": float(self.classification_loss_weight),
            "evidence": float(self.evidence_loss_weight),
            "reconstruction": float(self.reconstruction_loss_weight),
            "span_rerank": float(self.span_rerank_loss_weight),
        }

    def _record_component_losses(self, output: Dict[str, torch.Tensor]) -> None:
        raw_values = {
            "label": output["label_loss"],
            "explanation": output["explanation_loss"],
            "classification": output["classification_loss"],
            "evidence": output["evidence_loss"],
            "reconstruction": output["reconstruction_loss"],
            "span_rerank": output["span_rerank_loss"],
        }
        weights = self._component_loss_weights()
        weighted_total = 0.0
        row: Dict[str, float] = {}
        for name in self._COMPONENT_LOSS_NAMES:
            raw = float(raw_values[name].detach().float().mean().item())
            weighted = raw * float(weights[name])
            row[f"component_loss/raw_{name}"] = raw
            row[f"component_loss/weighted_{name}"] = weighted
            weighted_total += weighted
        row["component_loss/weighted_total"] = weighted_total
        row["component_loss/fusion_scale"] = float(self.evidence_guided_fusion_scale if self.evidence_guided_fusion else 0.0)
        row["component_loss/fusion_enabled"] = float(1.0 if self.evidence_guided_fusion else 0.0)
        row["component_loss/fusion_confidence_gate"] = float(
            1.0 if self.evidence_guided_fusion and self.evidence_guided_fusion_confidence_gate else 0.0
        )
        row["component_loss/evidence_negative_only_loss"] = float(1.0 if self.evidence_negative_only_loss else 0.0)
        row["component_loss/reconstruction_scope_explanation"] = float(
            1.0 if self.reconstruction_scope == "explanation" else 0.0
        )
        reconstruction_effective_weight = output.get("reconstruction_effective_weight")
        if reconstruction_effective_weight is not None:
            row["component_loss/reconstruction_effective_weight"] = float(
                reconstruction_effective_weight.detach().float().mean().item()
            )
            row["component_loss/weighted_reconstruction"] = (
                row["component_loss/raw_reconstruction"]
                * row["component_loss/reconstruction_effective_weight"]
            )
            weighted_total = sum(
                row[f"component_loss/weighted_{name}"]
                for name in self._COMPONENT_LOSS_NAMES
            )
            row["component_loss/weighted_total"] = weighted_total
        else:
            row["component_loss/reconstruction_effective_weight"] = float(self.reconstruction_loss_weight)
        reconstruction_ce_loss = output.get("reconstruction_ce_loss")
        if reconstruction_ce_loss is not None:
            row["component_loss/raw_reconstruction_ce"] = float(
                reconstruction_ce_loss.detach().float().mean().item()
            )
        reconstruction_margin_loss = output.get("reconstruction_margin_loss")
        if reconstruction_margin_loss is not None:
            row["component_loss/raw_reconstruction_margin"] = float(
                reconstruction_margin_loss.detach().float().mean().item()
            )
            row["component_loss/weighted_reconstruction_margin"] = (
                row["component_loss/raw_reconstruction_margin"]
                * float(self.reconstruction_margin_weight)
                * row["component_loss/reconstruction_effective_weight"]
            )
        row["component_loss/reconstruction_margin"] = float(self.reconstruction_margin)
        row["component_loss/reconstruction_margin_weight"] = float(self.reconstruction_margin_weight)
        row["component_loss/reconstruction_adaptive_enabled"] = float(1.0 if self.reconstruction_adaptive else 0.0)
        row["component_loss/reconstruction_target_share"] = float(
            self.reconstruction_adaptive_target_share if self.reconstruction_adaptive else 0.0
        )
        row["component_loss/label_conditioned_rationale"] = float(1.0 if self.label_conditioned_rationale else 0.0)
        row["component_loss/label_conditioned_rationale_scale"] = float(
            self.label_conditioned_rationale_scale if self.label_conditioned_rationale else 0.0
        )
        if abs(weighted_total) > 1e-12:
            for name in self._COMPONENT_LOSS_NAMES:
                row[f"component_loss/share_{name}"] = row[f"component_loss/weighted_{name}"] / weighted_total

        for key, value in row.items():
            self._component_loss_sums[key] = self._component_loss_sums.get(key, 0.0) + float(value)
        self._component_loss_count += 1

    def _pop_component_loss_logs(self) -> Dict[str, float]:
        if self._component_loss_count <= 0 or not self._component_loss_sums:
            return {}

        keys = sorted(self._component_loss_sums)
        sums = [float(self._component_loss_sums[key]) for key in keys]
        count = float(self._component_loss_count)
        logs = {
            key: float(value / max(1.0, count))
            for key, value in zip(keys, sums)
        }
        self._component_loss_sums = {}
        self._component_loss_count = 0
        return logs

    def log(self, logs: Dict[str, Any], *args, **kwargs):
        logs = dict(logs)
        logs.update(self._pop_component_loss_logs())
        return super().log(logs, *args, **kwargs)

    def _forward_label_explanation(
        self,
        model,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        model_inputs = dict(inputs)
        labels = model_inputs.pop("labels")
        label_positions = model_inputs.pop("label_positions")
        target_start_positions = model_inputs.pop("target_start_positions", label_positions)
        label_classes = model_inputs.pop("label_classes")
        evidence_mask = model_inputs.pop("evidence_mask", None)
        reconstruction_mask = model_inputs.pop("reconstruction_mask", None)
        use_evidence_loss = model_inputs.pop("use_evidence_loss", None)

        model_inputs["output_hidden_states"] = True
        outputs = model(**model_inputs)
        hidden_states = getattr(outputs, "hidden_states", None)
        if not hidden_states:
            raise RuntimeError("joint_evidence_explanation requires model outputs.hidden_states")
        last_hidden = hidden_states[-1]
        prompt_mask = model_inputs["attention_mask"].clone()
        seq_len = int(prompt_mask.shape[1])
        arange = torch.arange(seq_len, device=prompt_mask.device).unsqueeze(0)
        prompt_mask = prompt_mask * (arange < target_start_positions.unsqueeze(1)).long()
        head_model = _unwrap_model(model)
        pooled = _mean_pool_hidden(last_hidden=last_hidden, attention_mask=prompt_mask)
        classifier_head = head_model.joint_binary_classifier
        classifier_weight = classifier_head.weight
        classifier_logits = classifier_head(
            pooled.to(device=classifier_weight.device, dtype=classifier_weight.dtype)
        )
        class_weights = self.class_weights.to(classifier_logits.device) if self.class_weights is not None else None
        classification_loss, _ = _decoder_classification_loss(
            classifier_logits,
            label_classes,
            binary_classifier=True,
            class_weights=class_weights,
        )

        evidence_head = head_model.joint_evidence_classifier
        evidence_weight = evidence_head.weight
        evidence_logits = evidence_head(
            last_hidden.to(device=evidence_weight.device, dtype=evidence_weight.dtype)
        ).squeeze(-1)
        if evidence_mask is not None:
            evidence_mask = evidence_mask.to(evidence_logits.device)
        if use_evidence_loss is not None:
            use_evidence_loss = use_evidence_loss.to(evidence_logits.device)
        evidence_loss, _ = _decoder_evidence_token_loss(
            evidence_logits=evidence_logits,
            evidence_mask=evidence_mask,
            attention_mask=prompt_mask,
            use_evidence_loss=use_evidence_loss,
            alpha=self.evidence_alpha,
            beta=self.evidence_beta,
            negative_downsample_ratio=self.evidence_negative_downsample_ratio,
            allow_negative_only_samples=self.evidence_negative_only_loss,
            negative_only_max_tokens=self.evidence_negative_only_max_tokens,
        )

        span_rerank_loss = evidence_loss.new_zeros(())
        span_summary = None
        span_summary_active = None
        if self.span_rerank_loss_weight > 0.0 or self.evidence_guided_fusion:
            span_head = head_model.joint_span_reranker
            span_rerank_loss, span_summary, span_summary_active = _span_rerank_loss_and_summary(
                last_hidden=last_hidden,
                prompt_mask=prompt_mask,
                evidence_mask=evidence_mask,
                use_evidence_loss=use_evidence_loss,
                evidence_logits=evidence_logits,
                span_reranker=span_head,
                margin=self.span_rerank_margin,
                negatives_per_positive=self.span_rerank_negatives_per_positive,
                max_positive_spans=self.span_rerank_max_positive_spans,
                max_negative_spans=self.span_rerank_max_negative_spans,
                max_width=self.span_rerank_max_width,
                detach_scores_for_summary=self.evidence_guided_fusion_detach_scores,
            )

        logits = outputs.logits
        if self.evidence_guided_fusion:
            evidence_summary = _evidence_weighted_summary(
                last_hidden=last_hidden,
                prompt_mask=prompt_mask,
                evidence_logits=evidence_logits,
                detach_scores=self.evidence_guided_fusion_detach_scores,
            )
            fusion_confidence = (
                _evidence_confidence(
                    prompt_mask=prompt_mask,
                    evidence_logits=evidence_logits,
                    detach_scores=self.evidence_guided_fusion_detach_scores,
                )
                if self.evidence_guided_fusion_confidence_gate
                else None
            )
            fused_hidden = _apply_evidence_guided_fusion(
                last_hidden=last_hidden,
                evidence_summary=evidence_summary,
                span_summary=span_summary,
                span_summary_active=span_summary_active,
                fusion_confidence=fusion_confidence,
                fusion_proj=head_model.joint_evidence_fusion_proj,
                fusion_gate=head_model.joint_evidence_fusion_gate,
                scale=self.evidence_guided_fusion_scale,
            )
            output_embeddings = head_model.get_output_embeddings()
            if output_embeddings is None:
                raise RuntimeError("evidence_guided_fusion requires model.get_output_embeddings()")
            emb_weight = output_embeddings.weight
            logits = output_embeddings(fused_hidden.to(device=emb_weight.device, dtype=emb_weight.dtype))
        if self.label_conditioned_rationale:
            adapted_hidden = _apply_label_conditioned_rationale_adapter(
                last_hidden=last_hidden,
                label_classes=label_classes,
                label_positions=label_positions,
                label_embedding=head_model.joint_label_rationale_embedding,
                label_adapter=head_model.joint_label_rationale_adapter,
                scale=self.label_conditioned_rationale_scale,
            )
            output_embeddings = head_model.get_output_embeddings()
            if output_embeddings is None:
                raise RuntimeError("label_conditioned_rationale requires model.get_output_embeddings()")
            emb_weight = output_embeddings.weight
            logits = output_embeddings(adapted_hidden.to(device=emb_weight.device, dtype=emb_weight.dtype))
        if logits.ndim != 3:
            raise ValueError(f"Expected 3D logits, got shape={tuple(logits.shape)}")

        pred_positions = label_positions - 1
        if torch.any(pred_positions < 0):
            raise ValueError("label_positions must be >= 1 for causal next-token label loss.")

        batch_indices = torch.arange(logits.shape[0], device=logits.device)
        label_step_logits = logits[batch_indices, pred_positions]
        label_logits = torch.stack(
            (
                label_step_logits[:, self.zero_token_id],
                label_step_logits[:, self.one_token_id],
            ),
            dim=-1,
        )
        label_loss = F.cross_entropy(label_logits.float(), label_classes.long())

        explanation_labels = labels.clone()
        explanation_labels[batch_indices, label_positions] = -100
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = explanation_labels[:, 1:].contiguous()
        valid_explanation = shift_labels.ne(-100)
        if bool(valid_explanation.any().item()):
            explanation_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)).float(),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        else:
            explanation_loss = label_loss.new_zeros(())

        explanation_token_mask = labels.ne(-100) & (arange >= target_start_positions.unsqueeze(1))
        explanation_token_mask[batch_indices, label_positions] = False
        reconstruction_token_mask = explanation_token_mask
        if self.reconstruction_scope == "explanation":
            if reconstruction_mask is not None:
                reconstruction_token_mask = (
                    reconstruction_mask.to(device=labels.device).bool()
                    & labels.ne(-100)
                    & (arange >= target_start_positions.unsqueeze(1))
                )
                reconstruction_token_mask[batch_indices, label_positions] = False
        reconstruction_loss = label_loss.new_zeros(())
        reconstruction_ce_loss = label_loss.new_zeros(())
        reconstruction_margin_loss = label_loss.new_zeros(())
        if self.reconstruction_loss_weight > 0.0:
            reconstruction_head = head_model.joint_reconstruction_classifier
            # Keep DDP graph consistent across ranks even when a local batch has
            # no reconstruction tokens (common with small per-device batch sizes).
            reconstruction_loss = _zero_from_module(reconstruction_head, last_hidden)
            reconstruction_ce_loss = reconstruction_loss
            reconstruction_margin_loss = reconstruction_loss
            if bool(reconstruction_token_mask.any().item()):
                reconstruction_weight = reconstruction_head.weight
                if self.reconstruction_pooling == "last":
                    active_rows = reconstruction_token_mask.any(dim=1)
                    active_indices = torch.nonzero(active_rows, as_tuple=False).squeeze(-1)
                    masked_positions = reconstruction_token_mask.long() * arange
                    last_positions = masked_positions.max(dim=1).values
                    rec_hidden = last_hidden[active_indices, last_positions[active_indices]]
                    rec_labels = label_classes[active_indices]
                else:
                    rec_mask = reconstruction_token_mask.long()
                    active_rows = rec_mask.sum(dim=1) > 0
                    active_indices = torch.nonzero(active_rows, as_tuple=False).squeeze(-1)
                    rec_hidden_all = _mean_pool_hidden(last_hidden=last_hidden, attention_mask=rec_mask)
                    rec_hidden = rec_hidden_all[active_indices]
                    rec_labels = label_classes[active_indices]
                rec_logits = reconstruction_head(
                    rec_hidden.to(device=reconstruction_weight.device, dtype=reconstruction_weight.dtype)
                )
                rec_logits = rec_logits.float()
                rec_labels = rec_labels.to(rec_logits.device).long()
                reconstruction_ce_loss = F.cross_entropy(rec_logits, rec_labels)
                if self.reconstruction_margin_weight > 0.0 and self.reconstruction_margin > 0.0:
                    gold_logits = rec_logits.gather(1, rec_labels.unsqueeze(1)).squeeze(1)
                    other_logits = rec_logits.gather(1, (1 - rec_labels).unsqueeze(1)).squeeze(1)
                    margin_gap = gold_logits - other_logits
                    reconstruction_margin_loss = F.relu(
                        margin_gap.new_tensor(float(self.reconstruction_margin)) - margin_gap
                    ).mean()
                reconstruction_loss = reconstruction_ce_loss + (
                    float(self.reconstruction_margin_weight) * reconstruction_margin_loss
                )

        non_reconstruction_loss = (
            (self.classification_loss_weight * classification_loss)
            + (self.evidence_loss_weight * evidence_loss)
            + (self.label_loss_weight * label_loss)
            + (self.explanation_loss_weight * explanation_loss)
            + (self.span_rerank_loss_weight * span_rerank_loss)
        )
        reconstruction_effective_weight = reconstruction_loss.new_tensor(float(self.reconstruction_loss_weight))
        if self.reconstruction_adaptive and self.reconstruction_loss_weight > 0.0:
            adaptive_weight = (
                self.reconstruction_adaptive_target_share
                * non_reconstruction_loss.detach().float()
                / reconstruction_loss.detach().float().clamp_min(1e-8)
            )
            adaptive_weight = adaptive_weight.clamp(
                min=float(self.reconstruction_loss_weight),
                max=float(self.reconstruction_adaptive_max_weight),
            )
            reconstruction_effective_weight = adaptive_weight.to(
                device=reconstruction_loss.device,
                dtype=reconstruction_loss.dtype,
            )
        loss = non_reconstruction_loss + (reconstruction_effective_weight * reconstruction_loss)
        return {
            "loss": loss,
            "label_loss": label_loss,
            "explanation_loss": explanation_loss,
            "classification_loss": classification_loss,
            "evidence_loss": evidence_loss,
            "reconstruction_loss": reconstruction_loss,
            "reconstruction_ce_loss": reconstruction_ce_loss,
            "reconstruction_margin_loss": reconstruction_margin_loss,
            "reconstruction_effective_weight": reconstruction_effective_weight,
            "span_rerank_loss": span_rerank_loss,
            "label_logits": label_logits,
            "label_classes": label_classes,
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        output = self._forward_label_explanation(model, inputs)
        if bool(getattr(model, "training", False)):
            self._record_component_losses(output)
        if return_outputs:
            return output["loss"], {
                "logits": output["label_logits"],
                "label_loss": output["label_loss"],
                "explanation_loss": output["explanation_loss"],
                "classification_loss": output["classification_loss"],
                "evidence_loss": output["evidence_loss"],
                "reconstruction_loss": output["reconstruction_loss"],
                "reconstruction_effective_weight": output["reconstruction_effective_weight"],
                "span_rerank_loss": output["span_rerank_loss"],
            }
        return output["loss"]


def _to_dtype(dtype_str: Optional[str]) -> Optional[torch.dtype]:
    if not dtype_str or dtype_str == "auto":
        return None
    low = str(dtype_str).lower()
    if low in {"fp16", "float16", "half"}:
        return torch.float16
    if low in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if low in {"fp32", "float32"}:
        return torch.float32
    return None


def _resolve_quantization_config(
    model_cfg: Dict[str, Any],
    *,
    torch_dtype: Optional[torch.dtype],
) -> Optional[BitsAndBytesConfig]:
    load_in_4bit = bool(model_cfg.get("load_in_4bit", False))
    load_in_8bit = bool(model_cfg.get("load_in_8bit", False))
    if not load_in_4bit and not load_in_8bit:
        return None
    if load_in_4bit and load_in_8bit:
        raise ValueError("model.load_in_4bit and model.load_in_8bit cannot both be true")
    if BitsAndBytesConfig is None:
        raise ImportError("transformers BitsAndBytesConfig is required for k-bit loading.")
    if not torch.cuda.is_available():
        raise ValueError("k-bit quantized loading requires CUDA.")

    compute_dtype = _to_dtype(model_cfg.get("bnb_4bit_compute_dtype")) or torch_dtype or torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        bnb_4bit_quant_type=str(model_cfg.get("bnb_4bit_quant_type", "nf4")),
        bnb_4bit_use_double_quant=bool(model_cfg.get("bnb_4bit_use_double_quant", True)),
        bnb_4bit_compute_dtype=compute_dtype,
    )


def _uses_kbit_quantization(model_cfg: Dict[str, Any]) -> bool:
    return bool(model_cfg.get("load_in_4bit", False) or model_cfg.get("load_in_8bit", False))


def _resolve_quantized_device_map() -> Optional[Dict[str, int]]:
    if not torch.cuda.is_available():
        return None
    try:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    except Exception:
        local_rank = 0
    return {"": max(0, local_rank)}


def _maybe_peft_config(train_cfg: Dict[str, Any]) -> Optional[LoraConfig]:
    peft = (train_cfg or {}).get("peft", "").lower()
    if peft not in {"lora", "dora"}:
        return None
    if LoraConfig is None or TaskType is None:
        raise ImportError("peft is required for LoRA/Dora training but not installed.")
    r = int(train_cfg.get("lora_r", 8))
    alpha = int(train_cfg.get("lora_alpha", 16))
    dropout = float(train_cfg.get("lora_dropout", 0.05))
    target_modules = train_cfg.get(
        "lora_target_modules",
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    cfg_kwargs = {
        "r": r,
        "lora_alpha": alpha,
        "lora_dropout": dropout,
        "bias": train_cfg.get("lora_bias", "none"),
        "task_type": TaskType.CAUSAL_LM,
        "target_modules": target_modules,
    }
    sig = inspect.signature(LoraConfig)
    if "use_dora" in sig.parameters:
        cfg_kwargs["use_dora"] = peft == "dora"
    return LoraConfig(**cfg_kwargs)


def _create_tokenizer(model_name: str, hf_token: Optional[str]):
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=hf_token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def _load_base_model(
    model_name: str,
    hf_token: Optional[str],
    torch_dtype: Optional[torch.dtype],
    model_cfg: Optional[Dict[str, Any]] = None,
):
    model_cfg = model_cfg or {}
    kwargs = {"trust_remote_code": True, "token": hf_token}
    adapter_path = model_cfg.get("adapter_path")
    merge_adapter = bool(model_cfg.get("merge_adapter", False))
    quantization_config = _resolve_quantization_config(model_cfg, torch_dtype=torch_dtype)
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
        kwargs["device_map"] = _resolve_quantized_device_map()
        kwargs["low_cpu_mem_usage"] = True
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if adapter_path:
        if PeftModel is None:
            raise ImportError("peft is required to load adapter_path for warm-start training.")
        model = PeftModel.from_pretrained(model, str(adapter_path))
        if merge_adapter:
            model = model.merge_and_unload()
    return model


def _maybe_apply_peft(
    model,
    *,
    peft_cfg: Optional[LoraConfig],
    train_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
):
    if peft_cfg is None:
        if _uses_kbit_quantization(model_cfg):
            raise ValueError("k-bit quantized SFT requires train.peft=lora (QLoRA style training).")
        return model
    if get_peft_model is None:
        raise ImportError("peft is required for LoRA/Dora training but not installed.")

    if _uses_kbit_quantization(model_cfg):
        if prepare_model_for_kbit_training is None:
            raise ImportError("prepare_model_for_kbit_training is required for QLoRA but unavailable.")
        prep_kwargs = {}
        prep_sig = inspect.signature(prepare_model_for_kbit_training)
        if "use_gradient_checkpointing" in prep_sig.parameters:
            prep_kwargs["use_gradient_checkpointing"] = bool(train_cfg.get("gradient_checkpointing", False))
        model = prepare_model_for_kbit_training(model, **prep_kwargs)

    return get_peft_model(model, peft_cfg)


def _attach_trainer_processing_class(trainer_kwargs: Dict[str, Any], tokenizer) -> Dict[str, Any]:
    trainer_sig = inspect.signature(Trainer.__init__).parameters
    if "tokenizer" in trainer_sig:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_sig:
        trainer_kwargs["processing_class"] = tokenizer
    return trainer_kwargs


def train_with_trl(
    cfg: Dict[str, Any],
    run_dir: str,
    trainer_type: str,
    hf_token: Optional[str],
    deepspeed_cfg: Optional[str],
) -> str:
    text_col = _get_cfg(cfg, "data.text_col", "text")
    label_col = _get_cfg(cfg, "data.label_col", "label")
    target_col = _get_cfg(cfg, "data.target_col", label_col)
    reason_col = _get_cfg(cfg, "data.reason_col", None)
    compose_label_reason_target = bool(_get_cfg(cfg, "data.compose_label_reason_target", False))
    eval_compose_label_reason_target = _get_cfg(
        cfg, "data.eval_compose_label_reason_target", compose_label_reason_target
    )
    label_reason_target_format = str(
        _get_cfg(cfg, "data.label_reason_target_format", "anchored_braced_reason")
    )
    evidence_spans_col = str(_get_cfg(cfg, "data.evidence_spans_col", "spans"))
    evidences_col = str(_get_cfg(cfg, "data.evidences_col", "evidences"))
    use_evidence_loss_col = str(_get_cfg(cfg, "data.use_evidence_loss_col", "use_evidence_loss"))
    evidence_supervision_mode = str(
        _get_cfg(
            cfg,
            "data.evidence_supervision_mode",
            _get_cfg(cfg, "train.evidence_supervision_mode", "column"),
        )
    )
    evidence_supervise_empty_negatives = bool(
        _get_cfg(
            cfg,
            "data.evidence_supervise_empty_negatives",
            _get_cfg(cfg, "train.evidence_supervise_empty_negatives", False),
        )
    )
    max_evidence_items = int(_get_cfg(cfg, "data.max_evidence_items", 3))
    include_evidence_span_offsets = bool(_get_cfg(cfg, "data.include_evidence_span_offsets", True))
    explanation_label_anchor = bool(_get_cfg(cfg, "data.explanation_label_anchor", False))
    explanation_label_anchor_modality = str(_get_cfg(cfg, "data.explanation_label_anchor_modality", "sms"))
    metric_label_col = _get_cfg(cfg, "data.metric_label_col", label_col)
    prompt_cfg = _get_cfg(cfg, "prompt", {})
    prompt_builder = build_prompt_builder(prompt_cfg)
    train_csv = _get_cfg(cfg, "data.train_csv")
    eval_csv = _get_cfg(cfg, "data.eval_csv")
    if not train_csv:
        raise ValueError("data.train_csv is required.")
    if compose_label_reason_target and not reason_col:
        reason_col = target_col if target_col != label_col else "reason_value"

    # Backward compatibility: old configs with target_col=target_text can still
    # train on lean CSVs that only keep label/reason_value.
    if (not compose_label_reason_target) and str(target_col) == "target_text":
        try:
            train_preview = _read_csv_normalized(train_csv)
            if ("target_text" not in train_preview.columns) and {"label", "reason_value"}.issubset(set(train_preview.columns)):
                compose_label_reason_target = True
                reason_col = "reason_value"
        except Exception:
            pass

    supervised_col = label_col if compose_label_reason_target else target_col

    train_cfg = _get_cfg(cfg, "train", {})
    label_only_loss = bool(train_cfg.get("label_only_loss", False))
    joint_evidence_explanation = bool(train_cfg.get("joint_evidence_explanation", False))
    label_explanation_multitask = bool(
        train_cfg.get("label_explanation_multitask", False) or joint_evidence_explanation
    )
    label_loss_weight = float(train_cfg.get("label_loss_weight", 1.0))
    explanation_loss_weight = float(train_cfg.get("explanation_loss_weight", 0.1))
    peft_cfg = _maybe_peft_config(train_cfg)

    train_ds = eval_ds = None
    if not label_explanation_multitask:
        train_ds, eval_ds = build_sft_datasets(
            train_csv,
            text_col,
            supervised_col,
            prompt_builder,
            eval_csv,
            reason_col=reason_col,
            compose_label_reason_target=compose_label_reason_target,
            eval_compose_label_reason_target=eval_compose_label_reason_target,
            label_reason_target_format=label_reason_target_format,
            evidence_spans_col=evidence_spans_col,
            evidences_col=evidences_col,
            max_evidence_items=max_evidence_items,
            include_evidence_span_offsets=include_evidence_span_offsets,
            explanation_label_anchor=explanation_label_anchor,
            explanation_label_anchor_modality=explanation_label_anchor_modality,
        )

    model_type = _get_cfg(cfg, "model.model_type", "Decoder")
    model_cfg = _get_cfg(cfg, "model", {}) or {}
    model_name_cfg = _get_cfg(cfg, "model.model_name")
    model_id = _get_cfg(cfg, "model.model_id")
    model_name = resolve_model_name(model_type=model_type, model_name=model_name_cfg, model_id=model_id)
    torch_dtype = _to_dtype(_get_cfg(cfg, "model.dtype", "auto"))
    tokenizer = _create_tokenizer(model_name, hf_token)
    model = _load_base_model(model_name, hf_token, torch_dtype, model_cfg)

    run_seed = int(_get_cfg(cfg, "run.seed", 10))
    data_seed = int(_get_cfg(cfg, "run.data_seed", run_seed))
    max_length = int(train_cfg.get("max_length", 1024))
    batch_size = int(train_cfg.get("batch_size", 8))
    eval_batch_size = int(train_cfg.get("eval_batch_size", batch_size))
    dataloader_num_workers = int(train_cfg.get("dataloader_num_workers", 0))
    epochs = float(train_cfg.get("epochs", 1))
    lr = float(train_cfg.get("lr", 5e-5))
    grad_accum = int(train_cfg.get("grad_accum", 1))
    logging_steps = int(train_cfg.get("logging_steps", 10))
    save_steps = int(train_cfg.get("save_steps", 200))
    eval_steps = int(train_cfg.get("eval_steps", save_steps))
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.03))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    gradient_checkpointing = bool(train_cfg.get("gradient_checkpointing", False))
    length_bucket = bool(train_cfg.get("length_bucket", False))
    train_sampling_strategy = train_cfg.get("train_sampling_strategy")
    if train_sampling_strategy is None and length_bucket:
        train_sampling_strategy = "group_by_length"
    length_column_name = str(train_cfg.get("length_column_name", "length"))
    save_only_model = bool(train_cfg.get("save_only_model", False))
    ddp_find_unused_parameters = train_cfg.get("ddp_find_unused_parameters")
    load_best_model = bool(train_cfg.get("load_best_model_at_end", False))
    metric_for_best = train_cfg.get("metric_for_best_model")
    if metric_for_best is not None:
        metric_for_best = str(metric_for_best).strip() or None
    greater_is_better = train_cfg.get("greater_is_better")
    early_stop_patience = train_cfg.get("early_stopping_patience")
    if early_stop_patience is not None:
        early_stop_patience = int(early_stop_patience)
        if early_stop_patience <= 0:
            early_stop_patience = None
    early_stop_threshold = float(train_cfg.get("early_stopping_threshold", 0.0))
    early_stop_min_epochs = float(train_cfg.get("early_stopping_min_epochs", 0.0))
    early_stop_min_steps = int(train_cfg.get("early_stopping_min_steps", 0))
    eval_generate_metrics = bool(train_cfg.get("eval_generate_metrics", bool(eval_csv)))
    save_strategy = str(train_cfg.get("save_strategy", "steps")).lower()
    default_eval_strategy = "steps" if eval_csv else "no"
    eval_strategy_val = str(train_cfg.get("eval_strategy", default_eval_strategy)).lower()

    if not eval_csv:
        eval_strategy_val = "no"
    if not load_best_model and early_stop_patience is None:
        metric_for_best = None
        greater_is_better = None

    if (load_best_model or early_stop_patience is not None) and not eval_csv:
        raise ValueError("Early stopping / best-model loading requires data.eval_csv")
    if load_best_model and save_strategy != eval_strategy_val:
        raise ValueError("load_best_model_at_end requires matching save_strategy and eval_strategy")
    if (
        metric_for_best in {"eval_accuracy_strict", "eval_macro_f1_strict", "eval_positive_recall_strict"}
        and not eval_generate_metrics
    ):
        raise ValueError(
            "metric_for_best_model uses strict generation metric, but train.eval_generate_metrics is false."
        )

    # logging_dir: ensure uniqueness per run to avoid TensorBoard merge/overwrite
    user_logging_dir = train_cfg.get("logging_dir")
    if user_logging_dir:
        logging_dir = os.path.join(user_logging_dir, os.path.basename(run_dir))
    else:
        logging_dir = os.path.join(run_dir, "tb_logs")

    args_cls = SFTConfig if trainer_type == "trl" else TrainingArguments
    arg_kwargs = {
        "output_dir": run_dir,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": eval_batch_size,
        "gradient_accumulation_steps": grad_accum,
        "num_train_epochs": epochs,
        "learning_rate": lr,
        "seed": run_seed,
        "data_seed": data_seed,
        "warmup_ratio": warmup_ratio,
        "weight_decay": weight_decay,
        "logging_steps": logging_steps,
        # evaluation_strategy/ eval_strategy is set after we know which key is accepted
        "save_strategy": save_strategy,
        "save_total_limit": int(train_cfg.get("save_total_limit", 2)),
        "save_only_model": save_only_model,
        "fp16": torch_dtype == torch.float16,
        "bf16": torch_dtype == torch.bfloat16,
        "deepspeed": deepspeed_cfg,
        "load_best_model_at_end": load_best_model,
        "metric_for_best_model": metric_for_best,
        "greater_is_better": greater_is_better,
        # tensorboard is the sane default; can be overridden to ["none"] or ["wandb", ...]
        "report_to": train_cfg.get("report_to", "tensorboard"),
        "logging_dir": logging_dir,
        "dataloader_num_workers": dataloader_num_workers,
        "dataloader_pin_memory": bool(train_cfg.get("dataloader_pin_memory", torch.cuda.is_available())),
        "dataloader_persistent_workers": bool(train_cfg.get("dataloader_persistent_workers", dataloader_num_workers > 0)),
        "dataloader_prefetch_factor": (
            int(train_cfg.get("dataloader_prefetch_factor", 2)) if dataloader_num_workers > 0 else None
        ),
        "gradient_checkpointing": gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": False} if gradient_checkpointing else None,
        "remove_unused_columns": False,
        "train_sampling_strategy": train_sampling_strategy,
        "length_column_name": length_column_name,
    }
    if save_strategy == "steps":
        arg_kwargs["save_steps"] = save_steps
    if eval_strategy_val == "steps":
        arg_kwargs["eval_steps"] = eval_steps
    if ddp_find_unused_parameters is not None:
        arg_kwargs["ddp_find_unused_parameters"] = bool(ddp_find_unused_parameters)
    allowed = inspect.signature(args_cls).parameters
    # evaluation_strategy parameter is named differently in SFTConfig
    if "evaluation_strategy" in allowed:
        arg_kwargs["evaluation_strategy"] = eval_strategy_val
    elif "eval_strategy" in allowed:
        arg_kwargs["eval_strategy"] = eval_strategy_val
    # strip keys not supported by this args class
    arg_kwargs = {k: v for k, v in arg_kwargs.items() if k in allowed and v is not None}
    training_args = args_cls(**arg_kwargs)

    callbacks = []
    if eval_csv and eval_generate_metrics:
        callbacks.append(
            StrictGenerationValidationCallback(
                cfg=cfg,
                run_dir=run_dir,
                tokenizer=tokenizer,
                text_col=text_col,
                metric_label_col=metric_label_col,
                prompt_cfg=prompt_cfg,
            )
        )
    if early_stop_patience is not None:
        callbacks.append(
            WarmupEarlyStoppingCallback(
                early_stopping_patience=int(early_stop_patience),
                early_stopping_threshold=early_stop_threshold,
                min_epochs=early_stop_min_epochs,
                min_steps=early_stop_min_steps,
            )
        )
    if bool(train_cfg.get("write_epoch_classification_metrics", False)):
        callbacks.append(
            EpochClassificationEvalCallback(
                cfg=cfg,
                run_dir=run_dir,
                tokenizer=tokenizer,
                text_col=text_col,
                metric_label_col=metric_label_col,
                prompt_cfg=prompt_cfg,
            )
        )

    if label_explanation_multitask:
        if trainer_type != "hf":
            raise ValueError("train.label_explanation_multitask=true currently requires train.trainer=hf")
        if not compose_label_reason_target:
            raise ValueError("label_explanation_multitask requires data.compose_label_reason_target=true")
        if not reason_col:
            raise ValueError("label_explanation_multitask requires data.reason_col")

        inference_prompt_builder = build_inference_prompt_builder(prompt_cfg)
        zero_ids = tokenizer.encode("0", add_special_tokens=False)
        one_ids = tokenizer.encode("1", add_special_tokens=False)
        if len(zero_ids) != 1 or len(one_ids) != 1:
            raise ValueError(
                "label_explanation_multitask requires tokenizer to encode '0' and '1' as single tokens. "
                f"got zero_ids={zero_ids}, one_ids={one_ids}"
            )
        if joint_evidence_explanation:
            train_tok_ds, eval_tok_ds, label_stats = build_joint_evidence_explanation_tokenized_datasets(
                train_csv=train_csv,
                text_col=text_col,
                label_col=label_col,
                prompt_builder=prompt_builder,
                inference_prompt_builder=inference_prompt_builder,
                tokenizer=tokenizer,
                max_length=max_length,
                eval_csv=eval_csv,
                reason_col=reason_col,
                label_reason_target_format=label_reason_target_format,
                evidence_spans_col=evidence_spans_col,
                evidences_col=evidences_col,
                use_evidence_loss_col=use_evidence_loss_col,
                evidence_supervision_mode=evidence_supervision_mode,
                evidence_supervise_empty_negatives=evidence_supervise_empty_negatives,
                max_evidence_items=max_evidence_items,
                include_evidence_span_offsets=include_evidence_span_offsets,
                explanation_label_anchor=explanation_label_anchor,
                explanation_label_anchor_modality=explanation_label_anchor_modality,
            )
            print(
                "[joint_evidence_explanation] "
                f"train_kept={len(train_tok_ds)}/{label_stats['train_total']} "
                f"train_dropped_no_supervised={label_stats['train_dropped_no_supervised_token']} "
                f"train_dropped_bad_label={label_stats['train_dropped_bad_label_token']} "
                f"train_evidence_active={label_stats['train_evidence_active']} "
                f"train_evidence_negative_only={label_stats['train_evidence_negative_only']} "
                f"eval_kept={(len(eval_tok_ds) if eval_tok_ds is not None else 0)}/{label_stats['eval_total']} "
                f"eval_dropped_no_supervised={label_stats['eval_dropped_no_supervised_token']} "
                f"eval_dropped_bad_label={label_stats['eval_dropped_bad_label_token']} "
                f"eval_evidence_active={label_stats['eval_evidence_active']} "
                f"eval_evidence_negative_only={label_stats['eval_evidence_negative_only']} "
                f"evidence_supervision_mode={evidence_supervision_mode}"
            )
        else:
            train_tok_ds, eval_tok_ds, label_stats = build_label_explanation_tokenized_datasets(
                train_csv=train_csv,
                text_col=text_col,
                label_col=label_col,
                prompt_builder=prompt_builder,
                inference_prompt_builder=inference_prompt_builder,
                tokenizer=tokenizer,
                max_length=max_length,
                eval_csv=eval_csv,
                reason_col=reason_col,
                label_reason_target_format=label_reason_target_format,
                evidence_spans_col=evidence_spans_col,
                evidences_col=evidences_col,
                max_evidence_items=max_evidence_items,
                include_evidence_span_offsets=include_evidence_span_offsets,
                explanation_label_anchor=explanation_label_anchor,
                explanation_label_anchor_modality=explanation_label_anchor_modality,
                include_reconstruction_mask=(
                    float(train_cfg.get("reconstruction_loss_weight", 0.0)) > 0.0
                    and str(train_cfg.get("reconstruction_scope", "all")).strip().lower() == "explanation"
                ),
            )
            print(
                "[label_explanation_multitask] "
                f"train_kept={len(train_tok_ds)}/{label_stats['train_total']} "
                f"train_dropped_no_supervised={label_stats['train_dropped_no_supervised_token']} "
                f"train_dropped_bad_label={label_stats['train_dropped_bad_label_token']} "
                f"eval_kept={(len(eval_tok_ds) if eval_tok_ds is not None else 0)}/{label_stats['eval_total']} "
                f"eval_dropped_no_supervised={label_stats['eval_dropped_no_supervised_token']} "
                f"eval_dropped_bad_label={label_stats['eval_dropped_bad_label_token']}"
            )
        model = _maybe_apply_peft(model, peft_cfg=peft_cfg, train_cfg=train_cfg, model_cfg=model_cfg)
        collator = (
            JointEvidenceExplanationDataCollator(tokenizer=tokenizer)
            if joint_evidence_explanation
            else LabelExplanationDataCollator(tokenizer=tokenizer)
        )
        trainer_kwargs = _attach_trainer_processing_class(
            {
                "model": model,
                "args": training_args,
                "train_dataset": train_tok_ds,
                "eval_dataset": eval_tok_ds,
                "data_collator": collator,
                "callbacks": callbacks,
                "compute_metrics": _compute_label_eval_metrics,
            },
            tokenizer,
        )
        if joint_evidence_explanation:
            train_labels_for_weights = (
                _read_csv_normalized(train_csv)[label_col].fillna(0).astype(int).tolist()
            )
            class_weights = _decoder_compute_class_weights(train_labels_for_weights).tolist()
            trainer = JointEvidenceExplanationTrainer(
                zero_token_id=int(zero_ids[0]),
                one_token_id=int(one_ids[0]),
                label_loss_weight=label_loss_weight,
                explanation_loss_weight=explanation_loss_weight,
                classification_loss_weight=float(train_cfg.get("classification_loss_weight", 0.5)),
                evidence_loss_weight=float(train_cfg.get("evidence_loss_weight", 1.0)),
                class_weights=class_weights,
                evidence_alpha=float(train_cfg.get("evidence_alpha", 1.0)),
                evidence_beta=float(train_cfg.get("evidence_beta", 1.0)),
                evidence_negative_downsample_ratio=int(
                    train_cfg.get("evidence_negative_downsample_ratio", 8)
                ),
                evidence_negative_only_loss=bool(train_cfg.get("evidence_negative_only_loss", False)),
                evidence_negative_only_max_tokens=int(train_cfg.get("evidence_negative_only_max_tokens", 128)),
                reconstruction_loss_weight=float(train_cfg.get("reconstruction_loss_weight", 0.0)),
                reconstruction_pooling=str(train_cfg.get("reconstruction_pooling", "mean")),
                reconstruction_scope=str(train_cfg.get("reconstruction_scope", "all")),
                reconstruction_adaptive=bool(train_cfg.get("reconstruction_adaptive", False)),
                reconstruction_adaptive_target_share=float(
                    train_cfg.get("reconstruction_adaptive_target_share", 0.02)
                ),
                reconstruction_adaptive_max_weight=float(
                    train_cfg.get("reconstruction_adaptive_max_weight", 2.0)
                ),
                reconstruction_margin=float(train_cfg.get("reconstruction_margin", 0.0)),
                reconstruction_margin_weight=float(train_cfg.get("reconstruction_margin_weight", 0.0)),
                span_rerank_loss_weight=float(train_cfg.get("span_rerank_loss_weight", 0.0)),
                span_rerank_margin=float(train_cfg.get("span_rerank_margin", 1.0)),
                span_rerank_negatives_per_positive=int(
                    train_cfg.get("span_rerank_negatives_per_positive", 4)
                ),
                span_rerank_max_positive_spans=int(train_cfg.get("span_rerank_max_positive_spans", 3)),
                span_rerank_max_negative_spans=int(train_cfg.get("span_rerank_max_negative_spans", 12)),
                span_rerank_max_width=int(train_cfg.get("span_rerank_max_width", 32)),
                evidence_guided_fusion=bool(train_cfg.get("evidence_guided_fusion", False)),
                evidence_guided_fusion_detach_scores=bool(
                    train_cfg.get("evidence_guided_fusion_detach_scores", True)
                ),
                evidence_guided_fusion_scale=float(train_cfg.get("evidence_guided_fusion_scale", 0.1)),
                evidence_guided_fusion_confidence_gate=bool(
                    train_cfg.get("evidence_guided_fusion_confidence_gate", False)
                ),
                label_conditioned_rationale=bool(train_cfg.get("label_conditioned_rationale", False)),
                label_conditioned_rationale_scale=float(train_cfg.get("label_conditioned_rationale_scale", 0.05)),
                **trainer_kwargs,
            )
        else:
            trainer = LabelExplanationTrainer(
                zero_token_id=int(zero_ids[0]),
                one_token_id=int(one_ids[0]),
                label_loss_weight=label_loss_weight,
                explanation_loss_weight=explanation_loss_weight,
                reconstruction_loss_weight=float(train_cfg.get("reconstruction_loss_weight", 0.0)),
                reconstruction_pooling=str(train_cfg.get("reconstruction_pooling", "mean")),
                reconstruction_scope=str(train_cfg.get("reconstruction_scope", "all")),
                **trainer_kwargs,
            )
    elif label_only_loss:
        # Keep default behavior intact; only switch path when explicitly requested.
        print("[info] train.label_only_loss=true -> using label-only masked loss with HF Trainer.")
        inference_prompt_builder = build_inference_prompt_builder(prompt_cfg)
        train_tok_ds, eval_tok_ds, label_stats = build_label_only_tokenized_datasets(
            train_csv=train_csv,
            text_col=text_col,
            label_col=supervised_col,
            prompt_builder=prompt_builder,
            inference_prompt_builder=inference_prompt_builder,
            tokenizer=tokenizer,
            max_length=max_length,
            eval_csv=eval_csv,
            reason_col=reason_col,
            compose_label_reason_target=compose_label_reason_target,
            eval_compose_label_reason_target=eval_compose_label_reason_target,
            label_reason_target_format=label_reason_target_format,
            evidence_spans_col=evidence_spans_col,
            evidences_col=evidences_col,
            max_evidence_items=max_evidence_items,
            include_evidence_span_offsets=include_evidence_span_offsets,
            explanation_label_anchor=explanation_label_anchor,
            explanation_label_anchor_modality=explanation_label_anchor_modality,
        )
        print(
            "[label_only_loss] "
            f"train_kept={len(train_tok_ds)}/{label_stats['train_total']} "
            f"train_dropped={label_stats['train_dropped_no_supervised_token']} "
            f"eval_kept={(len(eval_tok_ds) if eval_tok_ds is not None else 0)}/{label_stats['eval_total']} "
            f"eval_dropped={label_stats['eval_dropped_no_supervised_token']}"
        )
        model = _maybe_apply_peft(model, peft_cfg=peft_cfg, train_cfg=train_cfg, model_cfg=model_cfg)
        collator = LabelOnlyDataCollator(tokenizer=tokenizer)
        trainer_kwargs = _attach_trainer_processing_class(
            {
                "model": model,
                "args": training_args,
                "train_dataset": train_tok_ds,
                "eval_dataset": eval_tok_ds,
                "data_collator": collator,
                "callbacks": callbacks,
            },
            tokenizer,
        )
        trainer = Trainer(**trainer_kwargs)
    elif trainer_type == "hf":
        model = _maybe_apply_peft(model, peft_cfg=peft_cfg, train_cfg=train_cfg, model_cfg=model_cfg)
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        def _tokenize_fn(batch):
            return tokenizer(
                batch["text"],
                padding="longest",
                truncation=True,
                max_length=max_length,
            )

        tokenized_train = train_ds.map(_tokenize_fn, batched=True, remove_columns=["text"])
        tokenized_eval = eval_ds.map(_tokenize_fn, batched=True, remove_columns=["text"]) if eval_ds else None
        trainer_kwargs = _attach_trainer_processing_class(
            {
                "model": model,
                "args": training_args,
                "train_dataset": tokenized_train,
                "eval_dataset": tokenized_eval,
                "data_collator": collator,
                "callbacks": callbacks,
            },
            tokenizer,
        )
        trainer = Trainer(**trainer_kwargs)
    else:
        model = _maybe_apply_peft(model, peft_cfg=peft_cfg, train_cfg=train_cfg, model_cfg=model_cfg)
        sft_sig = inspect.signature(SFTTrainer.__init__).parameters
        sft_kwargs = {
            "model": model,
            "train_dataset": train_ds,
            "eval_dataset": eval_ds,
            "args": training_args,
        }
        # tokenization arg name changed across TRL versions
        if "tokenizer" in sft_sig:
            sft_kwargs["tokenizer"] = tokenizer
        elif "processing_class" in sft_sig:
            sft_kwargs["processing_class"] = tokenizer

        if "dataset_text_field" in sft_sig:
            sft_kwargs["dataset_text_field"] = "text"
        elif "formatting_func" in sft_sig:
            # formatting_func should return string or list[str]; we return string to satisfy add_eos expectations
            sft_kwargs["formatting_func"] = lambda ex: " ".join(ex["text"]) if isinstance(ex["text"], list) else str(ex["text"])

        if "max_seq_length" in sft_sig:
            sft_kwargs["max_seq_length"] = max_length
        if "packing" in sft_sig:
            sft_kwargs["packing"] = False
        if "dataset_num_proc" in sft_sig:
            sft_kwargs["dataset_num_proc"] = None
        if callbacks:
            sft_kwargs["callbacks"] = callbacks

        trainer = SFTTrainer(**sft_kwargs)

    trainer.train()
    trainer.save_model(run_dir)
    tokenizer.save_pretrained(run_dir)
    return run_dir


def train_text_only_causal_lm(
    cfg: Dict[str, Any],
    run_dir: str,
    trainer_type: str,
    hf_token: Optional[str],
    deepspeed_cfg: Optional[str],
) -> str:
    text_col = _get_cfg(cfg, "data.text_col", "text")
    train_csv = _get_cfg(cfg, "data.train_csv")
    eval_csv = _get_cfg(cfg, "data.eval_csv")
    if not train_csv:
        raise ValueError("data.train_csv is required.")
    train_ds, eval_ds = build_text_only_datasets(train_csv, text_col, eval_csv)

    train_cfg = _get_cfg(cfg, "train", {})
    peft_cfg = _maybe_peft_config(train_cfg)

    model_type = _get_cfg(cfg, "model.model_type", "Decoder")
    model_cfg = _get_cfg(cfg, "model", {}) or {}
    model_name_cfg = _get_cfg(cfg, "model.model_name")
    model_id = _get_cfg(cfg, "model.model_id")
    model_name = resolve_model_name(model_type=model_type, model_name=model_name_cfg, model_id=model_id)
    torch_dtype = _to_dtype(_get_cfg(cfg, "model.dtype", "auto"))
    tokenizer = _create_tokenizer(model_name, hf_token)
    model = _load_base_model(model_name, hf_token, torch_dtype, model_cfg)

    run_seed = int(_get_cfg(cfg, "run.seed", 10))
    data_seed = int(_get_cfg(cfg, "run.data_seed", run_seed))
    max_length = int(train_cfg.get("max_length", 1024))
    batch_size = int(train_cfg.get("batch_size", 8))
    eval_batch_size = int(train_cfg.get("eval_batch_size", batch_size))
    dataloader_num_workers = int(train_cfg.get("dataloader_num_workers", 0))
    epochs = float(train_cfg.get("epochs", 1))
    lr = float(train_cfg.get("lr", 5e-6))
    grad_accum = int(train_cfg.get("grad_accum", 1))
    logging_steps = int(train_cfg.get("logging_steps", 10))
    save_steps = int(train_cfg.get("save_steps", 200))
    eval_steps = int(train_cfg.get("eval_steps", save_steps))
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.03))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
    gradient_checkpointing = bool(train_cfg.get("gradient_checkpointing", False))
    save_only_model = bool(train_cfg.get("save_only_model", False))
    group_by_length = bool(train_cfg.get("group_by_length", True))
    ddp_find_unused_parameters = train_cfg.get("ddp_find_unused_parameters")
    load_best_model = bool(train_cfg.get("load_best_model_at_end", False))
    metric_for_best = train_cfg.get("metric_for_best_model")
    if metric_for_best is not None:
        metric_for_best = str(metric_for_best).strip() or None
    greater_is_better = train_cfg.get("greater_is_better")
    early_stop_patience = train_cfg.get("early_stopping_patience")
    if early_stop_patience is not None:
        early_stop_patience = int(early_stop_patience)
        if early_stop_patience <= 0:
            early_stop_patience = None
    early_stop_threshold = float(train_cfg.get("early_stopping_threshold", 0.0))
    early_stop_min_epochs = float(train_cfg.get("early_stopping_min_epochs", 0.0))
    early_stop_min_steps = int(train_cfg.get("early_stopping_min_steps", 0))
    lr_scheduler_type = train_cfg.get("lr_scheduler_type", "cosine")
    max_steps = train_cfg.get("max_steps")
    save_strategy = str(train_cfg.get("save_strategy", "epoch")).lower()
    default_eval_strategy = "epoch" if eval_ds is not None else "no"
    eval_strategy = str(train_cfg.get("eval_strategy", default_eval_strategy)).lower()

    if eval_ds is None:
        eval_strategy = "no"
    if not load_best_model and early_stop_patience is None:
        metric_for_best = None
        greater_is_better = None
    if (load_best_model or early_stop_patience is not None) and eval_strategy == "no":
        raise ValueError("Early stopping / best-model loading requires data.eval_csv")
    if load_best_model and save_strategy != eval_strategy:
        raise ValueError("load_best_model_at_end requires matching save_strategy and eval_strategy")

    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    user_logging_dir = train_cfg.get("logging_dir")
    if user_logging_dir:
        logging_dir = os.path.join(user_logging_dir, os.path.basename(run_dir))
    else:
        logging_dir = os.path.join(run_dir, "tb_logs")

    arg_kwargs = {
        "output_dir": run_dir,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": eval_batch_size,
        "gradient_accumulation_steps": grad_accum,
        "num_train_epochs": epochs,
        "learning_rate": lr,
        "seed": run_seed,
        "data_seed": data_seed,
        "warmup_ratio": warmup_ratio,
        "weight_decay": weight_decay,
        "max_grad_norm": max_grad_norm,
        "logging_steps": logging_steps,
        "save_strategy": save_strategy,
        "save_total_limit": int(train_cfg.get("save_total_limit", 2)),
        "save_only_model": save_only_model,
        "fp16": torch_dtype == torch.float16,
        "bf16": torch_dtype == torch.bfloat16,
        "deepspeed": deepspeed_cfg,
        "load_best_model_at_end": load_best_model,
        "metric_for_best_model": metric_for_best,
        "greater_is_better": greater_is_better,
        "report_to": train_cfg.get("report_to", "tensorboard"),
        "logging_dir": logging_dir,
        "dataloader_num_workers": dataloader_num_workers,
        "gradient_checkpointing": gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": False} if gradient_checkpointing else None,
        "remove_unused_columns": False,
        "group_by_length": group_by_length,
        "lr_scheduler_type": lr_scheduler_type,
    }
    if max_steps is not None:
        arg_kwargs["max_steps"] = int(max_steps)
    if save_strategy == "steps":
        arg_kwargs["save_steps"] = save_steps
    if eval_strategy == "steps":
        arg_kwargs["eval_steps"] = eval_steps
    if ddp_find_unused_parameters is not None:
        arg_kwargs["ddp_find_unused_parameters"] = bool(ddp_find_unused_parameters)

    allowed = inspect.signature(TrainingArguments).parameters
    if "evaluation_strategy" in allowed:
        arg_kwargs["evaluation_strategy"] = eval_strategy
    elif "eval_strategy" in allowed:
        arg_kwargs["eval_strategy"] = eval_strategy
    arg_kwargs = {k: v for k, v in arg_kwargs.items() if k in allowed and v is not None}
    training_args = TrainingArguments(**arg_kwargs)

    callbacks = []
    if early_stop_patience is not None:
        callbacks.append(
            WarmupEarlyStoppingCallback(
                early_stopping_patience=int(early_stop_patience),
                early_stopping_threshold=early_stop_threshold,
                min_epochs=early_stop_min_epochs,
                min_steps=early_stop_min_steps,
            )
        )

    model = _maybe_apply_peft(model, peft_cfg=peft_cfg, train_cfg=train_cfg, model_cfg=model_cfg)

    def _tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            return_special_tokens_mask=True,
        )

    tokenized_train = train_ds.map(_tokenize_fn, batched=True, remove_columns=["text"])
    tokenized_eval = eval_ds.map(_tokenize_fn, batched=True, remove_columns=["text"]) if eval_ds else None
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer_kwargs = _attach_trainer_processing_class(
        {
            "model": model,
            "args": training_args,
            "train_dataset": tokenized_train,
            "eval_dataset": tokenized_eval,
            "data_collator": collator,
            "callbacks": callbacks,
        },
        tokenizer,
    )
    trainer = Trainer(**trainer_kwargs)

    trainer.train()
    trainer.save_model(run_dir)
    tokenizer.save_pretrained(run_dir)
    return run_dir


def run_training_from_config(
    cfg: Dict[str, Any],
    run_dir: str,
    trainer_choice: Optional[str],
    deepspeed_cfg: Optional[str],
    hf_token: Optional[str],
) -> str:
    trainer_type = (trainer_choice or _get_cfg(cfg, "train.trainer", "trl")).lower()
    if trainer_type not in {"trl", "hf"}:
        raise ValueError(f"Unsupported trainer_type: {trainer_type}")
    task = str(_get_cfg(cfg, "task", "train_sft")).lower()
    if task == "train_tapt":
        return train_text_only_causal_lm(cfg, run_dir, trainer_type, hf_token, deepspeed_cfg)
    return train_with_trl(cfg, run_dir, trainer_type, hf_token, deepspeed_cfg)
