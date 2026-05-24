from __future__ import annotations

import math
from typing import Any, Optional, Sequence

from transformers import AutoConfig, AutoTokenizer


_CONTEXT_LIMIT_KEYS = (
    "max_position_embeddings",
    "n_positions",
    "max_seq_len",
    "seq_length",
    "model_max_length",
)
_NESTED_CONFIG_KEYS = (
    "text_config",
    "language_config",
    "llm_config",
    "decoder",
)
_UNREASONABLE_CONTEXT_LIMIT = 1_000_000


def _to_reasonable_int(value: Any) -> Optional[int]:
    try:
        value_int = int(value)
    except Exception:
        return None
    if value_int <= 0 or value_int > _UNREASONABLE_CONTEXT_LIMIT:
        return None
    return value_int


def load_budget_tokenizer(model_name: str, hf_token: Optional[str]):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=hf_token,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _collect_context_candidates(config_dict: dict[str, Any], prefix: str = "") -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for key in _CONTEXT_LIMIT_KEYS:
        value = _to_reasonable_int(config_dict.get(key))
        if value is not None:
            candidates.append({"source": f"{prefix}{key}", "value": value})

    for nested_key in _NESTED_CONFIG_KEYS:
        nested_value = config_dict.get(nested_key)
        if isinstance(nested_value, dict):
            candidates.extend(
                _collect_context_candidates(nested_value, prefix=f"{prefix}{nested_key}.")
            )
    return candidates


def infer_context_window(
    model_name: str,
    hf_token: Optional[str],
    tokenizer=None,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []

    try:
        config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=True,
            token=hf_token,
        )
        config_dict = config.to_dict() if hasattr(config, "to_dict") else {}
        if isinstance(config_dict, dict):
            candidates.extend(_collect_context_candidates(config_dict))
    except Exception as exc:
        candidates.append({"source": "config_error", "value": None, "error": repr(exc)})

    if tokenizer is None:
        tokenizer = load_budget_tokenizer(model_name=model_name, hf_token=hf_token)
    tokenizer_limit = _to_reasonable_int(getattr(tokenizer, "model_max_length", None))
    if tokenizer_limit is not None:
        candidates.append({"source": "tokenizer.model_max_length", "value": tokenizer_limit})

    valid_candidates = [item for item in candidates if isinstance(item.get("value"), int)]
    context_window = min(item["value"] for item in valid_candidates) if valid_candidates else None
    return {
        "context_window": context_window,
        "candidates": candidates,
    }


def measure_prompt_token_lengths(tokenizer, inputs: Sequence[str], batch_size: int = 32) -> list[int]:
    lengths: list[int] = []
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    for start in range(0, len(inputs), batch_size):
        batch = list(inputs[start : start + batch_size])
        enc = tokenizer(
            batch,
            add_special_tokens=True,
            truncation=False,
            padding=False,
        )
        lengths.extend(len(ids) for ids in enc["input_ids"])
    return lengths


def empirical_quantile(values: Sequence[int], q: float) -> int:
    if not values:
        return 0
    if not (0.0 < q <= 1.0):
        raise ValueError("q must satisfy 0 < q <= 1")
    sorted_values = sorted(int(v) for v in values)
    idx = max(0, math.ceil(q * len(sorted_values)) - 1)
    return sorted_values[idx]


def summarize_prompt_lengths(lengths: Sequence[int], q: float = 0.95) -> dict[str, Any]:
    if not lengths:
        return {
            "num_rows": 0,
            "min": 0,
            "mean": 0.0,
            "p50": 0,
            "p90": 0,
            "p95": 0,
            "p99": 0,
            "max": 0,
            "target_quantile": q,
            "target_quantile_value": 0,
        }

    total = sum(int(v) for v in lengths)
    return {
        "num_rows": len(lengths),
        "min": min(lengths),
        "mean": total / float(len(lengths)),
        "p50": empirical_quantile(lengths, 0.50),
        "p90": empirical_quantile(lengths, 0.90),
        "p95": empirical_quantile(lengths, 0.95),
        "p99": empirical_quantile(lengths, 0.99),
        "max": max(lengths),
        "target_quantile": q,
        "target_quantile_value": empirical_quantile(lengths, q),
    }


def resolve_auto_max_input_tokens(
    model_name: str,
    inputs: Sequence[str],
    hf_token: Optional[str],
    max_new_tokens: int,
    margin: int = 64,
    quantile: float = 0.95,
    measure_batch_size: int = 32,
) -> dict[str, Any]:
    tokenizer = load_budget_tokenizer(model_name=model_name, hf_token=hf_token)
    lengths = measure_prompt_token_lengths(tokenizer=tokenizer, inputs=inputs, batch_size=measure_batch_size)
    summary = summarize_prompt_lengths(lengths=lengths, q=quantile)
    context_info = infer_context_window(model_name=model_name, hf_token=hf_token, tokenizer=tokenizer)
    context_window = context_info.get("context_window")

    available_input_budget = None
    if isinstance(context_window, int):
        available_input_budget = max(16, int(context_window) - int(max_new_tokens) - int(margin))

    target_quantile_value = int(summary["target_quantile_value"])
    resolved_max_input_tokens = (
        min(target_quantile_value, available_input_budget)
        if available_input_budget is not None
        else target_quantile_value
    )
    rows_over_resolved = sum(1 for value in lengths if int(value) > resolved_max_input_tokens)

    return {
        "model_name": model_name,
        "context_window": context_window,
        "context_candidates": context_info.get("candidates", []),
        "max_new_tokens": int(max_new_tokens),
        "margin": int(margin),
        "available_input_budget": available_input_budget,
        "resolved_max_input_tokens": int(resolved_max_input_tokens),
        "rows_over_resolved": rows_over_resolved,
        "share_over_resolved": (rows_over_resolved / float(len(lengths))) if lengths else 0.0,
        "length_summary": summary,
    }
