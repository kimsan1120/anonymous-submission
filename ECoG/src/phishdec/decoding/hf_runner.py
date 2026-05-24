import os
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from tqdm import tqdm

try:
    from safetensors.torch import load_file as _safe_load_file
except Exception:  
    _safe_load_file = None

try:
    from peft import PeftModel
except Exception:  
    PeftModel = None

from .parser import parse_pred


class _ForceChoiceOnFirstStep(LogitsProcessor):
    def __init__(self, prompt_seq_len: int, allowed_token_ids: list[int]):
        if not allowed_token_ids:
            raise ValueError("allowed_token_ids must not be empty")
        self.prompt_seq_len = int(prompt_seq_len)
        self.allowed_token_ids = sorted({int(token_id) for token_id in allowed_token_ids})

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if int(input_ids.shape[1]) != self.prompt_seq_len:
            return scores
        masked = torch.full_like(scores, float("-inf"))
        masked[:, self.allowed_token_ids] = scores[:, self.allowed_token_ids]
        return masked


def get_safe_max_len(tokenizer, user_cap=1024, fallback=4096, hard_cap=1_000_000):
    m = getattr(tokenizer, "model_max_length", None)
    try:
        m_int = int(m) if m is not None else fallback
    except Exception:
        m_int = fallback
    if m_int <= 0 or m_int > hard_cap:
        m_int = int(user_cap) if user_cap is not None else fallback
    return max(16, min(m_int, int(user_cap)))


def _load_tokenizer(model_name: str, hf_token: Optional[str]):
    tok = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=hf_token,
        padding_side="left",
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _infer_hidden_size(model: torch.nn.Module) -> int:
    config = getattr(model, "config", None)
    for attr in ("hidden_size", "n_embd", "d_model"):
        value = getattr(config, attr, None)
        if isinstance(value, int) and value > 0:
            return int(value)
    raise ValueError("Could not infer hidden size for joint fusion heads")


def _first_param_device_dtype(model: torch.nn.Module) -> tuple[torch.device, torch.dtype]:
    device = torch.device("cpu")
    dtype = torch.float32
    for param in model.parameters():
        device = param.device
        if param.is_floating_point():
            dtype = param.dtype
            break
    return device, dtype


def _load_yaml_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _candidate_checkpoint_dirs(model_name: str, adapter_path: Optional[str]) -> list[Path]:
    out: list[Path] = []
    for raw in (model_name, adapter_path):
        if not raw:
            continue
        path = Path(str(raw))
        if path.exists() and path.is_dir():
            out.append(path)
    return out


def _load_joint_tensor_state(checkpoint_dir: Path) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    safe_paths = [checkpoint_dir / "model.safetensors", checkpoint_dir / "adapter_model.safetensors"]
    bin_paths = [checkpoint_dir / "pytorch_model.bin", checkpoint_dir / "adapter_model.bin"]
    loaded: dict[str, torch.Tensor] = {}
    for safe_path in safe_paths:
        if safe_path.exists() and _safe_load_file is not None:
            loaded = _safe_load_file(str(safe_path), device="cpu")
            break
    if not loaded:
        for bin_path in bin_paths:
            if bin_path.exists():
                raw_loaded = torch.load(str(bin_path), map_location="cpu")
                if isinstance(raw_loaded, dict):
                    loaded = raw_loaded
                break
    for key, value in loaded.items():
        key_str = str(key)
        if key_str.startswith("joint_"):
            state[key_str] = value
            continue
        marker = ".joint_"
        if marker in key_str:
            normalized = key_str[key_str.index(marker) + 1 :]
            state[normalized] = value
    return state


def _has_joint_fusion_state(state: dict[str, torch.Tensor]) -> bool:
    required = {
        "joint_evidence_classifier.weight",
        "joint_evidence_classifier.bias",
        "joint_span_reranker.weight",
        "joint_span_reranker.bias",
        "joint_evidence_fusion_proj.weight",
        "joint_evidence_fusion_proj.bias",
        "joint_evidence_fusion_gate.weight",
        "joint_evidence_fusion_gate.bias",
    }
    return required.issubset(set(state.keys()))


def _has_joint_label_rationale_state(state: dict[str, torch.Tensor]) -> bool:
    required = {
        "joint_label_rationale_embedding.weight",
        "joint_label_rationale_adapter.weight",
        "joint_label_rationale_adapter.bias",
    }
    return required.issubset(set(state.keys()))


def _load_linear_from_state(
    module: torch.nn.Linear,
    state: dict[str, torch.Tensor],
    prefix: str,
) -> None:
    module_state = {
        "weight": state[f"{prefix}.weight"],
        "bias": state[f"{prefix}.bias"],
    }
    module.load_state_dict(module_state)


def _load_embedding_from_state(
    module: torch.nn.Embedding,
    state: dict[str, torch.Tensor],
    prefix: str,
) -> None:
    module.load_state_dict({"weight": state[f"{prefix}.weight"]})


def _attach_joint_label_rationale_adapter_for_generate(
    model: torch.nn.Module,
    checkpoint_dir: Path,
    label_token_ids: Optional[tuple[int, int]],
) -> Optional[dict[str, Any]]:
    train_cfg = _load_yaml_if_exists(checkpoint_dir / "config.yaml").get("train", {}) or {}
    if not bool(train_cfg.get("label_conditioned_rationale", False)):
        return None
    if label_token_ids is None:
        print(
            "[warn] label_conditioned_rationale=true but label token ids were not available; "
            "HF generate will use the base LM path."
        )
        return None

    state = _load_joint_tensor_state(checkpoint_dir)
    if not _has_joint_label_rationale_state(state):
        print(
            "[warn] label_conditioned_rationale=true but adapter weights were not found; "
            "HF generate will use the base LM path."
        )
        return None

    hidden_size = _infer_hidden_size(model)
    device, dtype = _first_param_device_dtype(model)
    head_model = model
    if not hasattr(head_model, "joint_label_rationale_embedding"):
        head_model.add_module("joint_label_rationale_embedding", torch.nn.Embedding(2, hidden_size))
    if not hasattr(head_model, "joint_label_rationale_adapter"):
        head_model.add_module("joint_label_rationale_adapter", torch.nn.Linear(hidden_size, hidden_size))

    _load_embedding_from_state(
        head_model.joint_label_rationale_embedding,
        state,
        "joint_label_rationale_embedding",
    )
    _load_linear_from_state(
        head_model.joint_label_rationale_adapter,
        state,
        "joint_label_rationale_adapter",
    )
    head_model.joint_label_rationale_embedding.to(device=device, dtype=dtype)
    head_model.joint_label_rationale_adapter.to(device=device, dtype=dtype)
    head_model.joint_label_rationale_embedding.eval()
    head_model.joint_label_rationale_adapter.eval()

    payload = {
        "scale": float(train_cfg.get("label_conditioned_rationale_scale", 0.05)),
        "zero_token_id": int(label_token_ids[0]),
        "one_token_id": int(label_token_ids[1]),
    }
    print(
        "[info] enabled label-conditioned rationale adapter for HF generate "
        f"(scale={payload['scale']})"
    )
    return payload


def _attach_joint_fusion_heads_for_generate(
    model: torch.nn.Module,
    checkpoint_dir: Path,
) -> Optional[dict[str, Any]]:
    train_cfg = _load_yaml_if_exists(checkpoint_dir / "config.yaml").get("train", {}) or {}
    if not bool(train_cfg.get("evidence_guided_fusion", False)):
        return None

    state = _load_joint_tensor_state(checkpoint_dir)
    if not _has_joint_fusion_state(state):
        print(
            "[warn] evidence_guided_fusion=true but joint fusion weights were not found; "
            "HF generate will use the base LM path."
        )
        return None

    hidden_size = _infer_hidden_size(model)
    device, dtype = _first_param_device_dtype(model)
    head_model = model
    if not hasattr(head_model, "joint_evidence_classifier"):
        head_model.add_module("joint_evidence_classifier", torch.nn.Linear(hidden_size, 1))
    if not hasattr(head_model, "joint_span_reranker"):
        head_model.add_module("joint_span_reranker", torch.nn.Linear(hidden_size, 1))
    if not hasattr(head_model, "joint_evidence_fusion_proj"):
        head_model.add_module("joint_evidence_fusion_proj", torch.nn.Linear(hidden_size, hidden_size))
    if not hasattr(head_model, "joint_evidence_fusion_gate"):
        head_model.add_module("joint_evidence_fusion_gate", torch.nn.Linear(hidden_size, hidden_size))

    _load_linear_from_state(head_model.joint_evidence_classifier, state, "joint_evidence_classifier")
    _load_linear_from_state(head_model.joint_span_reranker, state, "joint_span_reranker")
    _load_linear_from_state(head_model.joint_evidence_fusion_proj, state, "joint_evidence_fusion_proj")
    _load_linear_from_state(head_model.joint_evidence_fusion_gate, state, "joint_evidence_fusion_gate")

    for name in (
        "joint_evidence_classifier",
        "joint_span_reranker",
        "joint_evidence_fusion_proj",
        "joint_evidence_fusion_gate",
    ):
        getattr(head_model, name).to(device=device, dtype=dtype)
        getattr(head_model, name).eval()

    payload = {
        "scale": float(train_cfg.get("evidence_guided_fusion_scale", 0.1)),
        "span_max_width": int(train_cfg.get("span_rerank_max_width", 32)),
        "span_max_candidates": int(train_cfg.get("span_rerank_max_negative_spans", 12)),
        "confidence_gate": bool(train_cfg.get("evidence_guided_fusion_confidence_gate", False)),
    }
    print(
        "[info] enabled evidence-guided fusion for HF generate "
        f"(scale={payload['scale']}, span_max_width={payload['span_max_width']}, "
        f"span_max_candidates={payload['span_max_candidates']}, "
        f"confidence_gate={int(payload['confidence_gate'])})"
    )
    return payload


def _past_has_tokens(past_key_values: Any) -> bool:
    if past_key_values is None:
        return False
    get_seq_length = getattr(past_key_values, "get_seq_length", None)
    if callable(get_seq_length):
        try:
            return int(get_seq_length()) > 0
        except Exception:
            pass
    try:
        first = past_key_values[0]
        key = first[0] if isinstance(first, (tuple, list)) else first
        return int(key.shape[-2]) > 0
    except Exception:
        return True


def _evidence_summary_for_generate(
    last_hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    evidence_logits: torch.Tensor,
) -> torch.Tensor:
    prompt_mask = attention_mask.to(device=evidence_logits.device).bool()
    masked_logits = evidence_logits.float().masked_fill(~prompt_mask, -1.0e9)
    weights = torch.softmax(masked_logits, dim=1).to(last_hidden.dtype)
    return (weights.unsqueeze(-1).to(last_hidden.device) * last_hidden).sum(dim=1)


def _span_candidates_for_row(
    scores: torch.Tensor,
    prompt_mask: torch.Tensor,
    *,
    max_candidates: int,
    max_width: int,
) -> list[tuple[int, int]]:
    valid = torch.nonzero(prompt_mask, as_tuple=False).squeeze(-1)
    if valid.numel() == 0:
        return []
    min_pos = int(valid.min().item())
    max_pos_exclusive = int(valid.max().item()) + 1
    masked_scores = scores.float().masked_fill(~prompt_mask, -1.0e9)
    top_k = min(max(1, int(max_candidates)), int(valid.numel()))
    centers = torch.topk(masked_scores, k=top_k).indices.detach().cpu().tolist()
    widths = [4, 8, 16, max(1, int(max_width))]
    widths = sorted({max(1, min(int(w), int(max_width))) for w in widths if int(w) > 0})
    spans: list[tuple[int, int]] = []
    seen = set()
    for center_raw in centers:
        center = int(center_raw)
        for width in widths:
            half = max(0, width // 2)
            st = max(min_pos, center - half)
            ed = min(max_pos_exclusive, st + width)
            st = max(min_pos, ed - width)
            if ed <= st:
                continue
            span = (int(st), int(ed))
            if span in seen:
                continue
            seen.add(span)
            spans.append(span)
            if len(spans) >= int(max_candidates):
                return spans
    return spans


def _span_summary_for_generate(
    *,
    last_hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    evidence_logits: torch.Tensor,
    span_reranker: torch.nn.Linear,
    max_candidates: int,
    max_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, _, hidden_size = last_hidden.shape
    summaries = last_hidden.new_zeros((bsz, hidden_size))
    active = torch.zeros((bsz,), device=last_hidden.device, dtype=torch.bool)
    prompt_mask = attention_mask.to(device=evidence_logits.device).bool()
    for row_idx in range(bsz):
        spans = _span_candidates_for_row(
            evidence_logits[row_idx],
            prompt_mask[row_idx],
            max_candidates=max_candidates,
            max_width=max_width,
        )
        if not spans:
            continue
        span_reprs = torch.stack(
            [last_hidden[row_idx, st:ed].mean(dim=0) for st, ed in spans],
            dim=0,
        )
        weight = span_reranker.weight
        span_scores = span_reranker(span_reprs.to(device=weight.device, dtype=weight.dtype)).squeeze(-1).float()
        weights = torch.softmax(span_scores, dim=0).to(span_reprs.dtype)
        summaries[row_idx] = (weights.unsqueeze(-1).to(span_reprs.device) * span_reprs).sum(dim=0)
        active[row_idx] = True
    return summaries, active


def _replace_output_logits(outputs: Any, logits: torch.Tensor) -> Any:
    if isinstance(outputs, tuple):
        return (logits,) + tuple(outputs[1:])
    try:
        outputs.logits = logits
    except Exception:
        pass
    try:
        outputs["logits"] = logits
    except Exception:
        pass
    return outputs


def _wrap_forward_with_joint_fusion_for_generate(
    model: torch.nn.Module,
    payload: dict[str, Any],
) -> None:
    if bool(getattr(model, "_joint_fusion_generate_wrapped", False)):
        return

    original_forward = model.forward
    model._joint_fusion_generate_cache = {}

    def forward_with_joint_fusion(*args, **kwargs):
        past_key_values = kwargs.get("past_key_values")
        kwargs["output_hidden_states"] = True
        kwargs["return_dict"] = True
        outputs = original_forward(*args, **kwargs)
        hidden_states = getattr(outputs, "hidden_states", None)
        if not hidden_states:
            return outputs
        last_hidden = hidden_states[-1]
        if last_hidden.ndim != 3:
            return outputs

        head_device = last_hidden.device
        head_dtype = last_hidden.dtype
        for name in (
            "joint_evidence_classifier",
            "joint_span_reranker",
            "joint_evidence_fusion_proj",
            "joint_evidence_fusion_gate",
        ):
            module = getattr(model, name)
            if module.weight.device != head_device or module.weight.dtype != head_dtype:
                module.to(device=head_device, dtype=head_dtype)

        cache = getattr(model, "_joint_fusion_generate_cache", {})
        is_prefill = not _past_has_tokens(past_key_values)
        attention_mask = kwargs.get("attention_mask")
        if is_prefill or "summary" not in cache:
            if attention_mask is None or int(attention_mask.shape[1]) != int(last_hidden.shape[1]):
                prompt_mask = torch.ones(
                    last_hidden.shape[:2],
                    device=last_hidden.device,
                    dtype=torch.long,
                )
            else:
                prompt_mask = attention_mask.to(last_hidden.device)
            evidence_logits = model.joint_evidence_classifier(last_hidden).squeeze(-1)
            evidence_summary = _evidence_summary_for_generate(
                last_hidden=last_hidden,
                attention_mask=prompt_mask,
                evidence_logits=evidence_logits,
            )
            if bool(payload.get("confidence_gate", False)):
                confidence = torch.sigmoid(evidence_logits.float()).masked_fill(
                    prompt_mask.to(device=evidence_logits.device).bool().logical_not(),
                    0.0,
                ).max(dim=1).values
            else:
                confidence = None
            span_summary, span_active = _span_summary_for_generate(
                last_hidden=last_hidden,
                attention_mask=prompt_mask,
                evidence_logits=evidence_logits,
                span_reranker=model.joint_span_reranker,
                max_candidates=int(payload.get("span_max_candidates", 12)),
                max_width=int(payload.get("span_max_width", 32)),
            )
            summary = evidence_summary
            if bool(span_active.any().item()):
                summary = torch.where(
                    span_active.to(summary.device).unsqueeze(-1),
                    0.5 * (summary + span_summary.to(summary.device, dtype=summary.dtype)),
                    summary,
                )
            cache = {
                "summary": summary.detach(),
                "confidence": confidence.detach() if confidence is not None else None,
                "batch_size": int(last_hidden.shape[0]),
            }
            model._joint_fusion_generate_cache = cache
        else:
            summary = cache["summary"].to(device=last_hidden.device, dtype=last_hidden.dtype)
            if int(summary.shape[0]) != int(last_hidden.shape[0]):
                return outputs
            confidence = cache.get("confidence")

        proj = model.joint_evidence_fusion_proj(summary)
        gate = torch.sigmoid(model.joint_evidence_fusion_gate(last_hidden))
        delta = float(payload.get("scale", 0.1)) * gate * proj.unsqueeze(1)
        if confidence is not None:
            delta = confidence.to(device=delta.device, dtype=delta.dtype).view(-1, 1, 1) * delta
        fused_hidden = last_hidden + delta
        output_embeddings = model.get_output_embeddings()
        if output_embeddings is None:
            return outputs
        emb_weight = output_embeddings.weight
        fused_logits = output_embeddings(fused_hidden.to(device=emb_weight.device, dtype=emb_weight.dtype))
        return _replace_output_logits(outputs, fused_logits)

    model.forward = forward_with_joint_fusion
    model._joint_fusion_generate_wrapped = True


def _wrap_forward_with_joint_label_rationale_for_generate(
    model: torch.nn.Module,
    payload: dict[str, Any],
) -> None:
    if bool(getattr(model, "_joint_label_rationale_generate_wrapped", False)):
        return

    original_forward = model.forward
    model._joint_label_rationale_generate_cache = {}

    def forward_with_joint_label_rationale(*args, **kwargs):
        past_key_values = kwargs.get("past_key_values")
        kwargs["output_hidden_states"] = True
        kwargs["return_dict"] = True
        outputs = original_forward(*args, **kwargs)
        hidden_states = getattr(outputs, "hidden_states", None)
        if not hidden_states:
            return outputs
        last_hidden = hidden_states[-1]
        if last_hidden.ndim != 3:
            return outputs

        is_prefill = not _past_has_tokens(past_key_values)
        cache = getattr(model, "_joint_label_rationale_generate_cache", {})
        if is_prefill:
            model._joint_label_rationale_generate_cache = {}
            return outputs

        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            first_arg = args[0]
            if torch.is_tensor(first_arg):
                input_ids = first_arg
        if input_ids is None or not torch.is_tensor(input_ids) or input_ids.ndim != 2:
            return outputs

        bsz = int(last_hidden.shape[0])
        if "label_classes" not in cache:
            current_ids = input_ids[:, -1].detach()
            zero_id = int(payload["zero_token_id"])
            one_id = int(payload["one_token_id"])
            is_zero = current_ids == zero_id
            is_one = current_ids == one_id
            if not bool((is_zero | is_one).all().item()):
                return outputs
            label_classes = torch.where(
                is_one,
                torch.ones_like(current_ids, dtype=torch.long),
                torch.zeros_like(current_ids, dtype=torch.long),
            )
            cache = {"label_classes": label_classes.detach(), "batch_size": bsz}
            model._joint_label_rationale_generate_cache = cache
        else:
            label_classes = cache["label_classes"].to(device=last_hidden.device)
            if int(label_classes.shape[0]) != bsz:
                return outputs

        head_device = last_hidden.device
        head_dtype = last_hidden.dtype
        for name in ("joint_label_rationale_embedding", "joint_label_rationale_adapter"):
            module = getattr(model, name)
            if module.weight.device != head_device or module.weight.dtype != head_dtype:
                module.to(device=head_device, dtype=head_dtype)

        labels = label_classes.to(device=head_device).long().clamp(0, 1)
        label_delta = model.joint_label_rationale_adapter(
            model.joint_label_rationale_embedding(labels)
        ).to(dtype=last_hidden.dtype)
        adapted_hidden = last_hidden + float(payload.get("scale", 0.05)) * label_delta.unsqueeze(1)
        output_embeddings = model.get_output_embeddings()
        if output_embeddings is None:
            return outputs
        emb_weight = output_embeddings.weight
        adapted_logits = output_embeddings(adapted_hidden.to(device=emb_weight.device, dtype=emb_weight.dtype))
        return _replace_output_logits(outputs, adapted_logits)

    model.forward = forward_with_joint_label_rationale
    model._joint_label_rationale_generate_wrapped = True


def _maybe_enable_joint_label_rationale_for_generate(
    model: torch.nn.Module,
    model_name: str,
    adapter_path: Optional[str],
    label_token_ids: Optional[tuple[int, int]],
) -> bool:
    for checkpoint_dir in _candidate_checkpoint_dirs(model_name, adapter_path):
        payload = _attach_joint_label_rationale_adapter_for_generate(
            model,
            checkpoint_dir,
            label_token_ids,
        )
        if payload is not None:
            _wrap_forward_with_joint_label_rationale_for_generate(model, payload)
            return True
    return False


def _maybe_enable_joint_fusion_for_generate(
    model: torch.nn.Module,
    model_name: str,
    adapter_path: Optional[str],
) -> bool:
    for checkpoint_dir in _candidate_checkpoint_dirs(model_name, adapter_path):
        payload = _attach_joint_fusion_heads_for_generate(model, checkpoint_dir)
        if payload is not None:
            _wrap_forward_with_joint_fusion_for_generate(model, payload)
            return True
    return False


def _maybe_enable_joint_fusion_for_generate(
    model: torch.nn.Module,
    model_name: str,
    adapter_path: Optional[str],
) -> bool:
    for checkpoint_dir in _candidate_checkpoint_dirs(model_name, adapter_path):
        payload = _attach_joint_fusion_heads_for_generate(model, checkpoint_dir)
        if payload is not None:
            _wrap_forward_with_joint_fusion_for_generate(model, payload)
            return True
    return False


def _force_use_cache_for_eval(model: torch.nn.Module) -> None:
    seen: set[int] = set()
    queue: list[Any] = [model]

    while queue:
        current = queue.pop(0)
        if current is None:
            continue
        current_id = id(current)
        if current_id in seen:
            continue
        seen.add(current_id)

        config = getattr(current, "config", None)
        if config is not None:
            setattr(config, "use_cache", True)

        generation_config = getattr(current, "generation_config", None)
        if generation_config is not None:
            setattr(generation_config, "use_cache", True)

        get_base_model = getattr(current, "get_base_model", None)
        if callable(get_base_model):
            try:
                queue.append(get_base_model())
            except Exception:
                pass
        for attr in ("base_model", "model"):
            child = getattr(current, attr, None)
            if child is not None:
                queue.append(child)


def _load_model(
    model_name: str,
    hf_token: Optional[str],
    torch_dtype: Optional[torch.dtype],
    adapter_path: Optional[str],
    merge_adapter: bool,
    label_token_ids: Optional[tuple[int, int]] = None,
):
    kwargs = {"trust_remote_code": True, "token": hf_token}
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        low_cpu_mem_usage=True,
        **kwargs,
    )

    if adapter_path:
        if PeftModel is None:
            raise ImportError("peft is required to load adapter weights.")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        if merge_adapter:
            model = model.merge_and_unload()
    else:
        model = base_model

    _force_use_cache_for_eval(base_model)
    _force_use_cache_for_eval(model)

    label_adapter_enabled = _maybe_enable_joint_label_rationale_for_generate(
        model,
        model_name,
        adapter_path,
        label_token_ids,
    )
    if not label_adapter_enabled:
        _maybe_enable_joint_fusion_for_generate(model, model_name, adapter_path)
    model.eval()
    return model


def merge_adapter_to_dir(
    model_name: str,
    adapter_path: str,
    hf_token: Optional[str],
    torch_dtype: Optional[torch.dtype] = torch.float16,
    save_dir: Optional[str] = None,
) -> str:
    if PeftModel is None:
        raise ImportError("peft is required to merge adapters.")

    work_dir = save_dir or tempfile.mkdtemp(prefix="merged_adapter_")
    os.makedirs(work_dir, exist_ok=True)

    tokenizer = _load_tokenizer(model_name, hf_token)
    tokenizer.save_pretrained(work_dir)

    model = _load_model(model_name, hf_token, torch_dtype, adapter_path, merge_adapter=True)
    model.save_pretrained(work_dir)

    torch.cuda.empty_cache()
    return work_dir


def _normalize_eos_token_ids(eos_token_id: Any) -> set[int]:
    if eos_token_id is None:
        return set()
    if isinstance(eos_token_id, (list, tuple, set)):
        out = set()
        for item in eos_token_id:
            try:
                out.add(int(item))
            except Exception:
                continue
        return out
    try:
        return {int(eos_token_id)}
    except Exception:
        return set()


def _resolve_choice_token_ids(tokenizer, choices: list[str]) -> list[int]:
    token_ids: list[int] = []
    for choice in choices:
        text = str(choice)
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(
                f"Binary constrained decoding requires single-token choices. "
                f"choice={text!r} encoded_len={len(encoded)} ids={encoded}"
            )
        token_ids.append(int(encoded[0]))
    return token_ids


def _strip_existing_trailing_label_marker(text: str, marker: str) -> str:
    raw = str(text or "")
    marker_text = str(marker or "정답:\n")
    marker_key = marker_text.strip()
    if marker_key:
        idx = raw.find(marker_key)
        if idx >= 0:
            raw = raw[:idx]
    return raw.rstrip()


def _join_body_and_trailing_label_marker(body: str, marker: str) -> str:
    body_text = _strip_existing_trailing_label_marker(body, marker)
    marker_text = str(marker or "정답:\n")
    if body_text:
        return f"{body_text.rstrip()}\n{marker_text}"
    return marker_text


def run_with_hf_generate(
    model_name: str,
    inputs,
    hf_token: Optional[str],
    batch_size: int = 8,
    max_input_tokens: int = 1024,
    max_new_tokens: int = 8,
    torch_dtype: Optional[torch.dtype] = torch.float16,
    adapter_path: Optional[str] = None,
    merge_adapter: bool = False,
    empty_cache_each_batch: bool = False,
    on_sample: Optional[Callable[..., None]] = None,
    sample_offset: int = 0,
    temperature: float = 0.0,
    top_p: float = 1.0,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    repetition_penalty: float = 1.0,
    return_token_logprobs: bool = False,
    constrain_first_token_choices: Optional[list[str]] = None,
    constrain_trailing_binary_choices: Optional[list[str]] = None,
    trailing_binary_marker: str = "정답:\n",
):
    tokenizer = _load_tokenizer(model_name, hf_token)
    safe_len = get_safe_max_len(tokenizer, user_cap=max_input_tokens)
    choice_token_ids = (
        _resolve_choice_token_ids(tokenizer, constrain_first_token_choices)
        if constrain_first_token_choices
        else None
    )
    trailing_choice_token_ids = (
        _resolve_choice_token_ids(tokenizer, constrain_trailing_binary_choices)
        if constrain_trailing_binary_choices
        else None
    )
    if choice_token_ids is not None and trailing_choice_token_ids is not None:
        raise ValueError("Use only one of constrain_first_token_choices or constrain_trailing_binary_choices")
    try:
        label_token_ids = tuple(_resolve_choice_token_ids(tokenizer, ["0", "1"]))
    except ValueError:
        label_token_ids = None

    model = _load_model(
        model_name,
        hf_token,
        torch_dtype,
        adapter_path,
        merge_adapter,
        label_token_ids=label_token_ids,  
    )

    first_device = next(model.parameters()).device

    preds_all, gens_all = [], []
    tokens_all, token_logprobs_all = [], []
    do_sample = float(temperature) > 0.0 or float(top_p) < 1.0
    effective_temperature = float(temperature) if float(temperature) > 0.0 else 1.0
    eos_token_ids = _normalize_eos_token_ids(tokenizer.eos_token_id)
    pad_token_id = tokenizer.pad_token_id
    trailing_marker_token_budget = 0
    if trailing_choice_token_ids is not None:
        trailing_marker_token_budget = max(
            1,
            len(tokenizer.encode(str(trailing_binary_marker or "정답:\n"), add_special_tokens=False)),
        )
    if abs(float(presence_penalty)) > 1e-8 or abs(float(frequency_penalty)) > 1e-8:
        print(
            "[warn] HF generate ignores decode.presence_penalty and "
            "decode.frequency_penalty; continuing without them."
        )

    for s in tqdm(range(0, len(inputs), batch_size), desc="HF Decoding"):
        batch = inputs[s : s + batch_size]

        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=safe_len,
        )
        enc.pop("token_type_ids", None)
        enc = {k: v.to(first_device) for k, v in enc.items()}

        autocast_ctx = nullcontext()
        if torch.cuda.is_available() and torch_dtype in (torch.float16, torch.bfloat16):
            autocast_ctx = torch.autocast(device_type="cuda", dtype=torch_dtype)

        with torch.inference_mode(), autocast_ctx:
            if bool(getattr(model, "_joint_fusion_generate_wrapped", False)):
                model._joint_fusion_generate_cache = {}
            if bool(getattr(model, "_joint_label_rationale_generate_wrapped", False)):
                model._joint_label_rationale_generate_cache = {}
            body_max_new_tokens = int(max_new_tokens)
            if trailing_choice_token_ids is not None:
                body_max_new_tokens = max(1, int(max_new_tokens) - trailing_marker_token_budget - 1)
            generate_kwargs = dict(
                max_new_tokens=body_max_new_tokens,
                do_sample=do_sample,
                pad_token_id=pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=float(repetition_penalty),
            )
            if do_sample:
                generate_kwargs["temperature"] = effective_temperature
                generate_kwargs["top_p"] = float(top_p)
            if return_token_logprobs:
                generate_kwargs["return_dict_in_generate"] = True
                generate_kwargs["output_scores"] = True
            if choice_token_ids is not None:
                generate_kwargs["logits_processor"] = LogitsProcessorList([
                    _ForceChoiceOnFirstStep(
                        prompt_seq_len=int(enc["input_ids"].shape[1]),
                        allowed_token_ids=choice_token_ids,
                    )
                ])
            gen_out = model.generate(
                **enc,
                **generate_kwargs,
            )
            label_out = None
            label_step_logprobs = []
            label_sequences = None
            label_prompt_seq_len = None
            trailing_prefixes: list[str] = []
            if trailing_choice_token_ids is not None:
                body_sequences = gen_out.sequences if return_token_logprobs else gen_out
                body_prompt_seq_len = enc["input_ids"].shape[1]
                for i in range(len(batch)):
                    body_token_ids = body_sequences[i, int(body_prompt_seq_len) :]
                    body_txt = tokenizer.decode(body_token_ids, skip_special_tokens=True)
                    trailing_prefixes.append(_join_body_and_trailing_label_marker(body_txt, trailing_binary_marker))
                label_inputs = [
                    f"{str(prompt_text)}{str(prefix_text)}"
                    for prompt_text, prefix_text in zip(batch, trailing_prefixes)
                ]
                label_enc = tokenizer(
                    label_inputs,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=safe_len,
                )
                label_enc.pop("token_type_ids", None)
                label_enc = {k: v.to(first_device) for k, v in label_enc.items()}
                label_generate_kwargs = dict(
                    max_new_tokens=1,
                    do_sample=False,
                    pad_token_id=pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    logits_processor=LogitsProcessorList([
                        _ForceChoiceOnFirstStep(
                            prompt_seq_len=int(label_enc["input_ids"].shape[1]),
                            allowed_token_ids=trailing_choice_token_ids,
                        )
                    ]),
                )
                if return_token_logprobs:
                    label_generate_kwargs["return_dict_in_generate"] = True
                    label_generate_kwargs["output_scores"] = True
                label_out = model.generate(
                    **label_enc,
                    **label_generate_kwargs,
                )
                if return_token_logprobs:
                    label_step_logprobs = [
                        torch.log_softmax(step_scores.float(), dim=-1)
                        for step_scores in (list(getattr(label_out, "scores", ()) or ()))
                    ]
                    label_sequences = label_out.sequences
                else:
                    label_sequences = label_out
                label_prompt_seq_len = label_enc["input_ids"].shape[1]
        if return_token_logprobs:
            sequences = gen_out.sequences
            score_steps = list(getattr(gen_out, "scores", ()) or ())
            step_logprobs = [
                torch.log_softmax(step_scores.float(), dim=-1)
                for step_scores in score_steps
            ]
        else:
            sequences = gen_out
            step_logprobs = []
        
        
        
        
        
        prompt_seq_len = enc["input_ids"].shape[1]

        for i in range(len(batch)):
            gen_tokens = sequences[i, int(prompt_seq_len) :]
            txt = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            label_tokens = None
            if trailing_choice_token_ids is not None:
                if label_sequences is None or label_prompt_seq_len is None:
                    raise RuntimeError("Trailing binary constrained decode did not produce label sequences")
                label_tokens = label_sequences[i, int(label_prompt_seq_len) :]
                label_txt = tokenizer.decode(label_tokens, skip_special_tokens=True).strip()
                label_txt = label_txt[:1] if label_txt else "0"
                txt = f"{trailing_prefixes[i]}{label_txt}"
            sample_idx = sample_offset + s + i
            pred = parse_pred(txt, default=0)
            gens_all.append(txt)
            preds_all.append(pred)
            sample_tokens, sample_logprobs = None, None
            if return_token_logprobs:
                gen_token_ids = [int(tid) for tid in gen_tokens.tolist()]
                sample_token_ids = []
                sample_logprobs = []
                max_steps = min(len(step_logprobs), len(gen_token_ids))
                for step in range(max_steps):
                    token_id = int(gen_token_ids[step])
                    if pad_token_id is not None and token_id == int(pad_token_id):
                        break
                    sample_token_ids.append(token_id)
                    sample_logprobs.append(float(step_logprobs[step][i, token_id].item()))
                    if eos_token_ids and token_id in eos_token_ids:
                        break
                if trailing_choice_token_ids is not None and label_tokens is not None:
                    label_token_ids_i = [int(tid) for tid in label_tokens.tolist()]
                    label_steps = min(len(label_step_logprobs), len(label_token_ids_i))
                    for label_step in range(label_steps):
                        token_id = int(label_token_ids_i[label_step])
                        if pad_token_id is not None and token_id == int(pad_token_id):
                            break
                        sample_token_ids.append(token_id)
                        sample_logprobs.append(
                            float(label_step_logprobs[label_step][i, token_id].item())
                        )
                        break
                sample_tokens = tokenizer.convert_ids_to_tokens(sample_token_ids) if sample_token_ids else []
                tokens_all.append(sample_tokens)
                token_logprobs_all.append(sample_logprobs)
            if on_sample is not None:
                if return_token_logprobs:
                    try:
                        on_sample(sample_idx, pred, txt, sample_tokens, sample_logprobs)
                    except TypeError:
                        on_sample(sample_idx, pred, txt)
                else:
                    on_sample(sample_idx, pred, txt)

        del enc, gen_out, sequences
        if trailing_choice_token_ids is not None:
            try:
                del label_enc, label_out, label_sequences
            except Exception:
                pass
        if empty_cache_each_batch and torch.cuda.is_available():
            torch.cuda.empty_cache()

    if return_token_logprobs:
        return preds_all, gens_all, tokens_all, token_logprobs_all
    return preds_all, gens_all
