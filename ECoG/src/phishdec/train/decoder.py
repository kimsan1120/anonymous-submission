from __future__ import annotations

import ast
import json
import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  
    SummaryWriter = None
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from transformers.utils import ModelOutput

from phishdec.utils.model_registry import resolve_model_name
from phishdec.utils.seed import set_seed

try:
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
except Exception:  
    LoraConfig = None
    PeftModel = None
    TaskType = None
    get_peft_model = None


def _get_cfg(cfg: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur = cfg
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _read_csv_normalized(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = [str(col).replace("\ufeff", "").strip() for col in df.columns]
    return df


def _to_dtype(dtype_str: Optional[str]) -> Optional[torch.dtype]:
    if not dtype_str or str(dtype_str).lower() == "auto":
        return None
    low = str(dtype_str).lower()
    if low in {"fp16", "float16", "half"}:
        return torch.float16
    if low in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if low in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def _append_jsonl(path: str, row: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


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


def _tb_run_name(run_dir: str) -> str:
    return Path(run_dir).name


def _format_eta(seconds: float) -> str:
    if seconds <= 0 or not float(seconds) < float("inf"):
        return "00:00"
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _hmean(a: float, b: float) -> float:
    a = float(a)
    b = float(b)
    if a <= 0.0 or b <= 0.0:
        return 0.0
    return float((2.0 * a * b) / max(1e-12, a + b))


def _log_tradeoff_scalars(writer, split: str, metrics: Dict[str, Any], step: int) -> None:
    if writer is None or not metrics:
        return

    accuracy = float(metrics.get("accuracy", 0.0))
    macro_f1 = float(metrics.get("macro_f1", 0.0))
    weighted_macro_f1 = float(metrics.get("category_weighted_macro_f1", macro_f1))
    evidence_f1 = float(metrics.get("evidence_token_f1", 0.0))
    evidence_precision = float(metrics.get("evidence_token_precision", 0.0))
    evidence_recall = float(metrics.get("evidence_token_recall", 0.0))
    evidence_aware = float(metrics.get("evidence_aware_score", weighted_macro_f1))

    writer.add_scalars(
        f"tradeoff/{split}_core",
        {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_macro_f1": weighted_macro_f1,
            "evidence_token_f1": evidence_f1,
            "evidence_aware_score": evidence_aware,
        },
        step,
    )
    writer.add_scalars(
        f"tradeoff/{split}_evidence",
        {
            "precision": evidence_precision,
            "recall": evidence_recall,
            "f1": evidence_f1,
        },
        step,
    )

    recall_0 = float(metrics.get("label_0_recall", metrics.get("negative_recall", 0.0)))
    recall_1 = float(metrics.get("label_1_recall", metrics.get("positive_recall", 0.0)))
    if recall_0 > 0.0 or recall_1 > 0.0:
        writer.add_scalars(
            f"tradeoff/{split}_recall_balance",
            {
                "label_0_recall": recall_0,
                "label_1_recall": recall_1,
            },
            step,
        )

    writer.add_scalar(
        f"tradeoff/{split}_hmean_accuracy_evidence_f1",
        _hmean(accuracy, evidence_f1),
        step,
    )
    writer.add_scalar(
        f"tradeoff/{split}_hmean_macro_f1_evidence_f1",
        _hmean(macro_f1, evidence_f1),
        step,
    )
    writer.add_scalar(
        f"tradeoff/{split}_mean_accuracy_evidence_f1",
        float((accuracy + evidence_f1) / 2.0),
        step,
    )


@dataclass
class DistInfo:
    enabled: bool
    rank: int
    world_size: int
    local_rank: int
    is_main: bool


def _setup_distributed(timeout_minutes: int = 180) -> DistInfo:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    enabled = world_size > 1
    if enabled and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=timedelta(minutes=max(1, int(timeout_minutes))),
        )
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return DistInfo(
        enabled=enabled,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        is_main=(rank == 0),
    )


def _cleanup_distributed(info: DistInfo) -> None:
    if info.enabled and dist.is_initialized():
        dist.destroy_process_group()


def _rank0_print(info: DistInfo, message: str) -> None:
    if info.is_main:
        print(message)


def _barrier_if_needed(info: DistInfo) -> None:
    if info.enabled and dist.is_initialized():
        dist.barrier()


def _reduce_sum_scalar(value: float, device: torch.device, info: DistInfo) -> float:
    if not info.enabled:
        return float(value)
    tensor = torch.tensor([float(value)], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return summed / denom


def _infer_hidden_size(config: Any) -> int:
    for attr in ("hidden_size", "n_embd", "d_model"):
        value = getattr(config, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    raise ValueError("Could not infer hidden size from model config")


def _resolve_module_path(root: nn.Module, path: str) -> Optional[nn.Module]:
    cur: Any = root
    for part in path.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur if isinstance(cur, nn.Module) else None


def _get_backbone_module(model: nn.Module) -> nn.Module:
    module: nn.Module = model
    prefix = getattr(module, "base_model_prefix", "")
    if prefix and hasattr(module, prefix):
        pref_module = getattr(module, prefix)
        if isinstance(pref_module, nn.Module):
            module = pref_module

    
    visited = {id(module)}
    while True:
        advanced = False
        for candidate in ("model", "transformer", "backbone"):
            sub = getattr(module, candidate, None)
            if isinstance(sub, nn.Module) and id(sub) not in visited:
                module = sub
                visited.add(id(module))
                advanced = True
                break
        if not advanced:
            break
    return module


def _safe_list_literal(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
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


def _normalize_spans(
    spans_raw: Any,
    *,
    text: str,
    evidences_raw: Optional[Any] = None,
) -> List[Tuple[int, int]]:
    text = str(text or "")
    spans: List[Tuple[int, int]] = []
    candidates = _safe_list_literal(spans_raw)
    for item in candidates:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        try:
            st = int(item[0])
            ed = int(item[1])
        except Exception:
            continue
        st = max(0, min(len(text), st))
        ed = max(0, min(len(text), ed))
        if ed > st:
            spans.append((st, ed))

    if spans:
        return spans

    evidences = [str(v) for v in _safe_list_literal(evidences_raw) if str(v).strip()]
    used: List[Tuple[int, int]] = []
    for ev in evidences:
        pos = text.find(ev)
        while pos != -1:
            span = (pos, pos + len(ev))
            if all(span[1] <= s or span[0] >= e for s, e in used):
                used.append(span)
                break
            pos = text.find(ev, pos + 1)
    return used


def _maybe_peft_config(train_cfg: Dict[str, Any]) -> Optional[LoraConfig]:
    peft = str((train_cfg or {}).get("peft", "none")).lower()
    if peft not in {"lora", "dora"}:
        return None
    if LoraConfig is None or TaskType is None:
        raise ImportError("peft is required for LoRA/Dora training but not installed.")

    cfg_kwargs: Dict[str, Any] = {
        "r": int(train_cfg.get("lora_r", 16)),
        "lora_alpha": int(train_cfg.get("lora_alpha", 32)),
        "lora_dropout": float(train_cfg.get("lora_dropout", 0.05)),
        "bias": train_cfg.get("lora_bias", "none"),
        "task_type": TaskType.CAUSAL_LM,
        "target_modules": train_cfg.get(
            "lora_target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    }
    if "use_dora" in LoraConfig.__init__.__code__.co_varnames:
        cfg_kwargs["use_dora"] = peft == "dora"
    return LoraConfig(**cfg_kwargs)


@dataclass
class DecoderClassifierOutput(ModelOutput):
    logits: Optional[torch.Tensor] = None
    evidence_logits: Optional[torch.Tensor] = None
    pooled_output: Optional[torch.Tensor] = None


class TextLabelDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        text_col: str,
        label_col: str,
        category_col: Optional[str] = None,
        evidence_spans_col: Optional[str] = None,
        evidences_col: Optional[str] = None,
        use_evidence_loss_col: Optional[str] = None,
    ):
        required = [text_col, label_col]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in dataframe: {missing}")

        work = df.copy().reset_index(drop=True)
        work[text_col] = work[text_col].fillna("").astype(str)
        work[label_col] = work[label_col].astype(int)

        self.texts = work[text_col].tolist()
        self.labels = work[label_col].tolist()
        if category_col and category_col in work.columns:
            self.categories = work[category_col].fillna("").astype(str).tolist()
        else:
            self.categories = [""] * len(work)

        spans_col = evidence_spans_col if (evidence_spans_col and evidence_spans_col in work.columns) else None
        evidences_col_final = evidences_col if (evidences_col and evidences_col in work.columns) else None
        use_loss_col = use_evidence_loss_col if (use_evidence_loss_col and use_evidence_loss_col in work.columns) else None

        self.spans: List[List[Tuple[int, int]]] = []
        self.evidences: List[List[str]] = []
        self.use_evidence_loss: List[int] = []
        for idx in range(len(work)):
            text = str(self.texts[idx])
            label = int(self.labels[idx])
            spans_raw = work.iloc[idx][spans_col] if spans_col else None
            evidences_raw = work.iloc[idx][evidences_col_final] if evidences_col_final else None
            spans = _normalize_spans(spans_raw=spans_raw, text=text, evidences_raw=evidences_raw)
            evidences = [str(v).strip() for v in _safe_list_literal(evidences_raw) if str(v).strip()]

            if use_loss_col:
                raw_flag = work.iloc[idx][use_loss_col]
                try:
                    use_flag = int(raw_flag)
                except Exception:
                    use_flag = 0
            else:
                use_flag = 1 if (label == 1 and len(spans) > 0) else 0
            if label != 1:
                use_flag = 0
            if use_flag == 1 and not spans:
                use_flag = 0

            self.spans.append(spans)
            self.evidences.append(evidences)
            self.use_evidence_loss.append(use_flag)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return {
            "text": self.texts[index],
            "label": int(self.labels[index]),
            "category": str(self.categories[index]),
            "row_id": int(index),
            "spans": list(self.spans[index]),
            "evidences": list(self.evidences[index]),
            "use_evidence_loss": int(self.use_evidence_loss[index]),
        }


class TextClassificationCollator:
    def __init__(self, tokenizer, max_length: int, *, enable_evidence: bool = False):
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.enable_evidence = bool(enable_evidence)

    def __call__(self, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        texts = [str(row["text"]) for row in rows]
        labels = torch.tensor([int(row["label"]) for row in rows], dtype=torch.long)
        categories = [str(row.get("category", "")) for row in rows]
        row_ids = [int(row["row_id"]) for row in rows]
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_offsets_mapping=self.enable_evidence,
        )
        offset_mapping = enc.pop("offset_mapping", None)
        out: Dict[str, Any] = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
            "categories": categories,
            "row_ids": row_ids,
            "texts": texts,
            "gold_spans": [list(row.get("spans", []) or []) for row in rows],
        }
        if "token_type_ids" in enc:
            out["token_type_ids"] = enc["token_type_ids"]

        if self.enable_evidence:
            bsz, seqlen = int(enc["input_ids"].shape[0]), int(enc["input_ids"].shape[1])
            evidence_mask = torch.zeros((bsz, seqlen), dtype=torch.float32)
            use_evidence_loss = torch.zeros((bsz,), dtype=torch.float32)
            if offset_mapping is not None:
                for i, row in enumerate(rows):
                    spans = row.get("spans", []) or []
                    if int(row.get("use_evidence_loss", 0)) != 1 or not spans:
                        continue
                    token_offsets = offset_mapping[i].tolist()
                    for t, (st, ed) in enumerate(token_offsets):
                        if int(ed) <= int(st):
                            continue
                        hit = False
                        for s, e in spans:
                            if not (int(ed) <= int(s) or int(st) >= int(e)):
                                hit = True
                                break
                        if hit:
                            evidence_mask[i, t] = 1.0
                    if float(evidence_mask[i].sum().item()) > 0.0:
                        use_evidence_loss[i] = 1.0
            out["evidence_mask"] = evidence_mask
            out["use_evidence_loss"] = use_evidence_loss
            if offset_mapping is not None:
                out["offset_mapping"] = offset_mapping
        return out


class _CyclingPool:
    def __init__(self, indices: Sequence[int], rng: random.Random):
        self._base = list(indices)
        self._rng = rng
        self._buffer: List[int] = []
        self._refill()

    def _refill(self) -> None:
        self._buffer = self._base.copy()
        self._rng.shuffle(self._buffer)

    def draw(self) -> int:
        if not self._buffer:
            self._refill()
        return self._buffer.pop()


class BalancedLabelBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        labels: Sequence[int],
        batch_size: int,
        drop_last: bool = True,
        seed: int = 10,
    ):
        if batch_size <= 1:
            raise ValueError("batch_size must be > 1")
        self.labels = [int(label) for label in labels]
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0
        self.indices_by_label = {
            0: [idx for idx, label in enumerate(self.labels) if label == 0],
            1: [idx for idx, label in enumerate(self.labels) if label == 1],
        }
        if not self.indices_by_label[0] and not self.indices_by_label[1]:
            raise ValueError("No samples available for BalancedLabelBatchSampler")
        if self.drop_last:
            self.num_batches = len(self.labels) // self.batch_size
        else:
            self.num_batches = (len(self.labels) + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        available_labels = [label for label, indices in self.indices_by_label.items() if indices]
        pools = {
            label: _CyclingPool(indices=self.indices_by_label[label], rng=rng)
            for label in available_labels
        }
        for _ in range(self.num_batches):
            counts = {label: self.batch_size // max(1, len(available_labels)) for label in available_labels}
            remainder = self.batch_size - sum(counts.values())
            if remainder > 0:
                for label in rng.sample(available_labels, k=remainder):
                    counts[label] += 1
            batch: List[int] = []
            for label in available_labels:
                for _ in range(counts[label]):
                    batch.append(pools[label].draw())
            rng.shuffle(batch)
            if self.drop_last and len(batch) < self.batch_size:
                continue
            yield batch


class DecoderBinaryClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        *,
        backbone_lm: Optional[nn.Module] = None,
        torch_dtype: Optional[torch.dtype] = None,
        hf_token: Optional[str] = None,
        binary_classifier: bool = False,
        use_evidence_head: bool = True,
    ):
        super().__init__()
        if backbone_lm is not None:
            self.backbone_lm = backbone_lm
        else:
            kwargs: Dict[str, Any] = {"trust_remote_code": True}
            if torch_dtype is not None:
                kwargs["torch_dtype"] = torch_dtype
            if hf_token:
                kwargs["token"] = hf_token
            self.backbone_lm = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        hidden_size = _infer_hidden_size(self.backbone_lm.config)
        self.binary_classifier = bool(binary_classifier)
        self.use_evidence_head = bool(use_evidence_head)
        self.classifier = nn.Linear(hidden_size, 1 if self.binary_classifier else 2)
        self.evidence_classifier = nn.Linear(hidden_size, 1)
        lm_head = getattr(self.backbone_lm, "lm_head", None)
        if isinstance(lm_head, nn.Module):
            for param in lm_head.parameters():
                param.requires_grad = False

    def enable_gradient_checkpointing(self) -> None:
        if hasattr(self.backbone_lm, "gradient_checkpointing_enable"):
            self.backbone_lm.gradient_checkpointing_enable()
        if hasattr(self.backbone_lm, "config") and hasattr(self.backbone_lm.config, "use_cache"):
            self.backbone_lm.config.use_cache = False

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        compute_evidence: bool = True,
    ) -> DecoderClassifierOutput:
        backbone = _get_backbone_module(self.backbone_lm)
        model_inputs: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "return_dict": True,
        }
        if token_type_ids is not None:
            model_inputs["token_type_ids"] = token_type_ids
        outputs = backbone(**model_inputs)
        last_hidden = getattr(outputs, "last_hidden_state", None)
        if last_hidden is None:
            if isinstance(outputs, (tuple, list)) and outputs:
                last_hidden = outputs[0]
            else:
                raise RuntimeError("Backbone did not return last_hidden_state")
        pooled = _mean_pool(last_hidden=last_hidden, attention_mask=attention_mask)
        logits = self.classifier(pooled)
        evidence_logits = (
            self.evidence_classifier(last_hidden).squeeze(-1)
            if (self.use_evidence_head and compute_evidence)
            else None
        )
        return DecoderClassifierOutput(
            logits=logits,
            evidence_logits=evidence_logits,
            pooled_output=pooled,
        )


def _compute_class_weights(labels: Sequence[int]) -> torch.Tensor:
    counts = torch.bincount(torch.tensor([int(v) for v in labels], dtype=torch.long), minlength=2).float()
    total = counts.sum().clamp_min(1.0)
    return total / (counts.clamp_min(1.0) * float(len(counts)))


def _compute_binary_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    yt = [int(v) for v in y_true]
    yp = [int(v) for v in y_pred]
    metrics = {
        "accuracy": float(accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(yt, yp, average="weighted", zero_division=0)),
    }
    for label in (0, 1):
        metrics[f"label_{label}_precision"] = float(
            precision_score(yt, yp, pos_label=label, zero_division=0)
        )
        metrics[f"label_{label}_recall"] = float(
            recall_score(yt, yp, pos_label=label, zero_division=0)
        )
        metrics[f"label_{label}_f1"] = float(
            f1_score(yt, yp, labels=[label], average="macro", zero_division=0)
        )
    metrics["negative_precision"] = metrics["label_0_precision"]
    metrics["negative_recall"] = metrics["label_0_recall"]
    metrics["negative_f1"] = metrics["label_0_f1"]
    metrics["positive_precision"] = metrics["label_1_precision"]
    metrics["positive_recall"] = metrics["label_1_recall"]
    metrics["positive_f1"] = metrics["label_1_f1"]
    return metrics


def _compute_category_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    categories: Sequence[str],
) -> Dict[str, float]:
    if not categories:
        return {}
    frame = pd.DataFrame(
        {
            "y_true": [int(v) for v in y_true],
            "y_pred": [int(v) for v in y_pred],
            "category": [str(v or "").strip().lower() for v in categories],
        }
    )
    frame = frame[frame["category"] != ""]
    if frame.empty:
        return {}

    out: Dict[str, float] = {}
    weighted_macro_num = 0.0
    weighted_macro_den = 0.0
    for cat, group in frame.groupby("category"):
        metrics = _compute_binary_metrics(group["y_true"].tolist(), group["y_pred"].tolist())
        n = float(len(group))
        weighted_macro_num += n * float(metrics.get("macro_f1", 0.0))
        weighted_macro_den += n
        out[f"category_{cat}_size"] = n
        out[f"category_{cat}_macro_f1"] = float(metrics.get("macro_f1", 0.0))
        out[f"category_{cat}_positive_recall"] = float(metrics.get("label_1_recall", 0.0))

    if weighted_macro_den > 0:
        out["category_weighted_macro_f1"] = weighted_macro_num / weighted_macro_den
    return out


def _token_flags_to_char_spans(
    token_flags: Sequence[bool],
    token_offsets: Sequence[Sequence[int]],
    *,
    merge_gap: int = 1,
) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    cur: Optional[List[int]] = None
    for flag, off in zip(token_flags, token_offsets):
        if not flag:
            continue
        if not isinstance(off, (list, tuple)) or len(off) != 2:
            continue
        try:
            st = int(off[0])
            ed = int(off[1])
        except Exception:
            continue
        if ed <= st:
            continue
        if cur is None:
            cur = [st, ed]
            continue
        if st <= (cur[1] + int(merge_gap)):
            cur[1] = max(cur[1], ed)
        else:
            spans.append((int(cur[0]), int(cur[1])))
            cur = [st, ed]
    if cur is not None:
        spans.append((int(cur[0]), int(cur[1])))
    return spans


def _token_probs_to_scored_char_spans(
    token_probs: Sequence[float],
    token_offsets: Sequence[Sequence[int]],
    *,
    threshold: float,
    merge_gap: int = 1,
) -> List[Tuple[int, int, float]]:
    scored_spans: List[Tuple[int, int, float]] = []
    cur: Optional[List[float]] = None
    cur_scores: List[float] = []
    for prob, off in zip(token_probs, token_offsets):
        try:
            prob_f = float(prob)
        except Exception:
            continue
        if prob_f <= float(threshold):
            continue
        if not isinstance(off, (list, tuple)) or len(off) != 2:
            continue
        try:
            st = int(off[0])
            ed = int(off[1])
        except Exception:
            continue
        if ed <= st:
            continue
        if cur is None:
            cur = [st, ed]
            cur_scores = [prob_f]
            continue
        if st <= (int(cur[1]) + int(merge_gap)):
            cur[1] = max(int(cur[1]), ed)
            cur_scores.append(prob_f)
        else:
            if cur_scores:
                scored_spans.append((int(cur[0]), int(cur[1]), float(sum(cur_scores) / len(cur_scores))))
            cur = [st, ed]
            cur_scores = [prob_f]
    if cur is not None and cur_scores:
        scored_spans.append((int(cur[0]), int(cur[1]), float(sum(cur_scores) / len(cur_scores))))
    return scored_spans


def _classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    binary_classifier: bool,
    class_weights: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if binary_classifier:
        logit_1d = logits.view(-1).float()
        labels_f = labels.float()
        pos_weight = None
        if class_weights is not None and class_weights.numel() >= 2:
            denom = class_weights[0].clamp_min(1e-12)
            pos_weight = (class_weights[1] / denom).detach().float()
        ce_loss = F.binary_cross_entropy_with_logits(
            logit_1d,
            labels_f,
            pos_weight=pos_weight,
        )
        probs = torch.sigmoid(logit_1d)
        return ce_loss, probs
    ce_loss = F.cross_entropy(logits, labels, weight=class_weights)
    probs = torch.softmax(logits, dim=-1)[:, 1]
    return ce_loss, probs


def _evidence_token_loss(
    evidence_logits: Optional[torch.Tensor],
    evidence_mask: Optional[torch.Tensor],
    attention_mask: torch.Tensor,
    use_evidence_loss: Optional[torch.Tensor],
    *,
    alpha: float,
    beta: float,
    negative_downsample_ratio: int,
    allow_negative_only_samples: bool = False,
    negative_only_max_tokens: int = 128,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if (
        evidence_logits is None
        or evidence_mask is None
        or use_evidence_loss is None
        or evidence_logits.ndim != 2
    ):
        zero = attention_mask.sum() * 0.0
        return zero, {"evidence_active_samples": 0.0, "evidence_pos_tokens": 0.0, "evidence_neg_tokens": 0.0}

    bsz = int(evidence_logits.size(0))
    total_loss = evidence_logits.sum() * 0.0
    active = 0
    total_pos = 0
    total_neg = 0

    for i in range(bsz):
        if int(use_evidence_loss[i].item()) != 1:
            continue
        valid_mask = attention_mask[i] > 0
        pos_mask = (evidence_mask[i] > 0.5) & valid_mask
        neg_mask = (~pos_mask) & valid_mask
        pos_idx = torch.nonzero(pos_mask, as_tuple=False).squeeze(-1)
        neg_idx = torch.nonzero(neg_mask, as_tuple=False).squeeze(-1)
        if pos_idx.numel() == 0:
            if not allow_negative_only_samples or neg_idx.numel() == 0:
                continue
            max_neg = int(negative_only_max_tokens)
            if max_neg > 0 and neg_idx.numel() > max_neg:
                perm = torch.randperm(int(neg_idx.numel()), device=neg_idx.device)[:max_neg]
                neg_idx = neg_idx[perm]
            neg_logits = evidence_logits[i, neg_idx]
            neg_targets = torch.zeros_like(neg_logits)
            neg_loss = F.binary_cross_entropy_with_logits(neg_logits, neg_targets, reduction="mean")
            total_loss = total_loss + (float(beta) * neg_loss)
            active += 1
            total_neg += int(neg_idx.numel())
            continue

        if negative_downsample_ratio > 0 and neg_idx.numel() > 0:
            max_neg = int(negative_downsample_ratio * int(pos_idx.numel()))
            if max_neg > 0 and neg_idx.numel() > max_neg:
                perm = torch.randperm(int(neg_idx.numel()), device=neg_idx.device)[:max_neg]
                neg_idx = neg_idx[perm]

        pos_logits = evidence_logits[i, pos_idx]
        pos_targets = torch.ones_like(pos_logits)
        pos_loss = F.binary_cross_entropy_with_logits(pos_logits, pos_targets, reduction="mean")

        if neg_idx.numel() > 0:
            neg_logits = evidence_logits[i, neg_idx]
            neg_targets = torch.zeros_like(neg_logits)
            neg_loss = F.binary_cross_entropy_with_logits(neg_logits, neg_targets, reduction="mean")
        else:
            neg_loss = pos_loss * 0.0

        sample_loss = (float(alpha) * pos_loss) + (float(beta) * neg_loss)
        total_loss = total_loss + sample_loss
        active += 1
        total_pos += int(pos_idx.numel())
        total_neg += int(neg_idx.numel())

    if active == 0:
        
        
        return total_loss, {"evidence_active_samples": 0.0, "evidence_pos_tokens": 0.0, "evidence_neg_tokens": 0.0}

    return total_loss / float(active), {
        "evidence_active_samples": float(active),
        "evidence_pos_tokens": float(total_pos),
        "evidence_neg_tokens": float(total_neg),
    }


def _build_train_dataloader(
    dataset: TextLabelDataset,
    collator: TextClassificationCollator,
    *,
    batch_size: int,
    num_workers: int,
    seed: int,
    use_balanced_sampler: bool,
    dist_info: DistInfo,
) -> Tuple[DataLoader, Optional[Any]]:
    loader_kwargs: Dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    if dist_info.enabled:
        if use_balanced_sampler and dist_info.is_main:
            print("[info] DDP enabled: falling back to DistributedSampler instead of balanced label batching.")
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist_info.world_size,
            rank=dist_info.rank,
            shuffle=True,
            drop_last=True,
            seed=seed,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=True,
            collate_fn=collator,
            **loader_kwargs,
        )
        return loader, sampler

    if use_balanced_sampler:
        sampler = BalancedLabelBatchSampler(
            labels=dataset.labels,
            batch_size=batch_size,
            drop_last=True,
            seed=seed,
        )
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collator,
            **loader_kwargs,
        )
        return loader, sampler

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collator,
        **loader_kwargs,
    )
    return loader, None


def _build_eval_dataloader(
    dataset: TextLabelDataset,
    collator: TextClassificationCollator,
    *,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    loader_kwargs: Dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collator,
        **loader_kwargs,
    )


@dataclass
class EvalResult:
    metrics: Dict[str, float]
    predictions: pd.DataFrame


def _evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    *,
    loss_mode: str,
    binary_classifier: bool,
    class_weights: Optional[torch.Tensor],
    decision_threshold: float = 0.5,
    evidence_enabled: bool,
    label_loss_weight: float,
    evidence_lambda: float,
    evidence_alpha: float,
    evidence_beta: float,
    evidence_negative_downsample_ratio: int,
    evidence_threshold: float,
    evidence_metric_weight: float = 0.05,
    evidence_max_pred_spans: int = 3,
    evidence_warmup_epochs: int = 0,
    amp_dtype: Optional[torch.dtype],
) -> EvalResult:
    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[float] = []
    categories: List[str] = []
    row_ids: List[int] = []
    pred_evidence_spans_all: List[List[List[int]]] = []
    pred_evidence_texts_all: List[List[str]] = []
    gold_evidence_spans_all: List[List[List[int]]] = []
    total_loss = 0.0
    total_ce_loss = 0.0
    total_evidence_loss = 0.0
    total_samples = 0
    evi_tp = 0.0
    evi_fp = 0.0
    evi_fn = 0.0

    model.eval()
    autocast_enabled = torch.cuda.is_available() and amp_dtype in (torch.float16, torch.bfloat16)

    with torch.no_grad():
        for batch in dataloader:
            labels = batch["labels"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            evidence_mask = batch.get("evidence_mask")
            if evidence_mask is not None:
                evidence_mask = evidence_mask.to(device)
            use_evidence_loss = batch.get("use_evidence_loss")
            if use_evidence_loss is not None:
                use_evidence_loss = use_evidence_loss.to(device)

            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=amp_dtype)
                if autocast_enabled
                else nullcontext()
            )
            with autocast_ctx:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    compute_evidence=evidence_enabled,
                )
                logits = outputs.logits
                ce_loss, probs = _classification_loss(
                    logits=logits,
                    labels=labels,
                    binary_classifier=binary_classifier,
                    class_weights=class_weights,
                )
                if evidence_enabled:
                    evidence_loss, _ = _evidence_token_loss(
                        evidence_logits=outputs.evidence_logits,
                        evidence_mask=evidence_mask,
                        attention_mask=attention_mask,
                        use_evidence_loss=use_evidence_loss,
                        alpha=evidence_alpha,
                        beta=evidence_beta,
                        negative_downsample_ratio=evidence_negative_downsample_ratio,
                    )
                else:
                    evidence_loss = logits.sum() * 0.0
                loss = (
                    (float(label_loss_weight) * ce_loss)
                    + (float(evidence_lambda) * evidence_loss if evidence_enabled else 0.0)
                )

            if binary_classifier:
                preds = (probs > float(decision_threshold)).long()
            else:
                preds = torch.argmax(logits, dim=-1)
            batch_size = int(labels.size(0))

            total_loss += float(loss.detach().item()) * batch_size
            total_ce_loss += float(ce_loss.detach().item()) * batch_size
            total_evidence_loss += float(evidence_loss.detach().item()) * batch_size
            total_samples += batch_size

            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(preds.detach().cpu().tolist())
            y_prob.extend(probs.detach().cpu().tolist())
            categories.extend([str(value) for value in batch.get("categories", [""] * batch_size)])
            row_ids.extend([int(value) for value in batch.get("row_ids", list(range(batch_size)))])
            texts_batch = [str(value) for value in batch.get("texts", [""] * batch_size)]
            gold_spans_batch = batch.get("gold_spans", [[] for _ in range(batch_size)])
            offset_mapping_batch = batch.get("offset_mapping")

            pred_spans_batch: List[List[List[int]]] = [[] for _ in range(batch_size)]
            pred_texts_batch: List[List[str]] = [[] for _ in range(batch_size)]
            if evidence_enabled and outputs.evidence_logits is not None and offset_mapping_batch is not None:
                token_prob = torch.sigmoid(outputs.evidence_logits.detach()).cpu()
                attn_cpu = attention_mask.detach().cpu()
                for i in range(batch_size):
                    if int(preds[i].detach().cpu().item()) != 1:
                        continue
                    offs_i = offset_mapping_batch[i]
                    if isinstance(offs_i, torch.Tensor):
                        offs_i = offs_i.tolist()
                    probs_i = token_prob[i]
                    if isinstance(probs_i, torch.Tensor):
                        probs_i = probs_i.tolist()
                    attn_i = attn_cpu[i]
                    if isinstance(attn_i, torch.Tensor):
                        attn_i = attn_i.tolist()
                    valid_token_probs = [
                        float(prob_j) if int(attn_j) > 0 else 0.0
                        for prob_j, attn_j in zip(probs_i, attn_i)
                    ]
                    spans_i = _token_probs_to_scored_char_spans(
                        valid_token_probs,
                        offs_i,
                        threshold=float(evidence_threshold),
                    )
                    if int(evidence_max_pred_spans) > 0 and len(spans_i) > int(evidence_max_pred_spans):
                        spans_i = sorted(
                            spans_i,
                            key=lambda item: (-float(item[2]), int(item[0]), int(item[1])),
                        )[: int(evidence_max_pred_spans)]
                        spans_i = sorted(spans_i, key=lambda item: (int(item[0]), int(item[1])))
                    text_i = texts_batch[i]
                    clipped_spans: List[List[int]] = []
                    clipped_texts: List[str] = []
                    for st, ed, _score in spans_i:
                        st2 = max(0, min(len(text_i), int(st)))
                        ed2 = max(0, min(len(text_i), int(ed)))
                        if ed2 <= st2:
                            continue
                        snippet = text_i[st2:ed2]
                        if not snippet.strip():
                            continue
                        clipped_spans.append([st2, ed2])
                        clipped_texts.append(snippet)
                    pred_spans_batch[i] = clipped_spans
                    pred_texts_batch[i] = clipped_texts

            for i in range(batch_size):
                gs = gold_spans_batch[i] if i < len(gold_spans_batch) else []
                gs_norm: List[List[int]] = []
                if isinstance(gs, (list, tuple)):
                    for item in gs:
                        if isinstance(item, (list, tuple)) and len(item) == 2:
                            try:
                                gs_norm.append([int(item[0]), int(item[1])])
                            except Exception:
                                pass
                gold_evidence_spans_all.append(gs_norm)
            pred_evidence_spans_all.extend(pred_spans_batch)
            pred_evidence_texts_all.extend(pred_texts_batch)

            if (
                evidence_enabled
                and outputs.evidence_logits is not None
                and evidence_mask is not None
                and use_evidence_loss is not None
            ):
                token_pred = (torch.sigmoid(outputs.evidence_logits) > float(evidence_threshold)).float()
                valid = attention_mask.float()
                active = use_evidence_loss.view(-1, 1)
                target = evidence_mask.float()
                token_pred = token_pred * valid * active
                target = target * valid * active
                evi_tp += float((token_pred * target).sum().item())
                evi_fp += float((token_pred * (1.0 - target)).sum().item())
                evi_fn += float(((1.0 - token_pred) * target).sum().item())

    metrics = _compute_binary_metrics(y_true=y_true, y_pred=y_pred)
    metrics.update(_compute_category_metrics(y_true=y_true, y_pred=y_pred, categories=categories))
    if "category_weighted_macro_f1" not in metrics:
        metrics["category_weighted_macro_f1"] = float(metrics.get("macro_f1", 0.0))
    metrics["loss"] = total_loss / max(1, total_samples)
    metrics["ce_loss"] = total_ce_loss / max(1, total_samples)
    metrics["evidence_loss"] = total_evidence_loss / max(1, total_samples)
    evi_precision = evi_tp / max(1e-12, evi_tp + evi_fp)
    evi_recall = evi_tp / max(1e-12, evi_tp + evi_fn)
    evi_f1 = (2.0 * evi_precision * evi_recall) / max(1e-12, evi_precision + evi_recall)
    metrics["evidence_token_precision"] = float(evi_precision)
    metrics["evidence_token_recall"] = float(evi_recall)
    metrics["evidence_token_f1"] = float(evi_f1)
    metrics["evidence_aware_score"] = float(metrics.get("category_weighted_macro_f1", metrics.get("macro_f1", 0.0))) + (
        float(evidence_metric_weight) * float(metrics["evidence_token_f1"])
    )
    predictions = pd.DataFrame(
        {
            "row_id": row_ids,
            "label": y_true,
            "predicted": y_pred,
            "prob_label_1": y_prob,
            "category": categories,
            "pred_evidence_spans": [json.dumps(v, ensure_ascii=False) for v in pred_evidence_spans_all],
            "pred_evidence_texts": [json.dumps(v, ensure_ascii=False) for v in pred_evidence_texts_all],
            "gold_evidence_spans": [json.dumps(v, ensure_ascii=False) for v in gold_evidence_spans_all],
        }
    ).sort_values("row_id", kind="stable")
    return EvalResult(metrics=metrics, predictions=predictions)


def _save_checkpoint(
    model: nn.Module,
    tokenizer,
    checkpoint_dir: str,
    metadata: Dict[str, Any],
) -> None:
    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    base = _unwrap_model(model)
    base.backbone_lm.save_pretrained(str(path))
    tokenizer.save_pretrained(str(path))
    torch.save(
        {
            "classifier": base.classifier.state_dict(),
            "evidence_classifier": base.evidence_classifier.state_dict(),
        },
        str(path / "decoder_heads.pt"),
    )
    _write_json(str(path / "metadata.json"), metadata)


def _resolve_checkpoint_dir(model_ref: str) -> str:
    if not model_ref:
        raise ValueError("A checkpoint directory or run directory is required")

    candidate = Path(model_ref).expanduser()
    if candidate.is_dir() and (candidate / "decoder_heads.pt").exists():
        return str(candidate)

    direct_best = candidate / "best_checkpoint"
    if direct_best.is_dir() and (direct_best / "decoder_heads.pt").exists():
        return str(direct_best)

    summary_path = candidate / "summary.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        summary_checkpoint = summary.get("checkpoint_dir")
        if summary_checkpoint:
            summary_checkpoint_path = Path(str(summary_checkpoint)).expanduser()
            if summary_checkpoint_path.is_dir() and (summary_checkpoint_path / "decoder_heads.pt").exists():
                return str(summary_checkpoint_path)
        if direct_best.is_dir() and (direct_best / "decoder_heads.pt").exists():
            return str(direct_best)

    raise FileNotFoundError(
        f"Could not resolve decoder checkpoint from: {model_ref}. "
        "Expected a checkpoint dir with decoder_heads.pt or a run dir containing best_checkpoint/."
    )


def _load_decoder_checkpoint_metadata(checkpoint_dir: str) -> Dict[str, Any]:
    metadata_path = Path(checkpoint_dir) / "metadata.json"
    if not metadata_path.exists():
        return {}
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f) or {}


def load_decoder_classifier_for_eval(
    model_ref: str,
    *,
    device: torch.device,
    torch_dtype: Optional[torch.dtype] = None,
    hf_token: Optional[str] = None,
) -> Tuple[DecoderBinaryClassifier, Any, Dict[str, Any], str]:
    checkpoint_dir = _resolve_checkpoint_dir(model_ref)
    metadata = _load_decoder_checkpoint_metadata(checkpoint_dir)
    heads = torch.load(os.path.join(checkpoint_dir, "decoder_heads.pt"), map_location="cpu")

    peft_mode = str(metadata.get("peft_mode", "none")).lower()
    base_model_name = str(metadata.get("base_model_name", checkpoint_dir))
    binary_classifier = bool(metadata.get("binary_classifier", False))
    use_evidence_head = bool(metadata.get("evidence_enabled", True))

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    backbone_lm = None
    if peft_mode in {"lora", "dora"}:
        if PeftModel is None:
            raise ImportError("peft is required to load LoRA/Dora decoder checkpoints.")
        kwargs: Dict[str, Any] = {"trust_remote_code": True}
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
        if hf_token:
            kwargs["token"] = hf_token
        base_backbone = AutoModelForCausalLM.from_pretrained(base_model_name, **kwargs)
        backbone_lm = PeftModel.from_pretrained(base_backbone, checkpoint_dir)

    model = DecoderBinaryClassifier(
        model_name=base_model_name if backbone_lm is not None else checkpoint_dir,
        backbone_lm=backbone_lm,
        torch_dtype=torch_dtype,
        hf_token=hf_token,
        binary_classifier=binary_classifier,
        use_evidence_head=use_evidence_head,
    )
    model.classifier.load_state_dict(heads["classifier"])
    if "evidence_classifier" in heads:
        model.evidence_classifier.load_state_dict(heads["evidence_classifier"])
    model.to(device)
    model.eval()
    return model, tokenizer, metadata, checkpoint_dir


def evaluate_decoder_classifier_from_config(
    cfg: Dict[str, Any],
    run_dir: str,
    *,
    hf_token: Optional[str] = None,
) -> str:
    model_ref = _get_cfg(cfg, "model.checkpoint_dir") or _get_cfg(cfg, "model.model_name")
    if not model_ref:
        raise ValueError("model.checkpoint_dir or model.model_name is required")

    dtype = _to_dtype(_get_cfg(cfg, "model.dtype", "auto"))
    device_id = int(_get_cfg(cfg, "run.device_id", 0))
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    model, tokenizer, metadata, checkpoint_dir = load_decoder_classifier_for_eval(
        model_ref=model_ref,
        device=device,
        torch_dtype=dtype,
        hf_token=hf_token,
    )

    text_col = _get_cfg(cfg, "data.text_col", "text")
    label_col = _get_cfg(cfg, "data.label_col", "label")
    category_col = _get_cfg(cfg, "data.category_col", "category")
    evidence_spans_col = _get_cfg(cfg, "data.evidence_spans_col", "spans")
    evidences_col = _get_cfg(cfg, "data.evidences_col", "evidences")
    use_evidence_loss_col = _get_cfg(cfg, "data.use_evidence_loss_col", "use_evidence_loss")
    eval_csv = _get_cfg(cfg, "data.eval_csv")
    test_csv = _get_cfg(cfg, "data.test_csv")
    if not eval_csv and not test_csv:
        raise ValueError("At least one of data.eval_csv or data.test_csv is required")

    max_length = int(_get_cfg(cfg, "eval.max_length", _get_cfg(cfg, "train.max_length", 1024)))
    batch_size = int(_get_cfg(cfg, "eval.batch_size", _get_cfg(cfg, "train.eval_batch_size", 8)))
    num_workers = int(_get_cfg(cfg, "eval.dataloader_num_workers", _get_cfg(cfg, "train.dataloader_num_workers", 0)))
    loss_mode = str(_get_cfg(cfg, "eval.loss_mode", metadata.get("loss_mode", "weighted_ce"))).strip().lower()
    valid_loss_modes = {
        "weighted_ce",
        "bce",
        "weighted_bce",
        "multitask_evidence",
    }
    if loss_mode not in valid_loss_modes:
        raise ValueError(f"eval.loss_mode must be one of {sorted(valid_loss_modes)}")
    binary_classifier = bool(_get_cfg(cfg, "eval.binary_classifier", metadata.get("binary_classifier", False)))
    decision_threshold = float(_get_cfg(cfg, "eval.decision_threshold", metadata.get("decision_threshold", 0.5)))

    class_weights_raw = _get_cfg(cfg, "eval.class_weights", metadata.get("class_weights"))
    class_weights = (
        torch.tensor(class_weights_raw, dtype=torch.float32, device=device)
        if class_weights_raw is not None
        else None
    )
    evidence_enabled = bool(_get_cfg(cfg, "eval.evidence.enabled", metadata.get("evidence_enabled", False)))
    evidence_lambda = float(_get_cfg(cfg, "eval.evidence.lambda", metadata.get("evidence_lambda", 0.2)))
    evidence_alpha = float(_get_cfg(cfg, "eval.evidence.alpha", metadata.get("evidence_alpha", 1.0)))
    evidence_beta = float(_get_cfg(cfg, "eval.evidence.beta", metadata.get("evidence_beta", 0.25)))
    evidence_negative_downsample_ratio = int(
        _get_cfg(cfg, "eval.evidence.negative_downsample_ratio", metadata.get("evidence_negative_downsample_ratio", 0))
    )
    evidence_threshold = float(_get_cfg(cfg, "eval.evidence.threshold", metadata.get("evidence_threshold", 0.55)))
    evidence_metric_weight = float(
        _get_cfg(cfg, "eval.evidence.metric_weight", metadata.get("evidence_metric_weight", 0.05))
    )
    evidence_max_pred_spans = int(
        _get_cfg(cfg, "eval.evidence.max_pred_spans", metadata.get("evidence_max_pred_spans", 3))
    )

    collator = TextClassificationCollator(
        tokenizer=tokenizer,
        max_length=max_length,
        enable_evidence=evidence_enabled,
    )
    metrics_payload: Dict[str, Any] = {
        "checkpoint_dir": checkpoint_dir,
        "loss_mode": loss_mode,
        "class_weights": class_weights_raw,
    }

    for split_name, csv_path in (("eval", eval_csv), ("test", test_csv)):
        if not csv_path:
            continue
        df = _read_csv_normalized(csv_path)
        dataset = TextLabelDataset(
            df=df,
            text_col=text_col,
            label_col=label_col,
            category_col=category_col,
            evidence_spans_col=evidence_spans_col,
            evidences_col=evidences_col,
            use_evidence_loss_col=use_evidence_loss_col,
        )
        loader = _build_eval_dataloader(
            dataset=dataset,
            collator=collator,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        result = _evaluate(
            model=model,
            dataloader=loader,
            device=device,
            loss_mode=loss_mode,
            binary_classifier=binary_classifier,
            class_weights=class_weights,
            decision_threshold=decision_threshold,
            evidence_enabled=evidence_enabled,
            label_loss_weight=float(metadata.get("label_loss_weight", 1.0)),
            evidence_lambda=evidence_lambda,
            evidence_alpha=evidence_alpha,
            evidence_beta=evidence_beta,
            evidence_negative_downsample_ratio=evidence_negative_downsample_ratio,
            evidence_threshold=evidence_threshold,
            evidence_metric_weight=evidence_metric_weight,
            evidence_max_pred_spans=evidence_max_pred_spans,
            amp_dtype=dtype,
        )
        result.predictions.to_csv(os.path.join(run_dir, f"{split_name}_predictions.csv"), index=False)
        metrics_payload[split_name] = result.metrics

    _write_json(os.path.join(run_dir, "metrics.json"), metrics_payload)
    return run_dir


def run_decoder_classifier_from_config(
    cfg: Dict[str, Any],
    run_dir: str,
    *,
    hf_token: Optional[str] = None,
) -> str:
    ddp_timeout_minutes = int(_get_cfg(cfg, "run.ddp_timeout_minutes", 180))
    dist_info = _setup_distributed(timeout_minutes=ddp_timeout_minutes)
    writer = None
    try:
        seed = int(_get_cfg(cfg, "run.seed", 10))
        set_seed(seed=seed + dist_info.rank, deterministic=True, benchmark=False)

        model_type = _get_cfg(cfg, "model.model_type", "Decoder")
        model_name_cfg = _get_cfg(cfg, "model.model_name")
        model_id = _get_cfg(cfg, "model.model_id")
        model_name = resolve_model_name(model_type=model_type, model_name=model_name_cfg, model_id=model_id)
        dtype = _to_dtype(_get_cfg(cfg, "model.dtype", "auto"))

        text_col = _get_cfg(cfg, "data.text_col", "text")
        label_col = _get_cfg(cfg, "data.label_col", "label")
        category_col = _get_cfg(cfg, "data.category_col", "category")
        evidence_spans_col = _get_cfg(cfg, "data.evidence_spans_col", "spans")
        evidences_col = _get_cfg(cfg, "data.evidences_col", "evidences")
        use_evidence_loss_col = _get_cfg(cfg, "data.use_evidence_loss_col", "use_evidence_loss")
        train_csv = _get_cfg(cfg, "data.train_csv")
        eval_csv = _get_cfg(cfg, "data.eval_csv")
        test_csv = _get_cfg(cfg, "data.test_csv")
        if not train_csv:
            raise ValueError("data.train_csv is required")

        train_df = _read_csv_normalized(train_csv)
        eval_df = _read_csv_normalized(eval_csv) if eval_csv else None
        test_df = _read_csv_normalized(test_csv) if test_csv else None

        max_train_samples = int(_get_cfg(cfg, "data.max_train_samples", 0) or 0)
        max_eval_samples = int(_get_cfg(cfg, "data.max_eval_samples", 0) or 0)
        max_test_samples = int(_get_cfg(cfg, "data.max_test_samples", 0) or 0)
        if max_train_samples > 0:
            train_df = train_df.head(max_train_samples).copy()
        if eval_df is not None and max_eval_samples > 0:
            eval_df = eval_df.head(max_eval_samples).copy()
        if test_df is not None and max_test_samples > 0:
            test_df = test_df.head(max_test_samples).copy()

        loss_mode = str(_get_cfg(cfg, "train.loss_mode", "weighted_ce")).strip().lower()
        valid_loss_modes = {
            "weighted_ce",
            "bce",
            "weighted_bce",
            "multitask_evidence",
        }
        if loss_mode not in valid_loss_modes:
            raise ValueError(f"train.loss_mode must be one of {sorted(valid_loss_modes)}")
        use_weighted_ce = loss_mode in {"weighted_ce", "weighted_bce"}
        if loss_mode == "multitask_evidence":
            use_weighted_ce = True
        binary_classifier = bool(_get_cfg(cfg, "train.binary_classifier", False))
        decision_threshold = float(_get_cfg(cfg, "train.decision_threshold", 0.5))

        epochs = int(_get_cfg(cfg, "train.epochs", 3))
        batch_size = int(_get_cfg(cfg, "train.batch_size", 8))
        eval_batch_size = int(_get_cfg(cfg, "train.eval_batch_size", batch_size))
        grad_accum = int(_get_cfg(cfg, "train.grad_accum", 1))
        lr = float(_get_cfg(cfg, "train.lr", 2e-5))
        weight_decay = float(_get_cfg(cfg, "train.weight_decay", 0.01))
        warmup_ratio = float(_get_cfg(cfg, "train.warmup_ratio", 0.03))
        max_length = int(_get_cfg(cfg, "train.max_length", 1024))
        max_grad_norm = float(_get_cfg(cfg, "train.max_grad_norm", 1.0))
        num_workers = int(_get_cfg(cfg, "train.dataloader_num_workers", 0))
        use_balanced_sampler = bool(_get_cfg(cfg, "train.use_balanced_batch_sampler", True))
        gradient_checkpointing = bool(_get_cfg(cfg, "train.gradient_checkpointing", False))
        report_to = _get_cfg(cfg, "train.report_to", "tensorboard")
        logging_steps = int(_get_cfg(cfg, "train.logging_steps", 20))
        evidence_enabled = bool(_get_cfg(cfg, "train.evidence.enabled", False))
        label_loss_weight = float(_get_cfg(cfg, "train.label_loss_weight", 1.0))
        evidence_lambda = float(_get_cfg(cfg, "train.evidence.lambda", 0.2))
        evidence_alpha = float(_get_cfg(cfg, "train.evidence.alpha", 1.0))
        evidence_beta = float(_get_cfg(cfg, "train.evidence.beta", 0.25))
        evidence_negative_downsample_ratio = int(_get_cfg(cfg, "train.evidence.negative_downsample_ratio", 0))
        evidence_threshold = float(_get_cfg(cfg, "train.evidence.threshold", 0.55))
        evidence_metric_weight = float(_get_cfg(cfg, "train.evidence.metric_weight", 0.05))
        evidence_max_pred_spans = int(_get_cfg(cfg, "train.evidence.max_pred_spans", 3))
        evidence_warmup_epochs = int(_get_cfg(cfg, "train.evidence.warmup_epochs", 0))
        metric_for_best = str(_get_cfg(cfg, "train.metric_for_best_model", "macro_f1"))
        greater_is_better = bool(_get_cfg(cfg, "train.greater_is_better", True))
        early_stopping_threshold = float(_get_cfg(cfg, "train.early_stopping_threshold", 0.0))
        min_epoch_for_best_model = _get_cfg(cfg, "train.min_epoch_for_best_model")
        if min_epoch_for_best_model is None:
            min_epoch_for_best_model = (
                int(evidence_warmup_epochs) + 1
                if (evidence_enabled and metric_for_best == "evidence_aware_score")
                else 1
            )
        min_epoch_for_best_model = max(1, int(min_epoch_for_best_model))
        device = torch.device(
            f"cuda:{dist_info.local_rank}" if torch.cuda.is_available() else "cpu"
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=hf_token)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        model = DecoderBinaryClassifier(
            model_name=model_name,
            torch_dtype=dtype,
            hf_token=hf_token,
            binary_classifier=binary_classifier,
            use_evidence_head=evidence_enabled,
        )
        if gradient_checkpointing:
            model.enable_gradient_checkpointing()
        peft_cfg = _maybe_peft_config(_get_cfg(cfg, "train", {}) or {})
        peft_mode = str(_get_cfg(cfg, "train.peft", "none")).lower()
        if peft_cfg is not None:
            if get_peft_model is None:
                raise ImportError("peft is required for LoRA/Dora training but not installed.")
            model.backbone_lm = get_peft_model(model.backbone_lm, peft_cfg)
            _rank0_print(dist_info, f"[decoder_cls] enabled PEFT mode={peft_mode}")
        model.to(device)

        if dist_info.enabled:
            ddp_find_unused_parameters = bool(
                _get_cfg(cfg, "train.ddp_find_unused_parameters", False)
            )
            model = DDP(
                model,
                device_ids=[dist_info.local_rank] if torch.cuda.is_available() else None,
                output_device=dist_info.local_rank if torch.cuda.is_available() else None,
                find_unused_parameters=ddp_find_unused_parameters,
            )

        train_dataset = TextLabelDataset(
            df=train_df,
            text_col=text_col,
            label_col=label_col,
            category_col=category_col,
            evidence_spans_col=evidence_spans_col,
            evidences_col=evidences_col,
            use_evidence_loss_col=use_evidence_loss_col,
        )
        eval_dataset = (
            TextLabelDataset(
                df=eval_df,
                text_col=text_col,
                label_col=label_col,
                category_col=category_col,
                evidence_spans_col=evidence_spans_col,
                evidences_col=evidences_col,
                use_evidence_loss_col=use_evidence_loss_col,
            )
            if eval_df is not None
            else None
        )
        test_dataset = (
            TextLabelDataset(
                df=test_df,
                text_col=text_col,
                label_col=label_col,
                category_col=category_col,
                evidence_spans_col=evidence_spans_col,
                evidences_col=evidences_col,
                use_evidence_loss_col=use_evidence_loss_col,
            )
            if test_df is not None
            else None
        )
        collator = TextClassificationCollator(
            tokenizer=tokenizer,
            max_length=max_length,
            enable_evidence=evidence_enabled,
        )
        train_loader, train_sampler = _build_train_dataloader(
            dataset=train_dataset,
            collator=collator,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            use_balanced_sampler=use_balanced_sampler,
            dist_info=dist_info,
        )
        eval_loader = (
            _build_eval_dataloader(
                dataset=eval_dataset,
                collator=collator,
                batch_size=eval_batch_size,
                num_workers=num_workers,
            )
            if eval_dataset is not None
            else None
        )
        test_loader = (
            _build_eval_dataloader(
                dataset=test_dataset,
                collator=collator,
                batch_size=eval_batch_size,
                num_workers=num_workers,
            )
            if test_dataset is not None
            else None
        )

        class_weights = _compute_class_weights(train_dataset.labels) if use_weighted_ce else None
        class_weights_device = class_weights.to(device) if class_weights is not None else None
        label_counts = pd.Series(train_dataset.labels).value_counts().sort_index().to_dict()

        params = [param for param in model.parameters() if param.requires_grad]
        optimizer = AdamW(params=params, lr=lr, weight_decay=weight_decay)
        total_update_steps = max(1, (len(train_loader) * epochs) // max(1, grad_accum))
        warmup_steps = int(total_update_steps * warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_update_steps,
        )

        autocast_enabled = torch.cuda.is_available() and dtype in (torch.float16, torch.bfloat16)
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            scaler = torch.amp.GradScaler(
                "cuda",
                enabled=torch.cuda.is_available() and dtype == torch.float16,
            )
        else:
            scaler = GradScaler(enabled=torch.cuda.is_available() and dtype == torch.float16)

        if dist_info.is_main and _uses_tensorboard(report_to):
            user_logging_dir = _get_cfg(cfg, "train.logging_dir")
            writer_dir = (
                os.path.join(str(user_logging_dir), _tb_run_name(run_dir))
                if user_logging_dir
                else os.path.join(run_dir, "tb_logs")
            )
            if SummaryWriter is None:
                print("[warn] tensorboard requested but SummaryWriter is unavailable; skipping TB logging")
            else:
                Path(writer_dir).mkdir(parents=True, exist_ok=True)
                writer = SummaryWriter(log_dir=writer_dir)
                print(f"[info] tensorboard logging_dir={writer_dir}")

        _rank0_print(
            dist_info,
            (
                f"[decoder_cls] loss_mode={loss_mode} train_size={len(train_dataset)} "
                f"eval_size={len(eval_dataset) if eval_dataset is not None else 0} "
                f"test_size={len(test_dataset) if test_dataset is not None else 0} "
                f"binary_classifier={binary_classifier} evidence_enabled={evidence_enabled} "
                f"label_counts={{{', '.join(f'{k}: {v}' for k, v in label_counts.items())}}}"
            ),
        )

        best_score = -float("inf") if greater_is_better else float("inf")
        best_epoch = -1
        best_eval_metrics: Optional[Dict[str, float]] = None
        best_test_metrics: Optional[Dict[str, float]] = None
        epochs_without_improvement = 0
        train_log_path = os.path.join(run_dir, "train_epoch_metrics.jsonl")
        eval_log_path = os.path.join(run_dir, "eval_epoch_metrics.jsonl")
        test_log_path = os.path.join(run_dir, "test_epoch_metrics.jsonl")
        summary_path = os.path.join(run_dir, "summary.json")
        checkpoint_dir = os.path.join(run_dir, "best_checkpoint")
        train_state_path = os.path.join(run_dir, "train_state.json")
        _write_json(
            train_state_path,
            {
                "loss_mode": loss_mode,
                "binary_classifier": binary_classifier,
                "decision_threshold": decision_threshold,
                "evidence_enabled": evidence_enabled,
                "label_loss_weight": label_loss_weight,
                "evidence_lambda": evidence_lambda,
                "evidence_alpha": evidence_alpha,
                "evidence_beta": evidence_beta,
                "evidence_negative_downsample_ratio": evidence_negative_downsample_ratio,
                "evidence_threshold": evidence_threshold,
                "evidence_metric_weight": evidence_metric_weight,
                "evidence_max_pred_spans": evidence_max_pred_spans,
                "evidence_warmup_epochs": evidence_warmup_epochs,
                "peft_mode": peft_mode,
                "base_model_name": model_name,
                "train_csv": train_csv,
                "eval_csv": eval_csv,
                "test_csv": test_csv,
                "label_counts": {str(k): int(v) for k, v in label_counts.items()},
                "class_weights": class_weights.tolist() if class_weights is not None else None,
            },
        )

        train_start_time = time.perf_counter()
        for epoch in range(1, epochs + 1):
            if hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch)

            model.train()
            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0
            running_ce_loss = 0.0
            running_evidence_loss = 0.0
            seen_samples = 0
            epoch_start_time = time.perf_counter()
            log_window_start_time = epoch_start_time
            last_logged_global_micro_step = (epoch - 1) * len(train_loader)

            for step, batch in enumerate(train_loader, start=1):
                labels = batch["labels"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device)

                autocast_ctx = (
                    torch.autocast(device_type="cuda", dtype=dtype)
                    if autocast_enabled
                    else nullcontext()
                )
                with autocast_ctx:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        compute_evidence=evidence_enabled,
                    )
                    logits = outputs.logits
                    ce_loss, _ = _classification_loss(
                        logits=logits,
                        labels=labels,
                        binary_classifier=binary_classifier,
                        class_weights=class_weights_device,
                    )
                    evidence_mask = batch.get("evidence_mask")
                    if evidence_mask is not None:
                        evidence_mask = evidence_mask.to(device)
                    use_evidence_loss_batch = batch.get("use_evidence_loss")
                    if use_evidence_loss_batch is not None:
                        use_evidence_loss_batch = use_evidence_loss_batch.to(device)
                    evidence_loss, _ = _evidence_token_loss(
                        evidence_logits=outputs.evidence_logits if evidence_enabled else None,
                        evidence_mask=evidence_mask,
                        attention_mask=attention_mask,
                        use_evidence_loss=use_evidence_loss_batch,
                        alpha=evidence_alpha,
                        beta=evidence_beta,
                        negative_downsample_ratio=evidence_negative_downsample_ratio,
                    )
                    effective_evidence_lambda = (
                        float(evidence_lambda)
                        if (evidence_enabled and epoch > int(evidence_warmup_epochs))
                        else 0.0
                    )
                    loss = (
                        (float(label_loss_weight) * ce_loss)
                        + (effective_evidence_lambda * evidence_loss)
                    )

                loss_for_backward = loss / max(1, grad_accum)
                if scaler.is_enabled():
                    scaler.scale(loss_for_backward).backward()
                else:
                    loss_for_backward.backward()

                batch_size_now = int(labels.size(0))
                running_loss += float(loss.detach().item()) * batch_size_now
                running_ce_loss += float(ce_loss.detach().item()) * batch_size_now
                running_evidence_loss += float(evidence_loss.detach().item()) * batch_size_now
                seen_samples += batch_size_now

                should_step = (step % max(1, grad_accum) == 0) or (step == len(train_loader))
                if should_step:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    if max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                if dist_info.is_main and step % max(1, logging_steps) == 0:
                    now = time.perf_counter()
                    global_micro_step = ((epoch - 1) * len(train_loader)) + step
                    window_steps = max(1, global_micro_step - last_logged_global_micro_step)
                    window_elapsed = max(1e-6, now - log_window_start_time)
                    total_elapsed = max(1e-6, now - train_start_time)
                    epoch_elapsed = max(1e-6, now - epoch_start_time)
                    steps_per_sec = window_steps / window_elapsed
                    epoch_avg_steps_per_sec = step / epoch_elapsed
                    total_avg_steps_per_sec = global_micro_step / total_elapsed
                    eta_epoch = (len(train_loader) - step) / max(1e-6, epoch_avg_steps_per_sec)
                    eta_total = ((epochs * len(train_loader)) - global_micro_step) / max(
                        1e-6, total_avg_steps_per_sec
                    )
                    current_lr = float(optimizer.param_groups[0]["lr"]) if optimizer.param_groups else 0.0
                    print(
                        f"[epoch {epoch}/{epochs}] step={step}/{len(train_loader)} "
                        f"loss={float(loss.detach().item()):.4f} "
                        f"ce={float(ce_loss.detach().item()):.4f} "
                        f"evidence={float(evidence_loss.detach().item()):.4f} "
                        f"lr={current_lr:.2e} "
                        f"steps/s={steps_per_sec:.2f} "
                        f"eta_epoch={_format_eta(eta_epoch)} "
                        f"eta_total={_format_eta(eta_total)}",
                        flush=True,
                    )
                    log_window_start_time = now
                    last_logged_global_micro_step = global_micro_step

            train_row = {
                "epoch": epoch,
                "train_loss": 0.0,
                "train_ce_loss": 0.0,
                "train_evidence_loss": 0.0,
                "learning_rate": float(optimizer.param_groups[0]["lr"]) if optimizer.param_groups else 0.0,
            }

            running_loss = _reduce_sum_scalar(running_loss, device=device, info=dist_info)
            running_ce_loss = _reduce_sum_scalar(running_ce_loss, device=device, info=dist_info)
            running_evidence_loss = _reduce_sum_scalar(
                running_evidence_loss,
                device=device,
                info=dist_info,
            )
            seen_samples = int(_reduce_sum_scalar(seen_samples, device=device, info=dist_info))
            train_row["train_loss"] = running_loss / max(1, seen_samples)
            train_row["train_ce_loss"] = running_ce_loss / max(1, seen_samples)
            train_row["train_evidence_loss"] = running_evidence_loss / max(1, seen_samples)

            _barrier_if_needed(dist_info)
            stop_training = False
            if dist_info.is_main:
                _append_jsonl(train_log_path, train_row)
                if writer is not None:
                    writer.add_scalar("train/loss", train_row["train_loss"], epoch)
                    writer.add_scalar("train/ce_loss", train_row["train_ce_loss"], epoch)
                    writer.add_scalar("train/evidence_loss", train_row["train_evidence_loss"], epoch)
                    writer.add_scalar("train/lr", train_row["learning_rate"], epoch)

                eval_metrics_row = None
                if eval_loader is not None:
                    eval_result = _evaluate(
                        model=_unwrap_model(model),
                        dataloader=eval_loader,
                        device=device,
                        loss_mode=loss_mode,
                        binary_classifier=binary_classifier,
                        class_weights=class_weights_device,
                        decision_threshold=decision_threshold,
                        evidence_enabled=evidence_enabled,
                        label_loss_weight=label_loss_weight,
                        evidence_lambda=evidence_lambda,
                        evidence_alpha=evidence_alpha,
                        evidence_beta=evidence_beta,
                        evidence_negative_downsample_ratio=evidence_negative_downsample_ratio,
                        evidence_threshold=evidence_threshold,
                        evidence_metric_weight=evidence_metric_weight,
                        evidence_max_pred_spans=evidence_max_pred_spans,
                        evidence_warmup_epochs=evidence_warmup_epochs,
                        amp_dtype=dtype,
                    )
                    eval_metrics_row = {"epoch": epoch, "split": "eval", **eval_result.metrics}
                    _append_jsonl(eval_log_path, eval_metrics_row)
                    eval_result.predictions.to_csv(os.path.join(run_dir, f"eval_predictions_epoch_{epoch}.csv"), index=False)
                    if writer is not None:
                        for key, value in eval_result.metrics.items():
                            writer.add_scalar(f"eval/{key}", value, epoch)
                        _log_tradeoff_scalars(writer, "eval", eval_result.metrics, epoch)

                    missing_score_default = float("-inf") if greater_is_better else float("inf")
                    current_score = float(eval_result.metrics.get(metric_for_best, missing_score_default))
                    eligible_for_best = epoch >= int(min_epoch_for_best_model)
                    improved = False
                    if eligible_for_best:
                        improved = (
                            current_score > (best_score + early_stopping_threshold)
                            if greater_is_better
                            else current_score < (best_score - early_stopping_threshold)
                        )
                    if improved:
                        best_score = current_score
                        best_epoch = epoch
                        best_eval_metrics = dict(eval_result.metrics)
                        metadata = {
                            "epoch": epoch,
                            "metric_for_best_model": metric_for_best,
                            "best_score": best_score,
                            "loss_mode": loss_mode,
                            "binary_classifier": binary_classifier,
                            "decision_threshold": decision_threshold,
                            "evidence_enabled": evidence_enabled,
                            "label_loss_weight": label_loss_weight,
                            "evidence_lambda": evidence_lambda,
                            "evidence_alpha": evidence_alpha,
                            "evidence_beta": evidence_beta,
                            "evidence_negative_downsample_ratio": evidence_negative_downsample_ratio,
                            "evidence_threshold": evidence_threshold,
                            "evidence_metric_weight": evidence_metric_weight,
                            "evidence_max_pred_spans": evidence_max_pred_spans,
                            "evidence_warmup_epochs": evidence_warmup_epochs,
                            "peft_mode": peft_mode,
                            "base_model_name": model_name,
                            "class_weights": class_weights.tolist() if class_weights is not None else None,
                        }
                        _save_checkpoint(model=model, tokenizer=tokenizer, checkpoint_dir=checkpoint_dir, metadata=metadata)
                        eval_result.predictions.to_csv(os.path.join(run_dir, "best_eval_predictions.csv"), index=False)
                        epochs_without_improvement = 0

                        if test_loader is not None:
                            test_result = _evaluate(
                                model=_unwrap_model(model),
                                dataloader=test_loader,
                                device=device,
                                loss_mode=loss_mode,
                                binary_classifier=binary_classifier,
                                class_weights=class_weights_device,
                                decision_threshold=decision_threshold,
                                evidence_enabled=evidence_enabled,
                                label_loss_weight=label_loss_weight,
                                evidence_lambda=evidence_lambda,
                                evidence_alpha=evidence_alpha,
                                evidence_beta=evidence_beta,
                                evidence_negative_downsample_ratio=evidence_negative_downsample_ratio,
                                evidence_threshold=evidence_threshold,
                                evidence_metric_weight=evidence_metric_weight,
                                evidence_max_pred_spans=evidence_max_pred_spans,
                                evidence_warmup_epochs=evidence_warmup_epochs,
                                amp_dtype=dtype,
                            )
                            best_test_metrics = dict(test_result.metrics)
                            test_row = {"epoch": epoch, "split": "test", **test_result.metrics}
                            _append_jsonl(test_log_path, test_row)
                            test_result.predictions.to_csv(os.path.join(run_dir, "best_test_predictions.csv"), index=False)
                            if writer is not None:
                                for key, value in test_result.metrics.items():
                                    writer.add_scalar(f"test/{key}", value, epoch)
                                _log_tradeoff_scalars(writer, "test", test_result.metrics, epoch)
                    elif eligible_for_best:
                        epochs_without_improvement += 1
                elif best_epoch < 0:
                    best_epoch = epoch
                    best_eval_metrics = None
                    _save_checkpoint(
                        model=model,
                        tokenizer=tokenizer,
                        checkpoint_dir=checkpoint_dir,
                        metadata={
                            "epoch": epoch,
                            "loss_mode": loss_mode,
                            "binary_classifier": binary_classifier,
                            "decision_threshold": decision_threshold,
                            "evidence_enabled": evidence_enabled,
                            "label_loss_weight": label_loss_weight,
                            "evidence_lambda": evidence_lambda,
                            "evidence_alpha": evidence_alpha,
                            "evidence_beta": evidence_beta,
                            "evidence_negative_downsample_ratio": evidence_negative_downsample_ratio,
                            "evidence_threshold": evidence_threshold,
                            "evidence_metric_weight": evidence_metric_weight,
                            "evidence_max_pred_spans": evidence_max_pred_spans,
                            "evidence_warmup_epochs": evidence_warmup_epochs,
                            "peft_mode": peft_mode,
                            "base_model_name": model_name,
                        },
                    )

                summary = {
                    "model_name": model_name,
                    "loss_mode": loss_mode,
                    "binary_classifier": binary_classifier,
                    "decision_threshold": decision_threshold,
                    "evidence_enabled": evidence_enabled,
                    "label_loss_weight": label_loss_weight,
                    "evidence_lambda": evidence_lambda,
                    "evidence_alpha": evidence_alpha,
                    "evidence_beta": evidence_beta,
                    "evidence_negative_downsample_ratio": evidence_negative_downsample_ratio,
                    "evidence_threshold": evidence_threshold,
                    "evidence_metric_weight": evidence_metric_weight,
                    "evidence_max_pred_spans": evidence_max_pred_spans,
                    "evidence_warmup_epochs": evidence_warmup_epochs,
                    "min_epoch_for_best_model": min_epoch_for_best_model,
                    "peft_mode": peft_mode,
                    "best_epoch": best_epoch,
                    "metric_for_best_model": metric_for_best,
                    "greater_is_better": greater_is_better,
                    "best_score": best_score if best_epoch > 0 else None,
                    "best_eval_metrics": best_eval_metrics,
                    "best_test_metrics": best_test_metrics,
                    "label_counts": {str(k): int(v) for k, v in label_counts.items()},
                    "class_weights": class_weights.tolist() if class_weights is not None else None,
                    "checkpoint_dir": checkpoint_dir,
                }
                _write_json(summary_path, summary)
                if eval_metrics_row is not None:
                    print(
                        f"[epoch {epoch}] eval accuracy={eval_metrics_row.get('accuracy', 0.0):.4f} "
                        f"macro_f1={eval_metrics_row.get('macro_f1', 0.0):.4f} "
                        f"weighted_f1={eval_metrics_row.get('weighted_f1', 0.0):.4f} "
                        f"label_0_recall={eval_metrics_row.get('label_0_recall', 0.0):.4f} "
                        f"label_1_recall={eval_metrics_row.get('label_1_recall', 0.0):.4f}"
                    )

            if dist_info.enabled:
                stop_tensor = torch.tensor([1 if stop_training else 0], device=device, dtype=torch.int32)
                dist.broadcast(stop_tensor, src=0)
                stop_training = bool(stop_tensor.item())
            _barrier_if_needed(dist_info)
            if stop_training:
                _rank0_print(dist_info, f"[info] early stopping triggered at epoch {epoch}")
                break

        return run_dir
    finally:
        if writer is not None:
            writer.close()
        _cleanup_distributed(dist_info)
