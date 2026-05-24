import argparse
import json
import os
import random
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

try:
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
except Exception:  # pragma: no cover - optional dependency
    LoraConfig = None
    PeftModel = None
    TaskType = None
    get_peft_model = None

from phishdec.metrics.binary import compute_binary_metrics
from phishdec.utils.model_registry import resolve_model_name


def _safe_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_") or "model"


class EventLogger:
    def __init__(self, enable: bool, project: str, run_name: str, config: Dict[str, Any]):
        self.enabled = False

    def log(self, data: Dict[str, Any]):
        return None

    def log_confusion(self, y_true: Sequence[int], y_pred: Sequence[int], prefix: str):
        return None

    def finish(self):
        return None


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_out_csv(default_path: str, out_csv_arg: Optional[str]) -> str:
    path = out_csv_arg.strip() if out_csv_arg else default_path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path


def drop_unsupported_keys_by_model(model, batch: dict):
    no_tt_models = {"distilbert", "xlm-roberta", "funnel"}
    mt = getattr(getattr(model, "config", None), "model_type", None)
    if mt in no_tt_models and "token_type_ids" in batch:
        batch.pop("token_type_ids", None)


def _require_dir(path: str, msg: str):
    if not (path and os.path.isdir(path)):
        raise FileNotFoundError(f"{msg}: not found -> {path}")


def _print_dataset_stats(df: pd.DataFrame, name="eval"):
    n = len(df)
    pos = int((df["label"] == 1).sum())
    neg = n - pos
    print(f"[{name}] size={n}  pos={pos}  neg={neg}  pos_ratio={pos / n:.3f}")


def maybe_load_tapt_base(output_model_dir: str, model_name: str) -> Tuple[str, bool]:
    cand = os.path.join(output_model_dir, "tapt", model_name.replace("/", "_"))
    if os.path.isdir(cand):
        return cand, True
    print(f"[WARN] TAPT base not found -> {cand}. Falling back to pretrained base.")
    return model_name, False


def _compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    rec_pos = recall_score(y_true, y_pred, pos_label=1)
    rec_neg = recall_score(y_true, y_pred, pos_label=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    return acc, rec_pos, rec_neg, macro_f1, weighted_f1


def _log_eval_events(logger: EventLogger, y_true, y_pred, df=None, prefix="eval"):
    acc, rec_pos, rec_neg, macro_f1, weighted_f1 = _compute_metrics(y_true, y_pred)
    logger.log(
        {
            f"{prefix}/acc": acc,
            f"{prefix}/recall_pos": rec_pos,
            f"{prefix}/recall_neg": rec_neg,
            f"{prefix}/macro_f1": macro_f1,
            f"{prefix}/weighted_f1": weighted_f1,
        }
    )
    logger.log_confusion(y_true, y_pred, prefix)

    if df is not None and "category" in df.columns:
        for cat, sub in df.groupby("category"):
            y_c = sub["label"].astype(int).tolist()
            if "predicted" in sub.columns:
                p_c = sub["predicted"].astype(int).tolist()
            else:
                idx = sub.index.tolist()
                p_c = [int(y_pred[i]) for i in idx]
            a, rp, rn, mf, wf = _compute_metrics(y_c, p_c)
            logger.log(
                {
                    f"{prefix}/by_category/{cat}/acc": a,
                    f"{prefix}/by_category/{cat}/recall_pos": rp,
                    f"{prefix}/by_category/{cat}/recall_neg": rn,
                    f"{prefix}/by_category/{cat}/macro_f1": mf,
                    f"{prefix}/by_category/{cat}/weighted_f1": wf,
                }
            )


def safe_tokenize(tokenizer, texts: List[str], **kwargs):
    texts = [str(t) for t in texts]
    return tokenizer(texts, **kwargs)


def _to_dataset(
    df: pd.DataFrame, tokenizer, max_length: int, include_token_type_ids: bool
) -> TensorDataset:
    tokenized = safe_tokenize(
        tokenizer,
        df["text"].tolist(),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    labels = torch.tensor(df["label"].tolist())

    fields = [
        tokenized["input_ids"],
        tokenized["attention_mask"],
    ]
    if include_token_type_ids and "token_type_ids" in tokenized:
        fields.append(tokenized["token_type_ids"])
    fields.append(labels)
    return TensorDataset(*fields)


def train_peft_lora(
    model_name: str,
    base_model_path: str,
    tokenizer,
    train_df: pd.DataFrame,
    device,
    max_length: int,
    num_labels: int,
    epochs: int,
    batch_size: int,
    lr: float,
    logger: EventLogger,
):
    if get_peft_model is None or LoraConfig is None or TaskType is None:
        raise ImportError("peft is required for LoRA training but not installed.")

    if model_name.startswith(("monologg/distilkobert", "distilbert/distilbert-base-uncased")):
        target_modules = ["q_lin", "v_lin"]
    elif model_name.startswith(
        ("monologg/kobert", "google-bert/bert-base-multilingual-cased", "bert-base-multilingual-cased")
    ):
        target_modules = ["attention.self.query", "attention.self.value"]
    else:
        raise ValueError(f"Unknown target_modules for model: {model_name}")

    config = AutoConfig.from_pretrained(base_model_path, num_labels=num_labels)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_path, config=config, trust_remote_code=True
    ).to(device)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        target_modules=target_modules,
    )
    model = get_peft_model(base_model, lora_config).to(device)

    dataset = _to_dataset(train_df, tokenizer, max_length, include_token_type_ids=True)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available()
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    global_step = 0
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        loop = tqdm(dataloader, desc=f"PEFT-LoRA Epoch {epoch + 1}")
        for batch in loop:
            inputs = [b.to(device) for b in batch]
            input_ids, attention_mask, *rest = inputs
            labels = rest[-1]
            model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
            if len(rest) == 2:
                model_inputs["token_type_ids"] = rest[0]
            outputs = model(**model_inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            with torch.no_grad():
                prob = F.softmax(outputs.logits, dim=1)
                pred = torch.argmax(prob, dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)

            running_loss += loss.item()
            global_step += 1
            logger.log({"train/loss": float(loss.item()), "train/epoch": epoch + 1, "train/step": global_step})
            loop.set_postfix(loss=loss.item())

        epoch_loss = running_loss / max(1, len(dataloader))
        epoch_acc = correct / max(1, total)
        logger.log(
            {
                "train/epoch_loss": float(epoch_loss),
                "train/epoch_acc": float(epoch_acc),
                "epoch": epoch + 1,
                "mode": "lora",
            }
        )
    return model


def run_tapt_mlm(
    model_name: str,
    tokenizer,
    train_df: pd.DataFrame,
    device,
    max_length: int,
    epochs: int,
    batch_size: int,
    lr: float,
    mlm_probability: float,
    logger: EventLogger,
):
    mlm_model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True).to(device)
    mlm_model.train()

    tokenized = safe_tokenize(
        tokenizer,
        train_df["text"].astype(str).tolist(),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    features = [
        {"input_ids": ids, "attention_mask": mask}
        for ids, mask in zip(tokenized["input_ids"], tokenized["attention_mask"])
    ]
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability)
    loader = DataLoader(
        features,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    optim = torch.optim.AdamW(mlm_model.parameters(), lr=lr)
    for epoch in range(epochs):
        running = 0.0
        nsteps = 0
        loop = tqdm(loader, desc=f"TAPT-MLM Epoch {epoch + 1}")
        for batch in loop:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = mlm_model(**batch).loss
            loss.backward()
            optim.step()
            optim.zero_grad()
            running += loss.item()
            nsteps += 1
            logger.log({"mlm/train_loss": float(loss.item()), "mlm/epoch": epoch + 1})
            loop.set_postfix(loss=loss.item())
        logger.log({"mlm/epoch_loss": float(running / max(1, nsteps)), "epoch": epoch + 1, "mode": "tapt"})
    return mlm_model


def train_full_ft(
    model_name: str,
    tokenizer,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    device,
    max_length: int,
    epochs: int,
    batch_size: int,
    lr: float,
    logger: EventLogger,
) -> Tuple[Any, float, float, float]:
    tok_train = safe_tokenize(
        tokenizer,
        train_df["text"].tolist(),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    train_ds = TensorDataset(tok_train["input_ids"], tok_train["attention_mask"], torch.tensor(train_df["label"].tolist()))

    tok_valid = safe_tokenize(
        tokenizer,
        valid_df["text"].tolist(),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    config = AutoConfig.from_pretrained(model_name, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, config=config, trust_remote_code=True
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        running = 0.0
        correct = 0
        total = 0
        loop = tqdm(
            DataLoader(
                train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available()
            ),
            desc=f"FT Epoch {epoch + 1}",
        )
        for input_ids, attention_mask, labs in loop:
            input_ids, attention_mask, labs = (input_ids.to(device), attention_mask.to(device), labs.to(device))
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labs)
            out.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            with torch.no_grad():
                prob = F.softmax(out.logits, dim=1)
                pred = torch.argmax(prob, dim=1)
                correct += (pred == labs).sum().item()
                total += labs.size(0)
            running += out.loss.item()
            logger.log({"train/loss": float(out.loss.item()), "train/epoch": epoch + 1, "mode": "standard"})
        logger.log(
            {
                "train/epoch_loss": float(running / max(1, len(loop))),
                "train/epoch_acc": float(correct / max(1, total)),
                "epoch": epoch + 1,
                "mode": "standard",
            }
        )
    preds, _ = classification_response(model, tokenizer, valid_df, max_length, device, model_name=model_name)
    acc = accuracy_score(valid_df["label"], preds)
    rec = recall_score(valid_df["label"], preds, average="weighted")
    f1 = f1_score(valid_df["label"], preds, average="weighted")
    return model, acc, rec, f1


def classification_response(model, tokenizer, dataset, max_length, device, model_name: str = ""):
    if isinstance(dataset, dict):
        texts = dataset["text"]
    elif isinstance(dataset, pd.DataFrame):
        texts = dataset["text"].tolist()
    else:
        raise ValueError("Unsupported dataset format")

    texts = [str(t) for t in texts]
    inputs = safe_tokenize(
        tokenizer,
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=max_length,
    )
    models_without_token_type_ids = ["monologg/distilkobert", "distilbert/distilbert-base-uncased"]
    if any(model_name.startswith(m) for m in models_without_token_type_ids):
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]

    ordered_keys = ["input_ids", "attention_mask"]
    if "token_type_ids" in inputs:
        ordered_keys.append("token_type_ids")

    dataset_tensor = TensorDataset(*[inputs[k] for k in ordered_keys])

    dataloader = DataLoader(dataset_tensor, batch_size=32, num_workers=0, pin_memory=torch.cuda.is_available())

    preds, probs = [], []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in zip(ordered_keys, batch)}
            drop_unsupported_keys_by_model(model, batch)
            outputs = model(**batch)
            prob = F.softmax(outputs.logits, dim=1)
            pred = torch.argmax(prob, dim=1)
            preds.extend(pred.cpu().tolist())
            probs.extend(prob.cpu().tolist())
    return preds, probs


def _load_cls_df(path: str, text_col: str, label_col: str, require_label: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path, engine="c")
    missing = [c for c in (text_col, label_col) if require_label and c not in df.columns]
    if missing:
        raise ValueError(f"Columns {missing} not found in {path}")
    if text_col != "text":
        df = df.rename(columns={text_col: "text"})
    if require_label and label_col != "label":
        df = df.rename(columns={label_col: "label"})
    df = df.dropna(subset=["text"])
    if require_label:
        df = df.dropna(subset=["label"])
        df["label"] = pd.to_numeric(df["label"], errors="coerce")
        df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)
    df["text"] = df["text"].astype(str)
    return df.reset_index(drop=True)


def _build_artifact_paths(base_root: str, exp_name: str, model_name: str) -> Dict[str, str]:
    safe_root = Path(base_root)
    if exp_name:
        safe_root = safe_root / _safe_label(exp_name)
    safe_model = _safe_label(model_name)
    return {
        "tapt": str(safe_root / "tapt" / safe_model),
        "standard": str(safe_root / "standard" / safe_model),
        "lora": str(safe_root / "lora" / safe_model),
    }


def _save_tokenizer(tokenizer, save_dir: str):
    if "kobert" in tokenizer.name_or_path.lower():
        tokenizer.save_vocabulary(save_dir)
    else:
        tokenizer.save_pretrained(save_dir)


def _save_metrics_with_groups(
    eval_df: pd.DataFrame,
    preds: List[int],
    probs: List[List[float]],
    metrics_path: str,
) -> Dict[str, Any]:
    eval_df = eval_df.copy()
    eval_df["predicted"] = preds
    eval_df["proba_pos"] = [float(p[1]) for p in probs]

    y_true = eval_df["label"].astype(int).tolist()
    y_pred = [int(p) for p in preds]
    overall = compute_binary_metrics(y_true, y_pred)
    acc, rec_pos, rec_neg, macro_f1, weighted_f1 = _compute_metrics(y_true, y_pred)
    overall["sklearn"] = {
        "accuracy": acc,
        "recall_pos": rec_pos,
        "recall_neg": rec_neg,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }

    by_category = {}
    if "category" in eval_df.columns:
        for cat, sub in eval_df.groupby("category"):
            by_category[str(cat)] = compute_binary_metrics(
                sub["label"].astype(int).tolist(), sub["predicted"].astype(int).tolist()
            )
    by_subset = {}
    if "subset_label" in eval_df.columns:
        for sub, g in eval_df.groupby("subset_label"):
            by_subset[str(sub)] = compute_binary_metrics(
                g["label"].astype(int).tolist(), g["predicted"].astype(int).tolist()
            )
    overall["by_category"] = by_category
    overall["by_subset"] = by_subset

    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)
    return overall


def _import_legacy_backend():
    repo_root = Path(__file__).resolve().parents[3]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    import run_enc_models

    return run_enc_models


def _find_encoder_model_id(model_name: Optional[str]) -> Optional[int]:
    if not model_name:
        return None
    repo_root = Path(__file__).resolve().parents[3]
    models_path = repo_root / "models_args.json"
    with open(models_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for idx, candidate in enumerate(data.get("Encoder", [])):
        if candidate == model_name:
            return idx
    return None


def _prepare_legacy_csv(
    path: Optional[str],
    run_dir: str,
    split_name: str,
    text_col: str,
    label_col: str,
    require_label: bool,
) -> str:
    if not path:
        return ""
    if text_col == "text" and (not require_label or label_col == "label"):
        return path

    df = pd.read_csv(path, engine="c")
    if text_col not in df.columns:
        raise ValueError(f"Column `{text_col}` not found in {path}")
    rename_map = {}
    if text_col != "text":
        rename_map[text_col] = "text"
    if require_label:
        if label_col not in df.columns:
            raise ValueError(f"Column `{label_col}` not found in {path}")
        if label_col != "label":
            rename_map[label_col] = "label"
    elif label_col in df.columns and label_col != "label":
        rename_map[label_col] = "label"

    compat_df = df.rename(columns=rename_map)
    compat_dir = Path(run_dir) / "_legacy_compat"
    compat_dir.mkdir(parents=True, exist_ok=True)
    out_path = compat_dir / f"{split_name}.csv"
    compat_df.to_csv(out_path, index=False)
    return str(out_path)


def _write_metrics_from_predictions_csv(predictions_csv: str, metrics_path: str) -> Dict[str, Any]:
    df = pd.read_csv(predictions_csv)
    if "label" not in df.columns or "predicted" not in df.columns:
        raise ValueError(f"`label` and `predicted` columns are required in {predictions_csv}")

    y_true = pd.to_numeric(df["label"], errors="raise").astype(int).tolist()
    y_pred = pd.to_numeric(df["predicted"], errors="raise").astype(int).tolist()
    overall = compute_binary_metrics(y_true, y_pred)
    acc, rec_pos, rec_neg, macro_f1, weighted_f1 = _compute_metrics(y_true, y_pred)
    overall["sklearn"] = {
        "accuracy": acc,
        "recall_pos": rec_pos,
        "recall_neg": rec_neg,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }

    by_category = {}
    if "category" in df.columns:
        for cat, sub in df.groupby("category"):
            by_category[str(cat)] = compute_binary_metrics(
                pd.to_numeric(sub["label"], errors="raise").astype(int).tolist(),
                pd.to_numeric(sub["predicted"], errors="raise").astype(int).tolist(),
            )
    by_subset = {}
    if "subset_label" in df.columns:
        for sub_name, sub_df in df.groupby("subset_label"):
            by_subset[str(sub_name)] = compute_binary_metrics(
                pd.to_numeric(sub_df["label"], errors="raise").astype(int).tolist(),
                pd.to_numeric(sub_df["predicted"], errors="raise").astype(int).tolist(),
            )
    overall["by_category"] = by_category
    overall["by_subset"] = by_subset

    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)
    return overall


def _encoder_checkpoint_mode_dir(mode: str) -> str:
    return "lora" if mode == "peft" else mode


def _encoder_model_dir_name(model_name: str) -> str:
    return str(model_name).replace("/", "_")


def _resolve_encoder_output_model_root(
    *,
    output_model_root: Optional[str],
    checkpoint_dir: Optional[str],
    mode: str,
    model_name: str,
    eval_only: bool,
) -> str:
    if checkpoint_dir:
        if not eval_only:
            raise ValueError("model.checkpoint_dir is only supported when run.eval_only=true.")
        checkpoint_path = Path(str(checkpoint_dir))
        if not checkpoint_path.is_dir():
            raise FileNotFoundError(f"model.checkpoint_dir not found: {checkpoint_dir}")
        expected_mode_dir = _encoder_checkpoint_mode_dir(mode)
        expected_model_dir = _encoder_model_dir_name(model_name)
        if checkpoint_path.name != expected_model_dir or checkpoint_path.parent.name != expected_mode_dir:
            raise ValueError(
                "model.checkpoint_dir must point to "
                f".../{expected_mode_dir}/{expected_model_dir} for mode={mode}."
            )
        return str(checkpoint_path.parent.parent)

    if not output_model_root:
        raise ValueError("train.output_model_root is required unless model.checkpoint_dir is set.")
    return str(output_model_root)


def _encoder_checkpoint_paths(output_model_root: str, mode: str, model_name: str, use_tapt: bool) -> Dict[str, str]:
    artifacts = {
        "checkpoint_dir": os.path.join(
            output_model_root,
            _encoder_checkpoint_mode_dir(mode),
            _encoder_model_dir_name(model_name),
        ),
    }
    if use_tapt:
        artifacts["tapt_dir"] = os.path.join(
            output_model_root,
            "tapt",
            _encoder_model_dir_name(model_name),
        )
    return artifacts


def _copy_encoder_predictions(predictions_path: str, target_dir: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(predictions_path):
        return None
    target_predictions = os.path.join(target_dir, "predictions.csv")
    Path(target_predictions).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(predictions_path, target_predictions)
    metrics = _write_metrics_from_predictions_csv(target_predictions, os.path.join(target_dir, "metrics.json"))
    return {
        "predictions_csv": target_predictions,
        "metrics_json": os.path.join(target_dir, "metrics.json"),
        "metrics": metrics,
    }


def run_encoder_from_config(cfg: Dict[str, Any], run_dir: str) -> Dict[str, Any]:
    os.environ["TRANSFORMERS_TRUST_REMOTE_CODE"] = "true"
    torch.backends.cudnn.benchmark = True

    model_cfg = cfg.get("model", {}) or {}
    data_cfg = cfg.get("data", {}) or {}
    train_cfg = cfg.get("train", {}) or {}
    run_cfg = cfg.get("run", {}) or {}
    mode = (
        model_cfg.get("mode")
        or train_cfg.get("mode")
        or ("lora" if train_cfg.get("lora") else None)
        or ("standard" if train_cfg.get("standard") else None)
    )
    if mode == "lora":
        mode = "peft"
    if mode not in {"standard", "peft"}:
        raise ValueError("model.mode must be one of: standard|peft")

    eval_only = bool(run_cfg.get("eval_only", False))
    use_tapt = bool(train_cfg.get("tapt") or model_cfg.get("tapt", False))
    if use_tapt and mode == "standard":
        raise ValueError("tapt=true with mode=standard is not supported. Use mode=peft.")
    text_col = data_cfg.get("text_col", "text")
    label_col = data_cfg.get("label_col", "label")

    train_csv = data_cfg.get("train_csv")
    eval_csv = data_cfg.get("eval_csv")
    tapt_csv = data_cfg.get("tapt_csv") or train_csv

    if not eval_csv:
        raise ValueError("data.eval_csv is required.")
    if not eval_only and not train_csv:
        raise ValueError("data.train_csv is required when eval_only=False.")

    eval_csv_compat = _prepare_legacy_csv(
        eval_csv,
        run_dir=run_dir,
        split_name="eval",
        text_col=text_col,
        label_col=label_col,
        require_label=True,
    )
    train_csv_compat = _prepare_legacy_csv(
        train_csv,
        run_dir=run_dir,
        split_name="train",
        text_col=text_col,
        label_col=label_col,
        require_label=True,
    ) if train_csv else ""
    tapt_csv_compat = _prepare_legacy_csv(
        tapt_csv,
        run_dir=run_dir,
        split_name="tapt",
        text_col=text_col,
        label_col=label_col,
        require_label=False,
    ) if (use_tapt and tapt_csv) else ""

    legacy_backend = _import_legacy_backend()
    model_name = model_cfg.get("model_name")
    model_id = model_cfg.get("model_id")
    if model_id is None:
        model_id = _find_encoder_model_id(model_name)
    resolved_model_name = model_name or resolve_model_name(
        model_type=model_cfg.get("model_type", "Encoder"),
        model_name=model_name,
        model_id=model_id,
    )
    if not resolved_model_name:
        raise ValueError("model.model_name or model.model_id is required for encoder runs.")

    output_model_root = _resolve_encoder_output_model_root(
        output_model_root=train_cfg.get("output_model_root"),
        checkpoint_dir=model_cfg.get("checkpoint_dir"),
        mode=mode,
        model_name=resolved_model_name,
        eval_only=eval_only,
    )

    classification_output_root = (
        run_cfg.get("classification_output_root")
        or run_cfg.get("tapt_classification_output")
        or run_cfg.get("out_root")
        or run_dir
    )
    logging_root = train_cfg.get("logging_dir")
    event_log_dir = (
        os.path.join(str(logging_root), os.path.basename(run_dir))
        if logging_root
        else os.path.join(run_dir, "tb_logs")
    )

    def _build_backend_args(
        *,
        eval_set: str,
        classification_output_dir: str,
        current_event_log_dir: str,
        current_eval_only: bool,
    ) -> argparse.Namespace:
        return argparse.Namespace(
            model_id=model_id,
            model_name=resolved_model_name,
            eval_set=eval_set,
            train_data=train_csv_compat or None,
            max_length=int(model_cfg.get("max_length", train_cfg.get("max_length", 512))),
            standard=(mode == "standard"),
            peft=(mode == "peft"),
            lora=False,
            tapt=use_tapt,
            mlm_prob=float(train_cfg.get("mlm_probability", 0.01)),
            tapt_epochs=int(train_cfg.get("tapt_epochs", 1)),
            tapt_batch_size=int(train_cfg.get("tapt_batch_size", 128)),
            tapt_lr=float(train_cfg.get("tapt_lr", 1e-6)),
            tapt_data=tapt_csv_compat or None,
            output_model=output_model_root,
            tapt_classification_output=classification_output_dir,
            event_log_dir=current_event_log_dir,
            device_id=int(model_cfg.get("device_id", run_cfg.get("device_id", 0))),
            eval_mode=1 if current_eval_only else 0,
            seed=int(run_cfg.get("seed", 10)),
            valid_set="",
            decision_threshold=float(run_cfg.get("decision_threshold", 0.5)),
            tune_threshold_on_valid=bool(run_cfg.get("tune_threshold_on_valid", False)),
            clf_epochs=int(train_cfg.get("epochs", 3)),
            clf_batch_size=int(train_cfg.get("batch_size", 16)),
            clf_lr=float(train_cfg.get("lr", 1e-5)),
            skip_if_exists=bool(train_cfg.get("skip_if_exists", True)),
        )

    args = _build_backend_args(
        eval_set=eval_csv_compat,
        classification_output_dir=classification_output_root,
        current_event_log_dir=event_log_dir,
        current_eval_only=eval_only,
    )

    results = legacy_backend.run_from_args(args)

    model_tag = legacy_backend.build_model_tag(model_id, resolved_model_name)
    result_mode = "peft" if mode == "peft" else mode
    predictions_path = os.path.join(classification_output_root, f"{model_tag}_{result_mode}.csv")
    copied_outputs = _copy_encoder_predictions(predictions_path, run_dir)

    artifacts: Dict[str, Any] = {
        "model_name": resolved_model_name,
        "mode": mode,
        "output_model_root": output_model_root,
        "primary_eval_csv": eval_csv,
    }
    artifacts.update(_encoder_checkpoint_paths(output_model_root, mode, resolved_model_name, use_tapt))
    if copied_outputs:
        artifacts.update(copied_outputs)

    post_eval_entries = run_cfg.get("post_eval") or []
    if post_eval_entries:
        if eval_only:
            raise ValueError("run.post_eval is only supported when run.eval_only=false.")
        if not isinstance(post_eval_entries, list):
            raise ValueError("run.post_eval must be a list of mappings.")

        artifacts["post_eval"] = {}
        for index, entry in enumerate(post_eval_entries, start=1):
            if not isinstance(entry, dict):
                raise ValueError("Each run.post_eval entry must be a mapping.")
            post_eval_csv = entry.get("eval_csv")
            if not post_eval_csv:
                raise ValueError(f"run.post_eval[{index - 1}].eval_csv is required.")
            post_name = _safe_label(entry.get("name") or f"eval_{index}")
            post_text_col = entry.get("text_col", text_col)
            post_label_col = entry.get("label_col", label_col)
            post_eval_dir = os.path.join(run_dir, "post_eval", post_name)
            Path(post_eval_dir).mkdir(parents=True, exist_ok=True)
            post_eval_csv_compat = _prepare_legacy_csv(
                post_eval_csv,
                run_dir=post_eval_dir,
                split_name="eval",
                text_col=post_text_col,
                label_col=post_label_col,
                require_label=True,
            )
            post_event_log_dir = (
                os.path.join(str(logging_root), os.path.basename(run_dir), "post_eval", post_name)
                if logging_root
                else os.path.join(post_eval_dir, "tb_logs")
            )
            post_args = _build_backend_args(
                eval_set=post_eval_csv_compat,
                classification_output_dir=post_eval_dir,
                current_event_log_dir=post_event_log_dir,
                current_eval_only=True,
            )
            results[f"post_eval::{post_name}"] = legacy_backend.run_from_args(post_args)
            post_predictions_path = os.path.join(post_eval_dir, f"{model_tag}_{result_mode}.csv")
            post_outputs = _copy_encoder_predictions(post_predictions_path, post_eval_dir) or {}
            post_outputs.update(
                {
                    "name": post_name,
                    "eval_csv": post_eval_csv,
                    "raw_predictions_csv": post_predictions_path,
                }
            )
            artifacts["post_eval"][post_name] = post_outputs

    artifacts_path = os.path.join(run_dir, "artifacts.json")
    Path(artifacts_path).parent.mkdir(parents=True, exist_ok=True)
    with open(artifacts_path, "w", encoding="utf-8") as f:
        json.dump(artifacts, f, ensure_ascii=False, indent=2)

    return results
