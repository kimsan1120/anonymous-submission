from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


@dataclass
class EvalOutput:
    metrics: Dict[str, float]
    y_true: List[int]
    y_pred: List[int]
    y_prob: List[float]
    categories: List[str]


def compute_binary_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> Dict[str, float]:
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    return {
        "accuracy": float(accuracy_score(yt, yp)),
        "positive_precision": float(precision_score(yt, yp, pos_label=1, zero_division=0)),
        "positive_recall": float(recall_score(yt, yp, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
    }


def evaluate_binary_classifier(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
) -> EvalOutput:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[float] = []
    categories: List[str] = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            model_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "return_dict": True,
            }
            if token_type_ids is not None:
                model_inputs["token_type_ids"] = token_type_ids

            outputs = model(**model_inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[:, 1]
            preds = torch.argmax(logits, dim=-1)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            y_prob.extend(probs.cpu().tolist())
            categories.extend([str(v) for v in batch.get("categories", [""] * labels.size(0))])

    metrics = compute_binary_classification_metrics(y_true=y_true, y_pred=y_pred)
    return EvalOutput(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        categories=categories,
    )
