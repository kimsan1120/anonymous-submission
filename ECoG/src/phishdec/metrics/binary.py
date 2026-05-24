from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List

def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else 0.0

def compute_binary_metrics(y_true: List[int], y_pred: List[int]) -> Dict:
    assert len(y_true) == len(y_pred), "y_true/y_pred length mismatch"

    TP = sum((yt == 1 and yp == 1) for yt, yp in zip(y_true, y_pred))
    TN = sum((yt == 0 and yp == 0) for yt, yp in zip(y_true, y_pred))
    FP = sum((yt == 0 and yp == 1) for yt, yp in zip(y_true, y_pred))
    FN = sum((yt == 1 and yp == 0) for yt, yp in zip(y_true, y_pred))
    n = len(y_true)

    acc = _safe_div(TP + TN, n)
    prec1 = _safe_div(TP, TP + FP)
    rec1  = _safe_div(TP, TP + FN)
    f1_1  = _safe_div(2 * prec1 * rec1, prec1 + rec1)
    prec0 = _safe_div(TN, TN + FN)
    rec0  = _safe_div(TN, TN + FP)
    f1_0  = _safe_div(2 * prec0 * rec0, prec0 + rec0)

    macro_f1 = (f1_0 + f1_1) / 2.0

    return {
        "n": n,
        "confusion": {"TP": TP, "TN": TN, "FP": FP, "FN": FN},
        "accuracy": acc,
        "class_1": {"precision": prec1, "recall": rec1, "f1": f1_1},
        "class_0": {"precision": prec0, "recall": rec0, "f1": f1_0},
        "macro_f1": macro_f1,
    }