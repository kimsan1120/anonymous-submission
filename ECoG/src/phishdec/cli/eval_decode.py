import os
import argparse
import hashlib
import json
import tempfile
import re
from typing import Any, Dict, Optional

import numpy as np
import torch

from phishdec.utils.env import setup_env
from phishdec.utils.model_registry import resolve_model_name
from phishdec.prompting import render_row_template, resolve_instruction_texts
from phishdec.io.dataset import load_eval_csv
from phishdec.io.save import save_results_csv, save_evidence_w_explanation_csv
from phishdec.decoding.prompt_budget import resolve_auto_max_input_tokens
from phishdec.decoding.vllm_runner import run_with_vllm
from phishdec.decoding.hf_runner import (
    run_with_hf_generate,
    merge_adapter_to_dir,
    _load_model as _hf_load_model,
    _load_tokenizer as _hf_load_tokenizer,
    get_safe_max_len,
)
from phishdec.decoding.parser import parse_leading_binary_token, parse_trailing_binary_token
from phishdec.utils.seed import set_seed
from phishdec.metrics.binary import compute_binary_metrics


STEP1_CHECKLIST_ORDER = [
    "C1_LINK_PRESENT",
    "C2_INSTALL_OR_PERMISSION",
    "C3_CREDENTIAL_REQUEST",
    "C4_VALUE_TRANSFER",
    "C5_CONTACT_SOLICIT",
    "C6_ACTION_PRESSURE",
    "C7_IMPERSONATION_CONTEXT",
    "C8_OBFUSCATION",
    "C9_OVERSEA_CONTEXT",
    "C10_DELIVERY_CONTEXT",
    "C11_CARD_CONTEXT",
    "C12_LOAN_CONTEXT",
    "P1_DELIVERY",
    "CR1_CARD",
    "F1_BANK",
]
STEP1_ALLOWED_RESULT = {"PASS", "FAIL", "NA"}
STEP1_PIPE_ALLOWED_RESULT = {"PASS", "FAIL"}
STEP1_ALLOWED_CATEGORY = {"parcel", "credit", "finance"}
STEP1_PIPE_KEY_ORDER = [
    "RID",
    "IDX",
    "GC",
    "GL",
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "C7",
    "C8",
    "C9",
    "C10",
    "C11",
    "C12",
    "P1",
    "CR1",
    "F1",
]
STEP1_PIPE_RESULT_KEYS = [
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "C7",
    "C8",
    "C9",
    "C10",
    "C11",
    "C12",
    "P1",
    "CR1",
    "F1",
]
STEP1_PIPE_MODEL_OUTPUT_KEYS = [*STEP1_PIPE_RESULT_KEYS]
STEP1_OUTPUT_FORMATS = {"json", "pipe"}

STEP1_C5_TRIGGER_RE = re.compile(
    r"#PHONE|☎|전화|문의|연락|상담|고객센터|콜센터|카톡|카카오톡",
    flags=re.IGNORECASE,
)
STEP1_C5_CHANNEL_RE = re.compile(
    r"#PHONE|☎|전화|고객센터|콜센터|카톡|카카오톡",
    flags=re.IGNORECASE,
)
STEP1_C5_ACTION_RE = re.compile(
    r"문의\s*[:：]|연락\s*[:：]|상담문의|문의하여|문의해|문의주|연락주|연락\s*바랍니다|문의\s*바랍니다|전화\s*주세요|상담\s*가능|상담\s*요청|연락주세요|연락주셔요|신고\s*접수|접수\s*문의",
    flags=re.IGNORECASE,
)
STEP1_C7_TRIGGER_RE = re.compile(
    r"우체국|은행|저축은행|카드|국민카드|택배사|대한통운|금융|검찰|경찰|정부|국세청|법원|담당자|직원|수사관|팀장",
    flags=re.IGNORECASE,
)
STEP1_C7_ACTION_RE = re.compile(
    r"클릭|접속|확인|조회|설치|업데이트|입력|제공|회신|연락|상담|전화|문의|송금|이체|입금|납부|결제|인증|로그인",
    flags=re.IGNORECASE,
)
STEP1_C7_HIGH_RISK_ACTION_RE = re.compile(
    r"클릭|접속|설치|업데이트|입력|제공|인증|로그인|송금|이체|입금|납부|결제\s*(하|진행|요청|필요|바랍니다)|수수료\s*(납부|결제|입금)",
    flags=re.IGNORECASE,
)
STEP1_C7_ROLE_RE = re.compile(
    r"담당자|직원|수사관|팀장|주무관|관리자",
    flags=re.IGNORECASE,
)
STEP1_C7_NOTICE_TONE_RE = re.compile(
    r"안내|공지|알림|예정|완료|감사|문의처|고객센터",
    flags=re.IGNORECASE,
)
STEP1_C10_FORCE_PASS_RE = re.compile(
    r"해외배송|택배|배송|운송장|#TRACKING|등기|배달|수령|배송조회|반품|통관|관부가세|cj|대한통운|우체국",
    flags=re.IGNORECASE,
)
STEP1_C10_BLOCK_RE = re.compile(
    r"예방접종|병원|진료|예약|검진|의료",
    flags=re.IGNORECASE,
)
STEP1_F1_FORCE_PASS_RE = re.compile(
    r"저축은행|은행|계좌|입금|이체|송금|예금|출금|대출",
    flags=re.IGNORECASE,
)
STEP1_F1_CARD_ACCOUNT_ONLY_RE = re.compile(
    r"카드\s*결제\s*계좌|카드\s*결제계좌|카드\s*결제게좌",
    flags=re.IGNORECASE,
)

LEARNED_SCORE_RULES: dict[str, re.Pattern[str]] = {
    "rule_chat_channel": re.compile(
        r"카톡|카카오톡|오픈채팅|텔레그램|채널\s*추가|친구\s*추가|1\s*대\s*1\s*채팅|아이디\s*추가",
        flags=re.IGNORECASE,
    ),
    "rule_link_url": re.compile(
        r"https?://|www\.|링크|url|접속|클릭",
        flags=re.IGNORECASE,
    ),
    "rule_install_remote": re.compile(
        r"앱|어플|설치|업데이트|원격|팀뷰어|애니데스크|보안\s*프로그램|스크린\s*공유",
        flags=re.IGNORECASE,
    ),
    "rule_transfer_payment": re.compile(
        r"송금|이체|입금|출금|납부|결제|보안계좌|가상계좌|예치금|공탁",
        flags=re.IGNORECASE,
    ),
    "rule_loan_sales": re.compile(
        r"대출|한도|금리|저금리|대환|상환|승인|부결|심사",
        flags=re.IGNORECASE,
    ),
    "rule_sensitive_info": re.compile(
        r"인증번호|비밀번호|주민등록|신분증|계좌번호|카드번호|명의|otp|보안카드",
        flags=re.IGNORECASE,
    ),
    "rule_impersonation": re.compile(
        r"검찰|경찰|금감원|금융감독원|수사관|법원|국세청",
        flags=re.IGNORECASE,
    ),
    "rule_pressure": re.compile(
        r"지금|즉시|당장|바로|빨리|서둘러|오늘\s*안에|긴급",
        flags=re.IGNORECASE,
    ),
    "rule_fee_prepay": re.compile(
        r"수수료|선입금|보증금|법무비|예치금|공탁|상환처리|납부증명서",
        flags=re.IGNORECASE,
    ),
    "rule_safe_branch_visit": re.compile(
        r"지점\s*방문|영업점\s*방문|내점|창구",
        flags=re.IGNORECASE,
    ),
    "rule_safe_official_channel": re.compile(
        r"홈페이지|공식\s*앱|고객센터|대표번호|콜센터|ars|앱에서\s*확인",
        flags=re.IGNORECASE,
    ),
    "rule_safe_document_process": re.compile(
        r"증명서|발급|서류|팩스|신청서|제출|재발급",
        flags=re.IGNORECASE,
    ),
    "rule_safe_product_consult": re.compile(
        r"상품|가입|보험|예금|적금|청약|상담|안내",
        flags=re.IGNORECASE,
    ),
    "rule_safe_schedule": re.compile(
        r"영업시간|오전|오후|몇\s*시|까지|가능합니다|가능하십니다|방문하시면",
        flags=re.IGNORECASE,
    ),
}


def _get_cfg(cfg: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur = cfg
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _to_dtype(dtype_str: Optional[str]):
    if dtype_str is None:
        return None
    low = str(dtype_str).lower()
    if low in {"fp16", "float16", "half"}:
        return torch.float16
    if low in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if low in {"fp32", "float32"}:
        return torch.float32
    return None


def _count_cuda_visible_devices(cuda_devices: Any) -> Optional[int]:
    if cuda_devices is None:
        return None
    if isinstance(cuda_devices, (list, tuple)):
        count = len([v for v in cuda_devices if str(v).strip()])
        return count if count > 0 else None
    raw = str(cuda_devices).strip()
    if not raw:
        return None
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    return len(tokens) if tokens else None


def _resolve_n_devices(cfg: Dict[str, Any], cli_n_devices: Optional[int]) -> int:
    if cli_n_devices is not None:
        return max(1, int(cli_n_devices))

    model_n_devices = _get_cfg(cfg, "model.n_devices")
    if model_n_devices is not None:
        return max(1, int(model_n_devices))

    cuda_count = _count_cuda_visible_devices(_get_cfg(cfg, "run.cuda_visible_devices"))
    if cuda_count is not None:
        return max(1, cuda_count)

    return 1


def _safe_label(value: str) -> str:
    """
    Convert an arbitrary model name/path into a filesystem-safe label.
    """
    label = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")
    if len(label) > 80:
        digest = hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:10]
        label = f"{label[:48]}_{digest}"
    return label or "model"


def _resolve_finetuned_peft_paths(
    *,
    model_type: str,
    model_name: str,
    adapter_path: Optional[str],
) -> tuple[str, Optional[str]]:
    """
    Normalize Finetuned+PEFT loading paths.

    Common failure case:
    - model_name points to an adapter-only directory (contains adapter_config.json)
    - adapter_path is missing
    In that case we should load the base model from adapter_config and attach adapter_path.
    """
    if str(model_type).lower() != "finetuned":
        return model_name, adapter_path

    resolved_model_name = str(model_name)
    resolved_adapter_path = str(adapter_path) if adapter_path else None

    def _looks_like_full_model_dir(path: str) -> bool:
        if not os.path.isdir(path):
            return False
        required_any = (
            "config.json",
            "model.safetensors",
            "pytorch_model.bin",
        )
        if any(os.path.exists(os.path.join(path, name)) for name in required_any):
            return True
        return any(os.path.isdir(os.path.join(path, child)) for child in os.listdir(path) if str(child).startswith("checkpoint-"))

    candidate_adapter_dir: Optional[str] = None
    if resolved_adapter_path and os.path.isdir(resolved_adapter_path):
        adapter_cfg_path = os.path.join(resolved_adapter_path, "adapter_config.json")
        if os.path.exists(adapter_cfg_path):
            candidate_adapter_dir = resolved_adapter_path
        elif _looks_like_full_model_dir(resolved_adapter_path):
            print(
                "[PEFT] adapter_path does not contain adapter weights; "
                f"treating it as a full fine-tuned model directory: {resolved_adapter_path}"
            )
            return resolved_adapter_path, None
    elif os.path.isdir(resolved_model_name):
        adapter_cfg_path = os.path.join(resolved_model_name, "adapter_config.json")
        if os.path.exists(adapter_cfg_path):
            candidate_adapter_dir = resolved_model_name
            resolved_adapter_path = resolved_model_name

    if not candidate_adapter_dir:
        return resolved_model_name, resolved_adapter_path

    adapter_cfg_path = os.path.join(candidate_adapter_dir, "adapter_config.json")
    try:
        with open(adapter_cfg_path, "r", encoding="utf-8") as f:
            adapter_cfg = json.load(f) or {}
    except Exception as e:
        print(f"[Warn] Failed to read adapter config: {adapter_cfg_path} ({e})")
        return resolved_model_name, resolved_adapter_path

    base_model_name = adapter_cfg.get("base_model_name_or_path")
    if not isinstance(base_model_name, str) or not base_model_name.strip():
        return resolved_model_name, resolved_adapter_path

    if os.path.isdir(resolved_model_name):
        try:
            same_dir = os.path.samefile(resolved_model_name, candidate_adapter_dir)
        except Exception:
            same_dir = os.path.abspath(resolved_model_name) == os.path.abspath(candidate_adapter_dir)
        if same_dir:
            resolved_model_name = base_model_name.strip()
            print(
                "[PEFT] Detected adapter-only Finetuned model directory. "
                f"Using base model `{resolved_model_name}` with adapter `{candidate_adapter_dir}`."
            )

    return resolved_model_name, resolved_adapter_path


def _prepare_prompt_inputs(
    df,
    *,
    prompt_cfg: Dict[str, Any],
    text_col: str,
):
    if text_col not in df.columns:
        raise ValueError(f"`{text_col}` column not found. cols={list(df.columns)}")

    out_df = df.copy()
    use_instruction = bool(prompt_cfg.get("use_instruction", False))
    instruction_path = prompt_cfg.get("instruction_path", "")
    fmt = prompt_cfg.get("format", "plain")
    user_prefix = prompt_cfg.get("user_prefix", "문장: ")
    answer_prefix_raw = prompt_cfg.get("answer_prefix", None)
    if answer_prefix_raw is None:
        answer_prefix = "정답:"
    else:
        answer_prefix = str(answer_prefix_raw).rstrip()
    row_template = str(prompt_cfg.get("row_template", "")).strip()
    answer_with_space = f"{answer_prefix} " if answer_prefix else ""

    instruction_texts = [""] * len(out_df)
    instruction_paths = [""] * len(out_df)
    instruction_embeds_input = [False] * len(out_df)
    if use_instruction:
        if not instruction_path and not prompt_cfg.get("instruction_rules"):
            raise ValueError(
                "prompt.use_instruction=true but both prompt.instruction_path and "
                "prompt.instruction_rules are empty"
            )
        instruction_texts, instruction_paths, instruction_embeds_input = resolve_instruction_texts(
            df=out_df,
            prompt_cfg=prompt_cfg,
            text_col=text_col,
        )
        if any(instruction_paths):
            out_df["prompt_instruction_path"] = instruction_paths
            out_df["prompt_instruction_name"] = [os.path.basename(path) if path else "" for path in instruction_paths]

    if row_template:
        texts = render_row_template(out_df, row_template=row_template)
    else:
        texts = out_df[text_col].astype(str).tolist()

    if not use_instruction:
        inputs = texts
    else:
        inputs = []
        for idx, text in enumerate(texts):
            instruction_block = str(instruction_texts[idx]).rstrip()
            if instruction_embeds_input[idx]:
                if answer_with_space:
                    inputs.append(f"{instruction_block}\n{answer_with_space}")
                else:
                    inputs.append(f"{instruction_block}\n")
            else:
                if answer_with_space:
                    inputs.append(f"{instruction_block}\n{user_prefix}{text}\n{answer_with_space}")
                else:
                    inputs.append(f"{instruction_block}\n{user_prefix}{text}\n")
    return out_df, texts, inputs


def _last_nonpad_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    if attention_mask.ndim != 2:
        raise ValueError(f"Expected 2D attention_mask, got shape={tuple(attention_mask.shape)}")
    flipped = torch.flip(attention_mask.long(), dims=[1])
    last_offsets = torch.argmax(flipped, dim=1)
    return (attention_mask.shape[1] - 1 - last_offsets).long()


def _score_binary_choice_groups(
    *,
    model_name: str,
    hf_token: Optional[str],
    torch_dtype: Optional[torch.dtype],
    adapter_path: Optional[str],
    merge_adapter: bool,
    grouped_inputs: Dict[str, list[str]],
    batch_size: int,
    max_input_tokens: int,
    positive_choice: str = "1",
    negative_choice: str = "0",
) -> Dict[str, Dict[str, np.ndarray]]:
    tokenizer = _hf_load_tokenizer(model_name, hf_token)
    model = _hf_load_model(model_name, hf_token, torch_dtype, adapter_path, merge_adapter)
    safe_len = get_safe_max_len(tokenizer, user_cap=max_input_tokens)
    pos_ids = tokenizer.encode(str(positive_choice), add_special_tokens=False)
    neg_ids = tokenizer.encode(str(negative_choice), add_special_tokens=False)
    if len(pos_ids) != 1 or len(neg_ids) != 1:
        raise ValueError(
            "learned score adjustment requires single-token binary choices. "
            f"positive={positive_choice!r}:{pos_ids}, negative={negative_choice!r}:{neg_ids}"
        )
    pos_id = int(pos_ids[0])
    neg_id = int(neg_ids[0])
    first_device = next(model.parameters()).device

    results: Dict[str, Dict[str, np.ndarray]] = {}
    for group_name, inputs in grouped_inputs.items():
        prob1_all: list[float] = []
        margin_all: list[float] = []
        for start in range(0, len(inputs), batch_size):
            batch = inputs[start : start + batch_size]
            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=safe_len,
            )
            enc.pop("token_type_ids", None)
            enc = {k: v.to(first_device) for k, v in enc.items()}
            with torch.inference_mode():
                if torch.cuda.is_available() and torch_dtype in (torch.float16, torch.bfloat16):
                    with torch.autocast(device_type="cuda", dtype=torch_dtype):
                        outputs = model(**enc)
                else:
                    outputs = model(**enc)
            logits = outputs.logits
            last_idx = _last_nonpad_indices(enc["attention_mask"])
            batch_idx = torch.arange(logits.shape[0], device=logits.device)
            next_token_logits = logits[batch_idx, last_idx]
            label_logits = torch.stack(
                (next_token_logits[:, neg_id], next_token_logits[:, pos_id]),
                dim=-1,
            )
            probs1 = torch.softmax(label_logits.float(), dim=-1)[:, 1].detach().cpu().numpy()
            margins = (label_logits[:, 1] - label_logits[:, 0]).float().detach().cpu().numpy()
            prob1_all.extend(float(v) for v in probs1.tolist())
            margin_all.extend(float(v) for v in margins.tolist())
        results[group_name] = {
            "prob1": np.asarray(prob1_all, dtype=np.float64),
            "margin": np.asarray(margin_all, dtype=np.float64),
        }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def _build_learned_score_matrix(
    *,
    texts: list[str],
    prob1: np.ndarray,
    margin: np.ndarray,
) -> tuple[np.ndarray, list[str], Dict[str, np.ndarray]]:
    feature_rows: list[list[float]] = []
    risk_counts: list[float] = []
    safe_counts: list[float] = []
    for text_value, prob_value, margin_value in zip(texts, prob1.tolist(), margin.tolist()):
        raw_text = str(text_value or "")
        row = [
            float(prob_value),
            float(margin_value),
            float(abs(margin_value)),
            float(np.log1p(len(raw_text))),
        ]
        matched_values: list[float] = []
        for pattern in LEARNED_SCORE_RULES.values():
            matched_values.append(1.0 if pattern.search(raw_text) else 0.0)
        risk_count = float(sum(matched_values[:9]))
        safe_count = float(sum(matched_values[9:]))
        row.extend(matched_values)
        row.append(risk_count)
        row.append(safe_count)
        feature_rows.append(row)
        risk_counts.append(risk_count)
        safe_counts.append(safe_count)

    feature_names = [
        "base_prob1",
        "base_margin",
        "base_abs_margin",
        "text_len_log1p",
        *list(LEARNED_SCORE_RULES.keys()),
        "rule_risk_count",
        "rule_safe_count",
    ]
    return (
        np.asarray(feature_rows, dtype=np.float64),
        feature_names,
        {
            "risk_count": np.asarray(risk_counts, dtype=np.float64),
            "safe_count": np.asarray(safe_counts, dtype=np.float64),
        },
    )


def _select_score_threshold(
    *,
    y_true: np.ndarray,
    score: np.ndarray,
) -> dict[str, Any]:
    candidates = np.unique(
        np.concatenate(
            [
                np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
                np.round(score.astype(np.float64), 6),
            ]
        )
    )
    best: Optional[dict[str, Any]] = None
    for threshold in candidates.tolist():
        y_pred = (score >= float(threshold)).astype(np.int64)
        metrics = compute_binary_metrics(y_true.astype(int).tolist(), y_pred.astype(int).tolist())
        key = (
            float(metrics.get("macro_f1", 0.0)),
            float(metrics.get("accuracy", 0.0)),
            float(metrics.get("class_0", {}).get("recall", 0.0)),
            -float(threshold),
        )
        if best is None or key > best["key"]:
            best = {
                "threshold": float(threshold),
                "metrics": metrics,
                "key": key,
            }
    if best is None:
        raise ValueError("Failed to select score threshold.")
    return best


def _fit_learned_binary_score_adjuster(
    *,
    texts: list[str],
    labels: np.ndarray,
    prob1: np.ndarray,
    margin: np.ndarray,
    strength: float = 1.0,
) -> dict[str, Any]:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as e:
        raise ImportError("scikit-learn is required for decode.score_adjust.method=learned_logreg_v1") from e

    x_train, feature_names, aux = _build_learned_score_matrix(texts=texts, prob1=prob1, margin=margin)
    y_train = labels.astype(np.int64)
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    solver="lbfgs",
                    random_state=0,
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)
    strength_value = float(np.clip(float(strength), 0.0, 1.0))
    learned_train_score = pipeline.predict_proba(x_train)[:, 1].astype(np.float64)
    train_score = ((1.0 - strength_value) * prob1.astype(np.float64)) + (strength_value * learned_train_score)
    selected = _select_score_threshold(y_true=y_train, score=train_score)
    baseline_pred = (prob1 >= 0.5).astype(np.int64)
    baseline_metrics = compute_binary_metrics(y_train.tolist(), baseline_pred.tolist())
    logreg = pipeline.named_steps["logreg"]
    coef = logreg.coef_[0].astype(float).tolist() if getattr(logreg, "coef_", None) is not None else []
    coef_by_feature = {
        name: float(value) for name, value in zip(feature_names, coef)
    }
    return {
        "pipeline": pipeline,
        "threshold": float(selected["threshold"]),
        "feature_names": feature_names,
        "baseline_metrics": baseline_metrics,
        "train_metrics": selected["metrics"],
        "train_prob": train_score,
        "strength": strength_value,
        "coef_by_feature": coef_by_feature,
        "intercept": float(logreg.intercept_[0]) if getattr(logreg, "intercept_", None) is not None else 0.0,
        "aux": aux,
    }


def _apply_learned_binary_score_adjuster(
    *,
    score_pipeline,
    texts: list[str],
    prob1: np.ndarray,
    margin: np.ndarray,
    threshold: float,
    strength: float = 1.0,
) -> dict[str, Any]:
    x_eval, _, aux = _build_learned_score_matrix(texts=texts, prob1=prob1, margin=margin)
    strength_value = float(np.clip(float(strength), 0.0, 1.0))
    learned_prob = score_pipeline.predict_proba(x_eval)[:, 1].astype(np.float64)
    adjusted_prob = ((1.0 - strength_value) * prob1.astype(np.float64)) + (strength_value * learned_prob)
    adjusted_pred = (adjusted_prob >= float(threshold)).astype(np.int64)
    return {
        "prob": adjusted_prob,
        "pred": adjusted_pred,
        "strength": strength_value,
        "aux": aux,
    }


def _render_row_template(df, row_template: str) -> list[str]:
    return render_row_template(df=df, row_template=row_template)


def _resolve_instruction_texts(
    df, prompt_cfg: Dict[str, Any], text_col: str = "text"
) -> tuple[list[str], list[str], list[bool]]:
    return resolve_instruction_texts(df=df, prompt_cfg=prompt_cfg, text_col=text_col)


def _load_decode_progress(
    progress_path: str,
    total_n: int,
) -> dict[int, tuple[int, str, Optional[list[str]], Optional[list[Optional[float]]]]]:
    if not os.path.exists(progress_path):
        return {}

    recovered: dict[int, tuple[int, str, Optional[list[str]], Optional[list[Optional[float]]]]] = {}
    with open(progress_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                idx = int(item.get("idx"))
                pred = int(item.get("pred"))
                gen = str(item.get("gen", ""))
                gen_tokens_raw = item.get("gen_tokens")
                token_logprobs_raw = item.get("gen_token_logprobs")
            except Exception:
                continue
            gen_tokens: Optional[list[str]] = None
            if isinstance(gen_tokens_raw, list):
                gen_tokens = [str(token) for token in gen_tokens_raw]
            token_logprobs: Optional[list[Optional[float]]] = None
            if isinstance(token_logprobs_raw, list):
                token_logprobs = []
                for value in token_logprobs_raw:
                    if value is None:
                        token_logprobs.append(None)
                        continue
                    try:
                        token_logprobs.append(float(value))
                    except Exception:
                        token_logprobs.append(None)
            if 0 <= idx < total_n:
                recovered[idx] = (pred, gen, gen_tokens, token_logprobs)
    return recovered


def _append_decode_progress(
    progress_path: str,
    idx: int,
    pred: int,
    gen: str,
    *,
    gen_tokens: Optional[list[str]] = None,
    gen_token_logprobs: Optional[list[Optional[float]]] = None,
) -> None:
    row = {"idx": int(idx), "pred": int(pred), "gen": str(gen)}
    if gen_tokens is not None:
        row["gen_tokens"] = [str(token) for token in gen_tokens]
    if gen_token_logprobs is not None:
        normalized_logprobs: list[Optional[float]] = []
        for value in gen_token_logprobs:
            if value is None:
                normalized_logprobs.append(None)
                continue
            try:
                normalized_logprobs.append(float(value))
            except Exception:
                normalized_logprobs.append(None)
        row["gen_token_logprobs"] = normalized_logprobs
    with open(progress_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _strip_markdown_codeblock(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 2:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def _extract_fenced_json_block(text: str) -> Optional[str]:
    match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _infer_step1_output_format(instruction_path: str) -> str:
    base = os.path.basename(str(instruction_path or "")).lower()
    if "_pipe" in base:
        return "pipe"
    return "json"


def _parse_step1_json_object(gen_text: str) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    raw = str(gen_text or "")
    candidates: list[str] = []
    parse_errors: list[str] = []
    decoder = json.JSONDecoder()

    def _append_candidate(candidate: Optional[str]) -> None:
        if candidate is None:
            return
        cand = candidate.strip()
        if cand and cand not in candidates:
            candidates.append(cand)

    _append_candidate(raw)
    _append_candidate(_strip_markdown_codeblock(raw))
    _append_candidate(_extract_fenced_json_block(raw))

    first_brace = raw.find("{")
    if first_brace != -1:
        _append_candidate(raw[first_brace:])

    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj, None
            parse_errors.append(f"json.loads_not_object(type={type(obj).__name__})")
        except Exception as e:
            parse_errors.append(repr(e))

        try:
            obj_raw, _ = decoder.raw_decode(cand)
            if isinstance(obj_raw, dict):
                return obj_raw, None
            parse_errors.append(f"raw_decode_not_object(type={type(obj_raw).__name__})")
        except Exception as e:
            parse_errors.append(repr(e))

    if parse_errors:
        return None, f"invalid_json: {parse_errors[-1]}"
    return None, "invalid_json: no_json_candidate_found"


def _extract_pipe_segment(text: str) -> Optional[str]:
    if not text:
        return None
    candidates = [str(text), _strip_markdown_codeblock(str(text))]
    pattern = re.compile(r"B\s*\|(?P<body>.*?)\|\s*E\b", flags=re.IGNORECASE | re.DOTALL)
    for cand in candidates:
        match = pattern.search(cand)
        if match:
            return f"B|{match.group('body')}|E"
    return None


def _parse_step1_pipe_tokens(gen_text: str) -> tuple[Optional[list[str]], Optional[str]]:
    segment = _extract_pipe_segment(str(gen_text or ""))
    if segment is None:
        return None, "invalid_pipe: missing_B_to_E_segment"
    tokens = [tok.strip() for tok in segment.split("|") if tok.strip()]
    if not tokens:
        return None, "invalid_pipe: empty_tokens"
    return tokens, None


def _to_int_or_none(value: Any) -> Optional[int]:
    try:
        return int(str(value).strip())
    except Exception:
        return None


def _normalize_pass_fail(value: Any) -> str:
    s = str("" if value is None else value).strip()
    if not s:
        return "FAIL"

    up = s.upper()
    if re.search(r"\bPASS\b", up):
        return "PASS"
    if re.search(r"\bFAIL\b", up):
        return "FAIL"

    head = re.sub(r"[^A-Z가-힣]+", "", up)
    # Common typos / odd partials observed in logs.
    if head in {"PASS", "PAS", "PA", "P", "PARE"}:
        return "PASS"
    if head in {"FAIL", "FAL", "FA", "F", "FLIL", "FA일"}:
        return "FAIL"

    if len(head) <= 5 and head.startswith("P"):
        return "PASS"
    if len(head) <= 5 and head.startswith("F"):
        return "FAIL"

    return "FAIL"


def _is_empty_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and value != value:
        return True
    return str(value).strip() == ""


def _extract_step1_row_text(row: Dict[str, Any]) -> str:
    for key in ("text", "input_text", "input", "message", "content"):
        value = row.get(key)
        if not _is_empty_like(value):
            return str(value)

    best = ""
    for value in row.values():
        if isinstance(value, str) and len(value) > len(best):
            best = value
    return best


def _apply_step1_pipe_post_rules(row: Dict[str, Any], ordered: Dict[str, str]) -> Dict[str, str]:
    text = _extract_step1_row_text(row)
    label = _to_int_or_none(row.get("label"))

    has_c5_trigger = bool(STEP1_C5_TRIGGER_RE.search(text))
    has_c5_channel = bool(STEP1_C5_CHANNEL_RE.search(text))
    has_c5_action = bool(STEP1_C5_ACTION_RE.search(text))
    c5_should_pass = has_c5_action and (has_c5_trigger or has_c5_channel)

    if ordered.get("C5") == "PASS":
        if not c5_should_pass:
            ordered["C5"] = "FAIL"
    elif c5_should_pass:
        ordered["C5"] = "PASS"

    has_delivery = bool(STEP1_C10_FORCE_PASS_RE.search(text))
    has_delivery_block = bool(STEP1_C10_BLOCK_RE.search(text))
    if has_delivery and not has_delivery_block:
        ordered["C10"] = "PASS"
        ordered["P1"] = "PASS"
    elif has_delivery_block and not has_delivery:
        ordered["C10"] = "FAIL"
        ordered["P1"] = "FAIL"

    has_f1_trigger = bool(STEP1_F1_FORCE_PASS_RE.search(text))
    has_f1_card_account_only = bool(STEP1_F1_CARD_ACCOUNT_ONLY_RE.search(text))
    if has_f1_trigger and not has_f1_card_account_only:
        ordered["F1"] = "PASS"

    if ordered.get("C7") == "PASS":
        has_trigger = bool(STEP1_C7_TRIGGER_RE.search(text))
        has_action = bool(STEP1_C7_ACTION_RE.search(text))
        has_high_risk_action = bool(STEP1_C7_HIGH_RISK_ACTION_RE.search(text))
        has_role = bool(STEP1_C7_ROLE_RE.search(text))
        has_notice_tone = bool(STEP1_C7_NOTICE_TONE_RE.search(text))
        allow_contact_only = (label == 1) and has_role and has_c5_action
        c7_should_pass = has_trigger and has_action and (has_high_risk_action or allow_contact_only)
        if has_notice_tone and not has_high_risk_action and not allow_contact_only:
            c7_should_pass = False
        if not c7_should_pass:
            ordered["C7"] = "FAIL"

    return ordered


def _extract_pipe_kv_map(gen_text: str) -> dict[str, str]:
    raw = str(gen_text or "")
    candidates: list[str] = []
    segment = _extract_pipe_segment(raw)
    if segment:
        candidates.append(segment)
    stripped = _strip_markdown_codeblock(raw)
    if stripped and stripped not in candidates:
        candidates.append(stripped)
    if raw and raw not in candidates:
        candidates.append(raw)

    for cand in candidates:
        kv: dict[str, str] = {}
        for token in re.split(r"[|\n\r]+", cand):
            tok = token.strip()
            if not tok or tok.upper() in {"B", "E"}:
                continue
            if "=" not in tok:
                continue
            key_raw, value_raw = tok.split("=", 1)
            key = key_raw.strip().upper()
            value = value_raw.strip()
            if key:
                kv[key] = value
        if kv:
            return kv

    kv_regex: dict[str, str] = {}
    for match in re.finditer(r"\b([A-Za-z][A-Za-z0-9_]*)\s*=\s*([^|\n\r]+)", raw):
        key = match.group(1).strip().upper()
        value = match.group(2).strip()
        if key:
            kv_regex[key] = value
    return kv_regex


def _build_step1_pipe_wrapped_gen(gen_text: str, row: Dict[str, Any]) -> str:
    kv = _extract_pipe_kv_map(gen_text)

    if not _is_empty_like(row.get("row_idx")):
        rid_source = row.get("row_idx")
    elif not _is_empty_like(row.get("id")):
        rid_source = row.get("id")
    else:
        rid_source = 0
    rid = str(rid_source).strip()

    idx_int = _to_int_or_none(row.get("row_idx"))
    if idx_int is None:
        idx_int = _to_int_or_none(rid_source)
    idx = str(idx_int if idx_int is not None else 0)

    if not _is_empty_like(row.get("category")):
        gc = str(row.get("category")).strip()
    else:
        gc = str(kv.get("GC", "")).strip()

    gl_int = _to_int_or_none(row.get("label"))
    if gl_int not in (0, 1):
        gl_int = _to_int_or_none(kv.get("GL"))
    gl = str(gl_int if gl_int in (0, 1) else "")

    ordered: Dict[str, str] = {
        "RID": rid,
        "IDX": idx,
        "GC": gc,
        "GL": gl,
    }
    for key in STEP1_PIPE_MODEL_OUTPUT_KEYS:
        ordered[key] = _normalize_pass_fail(kv.get(key))

    ordered = _apply_step1_pipe_post_rules(row=row, ordered=ordered)

    return "B|" + "|".join(f"{key}={ordered.get(key, '')}" for key in STEP1_PIPE_KEY_ORDER) + "|E"


def _normalize_step1_pipe_gens(df, gens: list[str]) -> list[str]:
    rows = df.to_dict(orient="records")
    normalized: list[str] = []
    for i, gen_text in enumerate(gens):
        row = rows[i] if i < len(rows) else {}
        normalized.append(_build_step1_pipe_wrapped_gen(str(gen_text), row))
    return normalized


def _validate_step1_json_one(gen_text: str, row: Dict[str, Any]) -> list[str]:
    errors: list[str] = []
    obj, parse_error = _parse_step1_json_object(gen_text)
    if parse_error:
        return [parse_error]

    if not isinstance(obj, dict):
        return ["top_level_not_object"]

    required_top = {"id", "golden", "checklist", "notes"}
    missing_top = sorted(required_top - set(obj.keys()))
    if missing_top:
        errors.append(f"missing_top_keys={missing_top}")

    golden = obj.get("golden")
    if not isinstance(golden, dict):
        errors.append("golden_not_object")
    else:
        category = str(golden.get("category", ""))
        label = golden.get("label")
        if category not in STEP1_ALLOWED_CATEGORY:
            errors.append(f"invalid_golden_category={category!r}")
        if label not in (0, 1):
            errors.append(f"invalid_golden_label={label!r}")
        if "category" in row and str(row["category"]) != category:
            errors.append(f"golden_category_mismatch(row={row['category']!r}, out={category!r})")
        if "label" in row:
            try:
                row_label = int(row["label"])
            except Exception:
                row_label = row["label"]
            if row_label != label:
                errors.append(f"golden_label_mismatch(row={row_label!r}, out={label!r})")

    out_id = obj.get("id")
    if "row_idx" in row and str(out_id) != str(row["row_idx"]):
        errors.append(f"id_mismatch(row_idx={row['row_idx']!r}, out_id={out_id!r})")

    checklist = obj.get("checklist")
    if not isinstance(checklist, list):
        errors.append("checklist_not_list")
        return errors
    if len(checklist) != len(STEP1_CHECKLIST_ORDER):
        errors.append(
            f"invalid_checklist_length={len(checklist)} expected={len(STEP1_CHECKLIST_ORDER)}"
        )
        return errors

    for idx, expected_item_id in enumerate(STEP1_CHECKLIST_ORDER):
        item = checklist[idx]
        if not isinstance(item, dict):
            errors.append(f"checklist[{idx}]_not_object")
            continue

        item_id = item.get("item_id")
        if item_id != expected_item_id:
            errors.append(
                f"checklist[{idx}]_item_id_mismatch(expected={expected_item_id!r}, got={item_id!r})"
            )

        result = item.get("result")
        if result not in STEP1_ALLOWED_RESULT:
            errors.append(f"checklist[{idx}]_invalid_result={result!r}")

        rationale = item.get("rationale")
        if not isinstance(rationale, str):
            errors.append(f"checklist[{idx}]_rationale_not_string")

        evidence = item.get("evidence")
        if not isinstance(evidence, list):
            errors.append(f"checklist[{idx}]_evidence_not_list")
            continue
        if len(evidence) > 2:
            errors.append(f"checklist[{idx}]_evidence_too_many={len(evidence)}")
        if result == "PASS" and len(evidence) < 1:
            errors.append(f"checklist[{idx}]_pass_without_evidence")
        if any(not isinstance(e, str) for e in evidence):
            errors.append(f"checklist[{idx}]_evidence_non_string")

    return errors


def _validate_step1_pipe_one(gen_text: str, row: Dict[str, Any]) -> list[str]:
    errors: list[str] = []
    tokens, parse_error = _parse_step1_pipe_tokens(gen_text=gen_text)
    if parse_error:
        return [parse_error]
    if not tokens:
        return ["invalid_pipe: empty_tokens"]

    if tokens[0].upper() != "B":
        errors.append(f"invalid_begin_token={tokens[0]!r}")
    if tokens[-1].upper() != "E":
        errors.append(f"invalid_end_token={tokens[-1]!r}")

    kv_tokens = tokens[1:-1]
    if len(kv_tokens) != len(STEP1_PIPE_KEY_ORDER):
        errors.append(
            f"invalid_kv_count={len(kv_tokens)} expected={len(STEP1_PIPE_KEY_ORDER)}"
        )

    parsed: Dict[str, str] = {}
    for idx, expected_key in enumerate(STEP1_PIPE_KEY_ORDER):
        if idx >= len(kv_tokens):
            errors.append(f"missing_kv_key={expected_key!r}")
            continue
        token = kv_tokens[idx]
        if "=" not in token:
            errors.append(f"kv[{idx}]_missing_equal={token!r}")
            continue
        key_raw, value_raw = token.split("=", 1)
        key = key_raw.strip().upper()
        value = value_raw.strip()
        if key != expected_key:
            errors.append(f"kv[{idx}]_key_mismatch(expected={expected_key!r}, got={key!r})")
        if key in parsed:
            errors.append(f"duplicate_key={key!r}")
        parsed[key] = value

    rid = parsed.get("RID")
    if not rid:
        errors.append("invalid_rid_empty")
    elif "row_idx" in row and str(rid) != str(row["row_idx"]):
        errors.append(f"rid_mismatch(row_idx={row['row_idx']!r}, rid={rid!r})")

    idx_val = _to_int_or_none(parsed.get("IDX"))
    if idx_val is None:
        errors.append(f"invalid_idx={parsed.get('IDX')!r}")
    elif "row_idx" in row:
        row_idx_int = _to_int_or_none(row.get("row_idx"))
        if row_idx_int is not None and idx_val != row_idx_int:
            errors.append(f"idx_mismatch(row_idx={row_idx_int!r}, idx={idx_val!r})")

    gc = str(parsed.get("GC", ""))
    if gc not in STEP1_ALLOWED_CATEGORY:
        errors.append(f"invalid_gc={gc!r}")
    elif "category" in row and str(row["category"]) != gc:
        errors.append(f"gc_mismatch(row={row['category']!r}, gc={gc!r})")

    gl = _to_int_or_none(parsed.get("GL"))
    if gl not in (0, 1):
        errors.append(f"invalid_gl={parsed.get('GL')!r}")
    elif "label" in row:
        row_label = _to_int_or_none(row.get("label"))
        if row_label is not None and gl != row_label:
            errors.append(f"gl_mismatch(row={row_label!r}, gl={gl!r})")

    for key in STEP1_PIPE_RESULT_KEYS:
        result = parsed.get(key)
        if result not in STEP1_PIPE_ALLOWED_RESULT:
            errors.append(f"{key}_invalid_result={result!r}")

    return errors


def _validate_step1_outputs(df, gens, output_format: str = "json") -> list[dict[str, Any]]:
    rows = df.to_dict(orient="records")
    issues: list[dict[str, Any]] = []
    fmt = str(output_format or "json").lower()
    if fmt not in STEP1_OUTPUT_FORMATS:
        raise ValueError(f"Unsupported step1 output format: {fmt!r}")
    for i, (row, gen_text) in enumerate(zip(rows, gens)):
        if fmt == "pipe":
            errors = _validate_step1_pipe_one(gen_text=gen_text, row=row)
        else:
            errors = _validate_step1_json_one(gen_text=gen_text, row=row)
        if errors:
            issues.append(
                {
                    "row_number": i,
                    "row_idx": row.get("row_idx"),
                    "category": row.get("category"),
                    "label": row.get("label"),
                    "errors": errors,
                }
            )
    return issues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model_id", type=int, default=None)
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Preferred. Decoder/LLM: HF model id, Finetuned: local model dir.",
    )
    parser.add_argument("--model_type", type=str, default=None, help="Decoder|LLM|Finetuned|Encoder")
    parser.add_argument("--eval_set", type=str, default=None)
    parser.add_argument("--method", type=str, default=None, choices=["vllm_decode", "hf_decode"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out_dir", type=str, default="", help="(optional) run directory to write outputs/logs")
    parser.add_argument("--n_devices", type=int, default=None)
    parser.add_argument("--result_path", type=str, default=None)
    parser.add_argument("--hf_batch_size", type=int, default=None)
    parser.add_argument("--max_input_tokens", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--adapter_path", type=str, default=None, help="Optional PEFT adapter path")
    parser.add_argument(
        "--merge_adapter",
        action="store_true",
        help="Merge adapter weights into base model for HF decode (always merged for vLLM)",
    )
    args = parser.parse_args()

    cfg: Dict[str, Any] = {}
    if args.config:
        try:
            import yaml

            with open(args.config, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception as e:
            raise RuntimeError(f"Failed to read config: {args.config} ({e})")

    # env/seed
    hf_tok = setup_env()  # .env에서 hf_token 읽어오는 함수라고 가정

    backend = _get_cfg(cfg, "model.backend", "vllm")
    method = args.method or ("vllm_decode" if backend == "vllm" else "hf_decode")

    model_type = args.model_type if args.model_type is not None else _get_cfg(cfg, "model.model_type", "Decoder")
    model_name_cfg = _get_cfg(cfg, "model.model_name")
    model_name_arg = args.model_name if args.model_name is not None else None
    model_name_raw = model_name_arg or model_name_cfg
    model_id = args.model_id if args.model_id is not None else _get_cfg(cfg, "model.model_id")
    eval_set = args.eval_set or _get_cfg(cfg, "data.eval_csv")
    if eval_set is None:
        raise ValueError("`eval_set` is required (CLI args or config).")

    try:
        model_name = resolve_model_name(model_type=model_type, model_name=model_name_raw, model_id=model_id)
    except ValueError as e:
        raise ValueError("`model_name` (preferred) or `model_id` must be provided.") from e
    adapter_path = args.adapter_path or _get_cfg(cfg, "model.adapter_path")
    model_name, adapter_path = _resolve_finetuned_peft_paths(
        model_type=str(model_type),
        model_name=str(model_name),
        adapter_path=adapter_path,
    )
    model_label = _safe_label(model_name_raw if model_name_raw else model_id)
    if str(model_type).lower() == "finetuned" and (not adapter_path) and (not os.path.exists(model_name)):
        print(f"[Warn] Finetuned model directory not found: {model_name}")
    merge_adapter = bool(args.merge_adapter or _get_cfg(cfg, "model.merge_adapter", False))

    seed = args.seed if args.seed is not None else int(_get_cfg(cfg, "run.seed", 10))
    deterministic = bool(_get_cfg(cfg, "run.deterministic", True))
    benchmark = bool(_get_cfg(cfg, "run.benchmark", False))
    set_seed(seed, deterministic=deterministic, benchmark=benchmark)

    n_devices = _resolve_n_devices(cfg, args.n_devices)
    tensor_parallel_size = int(_get_cfg(cfg, "model.tensor_parallel_size", n_devices))
    pipeline_parallel_size = int(_get_cfg(cfg, "model.pipeline_parallel_size", 1))
    if tensor_parallel_size < 1 or pipeline_parallel_size < 1:
        raise ValueError("`model.tensor_parallel_size` and `model.pipeline_parallel_size` must be >= 1")
    if tensor_parallel_size * pipeline_parallel_size != n_devices:
        raise ValueError(
            "`n_devices` must equal `model.tensor_parallel_size * model.pipeline_parallel_size` "
            f"(got n_devices={n_devices}, tensor_parallel_size={tensor_parallel_size}, "
            f"pipeline_parallel_size={pipeline_parallel_size})"
        )
    torch_dtype = _to_dtype(_get_cfg(cfg, "model.dtype", None))
    max_input_tokens_cfg = (
        args.max_input_tokens
        if args.max_input_tokens is not None
        else _get_cfg(cfg, "decode.max_input_tokens", 1024)
    )
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else int(
        _get_cfg(cfg, "decode.max_new_tokens", 8)
    )
    decode_temperature = float(_get_cfg(cfg, "decode.temperature", 0.0))
    decode_top_p = float(_get_cfg(cfg, "decode.top_p", 1.0))
    decode_presence_penalty = float(_get_cfg(cfg, "decode.presence_penalty", 0.0))
    decode_frequency_penalty = float(_get_cfg(cfg, "decode.frequency_penalty", 0.0))
    decode_repetition_penalty = float(_get_cfg(cfg, "decode.repetition_penalty", 1.0))
    auto_max_input_tokens = False
    if isinstance(max_input_tokens_cfg, str):
        auto_mode = max_input_tokens_cfg.strip().lower()
        if auto_mode in {"auto", "auto_p95", "p95"}:
            auto_max_input_tokens = True
            max_input_tokens = None
        else:
            max_input_tokens = int(max_input_tokens_cfg)
    else:
        max_input_tokens = int(max_input_tokens_cfg)
    if not 0.0 < decode_top_p <= 1.0:
        raise ValueError("`decode.top_p` must satisfy 0 < top_p <= 1")
    max_input_tokens_quantile = float(_get_cfg(cfg, "decode.max_input_tokens_quantile", 0.95))
    max_input_tokens_margin = int(_get_cfg(cfg, "decode.max_input_tokens_margin", 64))
    max_input_tokens_measure_batch_size = int(_get_cfg(cfg, "decode.max_input_tokens_measure_batch_size", 32))
    hf_batch_size_cfg = _get_cfg(cfg, "decode.hf_batch_size")
    if hf_batch_size_cfg is None:
        hf_batch_size_cfg = _get_cfg(cfg, "decode.batch_size", 8)
    hf_batch_size = args.hf_batch_size if args.hf_batch_size is not None else int(hf_batch_size_cfg)
    if hf_batch_size < 1:
        raise ValueError("`decode.hf_batch_size` (or `decode.batch_size`) must be >= 1")
    allow_hf_fallback = bool(_get_cfg(cfg, "decode.allow_hf_fallback", True))
    pred_mode = str(_get_cfg(cfg, "decode.pred_mode", "parse_gen")).strip().lower()
    if pred_mode not in {"parse_gen", "label_col"}:
        raise ValueError("`decode.pred_mode` must be one of {'parse_gen', 'label_col'}")
    compute_metrics = bool(_get_cfg(cfg, "decode.compute_metrics", True))
    validate_step1 = bool(
        _get_cfg(cfg, "decode.validate_step1", _get_cfg(cfg, "decode.validate_step1_json", False))
    )
    fail_on_invalid_step1 = bool(
        _get_cfg(
            cfg,
            "decode.fail_on_invalid_step1",
            _get_cfg(cfg, "decode.fail_on_invalid_step1_json", True),
        )
    )
    step1_output_format_cfg = str(_get_cfg(cfg, "decode.step1_output_format", "auto")).lower()
    incremental_save = bool(_get_cfg(cfg, "decode.incremental_save", False))
    resume_from_checkpoint = bool(_get_cfg(cfg, "decode.resume_from_checkpoint", False))
    return_token_logprobs = bool(_get_cfg(cfg, "decode.return_token_logprobs", False))
    constrain_binary_output = bool(_get_cfg(cfg, "decode.constrain_binary_output", False))
    constrain_trailing_binary_output = bool(
        _get_cfg(cfg, "decode.constrain_trailing_binary_output", False)
    )
    trailing_binary_marker = str(_get_cfg(cfg, "decode.trailing_binary_marker", "정답:\n"))
    constrain_binary_choices_cfg = _get_cfg(cfg, "decode.constrain_binary_choices", ["0", "1"])
    if isinstance(constrain_binary_choices_cfg, (list, tuple)):
        constrain_binary_choices = [str(item) for item in constrain_binary_choices_cfg]
    else:
        constrain_binary_choices = [str(constrain_binary_choices_cfg)]
    if constrain_binary_output and method != "hf_decode":
        print("[warn] decode.constrain_binary_output is only supported for hf_decode right now.")
    if constrain_trailing_binary_output and method != "hf_decode":
        print("[warn] decode.constrain_trailing_binary_output is only supported for hf_decode right now.")
    if constrain_binary_output and constrain_trailing_binary_output:
        raise ValueError("Use only one of decode.constrain_binary_output or decode.constrain_trailing_binary_output")
    prediction_format = str(_get_cfg(cfg, "decode.prediction_format", "default")).strip().lower()
    if prediction_format not in {"default", "leading_binary", "trailing_binary"}:
        raise ValueError(
            "decode.prediction_format must be one of {'default', 'leading_binary', 'trailing_binary'}"
        )
    export_evidence_w_explanation_csv = bool(
        _get_cfg(cfg, "save.export_evidence_w_explanation_csv", False)
    )
    save_parsed_json_columns = bool(
        _get_cfg(cfg, "save.save_parsed_json_columns", False)
    )
    expand_parsed_json_fields = bool(
        _get_cfg(cfg, "save.expand_parsed_json_fields", False)
    )
    result_drop_columns_cfg = _get_cfg(cfg, "save.result_drop_columns", [])
    if isinstance(result_drop_columns_cfg, (str, bytes)):
        result_drop_columns = [str(result_drop_columns_cfg)]
    elif isinstance(result_drop_columns_cfg, (list, tuple)):
        result_drop_columns = [str(item) for item in result_drop_columns_cfg]
    else:
        result_drop_columns = []

    if args.result_path:
        os.makedirs(args.result_path, exist_ok=True)
        result_path = args.result_path
    elif args.config:
        result_path = os.path.dirname(os.path.abspath(args.config))
    else:
        result_path = "./outputs/results/causal_decode"

    # IO
    os.makedirs(result_path, exist_ok=True)
    out_file = os.path.join(
        result_path,
        f"results_{model_label}_{method}_{seed}.csv",
    )
    if os.path.exists(out_file):
        print(f"[Skip] exists: {out_file}")
        return
    progress_file = os.path.join(
        result_path,
        f"progress_{model_label}_{method}_{seed}.jsonl",
    )

    decode_model_for_vllm = model_name
    tmp_dir_ctx = None
    if adapter_path and method == "vllm_decode":
        tmp_dir_ctx = tempfile.TemporaryDirectory(prefix="merged_adapter_")
        decode_model_for_vllm = merge_adapter_to_dir(
            model_name=model_name,
            adapter_path=adapter_path,
            hf_token=hf_tok,
            torch_dtype=torch_dtype or torch.float16,
            save_dir=tmp_dir_ctx.name,
        )
    text_col = _get_cfg(cfg, "data.text_col", "text")
    label_col = _get_cfg(cfg, "data.label_col", "label")
    prompt_cfg = _get_cfg(cfg, "prompt", {}) or {}
    fmt = prompt_cfg.get("format", "plain")
    df = load_eval_csv(eval_set, text_col=text_col, label_col=label_col)
    df, texts, inputs = _prepare_prompt_inputs(df, prompt_cfg=prompt_cfg, text_col=text_col)

    score_adjust_cfg = _get_cfg(cfg, "decode.score_adjust", {}) or {}
    score_adjust_enabled = bool(score_adjust_cfg.get("enabled", False))
    score_adjust_method = str(score_adjust_cfg.get("method", "learned_logreg_v1")).strip().lower()
    score_adjust_source_csv = score_adjust_cfg.get("source_csv")
    score_adjust_text_col = str(score_adjust_cfg.get("text_col", text_col))
    score_adjust_label_col = str(score_adjust_cfg.get("label_col", label_col))
    score_adjust_strength = float(score_adjust_cfg.get("strength", 1.0))
    score_adjust_report_path = os.path.join(result_path, "score_adjust_report.json")
    step1_output_format = step1_output_format_cfg
    if step1_output_format == "auto":
        instruction_path_for_infer = str(prompt_cfg.get("instruction_path", ""))
        prompt_instruction_paths = df["prompt_instruction_path"].tolist() if "prompt_instruction_path" in df.columns else []
        if (not instruction_path_for_infer) and prompt_instruction_paths:
            instruction_path_for_infer = next((path for path in prompt_instruction_paths if path), "")
        step1_output_format = _infer_step1_output_format(instruction_path=instruction_path_for_infer)
    if step1_output_format not in STEP1_OUTPUT_FORMATS:
        raise ValueError(
            f"`decode.step1_output_format` must be one of {sorted(STEP1_OUTPUT_FORMATS)} or 'auto'. "
            f"got={step1_output_format_cfg!r}"
        )
    rows_records = df.to_dict(orient="records")

    if auto_max_input_tokens:
        prompt_length_report = resolve_auto_max_input_tokens(
            model_name=model_name,
            inputs=inputs,
            hf_token=hf_tok,
            max_new_tokens=max_new_tokens,
            margin=max_input_tokens_margin,
            quantile=max_input_tokens_quantile,
            measure_batch_size=max_input_tokens_measure_batch_size,
        )
        max_input_tokens = int(prompt_length_report["resolved_max_input_tokens"])
        prompt_length_report["applied_max_input_tokens"] = max_input_tokens
        prompt_length_report["measure_batch_size"] = max_input_tokens_measure_batch_size
        prompt_length_report["input_format"] = fmt
        prompt_length_report["num_inputs"] = len(inputs)
        prompt_length_report_path = os.path.join(result_path, "prompt_length_report.json")
        with open(prompt_length_report_path, "w", encoding="utf-8") as f:
            json.dump(prompt_length_report, f, ensure_ascii=False, indent=2)
        print(
            "[PromptBudget] "
            f"q={max_input_tokens_quantile:.2f} "
            f"resolved_max_input_tokens={max_input_tokens} "
            f"context_window={prompt_length_report.get('context_window')} "
            f"available_input_budget={prompt_length_report.get('available_input_budget')} "
            f"rows_over_resolved={prompt_length_report.get('rows_over_resolved')}/{len(inputs)}"
        )
        print("Saved:", prompt_length_report_path)

    score_adjust_payload: Optional[dict[str, Any]] = None
    if score_adjust_enabled:
        if score_adjust_method != "learned_logreg_v1":
            raise ValueError(
                "decode.score_adjust.method must be 'learned_logreg_v1' right now. "
                f"got={score_adjust_method!r}"
            )
        if pred_mode == "label_col":
            raise ValueError("decode.score_adjust is incompatible with decode.pred_mode=label_col")
        if prediction_format != "leading_binary":
            raise ValueError("decode.score_adjust currently requires decode.prediction_format=leading_binary")
        if not score_adjust_source_csv:
            raise ValueError("decode.score_adjust.enabled=true requires decode.score_adjust.source_csv")

        score_source_df = load_eval_csv(
            str(score_adjust_source_csv),
            text_col=score_adjust_text_col,
            label_col=score_adjust_label_col,
        )
        score_source_df, score_source_texts, score_source_inputs = _prepare_prompt_inputs(
            score_source_df,
            prompt_cfg=prompt_cfg,
            text_col=score_adjust_text_col,
        )
        grouped_scores = _score_binary_choice_groups(
            model_name=model_name,
            hf_token=hf_tok,
            torch_dtype=torch_dtype or torch.float16,
            adapter_path=adapter_path,
            merge_adapter=merge_adapter,
            grouped_inputs={
                "source": score_source_inputs,
                "eval": inputs,
            },
            batch_size=hf_batch_size,
            max_input_tokens=max_input_tokens,
        )
        score_source_labels = score_source_df["label"].fillna(0).astype(int).to_numpy(dtype=np.int64)
        score_fit = _fit_learned_binary_score_adjuster(
            texts=[str(v) for v in score_source_texts],
            labels=score_source_labels,
            prob1=grouped_scores["source"]["prob1"],
            margin=grouped_scores["source"]["margin"],
            strength=score_adjust_strength,
        )
        score_apply = _apply_learned_binary_score_adjuster(
            score_pipeline=score_fit["pipeline"],
            texts=[str(v) for v in texts],
            prob1=grouped_scores["eval"]["prob1"],
            margin=grouped_scores["eval"]["margin"],
            threshold=float(score_fit["threshold"]),
            strength=score_adjust_strength,
        )
        score_adjust_payload = {
            "source_csv": str(score_adjust_source_csv),
            "threshold": float(score_fit["threshold"]),
            "strength": float(score_fit["strength"]),
            "base_prob1": grouped_scores["eval"]["prob1"],
            "base_margin": grouped_scores["eval"]["margin"],
            "adjusted_prob1": score_apply["prob"],
            "adjusted_pred": score_apply["pred"],
            "base_score_pred": (grouped_scores["eval"]["prob1"] >= 0.5).astype(np.int64),
            "eval_aux": score_apply["aux"],
        }
        score_adjust_report = {
            "method": score_adjust_method,
            "source_csv": str(score_adjust_source_csv),
            "threshold": float(score_fit["threshold"]),
            "strength": float(score_fit["strength"]),
            "feature_names": score_fit["feature_names"],
            "coef_by_feature": score_fit["coef_by_feature"],
            "intercept": float(score_fit["intercept"]),
            "source_base_metrics": score_fit["baseline_metrics"],
            "source_adjusted_metrics": score_fit["train_metrics"],
        }
        if "label" in df.columns:
            eval_labels = df["label"].fillna(0).astype(int).to_numpy(dtype=np.int64)
            score_adjust_report["eval_base_metrics"] = compute_binary_metrics(
                eval_labels.tolist(),
                score_adjust_payload["base_score_pred"].astype(int).tolist(),
            )
            score_adjust_report["eval_adjusted_metrics"] = compute_binary_metrics(
                eval_labels.tolist(),
                score_adjust_payload["adjusted_pred"].astype(int).tolist(),
            )
        with open(score_adjust_report_path, "w", encoding="utf-8") as f:
            json.dump(score_adjust_report, f, ensure_ascii=False, indent=2)
        df = df.copy()
        df["base_prob1"] = [float(v) for v in score_adjust_payload["base_prob1"].tolist()]
        df["base_margin"] = [float(v) for v in score_adjust_payload["base_margin"].tolist()]
        df["base_score_pred"] = [int(v) for v in score_adjust_payload["base_score_pred"].tolist()]
        df["adjusted_prob1"] = [float(v) for v in score_adjust_payload["adjusted_prob1"].tolist()]
        df["adjusted_pred"] = [int(v) for v in score_adjust_payload["adjusted_pred"].tolist()]
        df["adjust_threshold"] = float(score_adjust_payload["threshold"])
        df["adjust_strength"] = float(score_adjust_payload["strength"])
        df["adjust_risk_count"] = [
            float(v) for v in score_adjust_payload["eval_aux"]["risk_count"].tolist()
        ]
        df["adjust_safe_count"] = [
            float(v) for v in score_adjust_payload["eval_aux"]["safe_count"].tolist()
        ]

    preds = gens = None
    gen_tokens = gen_token_logprobs = None
    checkpoint_enabled = bool(incremental_save or resume_from_checkpoint)
    restored: dict[int, tuple[int, str, Optional[list[str]], Optional[list[Optional[float]]]]] = {}
    if checkpoint_enabled:
        restored = _load_decode_progress(progress_file, total_n=len(inputs))
        if restored:
            print(
                f"[Resume] loaded {len(restored)} decoded samples from: {progress_file}"
            )

    try:
        if method == "vllm_decode":
            try:
                if return_token_logprobs:
                    preds, gens, gen_tokens, gen_token_logprobs = run_with_vllm(
                        model_name=decode_model_for_vllm,
                        inputs=inputs,
                        seed=seed,
                        n_devices=n_devices,
                        tensor_parallel_size=tensor_parallel_size,
                        pipeline_parallel_size=pipeline_parallel_size,
                        max_new_tokens=max_new_tokens,
                        dtype=_get_cfg(cfg, "model.dtype", "half"),
                        temperature=decode_temperature,
                        top_p=decode_top_p,
                        presence_penalty=decode_presence_penalty,
                        frequency_penalty=decode_frequency_penalty,
                        repetition_penalty=decode_repetition_penalty,
                        truncate_prompt_tokens=max_input_tokens,
                        return_token_logprobs=True,
                    )
                else:
                    preds, gens = run_with_vllm(
                        model_name=decode_model_for_vllm,
                        inputs=inputs,
                        seed=seed,
                        n_devices=n_devices,
                        tensor_parallel_size=tensor_parallel_size,
                        pipeline_parallel_size=pipeline_parallel_size,
                        max_new_tokens=max_new_tokens,
                        dtype=_get_cfg(cfg, "model.dtype", "half"),
                        temperature=decode_temperature,
                        top_p=decode_top_p,
                        presence_penalty=decode_presence_penalty,
                        frequency_penalty=decode_frequency_penalty,
                        repetition_penalty=decode_repetition_penalty,
                        truncate_prompt_tokens=max_input_tokens,
                    )
            except Exception as e:
                if allow_hf_fallback:
                    print(f"[vLLM FAILED] fallback HF. Reason: {repr(e)}")
                else:
                    raise RuntimeError(
                        f"[vLLM FAILED] fallback disabled by decode.allow_hf_fallback=false. "
                        f"reason={repr(e)}"
                    ) from e

        if preds is None or gens is None:
            if checkpoint_enabled:
                total_n = len(inputs)
                ordered_preds: list[Optional[int]] = [None] * total_n
                ordered_gens: list[Optional[str]] = [None] * total_n
                ordered_gen_tokens: Optional[list[Optional[list[str]]]] = (
                    [None] * total_n if return_token_logprobs else None
                )
                ordered_gen_token_logprobs: Optional[list[Optional[list[Optional[float]]]]] = (
                    [None] * total_n if return_token_logprobs else None
                )
                for idx, (pred_i, gen_i, gen_tokens_i, gen_token_logprobs_i) in restored.items():
                    ordered_preds[idx] = int(pred_i)
                    restored_gen = str(gen_i)
                    if step1_output_format == "pipe":
                        row_for_idx = rows_records[idx] if idx < len(rows_records) else {}
                        restored_gen = _build_step1_pipe_wrapped_gen(restored_gen, row_for_idx)
                    ordered_gens[idx] = restored_gen
                    if return_token_logprobs:
                        if ordered_gen_tokens is not None:
                            ordered_gen_tokens[idx] = (
                                [str(token) for token in gen_tokens_i]
                                if isinstance(gen_tokens_i, list)
                                else None
                            )
                        if ordered_gen_token_logprobs is not None:
                            ordered_gen_token_logprobs[idx] = (
                                list(gen_token_logprobs_i)
                                if isinstance(gen_token_logprobs_i, list)
                                else None
                            )
                pending_indices = [i for i in range(total_n) if ordered_preds[i] is None]
                pending_inputs = [inputs[i] for i in pending_indices]
                print(
                    f"[Resume] pending samples: {len(pending_inputs)}/{total_n} "
                    f"(progress file: {progress_file})"
                )

                def _on_sample(
                    local_idx: int,
                    pred_i: int,
                    gen_i: str,
                    sample_tokens: Optional[list[str]] = None,
                    sample_logprobs: Optional[list[Optional[float]]] = None,
                ) -> None:
                    global_idx = pending_indices[local_idx]
                    stored_gen = str(gen_i)
                    if step1_output_format == "pipe":
                        row_for_idx = rows_records[global_idx] if global_idx < len(rows_records) else {}
                        stored_gen = _build_step1_pipe_wrapped_gen(stored_gen, row_for_idx)
                    ordered_preds[global_idx] = int(pred_i)
                    ordered_gens[global_idx] = stored_gen
                    normalized_tokens: Optional[list[str]] = None
                    normalized_logprobs: Optional[list[Optional[float]]] = None
                    if return_token_logprobs:
                        if isinstance(sample_tokens, list):
                            normalized_tokens = [str(token) for token in sample_tokens]
                        if isinstance(sample_logprobs, list):
                            normalized_logprobs = []
                            for value in sample_logprobs:
                                if value is None:
                                    normalized_logprobs.append(None)
                                    continue
                                try:
                                    normalized_logprobs.append(float(value))
                                except Exception:
                                    normalized_logprobs.append(None)
                        if ordered_gen_tokens is not None:
                            ordered_gen_tokens[global_idx] = normalized_tokens
                        if ordered_gen_token_logprobs is not None:
                            ordered_gen_token_logprobs[global_idx] = normalized_logprobs
                    _append_decode_progress(
                        progress_file,
                        global_idx,
                        int(pred_i),
                        stored_gen,
                        gen_tokens=normalized_tokens,
                        gen_token_logprobs=normalized_logprobs,
                    )

                if pending_inputs:
                    if return_token_logprobs:
                        _preds_new, _gens_new, _tokens_new, _logprobs_new = run_with_hf_generate(
                            model_name=model_name,
                            inputs=pending_inputs,
                            hf_token=hf_tok,
                            batch_size=hf_batch_size,
                            max_input_tokens=max_input_tokens,
                            max_new_tokens=max_new_tokens,
                            torch_dtype=torch_dtype or torch.float16,
                            adapter_path=adapter_path,
                            merge_adapter=merge_adapter,
                            on_sample=_on_sample,
                            sample_offset=0,
                            temperature=decode_temperature,
                            top_p=decode_top_p,
                            presence_penalty=decode_presence_penalty,
                            frequency_penalty=decode_frequency_penalty,
                            repetition_penalty=decode_repetition_penalty,
                            return_token_logprobs=True,
                            constrain_first_token_choices=(
                                constrain_binary_choices if constrain_binary_output else None
                            ),
                            constrain_trailing_binary_choices=(
                                constrain_binary_choices if constrain_trailing_binary_output else None
                            ),
                            trailing_binary_marker=trailing_binary_marker,
                        )
                        if ordered_gen_tokens is not None:
                            for local_idx, global_idx in enumerate(pending_indices):
                                if ordered_gen_tokens[global_idx] is None and local_idx < len(_tokens_new):
                                    tokens_i = _tokens_new[local_idx]
                                    if isinstance(tokens_i, list):
                                        ordered_gen_tokens[global_idx] = [str(token) for token in tokens_i]
                        if ordered_gen_token_logprobs is not None:
                            for local_idx, global_idx in enumerate(pending_indices):
                                if (
                                    ordered_gen_token_logprobs[global_idx] is None
                                    and local_idx < len(_logprobs_new)
                                ):
                                    logprobs_i = _logprobs_new[local_idx]
                                    if isinstance(logprobs_i, list):
                                        normalized = []
                                        for value in logprobs_i:
                                            if value is None:
                                                normalized.append(None)
                                                continue
                                            try:
                                                normalized.append(float(value))
                                            except Exception:
                                                normalized.append(None)
                                        ordered_gen_token_logprobs[global_idx] = normalized
                    else:
                        _preds_new, _gens_new = run_with_hf_generate(
                            model_name=model_name,
                            inputs=pending_inputs,
                            hf_token=hf_tok,
                            batch_size=hf_batch_size,
                            max_input_tokens=max_input_tokens,
                            max_new_tokens=max_new_tokens,
                            torch_dtype=torch_dtype or torch.float16,
                            adapter_path=adapter_path,
                            merge_adapter=merge_adapter,
                            on_sample=_on_sample,
                            sample_offset=0,
                            temperature=decode_temperature,
                            top_p=decode_top_p,
                            presence_penalty=decode_presence_penalty,
                            frequency_penalty=decode_frequency_penalty,
                            repetition_penalty=decode_repetition_penalty,
                            constrain_first_token_choices=(
                                constrain_binary_choices if constrain_binary_output else None
                            ),
                            constrain_trailing_binary_choices=(
                                constrain_binary_choices if constrain_trailing_binary_output else None
                            ),
                            trailing_binary_marker=trailing_binary_marker,
                        )

                missing_indices = [
                    i
                    for i, (pred_i, gen_i) in enumerate(zip(ordered_preds, ordered_gens))
                    if pred_i is None or gen_i is None
                ]
                if missing_indices:
                    raise RuntimeError(
                        f"Decoding incomplete. Missing outputs for indices: "
                        f"{missing_indices[:20]}"
                    )

                preds = [int(v) for v in ordered_preds]
                gens = [str(v) for v in ordered_gens]
                if return_token_logprobs:
                    if ordered_gen_tokens is None:
                        gen_tokens = [None] * len(gens)
                    else:
                        gen_tokens = list(ordered_gen_tokens)
                    if ordered_gen_token_logprobs is None:
                        gen_token_logprobs = [None] * len(gens)
                    else:
                        gen_token_logprobs = list(ordered_gen_token_logprobs)
            else:
                if return_token_logprobs:
                    preds, gens, gen_tokens, gen_token_logprobs = run_with_hf_generate(
                        model_name=model_name,
                        inputs=inputs,
                        hf_token=hf_tok,
                        batch_size=hf_batch_size,
                        max_input_tokens=max_input_tokens,
                        max_new_tokens=max_new_tokens,
                        torch_dtype=torch_dtype or torch.float16,
                        adapter_path=adapter_path,
                        merge_adapter=merge_adapter,
                        temperature=decode_temperature,
                        top_p=decode_top_p,
                        presence_penalty=decode_presence_penalty,
                        frequency_penalty=decode_frequency_penalty,
                        repetition_penalty=decode_repetition_penalty,
                        return_token_logprobs=True,
                        constrain_first_token_choices=(
                            constrain_binary_choices if constrain_binary_output else None
                        ),
                        constrain_trailing_binary_choices=(
                            constrain_binary_choices if constrain_trailing_binary_output else None
                        ),
                        trailing_binary_marker=trailing_binary_marker,
                    )
                else:
                    preds, gens = run_with_hf_generate(
                        model_name=model_name,
                        inputs=inputs,
                        hf_token=hf_tok,
                        batch_size=hf_batch_size,
                        max_input_tokens=max_input_tokens,
                        max_new_tokens=max_new_tokens,
                        torch_dtype=torch_dtype or torch.float16,
                        adapter_path=adapter_path,
                        merge_adapter=merge_adapter,
                        temperature=decode_temperature,
                        top_p=decode_top_p,
                        presence_penalty=decode_presence_penalty,
                        frequency_penalty=decode_frequency_penalty,
                        repetition_penalty=decode_repetition_penalty,
                        constrain_first_token_choices=(
                            constrain_binary_choices if constrain_binary_output else None
                        ),
                        constrain_trailing_binary_choices=(
                            constrain_binary_choices if constrain_trailing_binary_output else None
                        ),
                        trailing_binary_marker=trailing_binary_marker,
                    )
                if step1_output_format == "pipe":
                    gens = _normalize_step1_pipe_gens(df=df, gens=[str(g) for g in gens])
    finally:
        if tmp_dir_ctx is not None:
            tmp_dir_ctx.cleanup()
    if step1_output_format == "pipe":
        gens = _normalize_step1_pipe_gens(df=df, gens=[str(g) for g in gens])

    invalid_parse_count = 0
    if prediction_format in {"leading_binary", "trailing_binary"} and pred_mode != "label_col" and label_col in df.columns:
        y_true_for_parse = df[label_col].fillna(0).astype(int).tolist()
        parsed_preds = []
        for gen_text, y_true_i in zip(gens, y_true_for_parse):
            try:
                if prediction_format == "leading_binary":
                    parsed_preds.append(int(parse_leading_binary_token(gen_text, strict=True)))
                else:
                    parsed_preds.append(int(parse_trailing_binary_token(gen_text, strict=True)))
            except Exception:
                parsed_preds.append(1 - int(y_true_i))
                invalid_parse_count += 1
        preds = parsed_preds
        if invalid_parse_count:
            print(
                f"[{prediction_format}-parse] invalid_or_nonbinary={invalid_parse_count}/{len(gens)} "
                "-> forced incorrect"
            )

    if pred_mode == "label_col":
        preds = df[label_col].fillna(0).astype(int).tolist()
    if score_adjust_payload is not None:
        df = df.copy()
        df["gen_pred"] = [int(p) for p in preds]
        preds = [int(v) for v in score_adjust_payload["adjusted_pred"].tolist()]
    if validate_step1:
        issues = _validate_step1_outputs(df=df, gens=gens, output_format=step1_output_format)
        report_path = os.path.join(result_path, "step1_validation_report.json")
        report_payload = {
            "format": step1_output_format,
            "total_rows": len(df),
            "invalid_rows": len(issues),
            "valid_rows": len(df) - len(issues),
            "issues": issues,
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_payload, f, ensure_ascii=False, indent=2)
        print("Saved:", report_path)
        if issues and fail_on_invalid_step1:
            preview = issues[:3]
            raise ValueError(
                f"Step1 {step1_output_format.upper()} validation failed: invalid_rows={len(issues)}/{len(df)}. "
                f"examples={preview}"
            )
    metrics_path = os.path.join(result_path, "metrics.json")
    if compute_metrics:
        y_true = df[label_col].fillna(0).astype(int).tolist()
        y_pred = [int(p) for p in preds]

        metrics = compute_binary_metrics(y_true, y_pred)
        by_category = {}
        if "category" in df.columns:
            for cat, g in df.groupby("category"):
                yt = g[label_col].fillna(0).astype(int).tolist()
                yp = [int(preds[i]) for i in g.index]
                by_category[str(cat)] = compute_binary_metrics(yt, yp)

        by_subset = {}
        if "subset_label" in df.columns:
            for sub, g in df.groupby("subset_label"):
                yt = g[label_col].fillna(0).astype(int).tolist()
                yp = [int(preds[i]) for i in g.index]
                by_subset[str(sub)] = compute_binary_metrics(yt, yp)

        by_length_sml = {}
        if "length_sml" in df.columns:
            for length_sml, g in df.groupby("length_sml"):
                yt = g[label_col].fillna(0).astype(int).tolist()
                yp = [int(preds[i]) for i in g.index]
                by_length_sml[str(length_sml)] = compute_binary_metrics(yt, yp)

        metrics["by_category"] = by_category
        metrics["by_subset"] = by_subset
        metrics["by_length_sml"] = by_length_sml
        metrics["invalid_parse_count"] = int(invalid_parse_count)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

    save_results_csv(
        df=df,
        preds=preds,
        gens=gens,
        out_file=out_file,
        gen_tokens=gen_tokens if return_token_logprobs else None,
        gen_token_logprobs=gen_token_logprobs if return_token_logprobs else None,
        save_parsed_json_columns=save_parsed_json_columns,
        expand_parsed_json_fields=expand_parsed_json_fields,
        drop_columns=result_drop_columns,
    )
    if export_evidence_w_explanation_csv:
        evidence_w_explanation_out_file = os.path.join(
            result_path,
            f"evidence_w_explanation_{model_label}_{method}_{seed}.csv",
        )
        save_evidence_w_explanation_csv(df=df, gens=gens, out_file=evidence_w_explanation_out_file)
    if compute_metrics:
        print("Saved:", metrics_path)
    else:
        print("[Skip] metrics disabled by decode.compute_metrics=false")
    if score_adjust_payload is not None:
        print("Saved:", score_adjust_report_path)
    print("Saved:", out_file)
    if export_evidence_w_explanation_csv:
        print("Saved:", evidence_w_explanation_out_file)


if __name__ == "__main__":
    main()
