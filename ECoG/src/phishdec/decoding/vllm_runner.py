import inspect
import os
from typing import Any




os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

_VLLM_IMPORT_ERR = None
try:  
    from vllm import LLM, SamplingParams
except Exception as e:  
    LLM = None
    SamplingParams = None
    _VLLM_IMPORT_ERR = e
from tqdm import tqdm
from .parser import parse_pred


os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
try:  
    import torch._dynamo as dynamo
    dynamo.config.suppress_errors = True
except Exception:
    pass


def _to_float_or_none(value):
    try:
        return float(value)
    except Exception:
        return None


def _extract_vllm_selected_logprobs(token_ids, step_logprobs):
    out = []
    for idx, token_id in enumerate(token_ids):
        selected = None
        step = step_logprobs[idx] if idx < len(step_logprobs) else None
        if isinstance(step, dict):
            selected = step.get(token_id)
            if selected is None:
                selected = step.get(str(token_id))
            if selected is None:
                for key, value in step.items():
                    try:
                        if int(key) == int(token_id):
                            selected = value
                            break
                    except Exception:
                        continue
        elif isinstance(step, (list, tuple)):
            for item in step:
                if getattr(item, "token_id", None) == int(token_id):
                    selected = item
                    break

        if selected is None:
            out.append(None)
            continue
        if hasattr(selected, "logprob"):
            out.append(_to_float_or_none(getattr(selected, "logprob")))
            continue
        out.append(_to_float_or_none(selected))
    return out


def _supports_kwarg(callable_obj, kwarg_name: str) -> bool:
    try:
        sig = inspect.signature(callable_obj)
    except Exception:
        return False
    params = sig.parameters
    if kwarg_name in params:
        return True
    return any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values())


def _resolve_max_model_len(
    max_model_len: int | None,
    truncate_prompt_tokens: int | None,
    max_new_tokens: int,
) -> int | None:
    if max_model_len is not None:
        return max(1024, int(max_model_len))
    if truncate_prompt_tokens is None:
        return None
    
    return max(1024, int(truncate_prompt_tokens) + int(max_new_tokens) + 128)


def _get_llm_tokenizer(llm: Any):
    getter = getattr(llm, "get_tokenizer", None)
    if callable(getter):
        try:
            tok = getter()
            if tok is not None:
                return tok
        except Exception:
            pass

    candidates = [
        "llm_engine.tokenizer.tokenizer",
        "llm_engine.tokenizer",
        "tokenizer",
    ]
    for path in candidates:
        cur = llm
        ok = True
        for part in path.split("."):
            if not hasattr(cur, part):
                ok = False
                break
            cur = getattr(cur, part)
        if ok and cur is not None:
            return cur
    return None


def _encode_text(tokenizer, text: str) -> list[int]:
    try:
        return list(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return list(tokenizer.encode(text))


def _decode_ids(tokenizer, token_ids: list[int]) -> str:
    try:
        return str(tokenizer.decode(token_ids, skip_special_tokens=False))
    except TypeError:
        return str(tokenizer.decode(token_ids))


def _truncate_prompts_by_tokens(
    prompts: list[str],
    tokenizer,
    max_input_tokens: int,
) -> tuple[list[str], int]:
    if max_input_tokens is None or max_input_tokens < 1:
        return prompts, 0
    out: list[str] = []
    truncated = 0
    limit = int(max_input_tokens)
    for prompt in prompts:
        token_ids = _encode_text(tokenizer, prompt)
        if len(token_ids) <= limit:
            out.append(prompt)
            continue
        new_prompt = _decode_ids(tokenizer, token_ids[:limit])
        out.append(new_prompt)
        truncated += 1
    return out, truncated


def _patch_transformers_tokenizer_compat():
    try:
        from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer
    except Exception:
        return

    def _special_tokens_map_extended(self):
        return {
            key: value
            for key, value in (getattr(self, "_special_tokens_map", {}) or {}).items()
            if value is not None
        }

    def _all_special_tokens_extended(self):
        special_map = _special_tokens_map_extended(self)
        ordered_keys = list(getattr(self, "SPECIAL_TOKENS_ATTRIBUTES", []) or [])
        ordered_keys.extend(key for key in special_map if key not in ordered_keys)

        values = []
        seen = set()
        for key in ordered_keys:
            value = special_map.get(key)
            items = value if isinstance(value, (list, tuple)) else [value]
            for item in items:
                if item is None:
                    continue
                marker = getattr(item, "content", str(item))
                if marker in seen:
                    continue
                seen.add(marker)
                values.append(item)
        return values

    if not hasattr(GPT2Tokenizer, "special_tokens_map_extended"):
        GPT2Tokenizer.special_tokens_map_extended = property(_special_tokens_map_extended)
    if not hasattr(GPT2Tokenizer, "all_special_tokens_extended"):
        GPT2Tokenizer.all_special_tokens_extended = property(_all_special_tokens_extended)
    if not hasattr(GPT2Tokenizer, "additional_special_tokens"):
        GPT2Tokenizer.additional_special_tokens = property(
            lambda self: list((_special_tokens_map_extended(self).get("additional_special_tokens") or []))
        )


def _patch_vllm_tqdm_compat():
    try:
        from vllm.model_executor.model_loader import weight_utils
    except Exception:
        return

    if getattr(weight_utils.DisabledTqdm, "_phishdec_compat_patched", False):
        return

    base_tqdm = weight_utils.tqdm

    class CompatibleDisabledTqdm(base_tqdm):
        _phishdec_compat_patched = True

        def __init__(self, *args, **kwargs):
            kwargs.setdefault("disable", True)
            super().__init__(*args, **kwargs)

    weight_utils.DisabledTqdm = CompatibleDisabledTqdm


def _resolve_vllm_model_source(model_name: str) -> str:
    if os.path.exists(model_name):
        return model_name

    try:
        from huggingface_hub import snapshot_download
    except Exception:
        return model_name

    hf_token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("hf_token")
    )
    return snapshot_download(
        repo_id=model_name,
        token=hf_token,
        resume_download=True,
    )


def run_with_vllm(
    model_name,
    inputs,
    seed,
    n_devices,
    tensor_parallel_size=None,
    pipeline_parallel_size=1,
    max_new_tokens=8,
    dtype="half",
    temperature=0.0,
    top_p=1.0,
    presence_penalty=0.0,
    frequency_penalty=0.0,
    repetition_penalty=1.0,
    truncate_prompt_tokens=None,
    max_model_len=None,
    batch_size=None,
    on_chunk=None,
    return_token_logprobs=False,
):
    if LLM is None or SamplingParams is None:
        raise ImportError(
            "vLLM backend requested but `vllm` is not installed. "
            "Install vllm or run with `--method hf_decode` / set `model.backend: hf`."
        ) from _VLLM_IMPORT_ERR
    _patch_transformers_tokenizer_compat()
    _patch_vllm_tqdm_compat()
    resolved_model_name = _resolve_vllm_model_source(model_name)
    effective_tp = int(tensor_parallel_size) if tensor_parallel_size is not None else int(n_devices)
    effective_pp = int(pipeline_parallel_size)
    resolved_max_model_len = _resolve_max_model_len(
        max_model_len=max_model_len,
        truncate_prompt_tokens=truncate_prompt_tokens,
        max_new_tokens=max_new_tokens,
    )
    llm_kwargs = {
        "model": resolved_model_name,
        "tokenizer": resolved_model_name,
        "tensor_parallel_size": effective_tp,
        "pipeline_parallel_size": effective_pp,
        "seed": seed,
        "dtype": dtype,
        "enforce_eager": True,
        "trust_remote_code": True,
    }
    if (
        resolved_max_model_len is not None
        and _supports_kwarg(LLM.__init__, "max_model_len")
    ):
        llm_kwargs["max_model_len"] = int(resolved_max_model_len)
    llm = LLM(**llm_kwargs)
    llm_tokenizer = _get_llm_tokenizer(llm)

    
    
    manual_prompt_cap = None
    if truncate_prompt_tokens is not None:
        manual_prompt_cap = int(truncate_prompt_tokens)
    elif resolved_max_model_len is not None:
        manual_prompt_cap = max(1, int(resolved_max_model_len) - int(max_new_tokens) - 1)

    if llm_tokenizer is not None and manual_prompt_cap is not None:
        new_inputs, truncated_n = _truncate_prompts_by_tokens(
            prompts=list(inputs),
            tokenizer=llm_tokenizer,
            max_input_tokens=manual_prompt_cap,
        )
        inputs = new_inputs
        if truncated_n > 0:
            print(
                f"[warn] Manually truncated {truncated_n}/{len(inputs)} prompts "
                f"to <= {manual_prompt_cap} tokens for vLLM compatibility."
            )

    sampling_kwargs = {
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "repetition_penalty": repetition_penalty,
        "logprobs": 1 if return_token_logprobs else None,
    }
    if truncate_prompt_tokens is not None and _supports_kwarg(
        SamplingParams.__init__, "truncate_prompt_tokens"
    ):
        sampling_kwargs["truncate_prompt_tokens"] = int(truncate_prompt_tokens)
    elif truncate_prompt_tokens is not None:
        print(
            "[warn] This vLLM version does not support `truncate_prompt_tokens`; "
            "using `max_model_len` guard instead."
        )

    try:
        sampling_params = SamplingParams(**sampling_kwargs)
    except TypeError as exc:
        
        
        if (
            "truncate_prompt_tokens" in sampling_kwargs
            and "truncate_prompt_tokens" in str(exc)
        ):
            sampling_kwargs.pop("truncate_prompt_tokens", None)
            print(
                "[warn] vLLM rejected `truncate_prompt_tokens`; retrying without it "
                "(using max_model_len guard)."
            )
            sampling_params = SamplingParams(**sampling_kwargs)
        else:
            raise

    preds, gens = [], []
    tokens_all, token_logprobs_all = [], []
    step = len(inputs) if batch_size is None else max(1, int(batch_size))
    for start in tqdm(range(0, len(inputs), step), desc="vLLM Decoding"):
        chunk_inputs = inputs[start : start + step]
        try:
            chunk_outputs = llm.generate(chunk_inputs, sampling_params)
        except Exception as exc:
            
            
            msg = str(exc)
            if (
                llm_tokenizer is not None
                and resolved_max_model_len is not None
                and ("maximum context length" in msg or "VLLMValidationError" in type(exc).__name__)
            ):
                retry_cap = max(1, int(resolved_max_model_len) - int(max_new_tokens) - 1)
                tightened_inputs, tightened_n = _truncate_prompts_by_tokens(
                    prompts=list(chunk_inputs),
                    tokenizer=llm_tokenizer,
                    max_input_tokens=retry_cap,
                )
                if tightened_n > 0:
                    print(
                        f"[warn] Retrying chunk with stricter prompt cap "
                        f"({retry_cap} tokens), truncated {tightened_n}/{len(chunk_inputs)} prompts."
                    )
                chunk_outputs = llm.generate(tightened_inputs, sampling_params)
            else:
                raise
        chunk_preds, chunk_gens = [], []
        chunk_tokens_all, chunk_token_logprobs_all = [], []
        for out in chunk_outputs:
            first = out.outputs[0]
            gen_text = first.text
            parsed = parse_pred(gen_text, default=0)
            chunk_gens.append(gen_text)
            chunk_preds.append(parsed)
            preds.append(parsed)
            gens.append(gen_text)
            if return_token_logprobs:
                token_ids = [int(tid) for tid in (getattr(first, "token_ids", []) or [])]
                tokens = getattr(first, "tokens", None)
                if not isinstance(tokens, list) or len(tokens) != len(token_ids):
                    tokens = [str(tid) for tid in token_ids]
                step_logprobs = list(getattr(first, "logprobs", []) or [])
                token_logprobs = _extract_vllm_selected_logprobs(token_ids, step_logprobs)
                chunk_tokens_all.append(tokens)
                chunk_token_logprobs_all.append(token_logprobs)
                tokens_all.append(tokens)
                token_logprobs_all.append(token_logprobs)
        if callable(on_chunk):
            on_chunk(
                start,
                chunk_preds,
                chunk_gens,
                chunk_tokens_all if return_token_logprobs else None,
                chunk_token_logprobs_all if return_token_logprobs else None,
            )
    if return_token_logprobs:
        return preds, gens, tokens_all, token_logprobs_all
    return preds, gens
