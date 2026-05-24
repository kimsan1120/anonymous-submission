import json
from typing import Any, Optional


def _load(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def get_decoder_model_name(model_id: int, path: str = "models_args.json"):
    return _load(path)["Decoder"][model_id]


def get_encoder_model_name(model_id: int, path: str = "models_args.json"):
    return _load(path)["Encoder"][model_id]


def get_finetuned_model_name(model_id: int, path: str = "models_args.json"):
    return _load(path)["Finetuned"][model_id]


def get_llm_model_name(model_id: int, path: str = "models_args.json"):
    return _load(path)["LLM"][model_id]


def get_model_name(model_type: str, model_id: int, path: str = "models_args.json") -> str:
    data = _load(path)
    key = model_type or "Decoder"
    key_norm = key.strip()
    if key_norm in data:
        return data[key_norm][model_id]
    
    return data["Decoder"][model_id]


def resolve_model_name(
    model_type: str,
    model_name: Optional[str] = None,
    model_id: Optional[int] = None,
    path: str = "models_args.json",
) -> str:
    """
    Resolve a model name from either:
    1) an explicit model_name (preferred), or
    2) a legacy model_id lookup via models_args.json.
    """
    if model_name:
        return str(model_name)
    if model_id is None:
        raise ValueError("Either `model_name` or `model_id` must be provided.")
    return get_model_name(model_type, int(model_id), path=path)
