"""Version-bound runtime tokenizer behavior; no fitted token-count offsets or tolerances."""
from __future__ import annotations


def runtime_adds_bos(model_info: dict, runtime_version: str, rules: dict) -> bool:
    if rules.get("runtime") != "ollama" or runtime_version != rules["version"]:
        raise ValueError("unverified runtime version for tokenizer contract")
    if (model_info.get("tokenizer.ggml.pre") in rules["forced_bos_pre"]
            or model_info.get("tokenizer.ggml.model") in rules["forced_bos_model"]):
        return True
    value = model_info.get(rules["fallback"])
    if type(value) is not bool:
        raise ValueError("explicit BOS metadata required for this tokenizer")
    return value
