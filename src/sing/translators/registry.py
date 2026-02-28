"""Translator registry helpers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from sing.translators.loader import LoadedTranslator, load_translator


def _require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"Registry field '{field_name}' must be a string, got {type(value).__name__}")
    parsed = value.strip()
    if not parsed:
        raise ValueError(f"Registry field '{field_name}' must be a non-empty string")
    return parsed


def _normalize_defaults(defaults: Mapping[str, Any]) -> dict[str, str]:
    normalized_unsorted: dict[str, str] = {}
    source_key_by_normalized: dict[str, str] = {}
    for model_name, translator in defaults.items():
        model_key = _require_non_empty_string(model_name, "defaults.<model_name>").lower()
        if model_key in source_key_by_normalized:
            raise ValueError(
                "Registry defaults contain ambiguous model keys "
                f"'{source_key_by_normalized[model_key]}' and '{model_name}'"
            )
        translator_name = _require_non_empty_string(translator, f"defaults.{model_name}")
        source_key_by_normalized[model_key] = str(model_name)
        normalized_unsorted[model_key] = translator_name
    return dict(sorted(normalized_unsorted.items(), key=lambda item: item[0]))


def read_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Translator registry not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        registry = yaml.safe_load(handle) or {}

    if not isinstance(registry, Mapping):
        raise TypeError(f"Translator registry must be a mapping: {path}")
    extra_top_keys = sorted(set(registry.keys()) - {"defaults"})
    if extra_top_keys:
        raise KeyError(f"Unexpected top-level keys in translator registry: {extra_top_keys}")

    defaults = registry.get("defaults", {})
    if not isinstance(defaults, Mapping):
        raise TypeError("translator registry 'defaults' must be a mapping of model -> translator")
    return {"defaults": _normalize_defaults(defaults)}


def load_default_translator(
    translators_root: Path,
    registry_path: Path,
    model_name: str,
    device,
) -> LoadedTranslator:
    registry = read_registry(registry_path)
    normalized_model_name = _require_non_empty_string(model_name, "model_name").lower()
    default_name = registry["defaults"].get(normalized_model_name)
    if default_name is None:
        raise KeyError(f"No default translator configured for model '{model_name}'")
    return load_translator(
        translators_root=translators_root,
        model_name=normalized_model_name,
        translator_name=str(default_name),
        device=device,
    )
