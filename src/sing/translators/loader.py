"""Translator loading from dedicated repository directory."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from sing.translators.architectures import build_translator, supported_architectures
from sing.utils import get_logger


@dataclass(frozen=True)
class TranslatorMetadata:
    model_name: str
    translator_name: str
    architecture: str
    embedding_backend: str
    in_dim: int
    out_dim: int
    hidden_dim: int
    checkpoint_file: str


@dataclass(frozen=True)
class LoadedTranslator:
    metadata: TranslatorMetadata
    model: torch.nn.Module


REQUIRED_KEYS = {
    "model_name",
    "translator_name",
    "architecture",
    "embedding_backend",
    "in_dim",
    "out_dim",
    "checkpoint_file",
}
OPTIONAL_KEYS = {"hidden_dim"}
ALLOWED_KEYS = REQUIRED_KEYS | OPTIONAL_KEYS
SUPPORTED_BACKEND_TOKENS = ("kakao", "karlo")
CHECKPOINT_STATE_DICT_KEYS = ("state_dict", "model_state_dict", "translator_state_dict", "model")


def _require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"Metadata field '{field_name}' must be a string, got {type(value).__name__}")
    parsed = value.strip()
    if not parsed:
        raise ValueError(f"Metadata field '{field_name}' must be a non-empty string")
    return parsed


def _require_positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Metadata field '{field_name}' must be an integer, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"Metadata field '{field_name}' must be > 0, got {value}")
    return value


def _validate_embedding_backend(embedding_backend: str) -> str:
    backend = embedding_backend.strip()
    if not any(token in backend.lower() for token in SUPPORTED_BACKEND_TOKENS):
        raise ValueError(
            f"Translator embedding_backend='{embedding_backend}' is not Kakao/Karlo compatible."
        )
    return backend


def _validate_architecture(architecture: str) -> str:
    arch = architecture.strip().lower()
    if arch not in supported_architectures():
        raise ValueError(
            f"Unsupported architecture '{architecture}'. Supported: {', '.join(supported_architectures())}"
        )
    return arch


def _validate_checkpoint_file(checkpoint_file: str) -> str:
    checkpoint_relpath = Path(checkpoint_file)
    if checkpoint_relpath.is_absolute():
        raise ValueError("metadata.checkpoint_file must be a relative path")
    if any(part == ".." for part in checkpoint_relpath.parts):
        raise ValueError("metadata.checkpoint_file must not contain parent traversal ('..')")
    if not checkpoint_relpath.parts:
        raise ValueError("metadata.checkpoint_file must not be empty")
    return str(checkpoint_relpath)


def _read_metadata(metadata_path: Path) -> TranslatorMetadata:
    with metadata_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, Mapping):
        raise TypeError(f"Translator metadata must be a mapping: {metadata_path}")

    missing = sorted(REQUIRED_KEYS.difference(data.keys()))
    if missing:
        raise KeyError(f"Missing metadata keys in {metadata_path}: {missing}")

    extra = sorted(set(data.keys()).difference(ALLOWED_KEYS))
    if extra:
        raise KeyError(f"Unexpected metadata keys in {metadata_path}: {extra}")

    model_name = _require_non_empty_string(data["model_name"], "model_name")
    translator_name = _require_non_empty_string(data["translator_name"], "translator_name")
    architecture = _validate_architecture(_require_non_empty_string(data["architecture"], "architecture"))
    embedding_backend = _validate_embedding_backend(
        _require_non_empty_string(data["embedding_backend"], "embedding_backend")
    )
    in_dim = _require_positive_int(data["in_dim"], "in_dim")
    out_dim = _require_positive_int(data["out_dim"], "out_dim")
    hidden_dim_raw = data.get("hidden_dim", in_dim)
    hidden_dim = _require_positive_int(hidden_dim_raw, "hidden_dim")
    checkpoint_file = _validate_checkpoint_file(
        _require_non_empty_string(data["checkpoint_file"], "checkpoint_file")
    )

    return TranslatorMetadata(
        model_name=model_name,
        translator_name=translator_name,
        architecture=architecture,
        embedding_backend=embedding_backend,
        in_dim=in_dim,
        out_dim=out_dim,
        hidden_dim=hidden_dim,
        checkpoint_file=checkpoint_file,
    )


def _strip_orig_mod_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    if not any(k.startswith("_orig_mod.") for k in state_dict):
        return state_dict
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}


def _safe_torch_load(checkpoint_path: Path) -> Any:
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


def _looks_like_state_dict(candidate: Mapping[str, Any]) -> bool:
    if not candidate:
        return False
    if not all(isinstance(k, str) for k in candidate):
        return False
    return any(torch.is_tensor(v) for v in candidate.values())


def _extract_state_dict(loaded_obj: Any, checkpoint_path: Path) -> dict[str, Any]:
    if not isinstance(loaded_obj, Mapping):
        raise ValueError(f"Checkpoint format not supported: {checkpoint_path}")

    direct_state_dict = dict(loaded_obj)
    if _looks_like_state_dict(direct_state_dict):
        return direct_state_dict

    for key in CHECKPOINT_STATE_DICT_KEYS:
        candidate = loaded_obj.get(key)
        if isinstance(candidate, Mapping):
            state_dict = dict(candidate)
            if _looks_like_state_dict(state_dict):
                return state_dict

    raise ValueError(
        f"Checkpoint does not contain a loadable state_dict under keys {CHECKPOINT_STATE_DICT_KEYS}: {checkpoint_path}"
    )


def _validate_checkpoint_tensors(
    state_dict: Mapping[str, Any], model_state_dict: Mapping[str, torch.Tensor], checkpoint_path: Path
) -> None:
    non_tensor_keys: list[str] = []
    bad_device_keys: list[str] = []
    bad_dtype_keys: list[str] = []

    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            non_tensor_keys.append(key)
            continue
        if value.device.type != "cpu":
            bad_device_keys.append(f"{key}:{value.device}")
        expected = model_state_dict.get(key)
        if expected is not None and torch.is_tensor(expected) and value.dtype != expected.dtype:
            bad_dtype_keys.append(f"{key}:{value.dtype}->{expected.dtype}")

    if non_tensor_keys:
        raise TypeError(
            f"Checkpoint contains non-tensor entries (unsupported): {checkpoint_path} keys={non_tensor_keys[:10]}"
        )
    if bad_device_keys:
        raise ValueError(
            f"Checkpoint tensors must be on CPU before loading: {checkpoint_path} keys={bad_device_keys[:10]}"
        )
    if bad_dtype_keys:
        raise ValueError(
            f"Checkpoint tensor dtypes do not match model parameters: {checkpoint_path} keys={bad_dtype_keys[:10]}"
        )


def _load_state_dict_with_fallback(
    model: torch.nn.Module, state_dict: dict[str, Any], logger_obj: Any, checkpoint_path: Path
) -> None:
    try:
        model.load_state_dict(state_dict, strict=True)
        return
    except RuntimeError as strict_error:
        logger_obj.warning(
            "Strict state_dict load failed for checkpoint %s; retrying with strict=False. Error: %s",
            checkpoint_path,
            strict_error,
        )

    incompatible = model.load_state_dict(state_dict, strict=False)
    logger_obj.warning(
        "Non-strict state_dict load used for checkpoint %s. missing_keys=%s unexpected_keys=%s",
        checkpoint_path,
        list(incompatible.missing_keys),
        list(incompatible.unexpected_keys),
    )


def load_translator(
    translators_root: Path,
    model_name: str,
    translator_name: str,
    device: torch.device,
    logger: Any | None = None,
) -> LoadedTranslator:
    """Load translator from translators/<model>/<name> directory."""
    logger_obj = logger or get_logger("sing.translators")
    translator_dir = translators_root / model_name / translator_name
    metadata_path = translator_dir / "metadata.yaml"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    metadata = _read_metadata(metadata_path)
    if metadata.model_name != model_name:
        raise ValueError(
            f"Metadata model_name='{metadata.model_name}' does not match requested model '{model_name}'"
        )
    if metadata.translator_name != translator_name:
        raise ValueError(
            f"Metadata translator_name='{metadata.translator_name}' does not match requested translator '{translator_name}'"
        )

    ckpt_path = translator_dir / metadata.checkpoint_file
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing translator checkpoint: {ckpt_path}")

    model = build_translator(
        architecture=metadata.architecture,
        in_dim=metadata.in_dim,
        hidden_dim=metadata.hidden_dim,
        out_dim=metadata.out_dim,
    )
    state_dict = _extract_state_dict(_safe_torch_load(ckpt_path), ckpt_path)
    state_dict = _strip_orig_mod_prefix(state_dict)
    _validate_checkpoint_tensors(state_dict, model.state_dict(), ckpt_path)
    _load_state_dict_with_fallback(model, state_dict, logger_obj, ckpt_path)
    model = model.to(device).eval()
    logger_obj.info(
        "Loaded translator model=%s translator=%s architecture=%s",
        model_name,
        translator_name,
        metadata.architecture,
    )
    return LoadedTranslator(metadata=metadata, model=model)
