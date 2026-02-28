"""Model registry for SING."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import timm
import torch
import torchvision
import yaml
from PIL import Image

from sing.models.wrappers.base import LoadedModel
from sing.models.wrappers.classifier_input import ClassifierInputWrapper
from sing.utils import get_logger

_MODEL_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "models.yaml"


@dataclass(frozen=True)
class ModelSpec:
    source: str
    model_id: str
    feature_dim: int
    feature_layer: str
    classifier_layer: str


def _resolve_attr(root: Any, path: str) -> Any:
    node = root
    for part in path.split("."):
        if not hasattr(node, part):
            raise ValueError(f"Missing path '{path}' on model.")
        node = getattr(node, part)
    return node


def _load_model_specs(config_path: Path) -> dict[str, ModelSpec]:
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    models_section = data.get("models")
    if not isinstance(models_section, dict):
        raise ValueError(f"Invalid model config format in {config_path}: missing 'models' mapping")

    specs: dict[str, ModelSpec] = {}
    for model_name, raw_spec in models_section.items():
        if not isinstance(raw_spec, dict):
            raise ValueError(f"Invalid config entry for model '{model_name}' in {config_path}")
        specs[str(model_name).lower()] = ModelSpec(
            source=str(raw_spec["source"]),
            model_id=str(raw_spec["model_id"]),
            feature_dim=int(raw_spec["feature_dim"]),
            feature_layer=str(raw_spec["feature_layer"]),
            classifier_layer=str(raw_spec["classifier_layer"]),
        )
    return specs


def _supported_models_from_config() -> tuple[str, ...]:
    return tuple(sorted(_load_model_specs(_MODEL_CONFIG_PATH).keys()))


SUPPORTED_MODELS = _supported_models_from_config()


def _build_torchvision(model_id: str, device: torch.device) -> tuple[torch.nn.Module, Any]:
    if model_id == "resnet50":
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        preprocess = torchvision.models.ResNet50_Weights.IMAGENET1K_V1.transforms()
    elif model_id == "densenet121":
        model = torchvision.models.densenet121(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1)
        preprocess = torchvision.models.DenseNet121_Weights.IMAGENET1K_V1.transforms()
    elif model_id == "vgg16":
        model = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
        preprocess = torchvision.models.VGG16_Weights.IMAGENET1K_V1.transforms()
    elif model_id == "vgg19":
        model = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1)
        preprocess = torchvision.models.VGG19_Weights.IMAGENET1K_V1.transforms()
    else:
        raise ValueError(f"Unsupported torchvision model_id '{model_id}'.")
    return model.to(device).eval(), preprocess


def _build_timm(model_id: str, device: torch.device) -> tuple[torch.nn.Module, Any]:
    model = timm.create_model(model_id, pretrained=True).to(device).eval()
    cfg = timm.data.resolve_model_data_config(model)
    preprocess = timm.data.create_transform(**cfg, is_training=False)
    return model, preprocess


def _build_transformers(model_id: str, device: torch.device) -> tuple[torch.nn.Module, Any]:
    if model_id != "facebook/dinov2-large-imagenet1k-1-layer":
        raise ValueError(f"Unsupported transformers model_id '{model_id}'.")
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    model = AutoModelForImageClassification.from_pretrained(model_id).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(model_id)

    def _preprocess(image: Any) -> torch.Tensor:
        out = processor(images=image, return_tensors="pt")
        return out["pixel_values"][0]

    return model, _preprocess


def _build_loaded_model(name: str, spec: ModelSpec, device: torch.device) -> LoadedModel:
    if spec.source == "torchvision":
        model, preprocess = _build_torchvision(spec.model_id, device)
    elif spec.source == "timm":
        model, preprocess = _build_timm(spec.model_id, device)
    elif spec.source == "transformers":
        model, preprocess = _build_transformers(spec.model_id, device)
    else:
        raise ValueError(f"Unsupported source '{spec.source}' for model '{name}'.")

    wrapper = ClassifierInputWrapper(
        model=model,
        feature_dim=spec.feature_dim,
        classifier_layer_name=spec.classifier_layer,
    )
    return LoadedModel(name=name, wrapper=wrapper, preprocess=preprocess)


def _validate_loaded_model(loaded: LoadedModel, spec: ModelSpec, device: torch.device) -> None:
    wrapper = loaded.wrapper
    if wrapper.feature_dim != spec.feature_dim:
        raise ValueError(f"Feature dim mismatch for model '{loaded.name}'.")
    if wrapper.feature_layer_name != spec.feature_layer:
        raise ValueError(f"Feature layer mismatch for model '{loaded.name}'.")
    if wrapper.classifier_layer_name != spec.classifier_layer:
        raise ValueError(f"Classifier layer mismatch for model '{loaded.name}'.")

    classifier_weight = wrapper.classifier_weight
    expected_weight = getattr(_resolve_attr(wrapper.model, spec.classifier_layer), "weight", None)
    if not isinstance(expected_weight, torch.Tensor):
        raise ValueError(f"Classifier layer '{spec.classifier_layer}' does not expose weight tensor.")
    flattened_from_conv = False
    if expected_weight.ndim == 4 and expected_weight.shape[-2:] == (1, 1):
        expected_weight = torch.flatten(expected_weight, 1)
        flattened_from_conv = True
    if expected_weight.shape != classifier_weight.shape:
        raise ValueError(f"Classifier weight shape mismatch for model '{loaded.name}'.")
    if not flattened_from_conv and expected_weight.data_ptr() != classifier_weight.data_ptr():
        raise ValueError(f"Classifier weight mapping mismatch for model '{loaded.name}'.")

    with torch.no_grad():
        sample = loaded.preprocess(Image.new("RGB", (512, 512))).unsqueeze(0).to(device)
        features = wrapper.extract_features(sample)
        logits = wrapper.logits_from_features(features)
    if logits.ndim != 2:
        raise ValueError(f"Logits must be 2D for model '{loaded.name}'.")


def load_model(model_name: str, device: torch.device, logger: Any | None = None) -> LoadedModel:
    """Load one configured model with strict config/runtime validation."""
    logger_obj = logger or get_logger("sing.models")
    specs = _load_model_specs(_MODEL_CONFIG_PATH)

    normalized = model_name.strip().lower()
    if normalized not in specs:
        raise ValueError(f"Unsupported model '{model_name}'. Supported: {tuple(sorted(specs.keys()))}")

    loaded = _build_loaded_model(normalized, specs[normalized], device)
    _validate_loaded_model(loaded, specs[normalized], device)
    logger_obj.info("Loaded model=%s on device=%s", loaded.name, str(device))
    return loaded
