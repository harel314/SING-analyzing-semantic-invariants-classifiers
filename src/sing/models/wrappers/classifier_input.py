"""Generic wrapper extracting classifier input features."""

from __future__ import annotations

from typing import Any

import torch


def _resolve_module(root: Any, module_path: str) -> Any:
    module = root
    for part in module_path.split("."):
        if not hasattr(module, part):
            raise ValueError(f"Missing module path '{module_path}' (stuck at '{part}')")
        module = getattr(module, part)
    return module


class ClassifierInputWrapper:
    """Capture classifier input via forward pre-hook."""

    feature_layer_name = "classifier_input"

    def __init__(self, model: torch.nn.Module, feature_dim: int, classifier_layer_name: str) -> None:
        self.model = model.eval()
        self.feature_dim = int(feature_dim)
        self.classifier_layer_name = classifier_layer_name

    def _classifier_layer(self) -> torch.nn.Module:
        layer = _resolve_module(self.model, self.classifier_layer_name)
        if not isinstance(layer, torch.nn.Module):
            raise ValueError(f"Classifier path '{self.classifier_layer_name}' is not a torch module")
        return layer

    def extract_features(self, inputs: torch.Tensor) -> torch.Tensor:
        layer = self._classifier_layer()
        captured: dict[str, torch.Tensor] = {}

        def _pre_hook(_module: torch.nn.Module, args: tuple[Any, ...]) -> None:
            if not args or not torch.is_tensor(args[0]):
                raise ValueError("Classifier pre-hook did not receive tensor input.")
            feat = args[0]
            if feat.ndim > 2:
                feat = torch.flatten(feat, 1)
            captured["features"] = feat

        handle = layer.register_forward_pre_hook(_pre_hook)
        try:
            with torch.no_grad():
                _ = self.model(inputs)
        finally:
            handle.remove()

        features = captured.get("features")
        if features is None:
            raise ValueError("Failed to capture classifier input features.")
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                f"Feature shape mismatch: expected (*, {self.feature_dim}), got {tuple(features.shape)}"
            )
        return features

    def logits_from_features(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                f"Classifier input mismatch: expected (*, {self.feature_dim}), got {tuple(features.shape)}"
            )
        layer = self._classifier_layer()
        if isinstance(layer, torch.nn.Conv2d):
            logits = layer(features.unsqueeze(-1).unsqueeze(-1))
            return torch.flatten(logits, 1)
        return layer(features)

    @property
    def classifier_weight(self) -> torch.Tensor:
        layer = self._classifier_layer()
        weight = getattr(layer, "weight", None)
        if not isinstance(weight, torch.Tensor):
            raise ValueError(f"Classifier layer '{self.classifier_layer_name}' has no tensor 'weight'.")
        if weight.ndim == 4 and weight.shape[-2:] == (1, 1):
            weight = torch.flatten(weight, 1)
        if weight.ndim != 2 or weight.shape[1] != self.feature_dim:
            raise ValueError(
                f"Classifier weight mismatch: expected (*, {self.feature_dim}), got {tuple(weight.shape)}"
            )
        return weight
