"""Base model wrapper contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import torch


class FeatureWrapper(Protocol):
    """Model wrapper protocol for SING feature workflows."""

    feature_dim: int
    feature_layer_name: str
    classifier_layer_name: str

    def extract_features(self, inputs: torch.Tensor) -> torch.Tensor: ...

    def logits_from_features(self, features: torch.Tensor) -> torch.Tensor: ...

    @property
    def classifier_weight(self) -> torch.Tensor: ...


@dataclass(frozen=True)
class LoadedModel:
    """Container returned by the model registry."""

    name: str
    wrapper: FeatureWrapper
    preprocess: Callable
