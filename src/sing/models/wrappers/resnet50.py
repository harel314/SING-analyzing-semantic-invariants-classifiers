"""ResNet50 wrapper for SING."""

from __future__ import annotations

import torch
import torchvision


class ResNet50Wrapper:
    """ResNet50 feature/logit adapter."""

    feature_dim = 2048
    feature_layer_name = "avgpool"
    classifier_layer_name = "fc"

    def __init__(self, model: torchvision.models.ResNet) -> None:
        self.model = model.eval()

    def extract_features(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.model.conv1(inputs)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        features = torch.flatten(x, 1)
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                f"ResNet50 feature shape mismatch: expected (*, {self.feature_dim}), "
                f"got {tuple(features.shape)}"
            )
        return features

    def logits_from_features(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                f"ResNet50 logits input mismatch: expected (*, {self.feature_dim}), "
                f"got {tuple(features.shape)}"
            )
        return self.model.fc(features)

    @property
    def classifier_weight(self) -> torch.Tensor:
        weight = self.model.fc.weight
        if weight.ndim != 2 or weight.shape[1] != self.feature_dim:
            raise ValueError(
                f"ResNet50 classifier weight mismatch: expected (*, {self.feature_dim}), "
                f"got {tuple(weight.shape)}"
            )
        return weight
