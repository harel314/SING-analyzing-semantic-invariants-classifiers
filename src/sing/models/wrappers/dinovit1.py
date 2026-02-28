"""DINO ViT-v1 wrapper for SING."""

from __future__ import annotations

import torch
import timm


class DinoVit1Wrapper:
    """ViT-L/16 feature/logit adapter."""

    feature_dim = 1024
    feature_layer_name = "cls_token_post_norm"
    classifier_layer_name = "head"

    def __init__(self, model: timm.models.vision_transformer.VisionTransformer) -> None:
        self.model = model.eval()

    def extract_features(self, inputs: torch.Tensor) -> torch.Tensor:
        tokens = self.model.forward_features(inputs)
        features: torch.Tensor
        if isinstance(tokens, dict):
            if "x_norm_clstoken" in tokens:
                features = tokens["x_norm_clstoken"]
            elif "x_norm" in tokens:
                features = tokens["x_norm"][:, 0]
            else:
                raise ValueError("Unsupported timm ViT feature dict format.")
        elif tokens.ndim == 3:
            features = tokens[:, 0]
        elif tokens.ndim == 2:
            features = tokens
        else:
            raise ValueError(f"Unexpected feature shape: {tuple(tokens.shape)}")
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                f"DINO ViT-v1 feature shape mismatch: expected (*, {self.feature_dim}), "
                f"got {tuple(features.shape)}"
            )
        return features

    def logits_from_features(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                f"DINO ViT-v1 logits input mismatch: expected (*, {self.feature_dim}), "
                f"got {tuple(features.shape)}"
            )
        return self.model.head(features)

    @property
    def classifier_weight(self) -> torch.Tensor:
        weight = self.model.head.weight
        if weight.ndim != 2 or weight.shape[1] != self.feature_dim:
            raise ValueError(
                f"DINO ViT-v1 classifier weight mismatch: expected (*, {self.feature_dim}), "
                f"got {tuple(weight.shape)}"
            )
        return weight
