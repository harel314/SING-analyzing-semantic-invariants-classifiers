"""Classifier-head projectors for principal/null spaces."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Projectors:
    v_null: torch.Tensor
    v_principal: torch.Tensor
    rank: int


def compute_projectors_from_weight(weight: torch.Tensor, tolerance: float = 1e-5) -> Projectors:
    """Compute principal/null bases from classifier weight matrix."""
    if weight.ndim != 2:
        raise ValueError("Classifier weight must be 2D")

    _, singular_values, vt = torch.linalg.svd(weight, full_matrices=True)
    v = vt.T
    rank = int((singular_values > tolerance).sum().item())
    v_principal = v[:, :rank]
    v_null = v[:, rank:]
    return Projectors(v_null=v_null, v_principal=v_principal, rank=rank)


def principal_component(features: torch.Tensor, v_null: torch.Tensor) -> torch.Tensor:
    """Remove null-space component from feature matrix."""
    if features.ndim == 1:
        features = features.unsqueeze(0)
    return features - (features @ v_null @ v_null.T)
