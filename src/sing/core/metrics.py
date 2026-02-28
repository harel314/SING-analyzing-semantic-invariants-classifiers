"""AS/IS metric functions."""

from __future__ import annotations

import torch


def _angle_deg(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = torch.nn.functional.normalize(a, dim=-1)
    b = torch.nn.functional.normalize(b, dim=-1)
    cos = torch.clamp(torch.sum(a * b, dim=-1), -1.0, 1.0)
    return torch.rad2deg(torch.acos(cos))


def compute_is(translated_original: torch.Tensor, translated_principal: torch.Tensor) -> torch.Tensor:
    """Image Score in degrees."""
    return _angle_deg(translated_original, translated_principal)


def compute_as(
    translated_original: torch.Tensor,
    translated_principal: torch.Tensor,
    text_embedding: torch.Tensor,
) -> torch.Tensor:
    """Attribute Score in degrees."""
    if text_embedding.ndim == 1:
        text_embedding = text_embedding.unsqueeze(0).expand_as(translated_original)
    return _angle_deg(translated_original, text_embedding) - _angle_deg(
        translated_principal, text_embedding
    )
