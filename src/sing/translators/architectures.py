"""Translator architectures for loading trained checkpoints."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearTranslator(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        del hidden_dim
        self.weight = nn.Parameter(torch.zeros(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class ThreeLayerTranslator(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class FourLayerTranslator(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ResidualTranslator(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.residual = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = h + self.residual(h)
        return self.output_proj(h)


ARCHITECTURE_BUILDERS = {
    "linear": LinearTranslator,
    "3layer": ThreeLayerTranslator,
    "4layer": FourLayerTranslator,
    "residual": ResidualTranslator,
}


def supported_architectures() -> tuple[str, ...]:
    return tuple(sorted(ARCHITECTURE_BUILDERS.keys()))


def build_translator(architecture: str, in_dim: int, hidden_dim: int, out_dim: int) -> nn.Module:
    arch = architecture.lower()
    builder = ARCHITECTURE_BUILDERS.get(arch)
    if builder is None:
        raise ValueError(
            f"Unsupported translator architecture: {architecture}. "
            f"Supported: {', '.join(supported_architectures())}"
        )
    return builder(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim)
