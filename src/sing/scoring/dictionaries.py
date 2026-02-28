"""Dictionary scoring for translated embeddings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class DictionaryScore:
    label: str
    original_similarity: float
    principal_similarity: float
    delta: float


def load_dictionary_embeddings(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    """Load dictionary embeddings JSON {label: [float,...]}."""
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Dictionary embedding file must contain a JSON object.")

    result: dict[str, torch.Tensor] = {}
    for label, values in raw.items():
        result[str(label)] = torch.tensor(values, dtype=torch.float32, device=device)
    return result


def score_against_dictionary(
    translated_original: torch.Tensor,
    translated_principal: torch.Tensor,
    dictionary_embeddings: dict[str, torch.Tensor],
) -> list[DictionaryScore]:
    original = torch.nn.functional.normalize(translated_original.squeeze(0), dim=-1)
    principal = torch.nn.functional.normalize(translated_principal.squeeze(0), dim=-1)

    scores: list[DictionaryScore] = []
    for label, emb in dictionary_embeddings.items():
        emb_n = torch.nn.functional.normalize(emb, dim=-1)
        sim_o = float(torch.dot(original, emb_n).item())
        sim_p = float(torch.dot(principal, emb_n).item())
        scores.append(
            DictionaryScore(
                label=label,
                original_similarity=sim_o,
                principal_similarity=sim_p,
                delta=sim_o - sim_p,
            )
        )
    return sorted(scores, key=lambda x: abs(x.delta), reverse=True)
