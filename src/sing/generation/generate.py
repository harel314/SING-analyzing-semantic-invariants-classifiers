"""Generation helpers for original and principal embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from sing.generation.unclip_wrapper import KakaoUnclipWrapper
from sing.utils import ensure_dir, get_logger


@dataclass(frozen=True)
class GenerationResult:
    seed: int
    original_path: Path
    principal_path: Path


def _build_output_path(output_dir: Path, seed: int, variant: str) -> Path:
    return output_dir / f"seed_{int(seed)}_{variant}.png"


def generate_seed_set(
    wrapper: KakaoUnclipWrapper,
    image: Image.Image,
    principal_embedding: torch.Tensor,
    seeds: list[int],
    output_dir: Path,
    logger: Any | None = None,
) -> list[GenerationResult]:
    logger_obj = logger or get_logger("sing.generation")
    ensure_dir(output_dir)
    principal_embedding_detached = principal_embedding.detach()

    results: list[GenerationResult] = []
    for seed in seeds:
        original_img = wrapper.generate_original(image=image, seed=seed)
        principal_img = wrapper.generate_from_embedding(
            embedding=principal_embedding_detached,
            seed=seed,
        )

        original_path = _build_output_path(output_dir=output_dir, seed=seed, variant="original")
        principal_path = _build_output_path(output_dir=output_dir, seed=seed, variant="principal")
        wrapper.save_image(original_img, original_path)
        wrapper.save_image(principal_img, principal_path)
        logger_obj.info("Generated images for seed=%d", seed)
        results.append(
            GenerationResult(seed=seed, original_path=original_path, principal_path=principal_path)
        )
    return results
