"""Kakao/Karlo UnCLIP wrapper for SING generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from diffusers import UnCLIPImageVariationPipeline

from sing.utils import get_logger


KAKAO_MODEL_ID = "kakaobrain/karlo-v1-alpha-image-variations"


class KakaoUnclipWrapper:
    """Thin wrapper for single-image and embedding-based generation."""

    def __init__(self, device: torch.device, torch_dtype: torch.dtype, logger: Any | None = None) -> None:
        self.logger = logger or get_logger("sing.generation")
        self.device = device
        self.torch_dtype = torch_dtype
        self._configure_determinism()
        self.pipe = UnCLIPImageVariationPipeline.from_pretrained(
            KAKAO_MODEL_ID,
            torch_dtype=torch_dtype,
        ).to(device)
        self.pipe.set_progress_bar_config(disable=True)
        self.logger.info("Loaded UnCLIP backend='%s' on %s", KAKAO_MODEL_ID, str(device))

    @staticmethod
    def default_dtype(device: torch.device) -> torch.dtype:
        if device.type == "cuda":
            return torch.float16
        return torch.float32

    def encode_image_embedding(self, image: Image.Image) -> torch.Tensor:
        pixel_values = self.pipe.feature_extractor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device=self.device, dtype=self.torch_dtype)
        with torch.no_grad():
            emb = self.pipe.image_encoder(pixel_values).image_embeds
        return emb

    def _configure_determinism(self) -> None:
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    def _make_generator(self, seed: int) -> torch.Generator:
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(seed))
        return generator

    def _normalize_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        normalized = embedding.detach()
        if normalized.ndim == 1:
            normalized = normalized.unsqueeze(0)
        return normalized.to(device=self.device, dtype=self.torch_dtype)

    @staticmethod
    def _generation_kwargs(
        decoder_steps: int,
        super_res_steps: int,
        guidance_scale: float,
    ) -> dict[str, int | float]:
        return {
            "decoder_num_inference_steps": int(decoder_steps),
            "super_res_num_inference_steps": int(super_res_steps),
            "decoder_guidance_scale": float(guidance_scale),
        }

    def generate_from_image(
        self,
        image: Image.Image,
        seed: int,
        decoder_steps: int = 25,
        super_res_steps: int = 7,
        guidance_scale: float = 8.0,
    ) -> Image.Image:
        generator = self._make_generator(seed)
        with torch.no_grad():
            out = self.pipe(
                image=image,
                generator=generator,
                output_type="pil",
                **self._generation_kwargs(
                    decoder_steps=decoder_steps,
                    super_res_steps=super_res_steps,
                    guidance_scale=guidance_scale,
                ),
            )
        return out.images[0]

    def generate_from_embedding(
        self,
        embedding: torch.Tensor,
        seed: int,
        decoder_steps: int = 25,
        super_res_steps: int = 7,
        guidance_scale: float = 8.0,
    ) -> Image.Image:
        normalized_embedding = self._normalize_embedding(embedding)
        generator = self._make_generator(seed)
        with torch.no_grad():
            out = self.pipe(
                image_embeddings=normalized_embedding,
                generator=generator,
                output_type="pil",
                **self._generation_kwargs(
                    decoder_steps=decoder_steps,
                    super_res_steps=super_res_steps,
                    guidance_scale=guidance_scale,
                ),
            )
        return out.images[0]

    def generate_original(self, image: Image.Image, seed: int) -> Image.Image:
        return self.generate_from_image(image=image, seed=seed)

    def save_image(self, image: Image.Image, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
