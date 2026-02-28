"""Single-image end-to-end analyzer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from sing.core.metrics import compute_as, compute_is
from sing.core.projectors import compute_projectors_from_weight, principal_component
from sing.generation.generate import GenerationResult, generate_seed_set
from sing.generation.unclip_wrapper import KakaoUnclipWrapper
from sing.models.registry import load_model
from sing.scoring.dictionaries import (
    DictionaryScore,
    load_dictionary_embeddings,
    score_against_dictionary,
)
from sing.translators.loader import load_translator
from sing.translators.registry import load_default_translator
from sing.utils import get_logger, resolve_device


@dataclass
class AnalysisOutput:
    model_name: str
    translator_name: str
    is_value: float
    as_value: float | None
    generated_files: list[GenerationResult]
    simple_scores: list[DictionaryScore]
    main_class_scores: list[DictionaryScore]


class SingleImageAnalyzer:
    """SING v1 single-image analyzer."""

    def __init__(self, repo_root: Path, device: str | None = None, logger: Any | None = None) -> None:
        self.repo_root = repo_root
        self.device = resolve_device(device)
        self.logger = logger or get_logger("sing.analyzer")

    def _load_unclip(self) -> KakaoUnclipWrapper:
        return KakaoUnclipWrapper(
            device=self.device,
            torch_dtype=KakaoUnclipWrapper.default_dtype(self.device),
            logger=self.logger,
        )

    def analyze(
        self,
        image_path: Path,
        model_name: str,
        seeds: list[int],
        output_dir: Path,
        translator_name: str = "default",
        use_registry: bool = True,
    ) -> AnalysisOutput:
        loaded_model = load_model(model_name=model_name, device=self.device, logger=self.logger)

        if use_registry:
            loaded_translator = load_default_translator(
                translators_root=self.repo_root / "translators",
                registry_path=self.repo_root / "translators" / "registry.yaml",
                model_name=loaded_model.name,
                device=self.device,
            )
        else:
            loaded_translator = load_translator(
                translators_root=self.repo_root / "translators",
                model_name=loaded_model.name,
                translator_name=translator_name,
                device=self.device,
            )

        image = Image.open(image_path).convert("RGB")
        input_tensor = loaded_model.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = loaded_model.wrapper.extract_features(input_tensor)
            projectors = compute_projectors_from_weight(
                loaded_model.wrapper.classifier_weight.detach().to(self.device)
            )
            principal_features = principal_component(features, projectors.v_null)
            translated_original = loaded_translator.model(features)
            translated_principal = loaded_translator.model(principal_features)

        is_value = float(compute_is(translated_original, translated_principal).mean().item())
        as_value = None

        simple_dict_path = self.repo_root / "misc" / "simple_dictionary_embeddings.json"
        main_dict_path = self.repo_root / "misc" / "main_classes_dictionary_embeddings.json"
        simple_scores: list[DictionaryScore] = []
        main_scores: list[DictionaryScore] = []
        if simple_dict_path.exists():
            simple_embeddings = load_dictionary_embeddings(simple_dict_path, device=self.device)
            simple_scores = score_against_dictionary(
                translated_original=translated_original,
                translated_principal=translated_principal,
                dictionary_embeddings=simple_embeddings,
            )
        if main_dict_path.exists():
            main_embeddings = load_dictionary_embeddings(main_dict_path, device=self.device)
            main_scores = score_against_dictionary(
                translated_original=translated_original,
                translated_principal=translated_principal,
                dictionary_embeddings=main_embeddings,
            )
            if main_scores:
                text_emb = load_dictionary_embeddings(main_dict_path, device=self.device)[
                    main_scores[0].label
                ].unsqueeze(0)
                as_value = float(
                    compute_as(
                        translated_original=translated_original,
                        translated_principal=translated_principal,
                        text_embedding=text_emb,
                    )
                    .mean()
                    .item()
                )

        unclip = self._load_unclip()
        generated = generate_seed_set(
            wrapper=unclip,
            image=image,
            principal_embedding=translated_principal.detach(),
            seeds=seeds,
            output_dir=output_dir,
            logger=self.logger,
        )
        return AnalysisOutput(
            model_name=loaded_model.name,
            translator_name=loaded_translator.metadata.translator_name,
            is_value=is_value,
            as_value=as_value,
            generated_files=generated,
            simple_scores=simple_scores,
            main_class_scores=main_scores,
        )
