"""SING HuggingFace Space — metrics-only demo (IS + AS)."""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from sing.core.metrics import compute_as, compute_is
from sing.core.projectors import compute_projectors_from_weight, principal_component
from sing.models.registry import load_model, SUPPORTED_MODELS
from sing.translators.registry import load_default_translator

REPO_ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# preload embeddings once
_classes_npz = np.load(REPO_ROOT / "misc" / "embeddings" / "imagenet_text_embeddings_fp16.npz", allow_pickle=True)
_attrs_npz = np.load(REPO_ROOT / "misc" / "embeddings" / "attribute_embeddings_broden_style_fp16.npz", allow_pickle=True)

class_labels = _classes_npz["classes"].astype(str)
attr_labels = _attrs_npz["classes"].astype(str)

_class_prompt_names = _classes_npz["prompts"].astype(str)
_attr_prompt_names = _attrs_npz["prompts"].astype(str)
_class_prompt_idx = int(np.where(_class_prompt_names == "mean")[0][0]) if (_class_prompt_names == "mean").any() else 0
_attr_prompt_idx = int(np.where(_attr_prompt_names == "mean")[0][0]) if (_attr_prompt_names == "mean").any() else 0

class_text_embeddings = torch.tensor(_classes_npz["embeddings"][:, _class_prompt_idx, :], dtype=torch.float32, device=DEVICE)
attr_text_embeddings = torch.tensor(_attrs_npz["embeddings"][:, _attr_prompt_idx, :], dtype=torch.float32, device=DEVICE)

# model cache
_model_cache: dict[str, object] = {}
_translator_cache: dict[str, object] = {}


def _get_model(model_name: str):
    if model_name not in _model_cache:
        _model_cache[model_name] = load_model(model_name=model_name, device=DEVICE)
    return _model_cache[model_name]


def _get_translator(model_name: str):
    if model_name not in _translator_cache:
        _translator_cache[model_name] = load_default_translator(
            translators_root=REPO_ROOT / "translators",
            registry_path=REPO_ROOT / "translators" / "registry.yaml",
            model_name=model_name,
            device=DEVICE,
        )
    return _translator_cache[model_name]


def run(image: Image.Image, model_name: str, topk: int) -> tuple[plt.Figure, str]:
    loaded_model = _get_model(model_name)
    loaded_translator = _get_translator(model_name)

    input_tensor = loaded_model.preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        features = loaded_model.wrapper.extract_features(input_tensor)
        classifier_weight = loaded_model.wrapper.classifier_weight.detach().to(DEVICE)
        projectors = compute_projectors_from_weight(classifier_weight)
        principal_features = principal_component(features, projectors.v_null)
        translated_original = loaded_translator.model(features)
        translated_principal = loaded_translator.model(principal_features)

        is_value = float(compute_is(translated_original, translated_principal).mean().item())
        as_classes = compute_as(translated_original, translated_principal, class_text_embeddings).detach().cpu().numpy()
        as_attrs = compute_as(translated_original, translated_principal, attr_text_embeddings).detach().cpu().numpy()

    topk_c = min(topk, as_classes.shape[0])
    topk_a = min(topk, as_attrs.shape[0])
    idx_c = np.argsort(np.abs(as_classes))[-topk_c:]
    idx_a = np.argsort(np.abs(as_attrs))[-topk_a:]

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    axes[0].bar(np.arange(topk_c), np.abs(as_classes[idx_c]))
    axes[0].set_title(f"{model_name} | top-{topk_c} |AS| classes")
    axes[0].set_xticks(np.arange(topk_c))
    axes[0].set_xticklabels(class_labels[idx_c], rotation=45, ha="right", fontsize=9)
    axes[0].set_ylabel("AS (deg)")

    axes[1].bar(np.arange(topk_a), np.abs(as_attrs[idx_a]), color="tab:orange")
    axes[1].set_title(f"{model_name} | top-{topk_a} |AS| attributes")
    axes[1].set_xticks(np.arange(topk_a))
    axes[1].set_xticklabels(attr_labels[idx_a], rotation=45, ha="right", fontsize=9)
    axes[1].set_ylabel("AS (deg)")

    axes[2].bar(["IS"], [is_value], color="tab:green")
    axes[2].set_title(f"{model_name} | IS")
    axes[2].set_ylabel("IS (deg)")

    plt.tight_layout()

    summary = f"**IS** = {is_value:.4f}°\n\n**Top class:** {class_labels[idx_c[-1]]} ({np.abs(as_classes[idx_c[-1]]):.4f}°)\n\n**Top attribute:** {attr_labels[idx_a[-1]]} ({np.abs(as_attrs[idx_a[-1]]):.4f}°)"
    return fig, summary


with gr.Blocks(title="SING — Semantic Invariants in Classifiers") as demo:
    gr.Markdown(
        """
# SING: Analyzing Semantic Invariants in Classifiers
**CVPR 2026** · [Paper](https://arxiv.org/abs/2603.14610) · [GitHub](https://github.com/harel314/SING-analyzing-semantic-invariants-classifiers)

Upload an image, select a model, and compute **IS** (Invariance Score) and **AS** (Attribute Score) —
metrics that quantify how much semantic information lives in the classifier's null-space.
        """
    )
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Input Image")
            model_dropdown = gr.Dropdown(
                choices=list(SUPPORTED_MODELS),
                value="resnet",
                label="Model",
            )
            topk_slider = gr.Slider(minimum=5, maximum=20, value=10, step=1, label="Top-K attributes/classes")
            run_btn = gr.Button("Run", variant="primary")
        with gr.Column(scale=2):
            summary_output = gr.Markdown(label="Scores")
            plot_output = gr.Plot(label="AS + IS")

    run_btn.click(fn=run, inputs=[image_input, model_dropdown, topk_slider], outputs=[plot_output, summary_output])

    gr.Examples(
        examples=[["samples/border_collie_n02106166.jpeg", "resnet", 10]],
        inputs=[image_input, model_dropdown, topk_slider],
    )

if __name__ == "__main__":
    demo.launch()
