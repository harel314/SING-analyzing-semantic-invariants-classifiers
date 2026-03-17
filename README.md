# SING: Analyzing Semantic Invariants in Classifiers

<p align="center">
  <a href="https://arxiv.org/abs/2603.14610"><img src="https://img.shields.io/badge/arXiv-2603.14610-b31b1b?style=flat-square&logo=arxiv&logoColor=white" alt="arXiv"></a>
  &nbsp;
  <a href="https://huggingface.co/papers/2603.14610"><img src="https://img.shields.io/badge/🤗%20HuggingFace-Paper-ffd21e?style=flat-square" alt="HuggingFace"></a>
  &nbsp;
  <a href="https://harel314.github.io/SING-analyzing-semantic-invariants-classifiers/"><img src="https://img.shields.io/badge/🌐%20Webpage-SING-5b7cfa?style=flat-square" alt="Webpage"></a>
  &nbsp;
  <a href="#"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab (Coming Soon)"></a>
</p>

<p align="center">
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="MIT License"></a>
  &nbsp;
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776ab?style=flat-square&logo=python&logoColor=white" alt="Python 3.9+">
  &nbsp;
  <img src="https://img.shields.io/badge/CVPR-2026-9b59b6?style=flat-square" alt="CVPR 2026">
</p>

> Official repository for the paper:  
> **SING: Analyzing Semantic Invariants in Classifiers**  
> Accepted to **CVPR 2026**.

![SING teaser](figs/teaser.png)

---

## Overview

All classifiers contain invariants induced by the geometry of their final linear head.  
SING (Semantic Interpretation of the Null-space Geometry) makes these invariants interpretable:

- it builds equivalent feature pairs through null-space analysis
- it translates features into a vision-language embedding space
- it quantifies semantic drift with `IS` and `AS`
- it visualizes the semantic effect of invariants on generated images

This repository provides a practical v1 flow for single-image analysis without requiring a dataset pipeline.

### Why it matters

- turns null-space geometry into human-readable semantic diagnostics
- supports class/model comparison through leakage into invariant directions
- provides both score-level (`IS`/`AS`) and image-level interpretation
- enables quick experiments from one image via CLI, notebooks, or Docker

---

## Method

Method diagram from the paper: [`figs/method.png`](figs/method.png)

SING follows four core steps:

1. Decompose classifier head weights with SVD into principal/null subspaces.  
2. Train/load a translator from classifier features to a multimodal embedding space.  
3. Build equivalent feature pairs via null-space projection/manipulation.  
4. Measure and visualize semantic variation with text/image projections.

---

## Colab Notebooks

| Notebook | Description | Link |
| --- | --- | --- |
| `notebooks/01_single_image_quickstart.ipynb` | Single-image walkthrough: load, project, translate, score, plot | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#) *(coming soon)* |

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Requirements

Core dependencies include:

- PyTorch
- torchvision / timm / transformers
- diffusers + accelerate
- numpy, pillow, pyyaml

> Note: generation uses `kakaobrain/karlo-v1-alpha-image-variations` via `diffusers`.
> Translators in this repo are aligned to this embedding backend.

---

## Usage

### 1) Run CLI on a single image

```bash
python -m sing.cli \
  --image samples/border_collie_n02106166.jpeg \
  --model resnet \
  --seeds 42 1337 \
  --output-dir outputs/demo
```

### 2) Run with Docker images

Use prebuilt images from Docker Hub:

- GPU: `harel314/sing:gpu-cuda12.4`
- CPU: `harel314/sing:cpu`

Create local cache once:

```bash
mkdir -p .cache/huggingface
```

#### GPU run (recommended)

```bash
docker run --rm --gpus all -e HF_HOME=/workspace/.cache/huggingface \
  -v "$PWD":/workspace -v "$PWD/.cache/huggingface":/workspace/.cache/huggingface -w /workspace \
  harel314/sing:gpu-cuda12.4 \
  --image samples/border_collie_n02106166.jpeg \
  --model resnet \
  --seeds 42 1337 \
  --output-dir outputs/demo
```

#### CPU run

```bash
docker run --rm -e HF_HOME=/workspace/.cache/huggingface \
  -v "$PWD":/workspace -v "$PWD/.cache/huggingface":/workspace/.cache/huggingface -w /workspace \
  harel314/sing:cpu \
  --image samples/border_collie_n02106166.jpeg \
  --model resnet \
  --seeds 42 1337 \
  --output-dir outputs/demo
```

#### Notebook/Remote IDE container

Start a long-running container and attach from Cursor Remote Explorer:

```bash
# GPU
docker run -d --name sing-dev --entrypoint /bin/bash --gpus all -p 8888:8888 \
  -e HF_HOME=/workspace/.cache/huggingface \
  -v "$PWD":/workspace -v "$PWD/.cache/huggingface":/workspace/.cache/huggingface -w /workspace \
  harel314/sing:gpu-cuda12.4 /bin/bash -lc "sleep infinity"

# CPU
docker run -d --name sing-dev-cpu --entrypoint /bin/bash -p 8889:8888 \
  -e HF_HOME=/workspace/.cache/huggingface \
  -v "$PWD":/workspace -v "$PWD/.cache/huggingface":/workspace/.cache/huggingface -w /workspace \
  harel314/sing:cpu /bin/bash -lc "sleep infinity"
```

### 3) Run notebook demo

Open:

```text
notebooks/01_single_image_quickstart.ipynb
```

### 4) Translators layout

```text
translators/<model_name>/linear/
  metadata.yaml
  best.pt
```

This repo includes translator checkpoints for many backbones, so you can choose between 13 translators in `translators/`, including:
`resnet`, `dinovit1`, `dinov2`, `vgg16`, `vgg19`, `convnext`, `deit`, `densenet`, `efficientnet`, `eva02_large`, `resnext`, `bitresnet`, `swin_large`.

---

## Repository Structure

```text
SING-analyzing-semantic-invariants-classifiers/
├── src/                     # Main Python package (sing)
├── configs/                 # Model/runtime/translator config files
├── translators/             # Linear translator checkpoints + metadata
├── samples/                 # Small example images for quick testing
├── notebooks/               # Reproducible notebook demos
├── tests/                   # Unit tests
├── docker/                  # Container notes/assets
├── misc/                    # Auxiliary resources and scripts
├── docs/                    # Agent notes + project TODO
├── index.html               # GitHub Pages landing file
└── README.md
```

---

## Citation

BibTeX will be added once camera-ready metadata is finalized.

```bibtex
@inproceedings{sing2026,
  title={SING: Analyzing Semantic Invariants in Classifiers},
  author={TBD},
  booktitle={CVPR},
  year={2026}
}
```

---

## Presentation Video

Coming soon.

---

## Contact

For questions and collaborations, open an issue in this repository.
