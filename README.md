# SING: Analyzing Semantic Invariants in Classifiers

v1 focuses on single-image analysis with a lightweight workflow:

1. load a classifier wrapper (`resnet` or `dinovit1`)
2. load a trained translator from `translators/`
3. load Kakao UnCLIP backend
4. generate original and principal (null-removed) images for selected seeds
5. compute IS/AS and dictionary scores

## Important backend note

- This project uses `kakaobrain/karlo-v1-alpha-image-variations`.
- Translators are expected to be trained for this embedding space.
- This is not OpenAI CLIP image generation.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Minimal CLI usage

```bash
python -m sing.cli \
  --image path/to/image.jpg \
  --model resnet \
  --seeds 42 1337 \
  --output-dir outputs/demo
```

## Translator assets

Place translator checkpoints under:

```text
translators/<model_name>/<translator_name>/
  metadata.yaml
  best.pt
```

See `translators/README.md` and `configs/translator_metadata_template.yaml`.

## Runtime

- GPU is preferred when available.
- CPU is supported for single-image runs.
- TPU is not required or used by the current runtime code.
