# Model Card (v1)

Supported classifier wrappers:
- `resnet` (ResNet50 ImageNet-1k)
- `dinovit1` (ViT-L/16 ImageNet-1k via timm)

Translator requirement:
- Must be provided externally and loaded from `translators/`.
- Must declare Kakao/Karlo embedding compatibility in metadata.

Generation backend:
- `kakaobrain/karlo-v1-alpha-image-variations`
