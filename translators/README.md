# Translators Layout

Expected structure:

```text
translators/
  registry.yaml
  resnet/
    linear/
      metadata.yaml
      best.pt
  dinovit1/
    linear/
      metadata.yaml
      best.pt
```

Required metadata fields:
- `model_name`
- `translator_name`
- `architecture`
- `embedding_backend` (must include `kakao` or `karlo`)
- `in_dim` (positive integer)
- `out_dim` (positive integer)
- `hidden_dim` (optional; positive integer; defaults to `in_dim`)
- `checkpoint_file` (relative path only; no absolute paths or `..`)

Validation notes:
- Metadata schema is strict: unknown keys are rejected.
- `model_name` and `translator_name` inside `metadata.yaml` must match their directory names.
- `architecture` must be one of: `linear`, `3layer`, `4layer`, `residual`.
- Registry lookup is case-insensitive by model name and rejects ambiguous case-collisions (for example both `ResNet` and `resnet`).
