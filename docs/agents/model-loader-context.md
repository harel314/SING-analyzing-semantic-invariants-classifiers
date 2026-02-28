# Agent Context: Model Loader (Review + Cleanup)

## Mission
Audit and refine existing model loading code for v1 SING.

## Scope
- Only `resnet` and `dinovit1`.
- Keep API stable:
  - `load_model(model_name, device)` -> loaded wrapper + preprocess.
- No expansion to more models.

## Files To Review
- `src/sing/models/wrappers/base.py`
- `src/sing/models/wrappers/resnet50.py`
- `src/sing/models/wrappers/dinovit1.py`
- `src/sing/models/registry.py`
- `configs/models.yaml`

## Constraints
- No `os.chdir`, no absolute paths, no `sys.path` hacks.
- Keep imports at top.
- Prefer small readable classes over clever abstractions.

## Required Checks
- Wrapper returns correct penultimate feature shape.
- Classifier weight points to actual final head weight.
- Registry errors are clear for unsupported names.
- CPU and CUDA device behavior is consistent.

## Deliverable
- Minimal patch improving readability/safety.
- Short note: what was fixed + residual risks.
