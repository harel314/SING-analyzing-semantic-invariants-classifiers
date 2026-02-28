# Agent Context: UnCLIP Generation (Review + Cleanup)

## Mission
Audit generation wrapper for correctness, stability, and deterministic seed behavior.

## Scope
- Keep backend: `kakaobrain/karlo-v1-alpha-image-variations`.
- Keep current single-image + seed-set flow.
- No feature expansion (no new generation modes).

## Files To Review
- `src/sing/generation/unclip_wrapper.py`
- `src/sing/generation/generate.py`
- `configs/runtime.yaml`

## Constraints
- No dataset assumptions.
- No absolute paths.
- Keep CPU/GPU behavior explicit and predictable.

## Required Checks
- Correct pipeline class/API use for image-variation model.
- Embedding encode/generate path works for single image.
- Seed determinism works per run.
- Output naming and save paths are stable.

## Deliverable
- Targeted patch with compatibility/stability improvements.
- Short note: what was fixed + any perf caveats on CPU.
