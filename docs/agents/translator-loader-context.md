# Agent Context: Translator Loader (Review + Cleanup)

## Mission
Audit translator loading stack and harden metadata/checkpoint handling.

## Scope
- Keep current `translators/` layout.
- Keep architecture names already supported.
- Keep Kakao/Karlo backend enforcement.

## Files To Review
- `src/sing/translators/architectures.py`
- `src/sing/translators/loader.py`
- `src/sing/translators/registry.py`
- `translators/registry.yaml`
- `configs/translator_metadata_template.yaml`
- `translators/README.md`

## Constraints
- No absolute paths.
- No fallback hacks outside metadata/registry.
- Keep strict, explicit error messages.

## Required Checks
- Metadata schema validation is complete and type-safe.
- Checkpoint loading supports plain `state_dict` and wrapped `{"state_dict": ...}`.
- `_orig_mod.` prefix stripping is safe.
- Registry lookup behavior is deterministic and clear.

## Deliverable
- Focused patch for robustness/readability.
- Short note: what was fixed + known edge cases.
