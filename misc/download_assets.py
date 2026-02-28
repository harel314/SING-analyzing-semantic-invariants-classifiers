"""Minimal helper to document asset placement."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare translators asset directories.")
    parser.add_argument("--output-root", type=Path, default=Path("translators"))
    args = parser.parse_args()

    for model in ("resnet", "dinovit1"):
        target = args.output_root / model / "default"
        target.mkdir(parents=True, exist_ok=True)
        print(f"Prepared: {target}")
    print("Place metadata.yaml and checkpoint (e.g., best.pt) in each directory.")


if __name__ == "__main__":
    main()
