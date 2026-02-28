"""CLI entrypoint for SING v1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sing.core.analyzer import SingleImageAnalyzer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SING single-image analyzer")
    parser.add_argument("--image", type=Path, required=True, help="Input image path")
    parser.add_argument("--model", type=str, choices=["resnet", "dinovit1"], required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1337])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/run"))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyzer = SingleImageAnalyzer(repo_root=args.repo_root.resolve(), device=args.device)
    result = analyzer.analyze(
        image_path=args.image,
        model_name=args.model,
        seeds=args.seeds,
        output_dir=args.output_dir,
    )
    payload = {
        "model_name": result.model_name,
        "translator_name": result.translator_name,
        "is_value": result.is_value,
        "as_value": result.as_value,
        "generated_files": [
            {
                "seed": item.seed,
                "original_path": str(item.original_path),
                "principal_path": str(item.principal_path),
            }
            for item in result.generated_files
        ],
        "simple_scores_top5": [score.__dict__ for score in result.simple_scores[:5]],
        "main_class_scores_top5": [score.__dict__ for score in result.main_class_scores[:5]],
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
