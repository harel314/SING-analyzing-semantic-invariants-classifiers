"""Convert embedding CSV files to compact NPZ format."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


#---------------------------- parsing --------------------------------#
def parse_embedding_str(embedding_str: str) -> np.ndarray:
    values = np.fromstring(embedding_str.strip(), sep=" ", dtype=np.float32)
    if values.size == 0:
        raise ValueError("Encountered empty embedding string.")
    return values


#---------------------------- conversion --------------------------------#
def convert_csv(csv_path: Path, output_path: Path, dtype: np.dtype = np.float16) -> None:
    classes: list[str] = []
    prompt_names: list[str] = []
    rows: list[np.ndarray] = []

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing header in CSV: {csv_path}")
        if "class" not in reader.fieldnames:
            raise ValueError(f"CSV must contain 'class' column: {csv_path}")

        prompt_names = [name for name in reader.fieldnames if name != "class"]
        if not prompt_names:
            raise ValueError(f"No prompt columns found in CSV: {csv_path}")

        for row in reader:
            classes.append(str(row["class"]))
            vectors = [parse_embedding_str(str(row[p])) for p in prompt_names]
            vec_lens = {v.size for v in vectors}
            if len(vec_lens) != 1:
                raise ValueError(f"Inconsistent embedding dims in row class={row['class']}")
            rows.append(np.stack(vectors, axis=0))

    embeddings = np.stack(rows, axis=0).astype(dtype, copy=False)  # (N, P, D)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        classes=np.array(classes, dtype=object),
        prompts=np.array(prompt_names, dtype=object),
        embeddings=embeddings,
    )


#---------------------------- cli --------------------------------#
def main() -> None:
    parser = argparse.ArgumentParser(description="Convert embedding CSV to compact NPZ.")
    parser.add_argument("--csv", type=Path, required=True, help="Input CSV path.")
    parser.add_argument("--out", type=Path, required=True, help="Output NPZ path.")
    args = parser.parse_args()
    convert_csv(args.csv, args.out)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
