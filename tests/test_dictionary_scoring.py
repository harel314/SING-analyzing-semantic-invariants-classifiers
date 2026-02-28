from pathlib import Path

import torch

from sing.scoring.dictionaries import load_dictionary_embeddings, score_against_dictionary


def test_dictionary_scoring_order(tmp_path: Path):
    data_file = tmp_path / "dict.json"
    data_file.write_text('{"a":[1,0], "b":[0,1]}', encoding="utf-8")
    embeddings = load_dictionary_embeddings(data_file, device=torch.device("cpu"))
    scores = score_against_dictionary(
        translated_original=torch.tensor([[1.0, 0.0]]),
        translated_principal=torch.tensor([[0.0, 1.0]]),
        dictionary_embeddings=embeddings,
    )
    assert scores[0].label in {"a", "b"}
    assert len(scores) == 2
