import torch

from sing.core.metrics import compute_as, compute_is


def test_compute_is_zero_for_identical_vectors():
    x = torch.tensor([[1.0, 0.0, 0.0]])
    is_value = compute_is(x, x)
    assert torch.allclose(is_value, torch.zeros_like(is_value))


def test_compute_as_shape():
    original = torch.tensor([[1.0, 0.0, 0.0]])
    principal = torch.tensor([[0.0, 1.0, 0.0]])
    text = torch.tensor([1.0, 0.0, 0.0])
    as_value = compute_as(original, principal, text)
    assert as_value.shape == (1,)
