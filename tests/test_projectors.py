import torch

from sing.core.projectors import compute_projectors_from_weight, principal_component


def test_projector_shapes_and_rank():
    w = torch.randn(4, 6)
    projectors = compute_projectors_from_weight(w)
    assert projectors.v_null.shape[0] == 6
    assert projectors.v_principal.shape[0] == 6
    assert 0 <= projectors.rank <= 4


def test_principal_component_reduces_null_projection():
    w = torch.randn(3, 5)
    p = compute_projectors_from_weight(w, tolerance=1e-12)
    features = torch.randn(2, 5)
    principal = principal_component(features, p.v_null)
    if p.v_null.numel() > 0:
        null_energy = torch.norm(principal @ p.v_null)
        assert float(null_energy.item()) < 1e-4
