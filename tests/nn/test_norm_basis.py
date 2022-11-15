import pytest
import torch

from allegro.nn import NormalizedBasis


@pytest.mark.parametrize("r_min", [0.2, 1.0, 3.0])
def test_normalized_basis(r_min):
    # Note that this parameter sharing is normally taken care of by `instantiate`
    nb = NormalizedBasis(r_min=r_min, r_max=5.0, original_basis_kwargs={"r_max": 5.0})
    rs = torch.empty(10 * nb.n)
    rs.uniform_(nb.r_min, nb.r_max)
    bvals = nb(rs)
    b_std, b_mean = torch.std_mean(bvals, dim=0)
    threshold = 2e-2  # pretty arbitrary, TODO
    assert torch.all(b_mean.abs() < threshold)
    assert torch.all((b_std - 1.0).abs() < threshold)
