import pytest
import torch

from allegro.nn import NormalizedBasis


@pytest.mark.parametrize("r_min", [0.2, 1.0, 3.0])
@pytest.mark.parametrize("norm_basis_mean_shift", [True, False])
def test_normalized_basis(r_min, norm_basis_mean_shift):
    # Note that this parameter sharing is normally taken care of by `instantiate`
    nb = NormalizedBasis(
        r_min=r_min,
        r_max=5.0,
        original_basis_kwargs={"r_max": 5.0, "num_bessels_per_basis": 8},
        norm_basis_mean_shift=norm_basis_mean_shift,
    )
    rs = torch.empty(100 * nb.n)
    rs.uniform_(nb.r_min, nb.r_max)
    bvals = nb(rs)
    threshold = 2e-2  # pretty arbitrary, TODO
    if norm_basis_mean_shift:
        b_std, b_mean = torch.std_mean(bvals, dim=0)
        assert torch.all(b_mean.abs() < threshold)
        assert torch.all((b_std - 1.0).abs() < threshold)
    else:
        b_rms = bvals.square().mean(dim=0).sqrt()
        assert torch.all((b_rms - 1.0).abs() < threshold)
