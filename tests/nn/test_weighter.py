import pytest

import torch

from e3nn import o3
from e3nn.util.test import assert_auto_jitable, assert_equivariant

from allegro.nn._strided import MakeWeightedChannels
from test_contract_basic import _strided_to_cat


@pytest.mark.parametrize("irreps_in", ["0e + 1o", "0e + 1o + 2e + 3o"])
@pytest.mark.parametrize("mul_out", [1, 5])
def test_make_weighter(
    irreps_in,
    mul_out,
):
    irreps_in = o3.Irreps(irreps_in)
    m_strided = MakeWeightedChannels(irreps_in=irreps_in, multiplicity_out=mul_out)
    m_cat = o3.Linear(
        irreps_in=irreps_in,
        irreps_out=o3.Irreps([(mul_out, ir) for _, ir in irreps_in]),
        shared_weights=False,
        internal_weights=False,
    )
    assert m_strided.weight_numel == m_cat.weight_numel
    batchdim = 7
    args_in = (
        irreps_in.randn(batchdim, -1),
        torch.randn(batchdim, m_strided.weight_numel),
    )

    def wrapper(x, w):
        # e3nn has the opposite weight convention
        w = (
            w.view(-1, len(irreps_in), mul_out)
            .transpose(-1, -2)
            .contiguous()
            .view(-1, m_strided.weight_numel)
        )
        return _strided_to_cat(irreps_in, mul_out, m_strided(x, w))

    assert_equivariant(
        wrapper,
        args_in=args_in,
        irreps_in=[irreps_in, o3.Irreps([(m_strided.weight_numel, "0e")])],
        irreps_out=m_cat.irreps_out,
    )
    assert_auto_jitable(m_strided)

    # Check same
    out_orig = m_cat(*args_in)
    out_opt = wrapper(*args_in)
    assert torch.allclose(out_orig, out_opt)
