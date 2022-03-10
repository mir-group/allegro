import pytest

import torch

from e3nn import o3

from nequip_allegro.nn._strided import Linear

from test_contract import _strided_to_cat


@pytest.mark.parametrize(
    "irreps_in", ["0e", "0e + 0o + 1e + 1o", "0e + 0e + 0e + 1o + 1o + 2e"]
)
@pytest.mark.parametrize("irreps_out", ["0e", "0e + 0o + 1e + 1o", "0e + 1o + 2e"])
@pytest.mark.parametrize("mul", [1, 2, 7])
@pytest.mark.parametrize("mulout", [1, 2, 7])
@pytest.mark.parametrize("pad", [1, 4])
@pytest.mark.parametrize("shared_weights", [False, True])
def test_like_linear(
    irreps_in,
    irreps_out,
    mul,
    mulout,
    pad,
    shared_weights,
):
    irreps_in = o3.Irreps(irreps_in)
    irreps_out = o3.Irreps(irreps_out)

    ours = Linear(
        irreps_in=o3.Irreps((mul, ir) for _, ir in irreps_in),
        irreps_out=o3.Irreps((mulout, ir) for _, ir in irreps_out),
        shared_weights=shared_weights,
        pad_to_alignment=pad,
    )
    batchdim = 7
    args_in = (
        torch.randn(batchdim, mul, ours.dim_in),
        torch.ones(
            ((batchdim,) if not shared_weights else tuple()) + (ours.weight_numel,)
        ),
    )

    e3nns = o3.Linear(
        irreps_in=o3.Irreps((mul, ir) for _, ir in irreps_in),
        irreps_out=o3.Irreps((mulout, ir) for _, ir in irreps_out),
        shared_weights=shared_weights,
        internal_weights=False,
    )
    assert e3nns.weight_numel == ours.weight_numel

    args_e3nn = (
        _strided_to_cat(irreps_in, mul, args_in[0]),
        args_in[1],
    )
    ours_out_strided = ours(*args_in)
    assert ours_out_strided.shape[-1] % pad == 0
    ours_out = _strided_to_cat(irreps_out, mulout, ours_out_strided)
    e3nn_out = e3nns(*args_e3nn)
    assert ours_out.shape == e3nn_out.shape
    assert torch.allclose(ours_out, e3nn_out, atol=1e-6)
