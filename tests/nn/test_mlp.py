import pytest

import torch

from e3nn import nn

from allegro.nn import ScalarMLPFunction


@pytest.mark.parametrize("hs", [[1, 4, 5, 8, 1], [10, 10], [13, 2, 7]])
@pytest.mark.parametrize("act", [None, "silu"])
@pytest.mark.parametrize("init", ["uniform", "orthogonal"])
def test_mlp(hs, act, init):
    bdim = 7
    data = torch.randn(bdim, hs[0])

    mlp = ScalarMLPFunction(
        mlp_input_dimension=None,
        mlp_latent_dimensions=hs,
        mlp_output_dimension=None,
        mlp_nonlinearity=act,
        mlp_initialization=init,
    )

    act = torch.nn.SiLU() if act == "silu" else None

    fc = nn.FullyConnectedNet(hs=hs, act=act)

    with torch.no_grad():
        for i, layer in enumerate(fc):
            getattr(mlp._forward, f"_weight_{i}").copy_(layer.weight)

    out = mlp(data)
    assert out.shape == (bdim, hs[-1])
    assert torch.allclose(
        out, fc(data), atol=2e-7 if torch.get_default_dtype() == torch.float32 else 1e-9
    )


def test_bias():
    bdim = 7
    hs = [4, 5, 5]
    data = torch.randn(bdim, hs[0])

    mlp = ScalarMLPFunction(
        mlp_input_dimension=None,
        mlp_latent_dimensions=hs,
        mlp_output_dimension=None,
        mlp_nonlinearity="silu",
        mlp_bias=True,
    )

    _ = mlp(data)
