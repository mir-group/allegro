import pytest

import torch

from allegro.nn import ScalarMLPFunction
from nequip.utils import torch_default_dtype


@pytest.mark.parametrize("hs", [[1, 4, 5, 8, 1], [10, 10], [13, 2, 7]])
@pytest.mark.parametrize("act", [None, "silu", "mish", "gelu"])
@pytest.mark.parametrize("model_dtype", [torch.float32, torch.float64])
def test_mlp(hs, act, model_dtype):
    bdim = 7

    with torch_default_dtype(model_dtype):
        data = torch.randn(bdim, hs[0])

        mlp = ScalarMLPFunction(
            mlp_input_dim=None,
            mlp_hidden_layer_dims=hs,
            mlp_output_dim=None,
            mlp_nonlinearity=act,
        )

    out = mlp(data)
    assert out.shape == (bdim, hs[-1])
