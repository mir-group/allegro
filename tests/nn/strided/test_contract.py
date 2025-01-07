import pytest

import torch

from e3nn import o3
from e3nn.util.test import assert_equivariant
from allegro.nn._strided import Contracter


@pytest.mark.parametrize("irreps_in1", ["0e + 0o + 1e + 1o", "2o + 1e + 0e"])
@pytest.mark.parametrize("irreps_in2", ["0e + 0o + 1e + 1o"])
@pytest.mark.parametrize("irreps_out", ["0e + 0o + 1e + 1o", "1o + 2e"])
@pytest.mark.parametrize(
    "mode,mul1,mul2,mulout",
    [
        ("uvw", 8, 8, 8),
        ("uvw", 8, 7, 3),
        ("uuu", 8, 8, 8),
        ("uvv", 8, 8, 8),
        ("uvv", 1, 8, 8),
        ("p", 3, 3, 3),
    ],
)
def test_contract(
    irreps_in1,
    irreps_in2,
    irreps_out,
    mode,
    mul1,
    mul2,
    mulout,
):
    irreps_in1 = o3.Irreps(irreps_in1)
    irreps_in2 = o3.Irreps(irreps_in2)
    irreps_out = o3.Irreps(irreps_out)
    instr = [
        (i_1, i_2, i_out)
        for i_1, (_, ir_1) in enumerate(irreps_in1)
        for i_2, (_, ir_2) in enumerate(irreps_in2)
        for i_out, (_, ir_out) in enumerate(irreps_out)
        if ir_out in ir_1 * ir_2
    ]
    c_base = Contracter(
        irreps_in1=o3.Irreps((mul1, ir) for _, ir in irreps_in1),
        irreps_in2=o3.Irreps((mul2, ir) for _, ir in irreps_in2),
        irreps_out=o3.Irreps((mulout, ir) for _, ir in irreps_out),
        instructions=instr,
        connection_mode=mode,
    )
    c_opt_mod = Contracter(
        irreps_in1=o3.Irreps((mul1, ir) for _, ir in irreps_in1),
        irreps_in2=o3.Irreps((mul2, ir) for _, ir in irreps_in2),
        irreps_out=o3.Irreps((mulout, ir) for _, ir in irreps_out),
        instructions=instr,
        connection_mode=mode,
    )
    with torch.no_grad():
        c_opt_mod.weights.copy_(c_base.weights)
    c_opt_mod = torch.jit.script(c_opt_mod)

    def c_opt(x, y, idx, dim, w=None):
        args = (x, y, idx, dim, w)
        if w is None:
            args = args[:-1]
        return c_opt_mod(*args)

    batchdim = 17
    scatter_dim = torch.tensor([batchdim], dtype=torch.long)
    scatter_idxs = torch.arange(batchdim)
    args_in = (
        irreps_in1.randn(batchdim, mul1, -1),
        irreps_in2.randn(batchdim, mul2, -1),
        scatter_idxs,
        scatter_dim,
        torch.randn(tuple(batchdim if e == -1 else e for e in c_base.weights.shape)),
    )
    args_in = args_in[:-1]

    for c in (c_base, c_opt):
        assert_equivariant(
            c,
            args_in=args_in,
            irreps_in=[irreps_in1, irreps_in2, None, None],
            irreps_out=irreps_out,
        )

    # Check grad
    if torch.get_default_dtype() == torch.float64:
        # check one input and a weight
        args_in[0].requires_grad_(True)
        torch.autograd.gradcheck(c_opt, args_in, fast_mode=True)
        args_in[0].requires_grad_(False)

    # Check same
    out_orig = c_base(*args_in)
    out_opt = c_opt(*args_in)
    assert torch.allclose(
        out_orig,
        out_opt[..., : out_orig.shape[-1]],
        atol={torch.float32: 1e-6, torch.float64: 1e-8}[torch.get_default_dtype()],
    )


def _strided_to_cat(irreps, mul, x):
    assert all(thismul == 1 for thismul, _ in irreps)
    # x is z u i
    assert x.ndim == 3
    assert x.shape[1] == mul
    # remove padding
    x = x[:, :, : irreps.dim]
    z = len(x)
    return torch.cat([x[:, :, s].reshape(z, -1) for s in irreps.slices()], dim=-1)


@pytest.mark.parametrize("irreps_in1", ["0e", "0e + 0o + 1e + 1o", "5o + 2e + 2o"])
@pytest.mark.parametrize("irreps_in2", ["0e", "0e + 0o + 1e + 1o", "2e + 4e + 2o"])
@pytest.mark.parametrize("irreps_out", ["0e", "0e + 0o + 1e + 1o", "1e + 5o + 3o"])
@pytest.mark.parametrize(
    "mode,mul1,mul2,mulout",
    [
        ("uuu", 8, 8, 8),
        ("uvw", 8, 8, 8),
        ("uvw", 8, 7, 3),
        ("uvv", 8, 8, 8),
        ("uvv", 1, 8, 8),
    ],
)
def test_like_tp(
    irreps_in1,
    irreps_in2,
    irreps_out,
    mode,
    mul1,
    mul2,
    mulout,
):
    irreps_in1 = o3.Irreps(irreps_in1)
    irreps_in2 = o3.Irreps(irreps_in2)
    irreps_out = o3.Irreps(irreps_out)
    instr = [
        (i_1, i_2, i_out)
        for i_1, (_, ir_1) in enumerate(irreps_in1)
        for i_2, (_, ir_2) in enumerate(irreps_in2)
        for i_out, (_, ir_out) in enumerate(irreps_out)
        if ir_out in ir_1 * ir_2
    ]
    if len(instr) == 0:
        pytest.skip("No instructions for this combo")
    c = Contracter(
        irreps_in1=o3.Irreps((mul1, ir) for _, ir in irreps_in1),
        irreps_in2=o3.Irreps((mul2, ir) for _, ir in irreps_in2),
        irreps_out=o3.Irreps((mulout, ir) for _, ir in irreps_out),
        instructions=instr,
        connection_mode=mode,
    )
    batchdim = 17
    scatter_dim = torch.tensor([batchdim], dtype=torch.long)
    scatter_idxs = torch.arange(batchdim)
    args_in = (
        torch.randn(batchdim, mul1, c.base_dim1),
        torch.randn(batchdim, mul2, c.base_dim2),
        scatter_idxs,
        scatter_dim,
        c.weights,
    )

    # TP
    tp = o3.TensorProduct(
        irreps_in1=o3.Irreps((mul1, ir) for _, ir in irreps_in1),
        irreps_in2=o3.Irreps((mul2, ir) for _, ir in irreps_in2),
        irreps_out=o3.Irreps((mulout, ir) for _, ir in irreps_out),
        instructions=[ins + (mode, True, 1.0) for ins in instr],
        shared_weights=True,
        internal_weights=False,
    )
    assert tp.weight_numel == c.weight_numel
    # to convert the weights, note that for Contracter
    # the weights are zuvwp. For TensorProduct, they are
    # catted uvw, so zpuvw
    weights_tp = args_in[-1]
    if len(instr) > 1:
        weights_tp = (
            weights_tp.detach()
            .reshape(c.weights.shape)  # zuvwp
            .permute(((-1,) + tuple(range(0, len(c.weights.shape) - 1))))
            .contiguous()
        )
    weights_tp = weights_tp.reshape(-1)
    args_in = args_in[:-1]

    args_tp = (
        _strided_to_cat(irreps_in1, mul1, args_in[0]),
        _strided_to_cat(irreps_in2, mul2, args_in[1]),
        weights_tp,
    )
    c_out = _strided_to_cat(irreps_out, mulout, c(*args_in))
    tp_out = tp(*args_tp)
    assert c_out.shape == tp_out.shape
    assert torch.allclose(
        c_out, tp_out, atol=2e-6 if torch.get_default_dtype() == torch.float32 else 1e-8
    )
