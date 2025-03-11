import pytest

import torch

from e3nn import o3
from e3nn.util.test import assert_equivariant
from nequip.utils import torch_default_dtype, dtype_from_name
from nequip.utils.versions import _TORCH_GE_2_6
from allegro.nn._strided import Contracter


@pytest.mark.parametrize("irreps_in1", ["0e + 0o + 1e + 1o", "2o + 1e + 0e"])
@pytest.mark.parametrize("irreps_in2", ["0e + 0o + 1e + 1o"])
@pytest.mark.parametrize("irreps_out", ["0e + 0o + 1e + 1o", "1o + 2e"])
@pytest.mark.parametrize("path_channel_coupling", [True, False])
@pytest.mark.parametrize("mul", [3, 8])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="kernel tests only work with GPUs"
)
@pytest.mark.skipif(
    not _TORCH_GE_2_6, reason="kernel tests only run for PyTorch >= 2.6"
)
def test_contract_kernel(
    irreps_in1,
    irreps_in2,
    irreps_out,
    path_channel_coupling,
    mul,
    dtype,
):
    assert torch.cuda.is_available()
    device = "cuda"

    with torch_default_dtype(dtype_from_name(dtype)):
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
            irreps_in1=o3.Irreps((1, ir) for _, ir in irreps_in1),
            irreps_in2=o3.Irreps((1, ir) for _, ir in irreps_in2),
            irreps_out=o3.Irreps((1, ir) for _, ir in irreps_out),
            mul=mul,
            instructions=instr,
            path_channel_coupling=path_channel_coupling,
            use_custom_kernels=False,
        ).to(device=device)
        c_kernel = Contracter(
            irreps_in1=o3.Irreps((1, ir) for _, ir in irreps_in1),
            irreps_in2=o3.Irreps((1, ir) for _, ir in irreps_in2),
            irreps_out=o3.Irreps((1, ir) for _, ir in irreps_out),
            mul=mul,
            instructions=instr,
            path_channel_coupling=path_channel_coupling,
            use_custom_kernels=True,
        ).to(device=device)

        with torch.no_grad():
            c_kernel.weights.copy_(c_base.weights)

        batchdim = 17
        scatter_dim = torch.tensor([batchdim], dtype=torch.long, device=device)
        scatter_idxs = torch.arange(batchdim, device=device)
        args_in = (
            irreps_in1.randn(batchdim, mul, -1, device=device),
            irreps_in2.randn(batchdim, mul, -1, device=device),
            scatter_idxs,
            scatter_dim,
        )

        for c in (c_base, c_kernel):
            assert_equivariant(
                c,
                args_in=args_in,
                irreps_in=[irreps_in1, irreps_in2, None, None],
                irreps_out=irreps_out,
            )

        atol = {torch.float32: 1e-6, torch.float64: 1e-10}[torch.get_default_dtype()]

        # check forward and gradients are the same with and without kernel
        for arg_idx in [0, 1]:
            args_in[arg_idx].requires_grad_(True)
            out_orig = c_base(*args_in)
            out_opt = c_kernel(*args_in)
            grad_output = torch.randn_like(out_opt.detach()).to(
                device=args_in[0].device
            )
            assert torch.allclose(out_orig, out_opt, atol=atol)

            grad_orig = torch.autograd.grad(out_orig, [args_in[arg_idx]], grad_output)[
                0
            ]
            grad_opt = torch.autograd.grad(out_opt, [args_in[arg_idx]], grad_output)[0]
            assert torch.allclose(grad_orig, grad_opt, atol=atol)
            args_in[arg_idx].requires_grad_(False)


@pytest.mark.parametrize("irreps_in1", ["0e + 0o + 1e + 1o", "2o + 1e + 0e"])
@pytest.mark.parametrize("irreps_in2", ["0e + 0o + 1e + 1o"])
@pytest.mark.parametrize("irreps_out", ["0e + 0o + 1e + 1o", "1o + 2e"])
@pytest.mark.parametrize("path_channel_coupling", [True, False])
@pytest.mark.parametrize("mul", [3, 8])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize(
    "device", ["cpu"] + ["cuda"] if torch.cuda.is_available() else []
)
def test_contract_jit(
    irreps_in1,
    irreps_in2,
    irreps_out,
    path_channel_coupling,
    mul,
    dtype,
    device,
):
    with torch_default_dtype(dtype_from_name(dtype)):
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
            irreps_in1=o3.Irreps((1, ir) for _, ir in irreps_in1),
            irreps_in2=o3.Irreps((1, ir) for _, ir in irreps_in2),
            irreps_out=o3.Irreps((1, ir) for _, ir in irreps_out),
            mul=mul,
            instructions=instr,
            path_channel_coupling=path_channel_coupling,
            use_custom_kernels=False,
        ).to(device=device)
        c_opt_mod = Contracter(
            irreps_in1=o3.Irreps((1, ir) for _, ir in irreps_in1),
            irreps_in2=o3.Irreps((1, ir) for _, ir in irreps_in2),
            irreps_out=o3.Irreps((1, ir) for _, ir in irreps_out),
            mul=mul,
            instructions=instr,
            path_channel_coupling=path_channel_coupling,
            use_custom_kernels=False,
        ).to(device=device)
        with torch.no_grad():
            c_opt_mod.weights.copy_(c_base.weights)
        c_opt_mod = torch.jit.script(c_opt_mod)

        def c_opt(x, y, idx, dim, w=None):
            args = (x, y, idx, dim, w)
            if w is None:
                args = args[:-1]
            return c_opt_mod(*args)

        batchdim = 17
        scatter_dim = torch.tensor([batchdim], dtype=torch.long, device=device)
        scatter_idxs = torch.arange(batchdim, device=device)
        args_in = (
            irreps_in1.randn(batchdim, mul, -1, device=device),
            irreps_in2.randn(batchdim, mul, -1, device=device),
            scatter_idxs,
            scatter_dim,
            torch.randn(
                tuple(batchdim if e == -1 else e for e in c_base.weights.shape)
            ),
        )
        args_in = args_in[:-1]

        for c in (c_base, c_opt):
            assert_equivariant(
                c,
                args_in=args_in,
                irreps_in=[irreps_in1, irreps_in2, None, None],
                irreps_out=irreps_out,
                # e3nn uses 1e-3, 1e-9
                tolerance={torch.float32: 1e-3, torch.float64: 1e-8}[
                    torch.get_default_dtype()
                ],
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
            atol={torch.float32: 1e-6, torch.float64: 1e-10}[torch.get_default_dtype()],
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
@pytest.mark.parametrize("mul", [3, 8])
@pytest.mark.parametrize("irrep_normalization", [None, "component"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize(
    "device", ["cpu"] + ["cuda"] if torch.cuda.is_available() else []
)
def test_like_tp(
    irreps_in1,
    irreps_in2,
    irreps_out,
    mul,
    irrep_normalization,
    dtype,
    device,
):
    """
    For compatibility with e3nn's TP, this test must use `path_channel_coupling` to match e3nn's "uuu" mode.
    """
    with torch_default_dtype(dtype_from_name(dtype)):
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
            irreps_in1=o3.Irreps((1, ir) for _, ir in irreps_in1),
            irreps_in2=o3.Irreps((1, ir) for _, ir in irreps_in2),
            irreps_out=o3.Irreps((1, ir) for _, ir in irreps_out),
            mul=mul,
            instructions=instr,
            path_channel_coupling=True,
            irrep_normalization=irrep_normalization,
            use_custom_kernels=False,
        ).to(device=device)
        print(c)
        # make input data
        batchdim = 1
        scatter_idxs = torch.arange(batchdim, device=device)
        scatter_dim = torch.tensor([batchdim], dtype=torch.long, device=device)
        tensor1 = torch.randn(batchdim, mul, c.base_dim1, device=device)
        tensor2 = torch.randn(batchdim, mul, c.base_dim2, device=device)

        # TP
        tp = o3.TensorProduct(
            irreps_in1=o3.Irreps((mul, ir) for _, ir in irreps_in1),
            irreps_in2=o3.Irreps((mul, ir) for _, ir in irreps_in2),
            irreps_out=o3.Irreps((mul, ir) for _, ir in irreps_out),
            instructions=[ins + ("uuu", True, 1.0) for ins in instr],
            irrep_normalization=(
                "none" if irrep_normalization is None else irrep_normalization
            ),
            path_normalization="none",
            shared_weights=True,
            internal_weights=False,
        ).to(device=device)
        print(tp)
        assert tp.weight_numel == c.weights.numel()
        # to convert the weights, note that for Contracter
        # the weights are zuvwp. For TensorProduct, they are
        # catted uvw, so zpuvw
        with torch.no_grad():
            weights_tp = c.weights.clone()
        if len(instr) > 1:
            # (u, p) -> (p, u)
            weights_tp = weights_tp.T
        # else weights are just (u,)
        weights_tp = weights_tp.reshape(-1)
        c_out = _strided_to_cat(
            irreps_out, mul, c(tensor1, tensor2, scatter_idxs, scatter_dim)
        )
        tp_out = tp(
            _strided_to_cat(irreps_in1, mul, tensor1),
            _strided_to_cat(irreps_in2, mul, tensor2),
            weights_tp,
        )
        assert c_out.shape == tp_out.shape
        max_err = torch.max(torch.abs(c_out - tp_out)).item()

        assert torch.allclose(
            c_out,
            tp_out,
            atol={torch.float32: 1e-6, torch.float64: 1e-10}[torch.get_default_dtype()],
        ), f"Max Error: {max_err}"
