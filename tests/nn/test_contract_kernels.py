import pytest

import torch

from e3nn import o3
from e3nn.util.test import assert_equivariant
from nequip.utils import torch_default_dtype, dtype_from_name
from nequip.utils.versions import _TORCH_GE_2_6
from allegro.nn._strided import Contracter

try:
    import triton  # noqa: F401

    _TRITON_INSTALLED = True
except ImportError:
    _TRITON_INSTALLED = False

try:
    import cuequivariance  # noqa: F401
    import cuequivariance_torch  # noqa: F401

    _CUEQ_INSTALLED = True
except ImportError:
    _CUEQ_INSTALLED = False


@pytest.mark.parametrize(
    "kernel_type",
    (["triton"] if (_TORCH_GE_2_6 and _TRITON_INSTALLED) else [])
    + (["cueq"] if _CUEQ_INSTALLED else []),
)
@pytest.mark.parametrize("irreps_in1", ["0e + 0o + 1e + 1o", "2o + 1e + 0e"])
@pytest.mark.parametrize("irreps_in2", ["0e + 0o + 1e + 1o"])
@pytest.mark.parametrize("irreps_out", ["0e + 0o + 1e + 1o", "1o + 2e"])
@pytest.mark.parametrize("path_channel_coupling", [True, False])
@pytest.mark.parametrize("mul", [3, 8])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="kernel tests only work with GPUs"
)
def test_contract_kernel(
    kernel_type,
    irreps_in1,
    irreps_in2,
    irreps_out,
    path_channel_coupling,
    mul,
    dtype,
):
    assert torch.cuda.is_available()
    device = "cuda"

    if kernel_type == "triton":
        from allegro.nn._strided._flashallegro import TritonContracter

        KernelClass = TritonContracter
    elif kernel_type == "cueq":
        from allegro.nn._strided._cueq_contracter import CuEquivarianceContracter

        KernelClass = CuEquivarianceContracter

    irreps_in1 = o3.Irreps(irreps_in1)
    irreps_in2 = o3.Irreps(irreps_in2)
    irreps_out = o3.Irreps(irreps_out)
    with torch_default_dtype(dtype_from_name(dtype)):
        c_base = Contracter(
            irreps_in1=o3.Irreps((1, ir) for _, ir in irreps_in1),
            irreps_in2=o3.Irreps((1, ir) for _, ir in irreps_in2),
            irreps_out=o3.Irreps((1, ir) for _, ir in irreps_out),
            mul=mul,
            path_channel_coupling=path_channel_coupling,
        ).to(device=device)
        c_kernel = KernelClass(
            irreps_in1=o3.Irreps((1, ir) for _, ir in irreps_in1),
            irreps_in2=o3.Irreps((1, ir) for _, ir in irreps_in2),
            irreps_out=o3.Irreps((1, ir) for _, ir in irreps_out),
            mul=mul,
            instructions=c_base.instructions,
            path_channel_coupling=path_channel_coupling,
        ).to(device=device)
        c_kernel.eval()

        with torch.no_grad():
            c_kernel.weights.copy_(c_base.weights)

        tol = {torch.float32: 1e-6, torch.float64: 1e-10}[torch.get_default_dtype()]
        torch.testing.assert_close(c_base.weights, c_kernel.weights, atol=tol, rtol=tol)

        num_edges = 17
        num_atoms = 5
        scatter_idxs = torch.randint(
            0, num_atoms, (num_edges,), dtype=torch.int64, device=device
        )
        args_in = (
            irreps_in1.randn(num_edges, mul, -1, device=device),
            irreps_in2.randn(num_edges, mul, -1, device=device),
            scatter_idxs,
            torch.tensor([num_atoms], dtype=torch.int64, device=device),
        )

        for c in (c_base, c_kernel):
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

        tol = {torch.float32: 1e-5, torch.float64: 1e-10}[torch.get_default_dtype()]

        # check forward and gradients are the same with and without kernel
        for arg_idx in [0, 1]:
            args_in[arg_idx].requires_grad_(True)
            out_orig = c_base(*args_in)
            out_opt = c_kernel(*args_in)
            grad_output = torch.randn_like(out_opt.detach()).to(
                device=args_in[0].device
            )
            torch.testing.assert_close(out_orig, out_opt, atol=tol, rtol=tol)

            grad_orig = torch.autograd.grad(out_orig, [args_in[arg_idx]], grad_output)[
                0
            ]
            grad_opt = torch.autograd.grad(out_opt, [args_in[arg_idx]], grad_output)[0]
            torch.testing.assert_close(grad_orig, grad_opt, atol=tol, rtol=tol)
            args_in[arg_idx].requires_grad_(False)
