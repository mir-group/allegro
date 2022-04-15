from typing import Tuple
from packaging import version

import torch

from e3nn.util.jit import compile_mode

_USE_PYG_SPARSE: bool = False

_TORCH_IS_GE_1_10: bool = version.parse(torch.__version__) >= version.parse("1.10.0")

if not _USE_PYG_SPARSE:

    class _ExplicitGradSpmm(torch.autograd.Function):
        @staticmethod
        def forward(ctx, sparse, a):
            ctx.save_for_backward(sparse)
            return torch.mm(sparse, a)

        @staticmethod
        def backward(ctx, grad_output):
            (sparse,) = ctx.saved_tensors
            return None, torch.mm(sparse.t(), grad_output)

    # TODO: support csr with similar method; wait for 1.10 probably
    @torch.jit.script
    def _remake_sparse_coo(i, v, shape: Tuple[int, int]):
        out = torch.sparse_coo_tensor(
            indices=i, values=v, size=shape, device=v.device, dtype=v.dtype
        )
        # mark it as coalesced, cause it is from when we build it in
        # ExplicitGradSpmm's __init__
        out._coalesced_(True)  # undocumented, AFAIK
        assert out.is_coalesced()
        return out

    @compile_mode("trace")
    class ExplicitGradSpmmCOO(torch.nn.Module):
        shape: Tuple[int, int]

        def __init__(self, mat: torch.Tensor):
            super().__init__()
            assert mat.is_sparse
            assert mat.ndim == 2
            mat = mat.coalesce()
            # To workaround https://github.com/pytorch/pytorch/issues/63987,
            # save indices and values explicitly
            self.register_buffer("_i", mat.indices())
            self.register_buffer("_v", mat.values())
            self.shape = tuple(mat.shape)

        def forward(self, x):
            # TODO: support csr
            sp = _remake_sparse_coo(self._i, self._v, self.shape)
            if self.training:
                # Use a custom autograd function for 2nd derivatives
                # torch.mm doesn't do double derivatives with sparse w3j
                tmp = _ExplicitGradSpmm.apply(sp, x)
            else:
                # For inference, assume only first derivatives necessary
                tmp = torch.mm(sp, x)
            return tmp

        def _make_tracing_inputs(self, n: int):
            return [
                {
                    "forward": (
                        torch.randn(
                            self.shape[-1],
                            3,
                            device=self._v.device,
                            dtype=self._v.dtype,
                        ),
                    )
                }
                for _ in range(n)
            ]

    if _TORCH_IS_GE_1_10:

        @torch.jit.script
        def _remake_sparse_csr(crow, col, v, shape: Tuple[int, int]) -> torch.Tensor:
            return torch.sparse_csr_tensor(
                crow_indices=crow,
                col_indices=col,
                values=v,
                size=shape,
                layout=torch.sparse_csr,
                device=v.device,
                dtype=v.dtype,
            )

        @compile_mode("trace")
        class ExplicitGradSpmmCSR(torch.nn.Module):
            shape: Tuple[int, int]

            def __init__(self, mat: torch.Tensor):
                super().__init__()
                assert mat.is_sparse_csr
                assert mat.ndim == 2
                # To workaround https://github.com/pytorch/pytorch/issues/63987,
                # save indices and values explicitly
                self.register_buffer("_crow", mat.crow_indices())
                self.register_buffer("_col", mat.col_indices())
                self.register_buffer("_v", mat.values())
                self.shape = tuple(mat.shape)

            def forward(self, x):
                # TODO: support csr
                sp = _remake_sparse_csr(self._crow, self._col, self._v, self.shape)
                if self.training:
                    # Use a custom autograd function for 2nd derivatives
                    # torch.mm doesn't do double derivatives with sparse w3j
                    tmp = _ExplicitGradSpmm.apply(sp, x)
                else:
                    # For inference, assume only first derivatives necessary
                    tmp = torch.mm(sp, x)
                return tmp

            def _make_tracing_inputs(self, n: int):
                return [
                    {
                        "forward": (
                            torch.randn(
                                self.shape[-1],
                                3,
                                device=self._v.device,
                                dtype=self._v.dtype,
                            ),
                        )
                    }
                    for _ in range(n)
                ]

    def ExplicitGradSpmm(mat):
        if mat.is_sparse:
            return ExplicitGradSpmmCOO(mat)
        elif _TORCH_IS_GE_1_10 and mat.is_sparse_csr:
            return ExplicitGradSpmmCSR(mat)
        else:
            raise TypeError

else:  # _USE_PYG_SPARSE

    from torch_sparse import SparseTensor
    from torch_sparse.matmul import spmm_add

    class ExplicitGradSpmm(torch.nn.Module):
        def __init__(self, mat):
            super().__init__()
            self.mat = SparseTensor.from_dense(mat.to_dense())

        def forward(self, x):
            return spmm_add(self.mat, x)
