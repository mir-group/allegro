from typing import List, Tuple
from math import sqrt

import torch

from e3nn import o3
from e3nn.util import prod

from ._layout import StridedLayout
from .._misc import _init_weight


class Contracter(torch.nn.Module):
    _weight_w3j_einstr: str
    _contract_einstr: str
    mul1: int
    mul2: int
    mulout: int
    base_dim1: int
    base_dim2: int
    base_dim_out: int
    weight_numel: int

    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        irreps_out,
        instructions: List[Tuple[int, int, int]],
        connection_mode: str,
        initialization: str = "uniform",
        normalization: str = "component",
    ):
        super().__init__()
        # -- Irrep management --
        irreps_in1 = o3.Irreps(irreps_in1)
        assert all(mul == irreps_in1[0].mul for mul, ir in irreps_in1)
        irreps_in2 = o3.Irreps(irreps_in2)
        assert all(mul == irreps_in2[0].mul for mul, ir in irreps_in2)
        irreps_out = o3.Irreps(irreps_out)
        assert all(mul == irreps_out[0].mul for mul, ir in irreps_out)
        layout_in1 = StridedLayout(irreps_in1)
        layout_in2 = StridedLayout(irreps_in2)
        layout_out = StridedLayout(irreps_out)
        self.mul1, self.base_dim1 = layout_in1.mul, layout_in1.base_dim
        self.mul2, self.base_dim2 = layout_in2.mul, layout_in2.base_dim
        self.mulout, self.base_dimout = layout_out.mul, layout_out.base_dim
        num_paths: int = len(instructions)

        # -- Make the w3j --
        w3j_index = []
        w3j_values = []

        for ins_i, ins in enumerate(instructions):
            mul_ir_in1 = layout_in1.base_irreps[ins[0]]
            mul_ir_in2 = layout_in2.base_irreps[ins[1]]
            mul_ir_out = layout_out.base_irreps[ins[2]]

            # Check instruction against the symmetric selection rules
            assert mul_ir_in1.ir.p * mul_ir_in2.ir.p == mul_ir_out.ir.p
            assert (
                abs(mul_ir_in1.ir.l - mul_ir_in2.ir.l)
                <= mul_ir_out.ir.l
                <= mul_ir_in1.ir.l + mul_ir_in2.ir.l
            )

            if mul_ir_in1.dim == 0 or mul_ir_in2.dim == 0 or mul_ir_out.dim == 0:
                raise ValueError

            this_w3j = o3.wigner_3j(mul_ir_in1.ir.l, mul_ir_in2.ir.l, mul_ir_out.ir.l)
            this_w3j_index = this_w3j.nonzero()
            w3j_values.append(
                this_w3j[
                    this_w3j_index[:, 0], this_w3j_index[:, 1], this_w3j_index[:, 2]
                ]
            )

            # Normalize the path through multiplying normalization constant with w3j
            if normalization == "component":
                w3j_norm_term = 2 * mul_ir_out.ir.l + 1
            elif normalization == "norm":
                w3j_norm_term = (2 * mul_ir_in1.ir.l + 1) * (2 * mul_ir_in2.ir.l + 1)
            else:
                raise ValueError
            alpha = sqrt(
                w3j_norm_term
                # Channel-mixing sum normalization term:
                / sum(
                    {
                        "uvw": (layout_in1.mul * layout_in2.mul),
                        "uvu": layout_in2.mul,
                        "uvv": layout_in1.mul,
                        "uuw": layout_in1.mul,
                        "uuu": 1,
                        "uvuv": 1,
                        # p means just to weight paths, so no change to normalization per-path
                        "p": 1,
                    }[connection_mode]
                    for i in instructions
                    if i[2] == ins[2]
                )
            )
            w3j_values[-1].mul_(alpha)

            this_w3j_index[:, 0] += layout_in1.base_irreps[: ins[0]].dim
            this_w3j_index[:, 1] += layout_in2.base_irreps[: ins[1]].dim
            this_w3j_index[:, 2] += layout_out.base_irreps[: ins[2]].dim
            # Now need to flatten the index to be for [pk][ij]
            w3j_index.append(
                torch.cat(
                    (
                        ins_i * layout_out.base_dim  # unweighted all go in first path
                        + this_w3j_index[:, 2].unsqueeze(-1),
                        this_w3j_index[:, 0].unsqueeze(-1) * layout_in2.base_dim
                        + this_w3j_index[:, 1].unsqueeze(-1),
                    ),
                    dim=1,
                )
            )
        del mul_ir_in1, mul_ir_in2, mul_ir_out, w3j_norm_term, this_w3j, this_w3j_index

        w3j = torch.sparse_coo_tensor(
            indices=torch.cat(w3j_index, dim=0).t(),
            values=torch.cat(w3j_values, dim=0),
            size=(
                num_paths * layout_out.base_dim,
                layout_in1.base_dim * layout_in2.base_dim,
            ),
        ).coalesce()

        # w3j is k,i,j, so this is whether, for nonzero entries,
        # the i index is always equal to the j index. If so, then
        # it is diagonal and we can eliminate the j dimension
        # in this case we are only taking diagonal (i == j)
        # entries from the outer product; but those values are just
        # the direct multiplication of the two tensors, eliminating
        # the need for the outer product.
        # obviously this only makes sense if they have the same size as well
        # this is more or less a test of whether this TP is an inner product
        w3j_i_indexes = torch.div(
            w3j.indices()[1], layout_in1.base_dim, rounding_mode="floor"
        )
        w3j_j_indexes = w3j.indices()[1] % layout_in1.base_dim
        w3j_is_ij_diagonal: bool = (
            layout_in1.base_dim == layout_in2.base_dim
        ) and torch.all(w3j_i_indexes == w3j_j_indexes)
        if w3j_is_ij_diagonal:
            # change the w3j to eliminate the dimension
            # now its just k,i
            w3j = torch.sparse_coo_tensor(
                indices=torch.stack((w3j.indices()[0], w3j_i_indexes)),
                values=w3j.values(),
                size=(
                    num_paths * layout_out.base_dim,
                    layout_in1.base_dim,
                ),
            )

        # in dense, must shape it for einsum:
        if w3j_is_ij_diagonal:
            kij_shape = (
                layout_out.base_dim,
                layout_in1.base_dim,
            )
        else:
            kij_shape = (
                layout_out.base_dim,
                layout_in1.base_dim,
                layout_in2.base_dim,
            )
        w3j = (
            w3j.to_dense()
            .reshape(((num_paths,) if num_paths > 1 else tuple()) + kij_shape)
            .contiguous()
        )
        del kij_shape
        self.register_buffer("w3j", w3j)

        # -- Make the channel mixing weights --
        u, v, w = "uuu" if connection_mode == "p" else connection_mode
        weight_label = {"uvw": "uvw", "uuu": "u", "uvv": "uv", "p": ""}[connection_mode]

        weight_shape = {
            "uvw": (layout_in1.mul, layout_in2.mul, layout_out.mul),
            "uuu": (layout_in1.mul,),
            "uvv": (layout_in1.mul, layout_in2.mul),
            "p": tuple(),
        }[connection_mode]
        if connection_mode == "p":
            assert num_paths > 1
        if num_paths > 1:
            # ^ if there's only one weighted path, the einsum simplifies without the p dimension
            weight_label = weight_label + "p"
            weight_shape = weight_shape + (num_paths,)

        self.weight_numel = abs(prod(weight_shape))
        self.weights = torch.nn.Parameter(torch.randn(weight_shape))
        _init_weight(self.weights, initialization=initialization)

        # -- Prepare the einstrings --
        j = "i" if w3j_is_ij_diagonal else "j"
        ij = "i" if w3j_is_ij_diagonal else "ij"
        p = "p" if num_paths > 1 else ""
        self._weight_w3j_einstr = (
            f"{weight_label},{p}k{ij}->{weight_label.rstrip('p')}k{ij}"
        )
        # note that PyTorch appears to contract left-to-right by default in C++
        # (which is all we get for the TorchScript backend, the default opt_einsum
        # support does not apply):
        # https://github.com/pytorch/pytorch/blob/ad39a2fc462fd14ad5442d2f21eed1d2c34a20eb/aten/src/ATen/native/Linear.cpp#L551-L552
        # We arange the einstr to do in order:
        #  zui,zuj->zuij (outer product)
        #  zuij,ijk->zuk  (matmul)
        self._contract_einstr = f"z{u}i,z{v}{j},{weight_label.rstrip('p')}k{ij}->z{w}k"

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # convert to strided shape
        x1 = x1.reshape(-1, self.mul1, self.base_dim1)
        x2 = x2.reshape(-1, self.mul2, self.base_dim2)
        # for shared weights, we can precontract weights and w3j so they can be frozen together
        # this is usually advantageous for inference, since the weights would have to be
        # multiplied in anyway at some point
        ww3j = torch.einsum(self._weight_w3j_einstr, self.weights, self.w3j)
        # now do the TP with the pre-contracted w3j
        out = torch.einsum(self._contract_einstr, x1, x2, ww3j)
        return out
