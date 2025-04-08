# This file is a part of the `allegro` package. Please see LICENSE and README at the root for information on using it.
import math
import torch
from e3nn.o3._irreps import Irreps
from e3nn.o3._wigner import wigner_3j
from nequip.nn import scatter
from typing import List, Tuple, Optional


class Contracter(torch.nn.Module):
    """Contracter for strided tensor product.

    Args:
        irreps_in1: input irreps for LHS
        irreps_in2: input irreps for RHS
        irreps_out: output irreps
        instructions: list of tuples of ints, each tuple specifies the input
            irreps to contract together and the output irrep to put them in.
            If None, all possible paths among the inputs leading to all outputs
            will be computed.
        path_channel_coupling: whether the weights provide path-channel couplings
    """

    _weight_w3j_einstr: str
    _contract_einstr: str
    mul: int
    base_dim1: int
    base_dim2: int
    base_dim_out: int
    num_paths: int

    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        irreps_out,
        mul: int,
        instructions: Optional[List[Tuple[int, int, int]]] = None,
        path_channel_coupling: bool = False,  # i.e. "p" vs "uuup" mode
        scatter_factor: Optional[float] = None,
        irrep_normalization: str = "component",
        use_custom_kernels: bool = False,
    ):
        super().__init__()

        # optional scatter factor (for fused scatter + index_select)
        self.scatter_factor = scatter_factor

        # -- Instruction management --
        if instructions is None:
            instructions = []
            for i_out, (_, ir_out) in enumerate(irreps_out):
                for i_1, (_, ir_1) in enumerate(irreps_in1):
                    for i_2, (_, ir_2) in enumerate(irreps_in2):
                        if ir_out in ir_1 * ir_2:
                            instructions.append((i_1, i_2, i_out))

        # -- Irrep management --
        assert mul > 0
        self.irreps_in1 = Irreps(irreps_in1)
        base_irreps1 = Irreps((1, ir) for _, ir in self.irreps_in1)
        dim1 = base_irreps1.dim
        assert all(m == 1 for m, ir in self.irreps_in1)
        self.irreps_in2 = Irreps(irreps_in2)
        base_irreps2 = Irreps((1, ir) for _, ir in self.irreps_in2)
        dim2 = base_irreps2.dim
        assert all(m == 1 for m, ir in irreps_in2)
        self.irreps_out = Irreps(irreps_out)
        base_irreps_out = Irreps((1, ir) for _, ir in self.irreps_out)
        dimout = base_irreps_out.dim
        assert all(m == 1 for m, ir in self.irreps_out)
        self.base_dim1 = dim1
        self.base_dim2 = dim2
        self.base_dim_out = dimout
        self.mul = mul
        self.num_paths: int = len(instructions)
        assert self.num_paths > 0, "No TP paths available"

        # -- Make the w3j --
        # list of tensors of shape [N, 3] containing i,j,k indexes
        w3j_index: List[torch.Tensor] = []
        # list of tensors of shape [N] containing w3j values
        w3j_values: List[torch.Tensor] = []

        for ins_i, ins in enumerate(instructions):
            ir_in1 = base_irreps1[ins[0]].ir
            ir_in2 = base_irreps2[ins[1]].ir
            ir_out = base_irreps_out[ins[2]].ir

            # Check instruction against the O3 selection rules
            assert ir_in1.p * ir_in2.p == ir_out.p
            assert abs(ir_in1.l - ir_in2.l) <= ir_out.l <= ir_in1.l + ir_in2.l

            this_w3j = wigner_3j(ir_in1.l, ir_in2.l, ir_out.l)
            this_w3j_index = this_w3j.nonzero()
            w3j_values.append(
                this_w3j[
                    this_w3j_index[:, 0], this_w3j_index[:, 1], this_w3j_index[:, 2]
                ]
            )

            # Normalize the path through multiplying normalization constant with w3j
            # "component" normalization
            # TODO what is the correct backwards normalization here?
            if irrep_normalization is None:
                w3j_norm_term = 1
            elif irrep_normalization == "component":
                w3j_norm_term = math.sqrt(2 * ir_out.l + 1)
            else:
                raise NotImplementedError(
                    f"`{irrep_normalization}` `irrep_normalization` is not implemented"
                )
            w3j_values[-1].mul_(w3j_norm_term)

            this_w3j_index[:, 0] += base_irreps1[: ins[0]].dim
            this_w3j_index[:, 1] += base_irreps2[: ins[1]].dim
            this_w3j_index[:, 2] += base_irreps_out[: ins[2]].dim
            w3j_index.append(this_w3j_index)
            del ir_in1, ir_in2, ir_out, w3j_norm_term, this_w3j, this_w3j_index

        # for every path, all i indexes are equal to all j indexes?
        # since these are coordinates of non-zero entries, if all nonzero entries
        # have i == j, then the matrix is diagonal
        # since we'll sum over the paths, if every path is diagonal, then the sum is diagonal
        self.w3j_is_ij_diagonal: bool = (base_irreps1.dim == base_irreps2.dim) and all(
            torch.all(e[:, 0] == e[:, 1]) for e in w3j_index
        )
        # in this case we are only taking diagonal (i == j)
        # entries from the outer product; but those values are just
        # the direct multiplication of the two tensors, eliminating
        # the need for computing the full outer product.

        if self.w3j_is_ij_diagonal:
            # pik
            w3j = torch.zeros(self.num_paths, base_irreps1.dim, base_irreps_out.dim)
            for path_index, (path_w3j_indexes, path_w3j_values) in enumerate(
                zip(w3j_index, w3j_values)
            ):
                w3j[
                    path_index,  # p
                    path_w3j_indexes[:, 0],  # i
                    path_w3j_indexes[:, 2],  # k
                ] = path_w3j_values
        else:
            # pijk
            w3j = torch.zeros(
                self.num_paths,
                base_irreps1.dim,
                base_irreps2.dim,
                base_irreps_out.dim,
            )

            for path_index, (path_w3j_indexes, path_w3j_values) in enumerate(
                zip(w3j_index, w3j_values)
            ):
                w3j[
                    path_index,  # p
                    path_w3j_indexes[:, 0],  # i
                    path_w3j_indexes[:, 1],  # j
                    path_w3j_indexes[:, 2],  # k
                ] = path_w3j_values

        # remove path dims of w3j if there's only one path
        if self.num_paths == 1:
            w3j = w3j.squeeze(dim=0)  # ik or ijk
        self.register_buffer("w3j", w3j)

        # === path mixing weights ===
        # "p" mode (p,) weights vs "uuup" mode (u,p) weights
        self.path_channel_coupling = path_channel_coupling
        weight_shape = (self.mul,) if self.path_channel_coupling else tuple()
        if self.num_paths > 1:
            weight_shape = weight_shape + (self.num_paths,)
        self.weights = torch.nn.Parameter(torch.randn(weight_shape))
        torch.nn.init.uniform_(self.weights, -math.sqrt(3), math.sqrt(3))

        # === get ww3j einstring ===
        ij = "i" if self.w3j_is_ij_diagonal else "ij"
        p = "p" if self.num_paths > 1 else ""
        u = "u" if self.path_channel_coupling else ""
        self._weight_w3j_einstr = f"{u}{p},{p}{ij}k->{u}{ij}k"

        # === condition for triggering kernel code path ===
        self.use_custom_kernels = (
            use_custom_kernels and not self.w3j_is_ij_diagonal and self.num_paths > 1
        )
        if self.use_custom_kernels:
            from ._flashallegro import FlashAllegroKernel

            # set up kernel
            self.custom_kernel = FlashAllegroKernel(
                w3j=self.w3j,
                path_channel_coupling=self.path_channel_coupling,
                base_dim_out=self.base_dim_out,
                base_dim1=self.base_dim1,
                base_dim2=self.base_dim2,
            )
        else:
            # we need the following to placate TorchScript
            self.custom_kernel = _dummy_custom_kernel

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        idxs: torch.Tensor,
        scatter_dim_size: int,
    ) -> torch.Tensor:
        # === optional scatter + index_select ===
        # normalize if normalization provided

        if self.scatter_factor is not None:
            x2 = self.scatter_factor * x2

        # scatter and index select
        x2_scatter = scatter(
            x2,
            idxs,
            dim=0,
            dim_size=scatter_dim_size,
        )
        x2 = torch.index_select(x2_scatter, 0, idxs)

        # === perform TP ===
        # convert to strided shape
        x1 = x1.reshape(-1, self.mul, self.base_dim1)
        x2 = x2.reshape(-1, self.mul, self.base_dim2)

        # take custom kernel path at runtime if kernel conditions are met at init and the input is on the GPU
        if self.use_custom_kernels and x1.is_cuda:
            out = self.custom_kernel(x1, x2, self.weights)
        else:
            # for shared weights, we can precontract weights and w3j so they can be frozen together
            # this is usually advantageous for inference, since the weights would have to be
            # multiplied in anyway at some point
            # `up, pijk -> uijk`` or `p, pijk -> ijk`
            if self.num_paths >= 1:
                ww3j = torch.einsum(self._weight_w3j_einstr, self.weights, self.w3j)
            else:
                # account for `_, ijk -> ijk`, i.e. single path case
                ww3j = self.w3j

            # now do the TP with the pre-contracted w3j
            if self.w3j_is_ij_diagonal:
                # zui, zui -> zui
                outer = x1 * x2
                if self.path_channel_coupling:
                    # zui1, uik -> zuk
                    out = torch.sum(outer.unsqueeze(-1) * ww3j, 2)
                else:
                    # zui, ik -> zuk
                    out = torch.mm(outer.view(-1, outer.size(2)), ww3j).view(
                        outer.size(0), outer.size(1), ww3j.size(1)
                    )
            else:
                # zui, zuj -> zuij
                outer = x1.unsqueeze(-1) * x2.unsqueeze(-2)
                if self.path_channel_coupling:
                    # zuij, uijk -> zuk
                    out = torch.sum(outer.unsqueeze(-1) * ww3j, (2, 3))
                else:
                    # (zu)(ij), (ij)k -> (zu)k -> zuk
                    out = torch.mm(
                        outer.view(
                            outer.size(0) * outer.size(1), outer.size(2) * outer.size(3)
                        ),
                        ww3j.view(-1, ww3j.size(2)),
                    ).view(-1, self.mul, ww3j.size(2))
        return out

    def extra_repr(self):
        return f"{self.irreps_in1} x {self.irreps_in2} -> {self.irreps_out} | {self.mul} channels | {self.num_paths} paths"


@torch.jit.unused
def _dummy_custom_kernel(
    x1: torch.Tensor, x2: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    return torch.Tensor()
