import math
import torch
from e3nn import o3
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
        self.irreps_in1 = o3.Irreps(irreps_in1)
        base_irreps1 = o3.Irreps((1, ir) for _, ir in self.irreps_in1)
        dim1 = base_irreps1.dim
        assert all(m == 1 for m, ir in self.irreps_in1)
        self.irreps_in2 = o3.Irreps(irreps_in2)
        base_irreps2 = o3.Irreps((1, ir) for _, ir in self.irreps_in2)
        dim2 = base_irreps2.dim
        assert all(m == 1 for m, ir in irreps_in2)
        self.irreps_out = o3.Irreps(irreps_out)
        base_irreps_out = o3.Irreps((1, ir) for _, ir in self.irreps_out)
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

            this_w3j = o3.wigner_3j(ir_in1.l, ir_in2.l, ir_out.l)
            this_w3j_index = this_w3j.nonzero()
            w3j_values.append(
                this_w3j[
                    this_w3j_index[:, 0], this_w3j_index[:, 1], this_w3j_index[:, 2]
                ]
            )

            # Normalize the path through multiplying normalization constant with w3j
            # "component" normalization
            # TODO what is the correct backwards normalization here?
            w3j_norm_term = math.sqrt(2 * ir_out.l + 1)
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
        w3j_is_ij_diagonal: bool = (base_irreps1.dim == base_irreps2.dim) and all(
            torch.all(e[:, 0] == e[:, 1]) for e in w3j_index
        )
        # in this case we are only taking diagonal (i == j)
        # entries from the outer product; but those values are just
        # the direct multiplication of the two tensors, eliminating
        # the need for computing the full outer product.

        if w3j_is_ij_diagonal:
            # i k p
            w3j = torch.zeros(base_irreps1.dim, base_irreps_out.dim, self.num_paths)
            for path_index, (path_w3j_indexes, path_w3j_values) in enumerate(
                zip(w3j_index, w3j_values)
            ):
                w3j[
                    path_w3j_indexes[:, 0],  # i
                    path_w3j_indexes[:, 2],  # k
                    path_index,  # p
                ] = path_w3j_values
        else:
            # i j k p
            w3j = torch.zeros(
                base_irreps1.dim, base_irreps2.dim, base_irreps_out.dim, self.num_paths
            )
            for path_index, (path_w3j_indexes, path_w3j_values) in enumerate(
                zip(w3j_index, w3j_values)
            ):
                w3j[
                    path_w3j_indexes[:, 0],  # i
                    path_w3j_indexes[:, 1],  # j
                    path_w3j_indexes[:, 2],  # k
                    path_index,  # p
                ] = path_w3j_values

        # remove path dims of w3j if there's only one path
        if self.num_paths == 1:
            w3j = w3j.squeeze(dim=-1)
        self.register_buffer("w3j", w3j)

        # -- Make the path mixing weights --
        # "p" mode (p,) weights vs "uuup" mode (u,p) weights
        if path_channel_coupling:
            weight_label = "u"
            weight_shape = (self.mul,)
        else:
            weight_label = ""
            weight_shape = tuple()
        if self.num_paths > 1:
            # either "p" or "up"
            weight_label = weight_label + "p"
            weight_shape = weight_shape + (self.num_paths,)
        self.weights = torch.nn.Parameter(torch.randn(weight_shape))
        with torch.no_grad():
            init_range = (
                math.sqrt(3 / self.mul) if path_channel_coupling else math.sqrt(3)
            )
            self.weights.uniform_(-init_range, init_range)
            del init_range

        # -- Prepare the einstrings --
        j = "i" if w3j_is_ij_diagonal else "j"
        ij = "i" if w3j_is_ij_diagonal else "ij"
        p = "p" if self.num_paths > 1 else ""
        self._weight_w3j_einstr = (
            f"{ij}k{p},{weight_label}->{weight_label.rstrip('p')}{ij}k"
        )
        # note that PyTorch appears to contract left-to-right by default in C++
        # (which is all we get for the TorchScript backend, the default opt_einsum
        # support does not apply):
        # https://github.com/pytorch/pytorch/blob/ad39a2fc462fd14ad5442d2f21eed1d2c34a20eb/aten/src/ATen/native/Linear.cpp#L551-L552
        # We arange the einstr to do in order:
        #  zui,zuj->zuij (outer product)
        #  zuij,(u)ijk->zuk  (matmul)
        self._outer_einstr = f"zui,zu{j}->zu{ij}"
        self._matmul_einstr = f"zu{ij},{weight_label.rstrip('p')}{ij}k->zuk"

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
        # for shared weights, we can precontract weights and w3j so they can be frozen together
        # this is usually advantageous for inference, since the weights would have to be
        # multiplied in anyway at some point
        ww3j = torch.einsum(self._weight_w3j_einstr, self.w3j, self.weights)
        # now do the TP with the pre-contracted w3j
        outer = torch.einsum(self._outer_einstr, x1, x2)
        out = torch.einsum(self._matmul_einstr, outer, ww3j)
        return out

    def extra_repr(self):
        return f"{self.irreps_in1} x {self.irreps_in2} -> {self.irreps_out} | {self.mul} channels | {self.num_paths} paths"
