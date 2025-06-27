# This file is a part of the `allegro` package. Please see LICENSE and README at the root for information on using it.
import torch

import cuequivariance as cue
import cuequivariance_torch as cuet

from nequip.nn import scatter
from ._contract import Contracter

import itertools
from typing import Dict


def allegro_tp_desc(
    irreps1: cue.Irreps,
    irreps2: cue.Irreps,
    irreps3: cue.Irreps,
    tp_path_channel_coupling: bool,
) -> cue.EquivariantPolynomial:
    """Construct the Allegro version of channelwise tensor product descriptor.

    subscripts: ``weights[u],lhs[iu],rhs[ju],output[ku]``

    Args:
        irreps1 (Irreps): Irreps of the first operand.
        irreps2 (Irreps): Irreps of the second operand.
        irreps3 (Irreps): Irreps of the output to consider.
    """
    common_mul = irreps1[0].mul

    if tp_path_channel_coupling:
        d = cue.SegmentedTensorProduct.from_subscripts("u,iu,ju,ku+ijk")
    else:
        d = cue.SegmentedTensorProduct.from_subscripts(",iu,ju,ku+ijk")

    for mul, ir in irreps1:
        assert mul == common_mul
        d.add_segment(1, (ir.dim, mul))
    for mul, ir in irreps2:
        assert mul == common_mul
        d.add_segment(2, (ir.dim, mul))
    for mul, ir in irreps3:
        d.add_segment(3, (ir.dim, common_mul))

    for (i3, (mul3, ir3)), (i1, (mul1, ir1)), (i2, (mul2, ir2)) in itertools.product(
        enumerate(irreps3), enumerate(irreps1), enumerate(irreps2)
    ):
        if ir3 in ir1 * ir2:
            for cg in cue.clebsch_gordan(ir1, ir2, ir3):
                d.add_path(None, i1, i2, i3, c=cg)

    return cue.EquivariantPolynomial(
        [
            cue.IrrepsAndLayout(irreps1.new_scalars(d.operands[0].size), cue.ir_mul),
            cue.IrrepsAndLayout(irreps1, cue.ir_mul),
            cue.IrrepsAndLayout(irreps2, cue.ir_mul),
        ],
        [cue.IrrepsAndLayout(irreps3, cue.ir_mul)],
        cue.SegmentedPolynomial.eval_last_operand(d),
    )


class CuEquivarianceContracter(Contracter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        ir1 = cue.Irreps("O3", [(self.mul, ir) for _, ir in self.irreps_in1])
        ir2 = cue.Irreps("O3", [(self.mul, ir) for _, ir in self.irreps_in2])
        irout = cue.Irreps("O3", [(self.mul, ir) for _, ir in self.irreps_out])
        self.cuet_sp = cuet.SegmentedPolynomial(
            allegro_tp_desc(
                ir1, ir2, irout, self.path_channel_coupling
            ).polynomial.flatten_coefficient_modes(),
            # self.w3j is of `model_dtype`
            math_dtype=self.w3j.dtype,
        )

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        idxs: torch.Tensor,
        scatter_dim_size: int,
    ) -> torch.Tensor:

        # NOTE: the reason for some duplicated code is because TorchScript doesn't support super() calls
        # see https://github.com/pytorch/pytorch/issues/42885

        if self.scatter_factor is not None:
            x2 = self.scatter_factor * x2

        x2_scatter = scatter(
            x2,
            idxs,
            dim=0,
            dim_size=scatter_dim_size,
        )

        if x1.is_cuda and self.num_paths >= 1:
            empty_dict: Dict[int, torch.Tensor] = {}  # for torchscript

            if self.path_channel_coupling:
                weights = self.weights.transpose(0, 1).reshape(1, -1)
            else:
                weights = self.weights.reshape(1, -1)

            cue_out_edges = self.cuet_sp(
                [
                    weights,
                    x1.transpose(1, 2).reshape(x1.size(0), -1),  # (edges, irreps * mul)
                    x2_scatter.transpose(1, 2).reshape(
                        scatter_dim_size, -1
                    ),  # (atoms, irreps * mul)
                ],
                {2: idxs},  # input indices
                empty_dict,  # output shapes
                empty_dict,  # output indices
            )[0]

            # reshape and transpose back to (edges, mul, irreps_out)
            return cue_out_edges.view(
                -1, cue_out_edges.shape[-1] // self.mul, self.mul
            ).transpose(1, 2)
        else:
            x2 = torch.index_select(x2_scatter, 0, idxs)
            return self._contract(x1, x2)
