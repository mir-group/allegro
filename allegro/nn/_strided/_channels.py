# This file is a part of the `allegro` package. Please see LICENSE and README at the root for information on using it.
import torch

from e3nn.util.jit import compile_mode


@compile_mode("script")
class MakeWeightedChannels(torch.nn.Module):
    weight_numel: int
    multiplicity_out: int
    weight_individual_irreps: bool
    alpha: float
    _num_irreps: int

    def __init__(
        self,
        irreps_in,
        multiplicity_out: int,
        alpha: float = 1.0,
        weight_individual_irreps: bool = True,
    ):
        super().__init__()
        assert all([mul == 1 for mul, ir in irreps_in])
        assert multiplicity_out >= 1
        self._num_irreps = len(irreps_in)
        self.multiplicity_out = multiplicity_out
        self.weight_individual_irreps = weight_individual_irreps
        self.alpha = alpha
        if not weight_individual_irreps:
            self.weight_numel = multiplicity_out
            self.register_buffer("_rtoi", torch.Tensor())
            return
        self.weight_numel = len(irreps_in) * multiplicity_out
        # Each edgewise output multiplicity is a per-irrep weighted sum over the input
        # So we need to apply the weight for the ith irrep to all DOF in that irrep
        rtoi = torch.zeros(self._num_irreps, irreps_in.dim)
        for i, this_slice in enumerate(irreps_in.slices()):
            rtoi[i, this_slice] = alpha
        self.register_buffer("_rtoi", rtoi, persistent=False)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(\n  num_irreps: {self._num_irreps}\n  multiplicity_out: {self.multiplicity_out}\n  weight_numel: {self.weight_numel}\n)"

    def forward(self, edge_attr, weights):
        if self.weight_individual_irreps:
            # weights are [z, u, r]
            # edge_attr are [z, i]
            # r runs over all irreps, which is why the weights need
            # to be indexed in order to go from [r] to [i]
            # [zu]r @ ri -> [zu]i -> zui
            aux = torch.mm(weights.reshape(-1, self._num_irreps), self._rtoi).view(
                edge_attr.size(0),
                self.multiplicity_out,
                self._rtoi.shape[1],
            )
            # zi,zui->zui
            return edge_attr.unsqueeze(1) * aux

        else:
            # weights are [z, u]
            # edge_attr are [z, i]
            # [z, u, 1] * [z, 1, i]
            return weights.unsqueeze(-1) * (self.alpha * edge_attr.unsqueeze(-2))
