from math import ceil

import torch

from e3nn.util.jit import compile_mode


@compile_mode("script")
class MakeWeightedChannels(torch.nn.Module):
    weight_numel: int
    multiplicity_out: int
    _num_irreps: int

    def __init__(
        self,
        irreps_in,
        multiplicity_out: int,
        pad_to_alignment: int = 1,
    ):
        super().__init__()
        assert all(mul == 1 for mul, ir in irreps_in)
        assert multiplicity_out >= 1
        # Each edgewise output multiplicity is a per-irrep weighted sum over the input
        # So we need to apply the weight for the ith irrep to all DOF in that irrep
        w_index = sum(([i] * ir.dim for i, (mul, ir) in enumerate(irreps_in)), [])
        # pad to padded length
        n_pad = (
            int(ceil(irreps_in.dim / pad_to_alignment)) * pad_to_alignment
            - irreps_in.dim
        )
        # use the last weight, what we use doesn't matter much
        w_index += [w_index[-1]] * n_pad
        self._num_irreps = len(irreps_in)
        self.register_buffer("_w_index", torch.as_tensor(w_index, dtype=torch.long))
        # there is
        self.multiplicity_out = multiplicity_out
        self.weight_numel = len(irreps_in) * multiplicity_out

    def forward(self, edge_attr, weights):
        # weights are [z, u, i]
        # edge_attr are [z, i]
        # i runs over all irreps, which is why the weights need
        # to be indexed in order to go from [num_i] to [i]
        return torch.einsum(
            "zi,zui->zui",
            edge_attr,
            weights.view(
                -1,
                self.multiplicity_out,
                self._num_irreps,
            )[:, :, self._w_index],
        )
