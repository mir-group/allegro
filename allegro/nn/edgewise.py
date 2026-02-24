# This file is a part of the `allegro` package. Please see LICENSE and README at the root for information on using it.
import torch

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin, scatter, AvgNumNeighborsNorm

from typing import Optional, Union, Dict, Sequence
from math import sqrt


class EdgewiseReduce(GraphModuleMixin, torch.nn.Module):
    """Like ``nequip.nn.AtomwiseReduce``, but accumulating per-edge data into per-atom data."""

    _factor: Optional[float]

    def __init__(
        self,
        field: str,
        out_field: Optional[str] = None,
        avg_num_neighbors: Union[float, Dict[str, float]] = None,
        type_names: Sequence[str] = None,
        reduce="sum",
        irreps_in={},
    ):
        super().__init__()
        assert reduce in ("sum", "mean", "min", "max")
        self.reduce = reduce
        self.field = field
        self.out_field = f"{reduce}_{field}" if out_field is None else out_field
        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out=(
                {self.out_field: irreps_in[self.field]}
                if self.field in irreps_in
                else {}
            ),
        )
        self.norm_module = AvgNumNeighborsNorm(
            avg_num_neighbors=avg_num_neighbors, type_names=type_names
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        # get destination nodes 🚂
        edge_dst = data[AtomicDataDict.EDGE_INDEX_KEY][0]
        edge_data = data[self.field]

        # === scatter ===
        out = scatter(
            edge_data,
            edge_dst,
            dim=0,
            dim_size=AtomicDataDict.num_nodes(data),
            reduce=self.reduce,
        )
        # === scale ===
        data[AtomicDataDict.NODE_FEATURES_KEY] = out
        data = self.norm_module(data)
        out = data[AtomicDataDict.NODE_FEATURES_KEY] / sqrt(2)
        # ^ factor of 2 to normalize dE/dr_i which includes both contributions from dE/dr_ij
        # and every other derivative against r_ji.

        data[self.out_field] = out
        return data
