# This file is a part of the `allegro` package. Please see LICENSE and README at the root for information on using it.
import torch

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin, scatter

from typing import Optional


class EdgewiseReduce(GraphModuleMixin, torch.nn.Module):
    """Like ``nequip.nn.AtomwiseReduce``, but accumulating per-edge data into per-atom data."""

    _factor: Optional[float]

    def __init__(
        self,
        field: str,
        out_field: Optional[str] = None,
        factor: Optional[float] = None,
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
        self._factor = None
        if factor is not None:
            self._factor = factor

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        # get destination nodes 🚂
        edge_dst = data[AtomicDataDict.EDGE_INDEX_KEY][0]
        edge_data = data[self.field]

        # === scale ===
        # for numerics it seems safer to make these smaller first before accumulating
        factor: Optional[float] = self._factor  # torchscript hack for typing
        if factor is not None:
            edge_data = edge_data * factor

        # === scatter ===
        out = scatter(
            edge_data,
            edge_dst,
            dim=0,
            dim_size=AtomicDataDict.num_nodes(data),
            reduce=self.reduce,
        )
        data[self.out_field] = out
        return data
