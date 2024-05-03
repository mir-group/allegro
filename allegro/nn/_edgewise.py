from typing import Optional, List
import math

import torch
from torch_runstats.scatter import scatter

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin

from .. import _keys


class EdgewiseReduce(GraphModuleMixin, torch.nn.Module):
    """Like ``nequip.nn.AtomwiseReduce``, but accumulating per-edge data into per-atom data."""

    _factor: Optional[float]

    def __init__(
        self,
        field: str,
        out_field: Optional[str] = None,
        normalize_edge_reduce: bool = True,
        avg_num_neighbors: Optional[float] = None,
        reduce="sum",
        irreps_in={},
    ):
        """Sum edges into nodes."""
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
        if normalize_edge_reduce and avg_num_neighbors is not None:
            self._factor = 1.0 / math.sqrt(avg_num_neighbors)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        # get destination nodes 🚂
        edge_dst = data[AtomicDataDict.EDGE_INDEX_KEY][0]

        out = scatter(
            data[self.field],
            edge_dst,
            dim=0,
            dim_size=len(data[AtomicDataDict.POSITIONS_KEY]),
            reduce=self.reduce,
        )

        factor: Optional[float] = self._factor  # torchscript hack for typing
        if factor is not None:
            out = out * factor

        data[self.out_field] = out

        return data


class EdgewiseEnergySum(GraphModuleMixin, torch.nn.Module):
    """Sum edgewise energies.

    Includes optional per-species-pair edgewise energy scales.
    """

    _factor: Optional[float]

    def __init__(
        self,
        num_types: int,
        avg_num_neighbors: Optional[float] = None,
        normalize_edge_energy_sum: bool = True,
        per_edge_species_scale: bool = False,
        per_edge_species_scales_mask: Optional[List[List[float]]] = None,
        irreps_in={},
    ):
        """Sum edges into nodes."""
        super().__init__()
        self._init_irreps(
            irreps_in=irreps_in,
            my_irreps_in={_keys.EDGE_ENERGY: "0e"},
            irreps_out={AtomicDataDict.PER_ATOM_ENERGY_KEY: "0e"},
        )

        self._factor = None
        if normalize_edge_energy_sum and avg_num_neighbors is not None:
            self._factor = 1.0 / math.sqrt(avg_num_neighbors)

        self.per_edge_species_scale = per_edge_species_scale
        self.has_per_edge_species_scales_mask = per_edge_species_scales_mask is not None
        if self.has_per_edge_species_scales_mask:
            if not self.per_edge_species_scale:
                raise Exception(
                    "If per_edge_species_scales_mask is used, then per_edge_species_scale must be True"
                )

        if self.per_edge_species_scale:
            self.per_edge_scales = torch.nn.Parameter(torch.ones(num_types, num_types))
            if self.has_per_edge_species_scales_mask:
                m = torch.as_tensor(
                    [[float(x) for x in y] for y in per_edge_species_scales_mask]
                )
                if m.shape[0] != num_types or m.shape[1] != num_types:
                    raise Exception(
                        f"per_edge_species_scales_mask should be ({num_types,num_types}), but is {m.shape}"
                    )
                for i in range(num_types):
                    for j in range(num_types):
                        if m[i, j] != m[j, i]:
                            raise Exception(
                                f"per_edge_species_scales_mask should be symmetric. Check elements {i},{j}"
                            )
                self.per_edge_scales_mask = torch.nn.Parameter(m, requires_grad=False)
        else:
            self.register_buffer("per_edge_scales", torch.Tensor())

        if not self.has_per_edge_species_scales_mask:
            self.register_buffer("per_edge_scales_mask", torch.Tensor())

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        edge_center = data[AtomicDataDict.EDGE_INDEX_KEY][0]
        edge_neighbor = data[AtomicDataDict.EDGE_INDEX_KEY][1]

        edge_eng = data[_keys.EDGE_ENERGY]
        species = data[AtomicDataDict.ATOM_TYPE_KEY].squeeze(-1)
        center_species = species[edge_center]
        neighbor_species = species[edge_neighbor]

        if self.per_edge_species_scale:
            if self.has_per_edge_species_scales_mask:
                edge_eng = edge_eng * (
                    self.per_edge_scales_mask[center_species, neighbor_species]
                    * self.per_edge_scales[center_species, neighbor_species]
                ).unsqueeze(-1)
            else:
                edge_eng = edge_eng * self.per_edge_scales[
                    center_species, neighbor_species
                ].unsqueeze(-1)

        atom_eng = scatter(edge_eng, edge_center, dim=0, dim_size=len(species))
        factor: Optional[float] = self._factor  # torchscript hack for typing
        if factor is not None:
            atom_eng = atom_eng * factor

        data[AtomicDataDict.PER_ATOM_ENERGY_KEY] = atom_eng

        return data
