from typing import Optional, List, Dict, Union

import math

import torch

from e3nn import o3

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin
from nequip.scripts.deploy import _set_deploy_metadata


class AllegroBesselBasis(GraphModuleMixin, torch.nn.Module):
    r_max: float
    PolynomialCutoff_p: float
    _per_edge_type: bool
    _num_types: int

    def __init__(
        self,
        r_max: float,
        bessel_frequency_cutoff: Optional[float] = None,
        num_bessels_per_basis: Optional[int] = None,
        per_edge_type_cutoff: Optional[
            Dict[str, Union[float, Dict[str, float]]]
        ] = None,
        type_names: Optional[List[str]] = None,
        trainable: bool = True,
        PolynomialCutoff_p: float = 6.0,
        irreps_in=None,
    ):
        r"""Modified Radial Bessel Basis with smooth Polynomial Cutoff, derived from DimeNet: https://arxiv.org/abs/2003.03123

        This basis does not carry the r_max scale dependence of the bessel basis. It is essentially a "normalized sinc basis":
        https://en.wikipedia.org/wiki/Sinc_function


        Parameters
        ----------
        r_max : float
            Cutoff radius

        bessel_frequency_cutoff : float, optional
            The highest spatial frequency in the Bessel basis, which indirectly sets `num_bessels_per_basis`.

        num_bessels_per_basis : int, optional
            Number of Bessel Basis functions. Required if `bessel_frequency_cutoff` not provided.

        per_type_cutoff : Dict[str, Union[float, Dict[str, float]]], optional
            If provided, smaller cutoffs than `r_max` can be used on a per-edge-type basis.
            This also as a side effect means that there are separate coefficients for each edge type.

        trainable : bool
            Whether the :math:`n \pi` coefficients are trainable or not.
        """
        super().__init__()

        self.trainable = trainable
        self.r_max = float(r_max)

        if bessel_frequency_cutoff is not None:
            assert per_edge_type_cutoff is None
            assert num_bessels_per_basis is None
            # max freq is n pi / r_max
            # => n = (r_max / pi) * bessel_frequency_cutoff
            num_bessels_per_basis = int(
                math.ceil((self.r_max * bessel_frequency_cutoff) / math.pi)
            )
            # ^ ceil to ensure at least some basis functions
        assert num_bessels_per_basis is not None
        self.num_bessels_per_basis = num_bessels_per_basis
        self.num_basis = num_bessels_per_basis
        self.PolynomialCutoff_p = PolynomialCutoff_p

        self._num_types = len(type_names)
        self._per_edge_type = False
        if per_edge_type_cutoff is not None:
            self._per_edge_type = True
            # map dicts from type name to thing into lists
            per_edge_type_cutoff = {
                k: (
                    [e[t] for t in type_names]
                    if isinstance(e, dict)
                    else [e] * self._num_types
                )
                for k, e in per_edge_type_cutoff.items()
            }
            per_edge_type_cutoff = [per_edge_type_cutoff[k] for k in type_names]
            per_edge_type_cutoff = torch.as_tensor(per_edge_type_cutoff)
            assert per_edge_type_cutoff.shape == (self._num_types, self._num_types)
            assert torch.all(per_edge_type_cutoff > 0)
            assert torch.all(per_edge_type_cutoff <= r_max)
            self.register_buffer("per_edge_type_cutoff", per_edge_type_cutoff)
            _set_deploy_metadata(
                "per_edge_type_cutoff",
                " ".join(str(e.item()) for e in per_edge_type_cutoff.view(-1)),
            )

            bessel_weights = torch.linspace(
                start=1.0, end=num_bessels_per_basis, steps=num_bessels_per_basis
            ).unsqueeze(0) * (math.pi / per_edge_type_cutoff.unsqueeze(-1))
            # ^ [n_type, n_type, n_bessel]
            # flatten for how its used in forward:
            bessel_weights = bessel_weights.view(-1, num_bessels_per_basis)
            rmax_recip = per_edge_type_cutoff.reciprocal()
            # flatten for how its used in forward:
            rmax_recip = rmax_recip.view(-1)
        else:
            # We have one set of weights:
            bessel_weights = torch.linspace(
                start=1.0, end=num_bessels_per_basis, steps=num_bessels_per_basis
            ) * (math.pi / self.r_max)
            rmax_recip = torch.as_tensor(1.0 / r_max)
        self.register_buffer("_rmax_recip", rmax_recip)

        if self.trainable:
            self.bessel_weights = torch.nn.Parameter(bessel_weights)
        else:
            self.register_buffer("bessel_weights", bessel_weights)

        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={
                AtomicDataDict.EDGE_EMBEDDING_KEY: o3.Irreps(
                    [(self.num_basis, (0, 1))]
                ),
                AtomicDataDict.EDGE_CUTOFF_KEY: "0e",
            },
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        model_dtype = data.get(
            AtomicDataDict.MODEL_DTYPE_KEY, data[AtomicDataDict.POSITIONS_KEY]
        ).dtype
        data = AtomicDataDict.with_edge_vectors(data, with_lengths=True)
        rmax_recip = self._rmax_recip
        bessel_weights = self.bessel_weights
        if self._per_edge_type:
            edge_type = torch.index_select(
                data[AtomicDataDict.ATOM_TYPE_KEY],
                0,
                data[AtomicDataDict.EDGE_INDEX_KEY].reshape(-1),
            ).view(2, -1)
            # convert into row-major NxN matrix index
            edge_type = edge_type[0] * self._num_types + edge_type[1]
            # need to go from [n_type, n_type] to [n_edge, 1]
            rmax_recip = torch.index_select(rmax_recip, 0, edge_type).unsqueeze(-1)
            bessel_weights = torch.index_select(bessel_weights, 0, edge_type)

        r = data[AtomicDataDict.EDGE_LENGTH_KEY].view(-1, 1)  # [z, 1]

        x = r * rmax_recip

        # [z, 1] * [z, num_basis] = [z, num_basis]
        bessel = torch.sin(r * bessel_weights) / (math.pi * x)
        # ^ pi is what the 0th Bessel goes to at 0

        # cutoff
        p = self.PolynomialCutoff_p

        out = 1.0
        out = out - (((p + 1.0) * (p + 2.0) / 2.0) * torch.pow(x, p))
        out = out + (p * (p + 2.0) * torch.pow(x, p + 1.0))
        out = out - ((p * (p + 1.0) / 2) * torch.pow(x, p + 2.0))

        cutoff = out * (x < 1.0)

        data[AtomicDataDict.EDGE_EMBEDDING_KEY] = (
            bessel.view(-1, self.num_basis) * cutoff
        ).to(model_dtype)
        data[AtomicDataDict.EDGE_CUTOFF_KEY] = cutoff
        return data
