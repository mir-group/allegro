# This file is a part of the `allegro` package. Please see LICENSE and README at the root for information on using it.
import torch

from e3nn.o3._irreps import Irreps
from e3nn.o3._spherical_harmonics import SphericalHarmonics
from e3nn.util.jit import compile_mode

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin, ScalarMLPFunction, with_edge_vectors_

from ._strided import MakeWeightedChannels

from typing import Union


@compile_mode("script")
class TwoBodySphericalHarmonicTensorEmbed(GraphModuleMixin, torch.nn.Module):
    """Construct two-body tensor embedding as weighted spherical harmonics.

    Constructs tensor basis as spherical harmonic projections of edge vectors,
    and tensor embedding as weighted tensor basis (weights learnt from scalar embedding).

    Args:
        irreps_edge_sh (int, str, or o3.Irreps): if int, will be treated as lmax for o3.Irreps.spherical_harmonics(lmax)
        num_tensor_features (int): number of tensor feature channels
    """

    num_tensor_features: int

    def __init__(
        self,
        irreps_edge_sh: Union[int, str, Irreps],
        num_tensor_features: int,
        forward_weight_init: bool = True,
        # bookkeeping args
        scalar_embedding_in_field: str = AtomicDataDict.EDGE_EMBEDDING_KEY,
        tensor_basis_out_field: str = AtomicDataDict.EDGE_ATTRS_KEY,
        tensor_embedding_out_field: str = AtomicDataDict.EDGE_FEATURES_KEY,
        irreps_in=None,
        # optional hyperparameters
        edge_sh_normalization: str = "component",
        edge_sh_normalize: bool = True,
        weight_individual_irreps: bool = True,
    ):
        super().__init__()

        self.scalar_embedding_in_field = scalar_embedding_in_field
        self.tensor_basis_out_field = tensor_basis_out_field
        self.tensor_embedding_out_field = tensor_embedding_out_field

        if isinstance(irreps_edge_sh, int):
            irreps_edge_sh = Irreps.spherical_harmonics(irreps_edge_sh)
        else:
            irreps_edge_sh = Irreps(irreps_edge_sh)
        self.sh = SphericalHarmonics(
            irreps_edge_sh, edge_sh_normalize, edge_sh_normalization
        )
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[self.scalar_embedding_in_field],
            irreps_out={
                self.tensor_basis_out_field: irreps_edge_sh,
                self.tensor_embedding_out_field: irreps_edge_sh,
            },
        )

        # use learned weights from two-body scalar track to weight
        # initial spherical harmonics embedding
        self._edge_weighter = MakeWeightedChannels(
            irreps_in=irreps_edge_sh,
            multiplicity_out=num_tensor_features,
            weight_individual_irreps=weight_individual_irreps,
        )

        # hardcode a linear projection
        self.env_embed_linear = ScalarMLPFunction(
            input_dim=self.irreps_in[self.scalar_embedding_in_field].num_irreps,
            output_dim=self._edge_weighter.weight_numel,
            forward_weight_init=forward_weight_init,
        )
        assert not self.env_embed_linear.is_nonlinear

        self._output_dtype = torch.get_default_dtype()

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = with_edge_vectors_(data, with_lengths=False)
        # compute weights from scalar embedding
        edge_invariants = data[self.scalar_embedding_in_field]
        weights = self.env_embed_linear(edge_invariants)
        # store unweighted spherical harmonics embedding as two-body tensor basis
        edge_vec = data[AtomicDataDict.EDGE_VECTORS_KEY]
        edge_sh = self.sh(edge_vec).to(self._output_dtype)
        data[self.tensor_basis_out_field] = edge_sh
        # store two-body tensor features (weighted spherical harmonics embedding)
        data[self.tensor_embedding_out_field] = self._edge_weighter(edge_sh, weights)
        return data
