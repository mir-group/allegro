# This file is a part of the `allegro` package. Please see LICENSE and README at the root for information on using it.
import torch

from e3nn.o3._irreps import Irreps
from e3nn.util.jit import compile_mode

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin, ScalarMLPFunction

from typing import List


@compile_mode("script")
class ProductTypeEmbedding(GraphModuleMixin, torch.nn.Module):
    """Take a radial edge embedding and combine it with type embedding information through an elementwise product in an embedding space.

    Args:
        type_names (List[str]): list of atom type names
        initial_embedding_dim (int): the dimension of the initial embedding space
    """

    num_types: int

    def __init__(
        self,
        type_names: List[str],
        initial_embedding_dim: int,
        forward_weight_init: bool = True,
        # bookkeeping
        edge_type_field: str = AtomicDataDict.EDGE_TYPE_KEY,
        radial_features_in_field=AtomicDataDict.EDGE_EMBEDDING_KEY,
        edge_embed_out_field=AtomicDataDict.EDGE_EMBEDDING_KEY,
        irreps_in=None,
    ):
        super().__init__()
        self._init_irreps(irreps_in=irreps_in)

        # bookkeeping
        self.edge_type_field = edge_type_field
        self.in_field = radial_features_in_field
        self.out_field = edge_embed_out_field
        self.num_types = len(type_names)

        # == type embedding ==
        assert (
            initial_embedding_dim % 2 == 0
        ), "`initial_embedding_dim` must be an even number"

        self.center_embed = torch.nn.Embedding(
            num_embeddings=self.num_types,
            embedding_dim=initial_embedding_dim // 2,
        )
        self.neighbor_embed = torch.nn.Embedding(
            num_embeddings=self.num_types,
            embedding_dim=initial_embedding_dim // 2,
        )

        # == radial basis linear projection ==
        self.basis_linear = ScalarMLPFunction(
            input_dim=self.irreps_in[self.in_field].num_irreps,
            output_dim=initial_embedding_dim,
            forward_weight_init=forward_weight_init,
        )
        assert not self.basis_linear.is_nonlinear

        self.irreps_out[self.out_field] = Irreps([(initial_embedding_dim, (0, 1))])

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        # == embed atom types ==
        if self.edge_type_field in data:
            edge_types = data[self.edge_type_field]
        else:
            edge_types = torch.index_select(
                data[AtomicDataDict.ATOM_TYPE_KEY].reshape(-1),
                0,
                data[AtomicDataDict.EDGE_INDEX_KEY].reshape(-1),
            ).view(2, -1)
        type_embed = torch.cat(
            (self.center_embed(edge_types[0]), self.neighbor_embed(edge_types[1])),
            dim=-1,
        )
        # project radial basis out to type embedding dimension and multiply
        basis = self.basis_linear(data[self.in_field])
        data[self.out_field] = type_embed * basis
        return data
