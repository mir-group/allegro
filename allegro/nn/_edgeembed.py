import torch

from e3nn import o3
from e3nn.util.jit import compile_mode

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin

from ._fc import ScalarMLPFunction

from typing import List


@compile_mode("script")
class ProductTypeEmbedding(GraphModuleMixin, torch.nn.Module):
    """Take a radial edge embedding and combine it with type embedding information through an elementwise product in an embedding space.

    Args:
        initial_scalar_embedding_dim (int): the dimension of the embedding space
        basis_mlp_kwargs: options for the MLP embedding the previous radial basis. By default no hidden layers or nonlinearities (linear projection).
    """

    num_types: int

    def __init__(
        self,
        type_names: List[str],
        initial_scalar_embedding_dim: int,
        radial_basis_mlp=ScalarMLPFunction,
        radial_basis_mlp_kwargs={},
        irreps_in=None,
    ):
        super().__init__()
        self._init_irreps(irreps_in=irreps_in)
        self.num_types = len(type_names)
        assert initial_scalar_embedding_dim % 2 == 0
        self.type_embeddings = torch.nn.Parameter(
            torch.randn(2, self.num_types, initial_scalar_embedding_dim // 2)
        )

        # default
        opts = dict(mlp_latent_dimensions=[], mlp_nonlinearity=None)
        opts.update(radial_basis_mlp_kwargs)
        self.basis_mlp = radial_basis_mlp(
            mlp_input_dimension=self.irreps_in[
                AtomicDataDict.EDGE_EMBEDDING_KEY
            ].num_irreps,  # TODO check scalar
            mlp_output_dimension=initial_scalar_embedding_dim,
            **opts,
        )

        self.irreps_out[AtomicDataDict.EDGE_EMBEDDING_KEY] = o3.Irreps(
            [(initial_scalar_embedding_dim, (0, 1))]
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = AtomicDataDict.with_edge_vectors(data, with_lengths=True)
        # embed types
        atom_types = data[AtomicDataDict.ATOM_TYPE_KEY].squeeze(-1)
        edge_types = torch.index_select(
            atom_types, 0, data[AtomicDataDict.EDGE_INDEX_KEY].reshape(-1)
        ).view(2, -1)
        center_types = edge_types[0]
        neighbor_types = edge_types[1]
        center_embed = torch.index_select(
            self.type_embeddings[0],
            0,
            center_types,
        )
        neighbor_embed = torch.index_select(
            self.type_embeddings[1],
            0,
            neighbor_types,
        )

        # project basis out to type embedding dimension
        basis = self.basis_mlp(data[AtomicDataDict.EDGE_EMBEDDING_KEY])
        type_embed = torch.cat(
            (center_embed, neighbor_embed),
            dim=-1,
        )
        edge_embed = type_embed * basis

        data[AtomicDataDict.EDGE_EMBEDDING_KEY] = edge_embed
        return data
