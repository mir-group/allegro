import math

import torch

from e3nn import o3
from e3nn.util.jit import compile_mode

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin
from nequip.nn.radial_basis import BesselBasis

from ._fc import ScalarMLPFunction


@compile_mode("script")
class EdgeEmbedding(GraphModuleMixin, torch.nn.Module):
    num_types: int
    type_embedding_mode: str

    def __init__(
        self,
        num_types: int,
        type_embedding_mode: str = "cat",
        type_embedding_init: str = "normal",
        basis=BesselBasis,
        basis_kwargs={},
        irreps_in=None,
    ):
        super().__init__()
        self._init_irreps(irreps_in=irreps_in)
        self.num_types = num_types
        self.type_embedding_mode = type_embedding_mode
        if type_embedding_mode == "cat":
            norm_const = math.sqrt(num_types)
            self.type_embeddings = torch.nn.Parameter(
                norm_const
                * torch.cat(
                    [torch.eye(num_types).unsqueeze(0) for _ in range(2)], dim=0
                )
            )
        elif type_embedding_mode == "prod":
            self.type_embeddings = torch.nn.Parameter(
                num_types
                * torch.eye(num_types**2).reshape(
                    num_types, num_types, num_types**2
                )
            )
        else:
            raise NotImplementedError
        with torch.no_grad():
            if type_embedding_init == "eye":
                pass
            elif type_embedding_init == "normal":
                self.type_embeddings.normal_()
            else:
                raise NotImplementedError
        self.basis = basis(**basis_kwargs)
        self.irreps_out[AtomicDataDict.EDGE_EMBEDDING_KEY] = o3.Irreps(
            [
                (
                    self.basis.num_basis
                    + (
                        2 * num_types
                        if self.type_embedding_mode == "cat"
                        else num_types**2
                    ),
                    (0, 1),
                )
            ]
        )
        self.embed_basis = ScalarMLPFunction(
            mlp_input_dimension=self.basis.num_basis,
            mlp_latent_dimensions=[],
            mlp_output_dimension=self.basis.num_basis,
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = AtomicDataDict.with_edge_vectors(data, with_lengths=True)
        edge_length = data[AtomicDataDict.EDGE_LENGTH_KEY]
        edge_center, edge_neighbor = (
            data[AtomicDataDict.EDGE_INDEX_KEY][0],
            data[AtomicDataDict.EDGE_INDEX_KEY][1],
        )
        atom_types = data[AtomicDataDict.ATOM_TYPE_KEY].squeeze(-1)
        basis = self.embed_basis(self.basis(edge_length))
        if self.type_embedding_mode == "cat":
            edge_embed = torch.cat(
                (
                    basis,
                    self.type_embeddings[0, atom_types[edge_center]],
                    self.type_embeddings[1, atom_types[edge_neighbor]],
                ),
                dim=-1,
            )
        elif self.type_embedding_mode == "prod":
            edge_embed = torch.cat(
                (
                    basis,
                    self.type_embeddings[
                        atom_types[edge_center], atom_types[edge_neighbor]
                    ],
                ),
                dim=-1,
            )
        else:
            assert False
        data[AtomicDataDict.EDGE_EMBEDDING_KEY] = edge_embed
        return data
