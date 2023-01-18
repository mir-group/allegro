import torch

from e3nn import o3
from e3nn.util.jit import compile_mode, compile

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin
from nequip.nn.cutoffs import PolynomialCutoff

from ._norm_basis import BesselBasis
from ._fc import ScalarMLPFunction


@compile_mode("script")
class EdgeEmbedding(GraphModuleMixin, torch.nn.Module):
    num_types: int

    _embed_prod: bool

    def __init__(
        self,
        num_types: int,
        type_embedding_dim: int,
        typexbasis_mode: str = "product",
        basis=BesselBasis,
        basis_kwargs={},
        basis_mlp=ScalarMLPFunction,
        basis_mlp_kwargs={},
        typexbasis_mlp=ScalarMLPFunction,
        typexbasis_mlp_kwargs={},
        cutoff=PolynomialCutoff,
        cutoff_kwargs={},
        irreps_in=None,
    ):
        super().__init__()
        self._init_irreps(irreps_in=irreps_in)
        self.num_types = num_types
        assert type_embedding_dim % 2 == 0
        self.type_embeddings = torch.nn.Parameter(
            torch.randn(2, num_types, type_embedding_dim // 2)
        )
        assert typexbasis_mode in ("product", "cat")
        self._embed_prod = typexbasis_mode == "product"

        self.basis = compile(basis(**basis_kwargs))
        self.cutoff = compile(cutoff(**cutoff_kwargs))

        # default to an extra linear layer, since it costs nothing in inference
        opts = dict(mlp_latent_dimensions=[type_embedding_dim])
        opts.update(basis_mlp_kwargs)
        self.basis_mlp = basis_mlp(
            mlp_input_dimension=self.basis.num_basis,
            mlp_output_dimension=type_embedding_dim,
            **opts,
        )
        # default to an extra linear layer, since it costs nothing in inference
        opts = dict(mlp_latent_dimensions=[type_embedding_dim])
        opts.update(typexbasis_mlp_kwargs)
        self.typexbasis_mlp = typexbasis_mlp(
            mlp_input_dimension=type_embedding_dim * (1 if self._embed_prod else 2),
            mlp_output_dimension=type_embedding_dim,
            **opts,
        )

        self.irreps_out[AtomicDataDict.EDGE_EMBEDDING_KEY] = o3.Irreps(
            [
                (
                    type_embedding_dim,
                    (0, 1),
                )
            ]
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = AtomicDataDict.with_edge_vectors(data, with_lengths=True)
        edge_length = data[AtomicDataDict.EDGE_LENGTH_KEY]
        edge_center, edge_neighbor = (
            data[AtomicDataDict.EDGE_INDEX_KEY][0],
            data[AtomicDataDict.EDGE_INDEX_KEY][1],
        )
        # embed types
        cutoff = self.cutoff(edge_length)
        basis = self.basis(edge_length)
        basis = self.basis_mlp(basis)
        atom_types = data[AtomicDataDict.ATOM_TYPE_KEY].squeeze(-1)

        edge_types = torch.index_select(
            atom_types, 0, data[AtomicDataDict.EDGE_INDEX_KEY].view(-1)
        ).view(2, -1)
        center_types = edge_types[0]
        neighbor_types = edge_types[1]
        center_embed = torch.index_select(
            self.type_embeddings[0],
            0,
            center_types,
        )
        neighbor_embed = torch.index_select(
            self.type_embeddings[0],
            0,
            neighbor_types,
        )

        if self._embed_prod:
            type_embed = torch.cat(
                (center_embed, neighbor_embed),
                dim=-1,
            )
            edge_embed = self.typexbasis_mlp(type_embed * basis)
        else:
            edge_embed = torch.cat(
                (basis, center_embed, neighbor_embed),
                dim=-1,
            )
            edge_embed = self.typexbasis_mlp(edge_embed)

        data[AtomicDataDict.EDGE_EMBEDDING_KEY] = edge_embed * cutoff.unsqueeze(-1)
        return data
