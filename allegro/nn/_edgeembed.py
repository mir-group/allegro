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

    def __init__(
        self,
        num_types: int,
        type_embedding_dim: int,
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
        self.type_embeddings = torch.nn.Parameter(
            torch.randn(2, num_types, type_embedding_dim)
        )

        self.basis = compile(basis(**basis_kwargs))
        self.cutoff = compile(cutoff(**cutoff_kwargs))

        full_emb_dim = type_embedding_dim * 2
        opts = dict(mlp_latent_dimensions=[])
        opts.update(basis_mlp_kwargs)
        self.basis_mlp = basis_mlp(
            mlp_input_dimension=self.basis.num_basis,
            mlp_output_dimension=full_emb_dim,
            **opts,
        )
        opts = dict(mlp_latent_dimensions=[])
        opts.update(typexbasis_mlp_kwargs)
        self.typexbasis_mlp = typexbasis_mlp(
            mlp_input_dimension=full_emb_dim, mlp_output_dimension=full_emb_dim, **opts
        )

        self.irreps_out[AtomicDataDict.EDGE_EMBEDDING_KEY] = o3.Irreps(
            [
                (
                    full_emb_dim,
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
        atom_types = data[AtomicDataDict.ATOM_TYPE_KEY].squeeze(-1)
        type_embed = torch.cat(
            (
                self.type_embeddings[0, atom_types[edge_center]],
                self.type_embeddings[1, atom_types[edge_neighbor]],
            ),
            dim=-1,
        )
        basis = self.basis_mlp(basis)
        edge_embed = self.typexbasis_mlp(type_embed * basis)

        data[AtomicDataDict.EDGE_EMBEDDING_KEY] = edge_embed * cutoff.unsqueeze(-1)
        return data
