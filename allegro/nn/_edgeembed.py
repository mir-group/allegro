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

        if typexbasis_mode == "product":
            # default
            opts = dict(mlp_latent_dimensions=[], mlp_nonlinearity=None)
            opts.update(basis_mlp_kwargs)
            self.basis_mlp = basis_mlp(
                mlp_input_dimension=self.basis.num_basis,
                mlp_output_dimension=type_embedding_dim,
                **opts,
            )
        elif typexbasis_mode == "cat":
            self.basis_mlp = torch.nn.Identity()
        else:
            raise NotImplementedError

        self.irreps_out[AtomicDataDict.EDGE_EMBEDDING_KEY] = o3.Irreps(
            [
                (
                    type_embedding_dim
                    if typexbasis_mode == "product"
                    else type_embedding_dim + self.basis.num_basis,
                    (0, 1),
                )
            ]
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

        # do basis and cutoffs
        edge_length = data[AtomicDataDict.EDGE_LENGTH_KEY]
        cutoff = self.cutoff(edge_length)
        basis = self.basis(edge_length)

        # combine type and basis embedings
        if self._embed_prod:
            # project basis out to type embedding dimension
            basis = self.basis_mlp(basis)
            type_embed = torch.cat(
                (center_embed, neighbor_embed),
                dim=-1,
            )
            edge_embed = type_embed * basis
        else:
            edge_embed = torch.cat(
                (basis, center_embed, neighbor_embed),
                dim=-1,
            )
            edge_embed = edge_embed

        data[AtomicDataDict.EDGE_EMBEDDING_KEY] = edge_embed * cutoff.unsqueeze(-1)
        return data
