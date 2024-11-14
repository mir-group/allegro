import torch

from e3nn import o3
from e3nn.util.jit import compile_mode

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin, SequentialGraphNetwork
from nequip.nn.embedding import PolynomialCutoff, BesselEdgeLengthEncoding
from nequip.utils.global_dtype import _GLOBAL_DTYPE

from ._edgeembed import ProductTypeEmbedding
from ._fc import ScalarMLP
from .spline import PerClassSpline

from typing import Sequence


def TwoBodyBesselScalarEmbed(
    type_names: Sequence[str],
    # bessel encoding
    num_bessels: int = 8,
    bessel_trainable: bool = False,
    polynomial_cutoff_p: int = 6,
    # two body MLP
    two_body_embedding_dim: int = 32,
    two_body_mlp_hidden_layer_depth: int = 2,
    two_body_mlp_hidden_layer_width: int = 64,
    two_body_mlp_nonlinear: bool = True,
    module_output_dim: int = 64,
    scalar_embed_field: str = AtomicDataDict.EDGE_EMBEDDING_KEY,
    irreps_in=None,
) -> SequentialGraphNetwork:
    """Two-body Bessel scalar embedding.

    The radial edge lengths are encoded with a Bessel basis, which is then projected to ``two_body_embedding_dim``.
    The center-neighbor atom types are embedded with weights to the same ``two_body_embedding_dim``.
    The radial embedding and center-neighbor type embedding are multiplied (to impose that the embedding smoothly goes to zero at the cutoff).
    The product then goes through a multilayer perception (``two_body_mlp``) to form a two-body scalar embedding (users shouldn't have to worry about the output dimesion -- that is handled internally by the model using this module).

    The following arguments are the ones that users should configure, the rest are used internally for composing this module into a model.

    Args:
        num_bessels (int): number of Bessel basis functions
        bessel_trainable (int): whether Bessel roots are trainable
        polynomial_cutoff_p (int): p-exponent used in polynomial cutoff function, smaller p corresponds to stronger decay with distance
        two_body_embedding_dim: int = 32,
        two_body_mlp_hidden_layer_depth: int = 2,
        two_body_mlp_hidden_layer_width: int = 64,
        two_body_mlp_nonlinear (bool): whether the two-body MLP has a ``silu`` nonlinearity
    """
    # the following args are for internal use in model building:
    # `type_names`, `module_output_dim`, `scalar_embed_field`, `irreps_in`
    # and so are not explained in the docstring

    bessel_encode = BesselEdgeLengthEncoding(
        cutoff=PolynomialCutoff(polynomial_cutoff_p),
        num_bessels=num_bessels,
        trainable=bessel_trainable,
        edge_invariant_field=scalar_embed_field,
        irreps_in=irreps_in,
    )
    type_embed = ProductTypeEmbedding(
        type_names=type_names,
        initial_embedding_dim=two_body_embedding_dim,
        radial_features_in_field=scalar_embed_field,
        edge_embed_out_field=scalar_embed_field,
        irreps_in=bessel_encode.irreps_out,
    )

    twobody_mlp = ScalarMLP(
        # input dims from previous module
        field=scalar_embed_field,
        mlp_output_dim=module_output_dim,
        mlp_hidden_layer_depth=two_body_mlp_hidden_layer_depth,
        mlp_hidden_layer_width=two_body_mlp_hidden_layer_width,
        mlp_nonlinearity="silu" if two_body_mlp_nonlinear else None,
        irreps_in=type_embed.irreps_out,
    )

    return SequentialGraphNetwork(
        {
            "bessel_encode": bessel_encode,
            "type_embed": type_embed,
            "twobody_mlp": twobody_mlp,
        }
    )


@compile_mode("script")
class TwoBodySplineScalarEmbed(GraphModuleMixin, torch.nn.Module):
    r"""Two-body scalar embedding based on B-splines for every edge type (pair of center-neighbor types).

    Args:
        spline_grid (int): number of spline basis grid centers in [0, 1]
        spline_span (int): number of spline basis functions that overlap on spline grid centers
    """

    def __init__(
        self,
        type_names: Sequence[str],
        # spline params
        spline_grid: int = 5,
        spline_span: int = 3,
        # model builder params
        module_output_dim: int = 64,
        # bookkeeping
        scalar_embed_field: str = AtomicDataDict.EDGE_EMBEDDING_KEY,
        edge_type_field: str = AtomicDataDict.EDGE_TYPE_KEY,
        norm_length_field: str = AtomicDataDict.NORM_LENGTH_KEY,
        irreps_in=None,
    ):
        super().__init__()

        # === bookkeeping ===
        self.num_types = len(type_names)
        self.scalar_embed_field = scalar_embed_field
        self.edge_type_field = edge_type_field
        self.norm_length_field = norm_length_field

        # === instantiate spline module ===
        self.spline = PerClassSpline(
            num_classes=self.num_types * self.num_types,
            num_channels=module_output_dim,
            spline_grid=spline_grid,
            spline_span=spline_span,
            dtype=_GLOBAL_DTYPE,
        )

        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={
                self.scalar_embed_field: o3.Irreps([(module_output_dim, (0, 1))]),
            },
        )

        self._output_dtype = torch.get_default_dtype()

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        # get edge_types
        if self.edge_type_field in data:
            edge_types = data[self.edge_type_field]
        else:
            edge_types = torch.index_select(
                data[AtomicDataDict.ATOM_TYPE_KEY].reshape(-1),
                0,
                data[AtomicDataDict.EDGE_INDEX_KEY].reshape(-1),
            ).view(2, -1)
        # convert into row-major NxN matrix index
        edge_types = edge_types[0] * self.num_types + edge_types[1]

        # apply spline
        x = data[self.norm_length_field]
        data[self.scalar_embed_field] = self.spline(x, edge_types).to(
            self._output_dtype
        )
        return data
