from nequip.data import AtomicDataDict
from nequip.nn import SequentialGraphNetwork

from nequip.nn.embedding import PolynomialCutoff, BesselEdgeLengthEncoding
from ._edgeembed import ProductTypeEmbedding
from ._fc import ScalarMLP

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
