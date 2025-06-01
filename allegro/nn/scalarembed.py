# This file is a part of the `allegro` package. Please see LICENSE and README at the root for information on using it.
from math import sqrt
import torch

from e3nn.o3._irreps import Irreps
from e3nn.util.jit import compile_mode

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin, SequentialGraphNetwork
from nequip.nn.embedding import PolynomialCutoff, BesselEdgeLengthEncoding
from nequip.utils.global_dtype import _GLOBAL_DTYPE

from ._edgeembed import ProductTypeEmbedding
from .spline import PerClassSpline

from typing import Sequence


def TwoBodyBesselScalarEmbed(
    type_names: Sequence[str],
    # bessel encoding
    num_bessels: int = 8,
    bessel_trainable: bool = False,
    polynomial_cutoff_p: int = 6,
    # model builder params
    module_output_dim: int = 64,
    forward_weight_init: bool = True,
    # bookkeeping
    scalar_embed_field: str = AtomicDataDict.EDGE_EMBEDDING_KEY,
    irreps_in=None,
) -> SequentialGraphNetwork:
    """Two-body Bessel scalar embedding.

    The radial edge lengths are encoded with a Bessel basis, which is then projected to ``two_body_embedding_dim``.
    The center-neighbor atom types are embedded with weights to the same ``two_body_embedding_dim``.
    The radial embedding and center-neighbor type embedding are multiplied.

    This module can be used for the ``scalar_embed`` argument of the ``AllegroModel`` in the config as follows.

    .. code-block:: yaml

      model:
        _target_: allegro.model.AllegroModel
        # other Allegro model parameters
        scalar_embed:
          _target_: allegro.nn.TwoBodyBesselScalarEmbed
          num_bessels: 8
          bessel_trainable: false
          polynomial_cutoff_p: 6

    Args:
        num_bessels (int): number of Bessel basis functions (default ``8``)
        bessel_trainable (int): whether Bessel roots are trainable (default ``False``)
        polynomial_cutoff_p (int): p-exponent used in polynomial cutoff function, smaller p corresponds to stronger decay with distance (default ``6``)
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
        initial_embedding_dim=module_output_dim,
        forward_weight_init=forward_weight_init,
        radial_features_in_field=scalar_embed_field,
        edge_embed_out_field=scalar_embed_field,
        irreps_in=bessel_encode.irreps_out,
    )

    return SequentialGraphNetwork(
        {
            "bessel_encode": bessel_encode,
            "type_embed": type_embed,
        }
    )


@compile_mode("script")
class TwoBodySplineScalarEmbed(GraphModuleMixin, torch.nn.Module):
    r"""Two-body spline scalar embedding.

    This module can be used for the ``scalar_embed`` argument of the ``AllegroModel`` in the config as follows.

    .. code-block:: yaml

      model:
        _target_: allegro.model.AllegroModel
        # other Allegro model parameters
        scalar_embed:
          _target_: allegro.nn.TwoBodySplineScalarEmbed
          num_splines: 16
          spline_span: 12

    Args:
        num_splines (int): number of spline basis functions
        spline_span (int): number of spline basis functions that overlap on spline grid centers
    """

    def __init__(
        self,
        type_names: Sequence[str],
        # spline params
        num_splines: int = 16,
        spline_span: int = 12,
        # model builder params
        module_output_dim: int = 64,
        forward_weight_init: bool = True,
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
            num_splines=num_splines,
            spline_span=spline_span,
            dtype=_GLOBAL_DTYPE,
        )

        # === embedding weight init ===
        # this should in principle be done in the spline module, but we might as well do it here instead of passing the argument on
        if forward_weight_init:
            # since splines have finite support, we only account for overlapping splines
            # the overlap is approximately `spline_span` (though it should be less)
            bound = sqrt(3 / spline_span)
        else:
            bound = sqrt(3 / self.spline.num_channels)
        torch.nn.init.uniform_(self.spline.class_embed.weight, a=-bound, b=bound)
        del bound

        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={
                self.scalar_embed_field: Irreps([(module_output_dim, (0, 1))]),
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
