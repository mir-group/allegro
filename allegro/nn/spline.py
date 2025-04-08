# This file is a part of the `allegro` package. Please see LICENSE and README at the root for information on using it.
from math import pi
import torch
from e3nn.util.jit import compile_mode
from nequip.utils.global_dtype import _GLOBAL_DTYPE


@compile_mode("script")
class PerClassSpline(torch.nn.Module):
    """Module implementing the spline required for a two-body scalar embedding.

    Per-class splines with finite support for [0, 1], and will go to zero smoothly at 1.

    Args:
        num_classes (int)  : number of classes or categories (for ``index_select`` operation)
        num_channels (int) : number of output channels
        num_splines (int)  : number of spline basis functions
        spline_span (int)  : number of spline basis functions that overlap on spline grid points
    """

    def __init__(
        self,
        num_classes: int,
        num_channels: int,
        num_splines: int,
        spline_span: int,
        dtype: torch.dtype = _GLOBAL_DTYPE,
    ):
        super().__init__()

        # === sanity check ===
        assert 0 <= spline_span <= num_splines
        assert num_splines > 0

        # === save inputs parameters ===
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.num_splines = num_splines
        self.spline_span = spline_span
        self.dtype = dtype

        # === spline grid parameters ===
        lower = (
            torch.arange(
                -self.spline_span, self.num_splines - spline_span, dtype=self.dtype
            )
            / self.num_splines
        )
        diff = (self.spline_span + 1) / self.num_splines

        self.register_buffer("lower", lower)
        self.register_buffer("upper", lower + diff)
        self._const = 2 * pi / diff

        # === use torch.nn.Embedding for spline weights ===
        self.class_embed = torch.nn.Embedding(
            num_embeddings=self.num_classes,
            embedding_dim=self.num_channels * self.num_splines,
            dtype=dtype,
        )

    def extra_repr(self) -> str:
        msg = f"num classes : {self.num_classes}\n"
        msg += f"num channels: {self.num_channels}\n"
        msg += f"num splines : {self.num_splines}\n"
        msg += f"spline span : {self.spline_span}"
        return msg

    def forward(self, x: torch.Tensor, classes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor)      : input tensor with shape (z, 1)
            classes (torch.Tensor): class tensor with shape (z,) whose values are integer indices from 0 to num_classes - 1
        """
        # index out weights based on classes: -> (z, num_channels, num_splines)
        spline_weights = self.class_embed(classes).view(
            classes.size(0), self.num_channels, self.num_splines
        )
        spline_basis = self._get_basis(x)
        # (z, num_channels, num_splines), (z, num_splines) -> (z, num_channels)
        return torch.bmm(spline_weights, spline_basis.unsqueeze(-1)).squeeze(-1)

    def _get_basis(self, x: torch.Tensor) -> torch.Tensor:
        # construct spline basis
        # x: (z, 1) -> spline_basis: (z, num_splines)
        normalized_x = self._const * (
            torch.clamp(x, min=self.lower, max=self.upper) - self.lower
        )
        return 0.25 * (1 - torch.cos(normalized_x)).square()
