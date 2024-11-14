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
        spline_grid (int)  : number of spline grid centers in [0, 1]
        spline_span (int)  : number of spline basis functions that overlap on spline grid points
        lower_cutoff (bool): whether to impose that the output goes to zero smoothly at input <= 0
    """

    def __init__(
        self,
        num_classes: int,
        num_channels: int,
        spline_grid: int,
        spline_span: int,
        lower_cutoff: bool = False,
        dtype: torch.dtype = _GLOBAL_DTYPE,
    ):
        # === initialize and save inputs parameters ===
        super().__init__()
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.spline_grid = spline_grid
        self.spline_span = spline_span
        self.grid_dim = self.spline_grid + self.spline_span
        self.lower_cutoff = lower_cutoff
        self.dtype = dtype

        # === spline grid parameters ===
        # note that num_splines = spline_grid + spline_span
        if lower_cutoff:
            lower = (
                torch.arange(-spline_span, spline_grid, dtype=self.dtype) / spline_grid
            )
            lower = lower + spline_span / spline_grid
            lower = lower * spline_grid / (spline_grid + 2 * spline_span)
            diff = (spline_span + 1) / (spline_grid + 2 * spline_span)
        else:
            lower = torch.arange(-spline_span, spline_grid, dtype=self.dtype) / (
                spline_grid + spline_span
            )
            diff = (spline_span + 1) / (spline_grid + spline_span)

        self.register_buffer("lower", lower)
        self.register_buffer("upper", lower + diff)
        self._const = 2 * pi / diff

        # === use torch.nn.Embedding for spline weights ===
        self.class_embed = torch.nn.Embedding(
            num_embeddings=self.num_classes,
            embedding_dim=self.num_channels * self.grid_dim,
            dtype=dtype,
        )

    def extra_repr(self) -> str:
        msg = f"num classes : {self.num_classes}\n"
        msg += f"num channels: {self.num_channels}\n"
        msg += f"spline grid : {self.spline_grid}\n"
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
            classes.size(0), self.num_channels, self.grid_dim
        )
        spline_basis = self._get_basis(x)
        return torch.einsum("ens, es -> en", spline_weights, spline_basis)

    def _get_basis(self, x: torch.Tensor) -> torch.Tensor:
        # construct spline basis
        # x: (z, 1) -> spline_basis: (z, num_splines)
        normalized_x = self._const * (
            torch.clamp(x, min=self.lower, max=self.upper) - self.lower
        )
        return 0.25 * (1 - torch.cos(normalized_x)).square()
