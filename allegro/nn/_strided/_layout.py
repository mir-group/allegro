"""Tools for handing the strided irreps layout."""

import math

import torch

from e3nn.o3 import Irreps


class StridedLayout:
    """Utility class to represent a strided layout of a tensor whose irreps all have the same multiplicity."""

    irreps: Irreps
    base_irreps: Irreps
    pad_to_multiple: int
    dim: int
    base_dim: int
    mul: int

    def __init__(self, irreps: Irreps, pad_to_multiple: int = 1):
        irreps = Irreps(irreps)
        if not self.can_be_strided(irreps):
            raise ValueError(f"Irreps `{irreps}` cannot be strided.")
        self.irreps = irreps
        self.base_irreps = Irreps([(1, ir) for _, ir in irreps])
        self.mul = self.irreps[0].mul if len(irreps) > 0 else 0
        assert self.irreps.dim == self.base_irreps.dim * self.mul
        self.pad_to_multiple = pad_to_multiple
        assert self.pad_to_multiple in (1, 2, 4, 8)

        self.base_dim = int(
            math.ceil(self.base_irreps.dim / self.pad_to_multiple)
            * self.pad_to_multiple
        )
        pad_by = self.base_dim - self.base_irreps.dim
        self.dim = self.base_dim * self.mul

        # indexes to convert
        self.indexes_to_strided = torch.zeros(self.dim, dtype=torch.long)
        self.indexes_to_catted = torch.zeros(self.irreps.dim, dtype=torch.long)
        i: int = 0
        for mul_i in range(self.mul):
            for irrep_i, (_, irrep) in enumerate(self.base_irreps):
                strided_indexes = torch.arange(start=i, end=i + irrep.dim)
                catted_indexes = (
                    torch.arange(irrep.dim)
                    + self.irreps[:irrep_i].dim
                    + irrep.dim * mul_i
                )
                self.indexes_to_strided[strided_indexes] = catted_indexes
                self.indexes_to_catted[catted_indexes] = strided_indexes
                i += irrep.dim
            # pad out this line of the [mul, k] shape
            i += pad_by

        # They should be inverses:
        assert torch.all(
            self.indexes_to_strided[self.indexes_to_catted]
            == torch.arange(self.irreps.dim)
        )

    @staticmethod
    def can_be_strided(irreps: Irreps) -> bool:
        """Check whether ``irreps`` is compatible with strided layout."""
        irreps = Irreps(irreps)
        if len(irreps) == 0:
            return True
        return all(irreps[0].mul == mul for mul, ir in irreps)

    def to_strided(self, x: torch.Tensor) -> torch.Tensor:
        """Convert a tensor from default to strided layout."""
        return x[..., self.indexes_to_strided]

    def to_catted(self, x: torch.Tensor) -> torch.Tensor:
        """Convert a tensor from strided to default layout."""
        return x[..., self.indexes_to_catted]
