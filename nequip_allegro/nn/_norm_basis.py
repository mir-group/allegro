import torch

from nequip.nn.radial_basis import BesselBasis


class NormalizedBasis(torch.nn.Module):
    """Normalized version of a given radial basis.

    Args:
        basis (constructor): callable to build the underlying basis
        basis_kwargs (dict): parameters for the underlying basis
        n (int, optional): the number of samples to use for the estimated statistics
        r_min (float): the lower bound of the uniform square bump distribution for inputs
        r_max (float): the upper bound of the same
    """

    num_basis: int

    def __init__(
        self,
        r_max: float,
        r_min: float = 0.0,
        original_basis=BesselBasis,
        original_basis_kwargs: dict = {},
        n: int = 4000,
        norm_basis_mean_shift: bool = True,
    ):
        super().__init__()
        self.basis = original_basis(**original_basis_kwargs)
        self.r_min = r_min
        self.r_max = r_max
        assert self.r_min >= 0.0
        assert self.r_max > r_min
        self.n = n

        self.num_basis = self.basis.num_basis

        # Uniform distribution on [r_min, r_max)
        with torch.no_grad():
            # don't take 0 in case of weirdness like bessel at 0
            rs = torch.linspace(r_min, r_max, n + 1)[1:]
            bs = self.basis(rs)
            assert bs.ndim == 2 and len(bs) == n
            if norm_basis_mean_shift:
                basis_std, basis_mean = torch.std_mean(bs, dim=0)
            else:
                basis_std = bs.square().mean().sqrt()
                basis_mean = torch.as_tensor(
                    0.0, device=basis_std.device, dtype=basis_std.dtype
                )

        self.register_buffer("_mean", basis_mean)
        self.register_buffer("_inv_std", torch.reciprocal(basis_std))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.basis(x) - self._mean) * self._inv_std
