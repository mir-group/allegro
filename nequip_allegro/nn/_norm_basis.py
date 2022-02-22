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
        r_min: float,
        r_max: float,
        original_basis=BesselBasis,
        original_basis_kwargs: dict = {},
        n: int = 4000,
    ):
        super().__init__()
        self.basis = original_basis(**original_basis_kwargs)
        self.r_min = r_min
        self.r_max = r_max
        self.n = n

        self.num_basis = self.basis.num_basis

        # Uniform distribution on [r_min, r_max)
        with torch.no_grad():
            rs = torch.linspace(r_min, r_max, n)
            basis_std, basis_mean = torch.std_mean(self.basis(rs), dim=0)

        self.register_buffer("_mean", basis_mean)
        self.register_buffer("_inv_std", torch.reciprocal(basis_std))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.basis(x) - self._mean) * self._inv_std
