import math

import torch


class BesselBasis(torch.nn.Module):
    r_max: float
    prefactor: float
    cutoff_p: float

    def __init__(
        self,
        r_max: float,
        num_bessels_per_basis: int = 8,
        num_bases: int = 1,
        trainable: bool = True,
    ):
        r"""Radial Bessel Basis with Polynomial Cutoff, as proposed in DimeNet: https://arxiv.org/abs/2003.03123


        Parameters
        ----------
        r_max : float
            Cutoff radius

        num_basis : int
            Number of Bessel Basis functions

        trainable : bool
            Train the :math:`n \pi` part or not.
        """
        super().__init__()

        self.trainable = trainable
        self.num_bases = num_bases
        assert num_bases >= 1
        if num_bases > 1:
            assert trainable
        self.num_bessels_per_basis = num_bessels_per_basis
        self.num_basis = num_bessels_per_basis * num_bases
        self.r_max = float(r_max)
        self.prefactor = math.sqrt(2.0 / self.r_max)

        bessel_weights = (
            (
                torch.linspace(
                    start=1.0, end=num_bessels_per_basis, steps=num_bessels_per_basis
                )
                * (math.pi / self.r_max)
            )
            .unsqueeze(0)
            .expand(num_bases, num_bessels_per_basis)
            .clone(memory_format=torch.contiguous_format)
        )
        if self.trainable:
            self.bessel_weights = torch.nn.Parameter(bessel_weights)
        else:
            self.register_buffer("bessel_weights", bessel_weights)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """
        Evaluate basis for input x.

        Parameters
        ----------
        r : torch.Tensor
            Values to embed

        Returns
        -------
            basis
        """
        r = r.view(-1, 1, 1)

        # [z, 1, 1] * [num_bases, num_basis] = [z, num_bases, num_basis]
        bessel = (self.prefactor / r) * torch.sin(r * self.bessel_weights)

        return bessel.view(-1, self.num_basis)


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
    norm_basis_mean_shift: bool

    def __init__(
        self,
        r_max: float,
        r_min: float = 0.0,
        original_basis=BesselBasis,
        original_basis_kwargs: dict = {},
        n: int = 4000,
        norm_basis_mean_shift: bool = False,
    ):
        super().__init__()
        self.basis = original_basis(**original_basis_kwargs)
        self.r_min = r_min
        self.r_max = r_max
        assert self.r_min >= 0.0
        assert self.r_max > r_min
        self.n = n

        self.num_basis = self.basis.num_basis
        self.norm_basis_mean_shift = norm_basis_mean_shift

        # Uniform distribution on [r_min, r_max)
        with torch.no_grad():
            # don't take 0 in case of weirdness like bessel at 0
            rs = torch.linspace(r_min, r_max, n + 1)[1:]
            bs = self.basis(rs)
            assert bs.ndim == 2 and len(bs) == n
            if norm_basis_mean_shift:
                basis_std, basis_mean = torch.std_mean(bs, dim=0)
            else:
                basis_std = bs.square().mean(dim=0).sqrt()
                basis_mean = torch.Tensor()

        self.register_buffer("_mean", basis_mean)
        self.register_buffer("_inv_std", torch.reciprocal(basis_std))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        basis = self.basis(x)
        if self.norm_basis_mean_shift:
            basis = basis - self._mean
        return basis * self._inv_std
