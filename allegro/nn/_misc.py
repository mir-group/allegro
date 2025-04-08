# This file is a part of the `allegro` package. Please see LICENSE and README at the root for information on using it.
import math

import torch


class ScalarMultiply(torch.nn.Module):
    alpha: float

    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * x


def _init_weight(w: torch.Tensor, initialization: str, allow_orthogonal: bool = False):
    with torch.no_grad():
        if initialization == "normal":
            w.normal_()
        elif initialization == "uniform":
            # these values give < x^2 > = 1
            w.uniform_(-math.sqrt(3), math.sqrt(3))
        elif allow_orthogonal and initialization == "orthogonal":
            # this rescaling gives < x^2 > = 1
            torch.nn.init.orthogonal_(w, gain=math.sqrt(max(w.shape)))
        else:
            raise NotImplementedError(f"Invalid initialization {initialization}")
