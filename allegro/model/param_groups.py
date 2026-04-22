# This file is a part of the `allegro` package. Please see LICENSE and README at the root for information on using it.
import torch


def MuonParamGroups(
    model: torch.nn.Module,
    muon: dict,
    adam: dict,
):
    """
    Build optimizer parameter groups, splitting parameters between a Muon-based optimizer
    and Adam (or Adam-like) optimizer.

    This parameter group function is intended for use with the ``nequip.train.MuonWithAuxAdam``
    optimizer, where a subset of parameters is updated using Muon and the remainder with Adam.

    Assigned to Adam group:
      - Any parameter whose name contains the substring ``"readout"`` or ``"energy_output"`` or ``"pair_potential"``.
      - Any parameter not matching the Muon-specific rules below.

    Assigned to Muon group:
      - MLP weight matrices.

    Args:
        model (torch.nn.Module): The model to optimize.
        muon (dict): Muon config parameters.
        adam (dict): Adam config parameters.
    """
    muon_weights = []
    adam_weights = []

    for name, param in model.named_parameters():
        if "readout" in name or "energy_output" in name or "pair_potential" in name:
            adam_weights.append(param)
        elif param.ndim == 2 and "mlp" in name:
            muon_weights.append(param)
        else:
            adam_weights.append(param)

    param_groups = [
        dict(params=muon_weights, use_muon=True, e3nn_reshaping={}, **muon),
        dict(params=adam_weights, use_muon=False, **adam),
    ]

    return param_groups
