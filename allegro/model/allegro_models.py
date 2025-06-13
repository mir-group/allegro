# This file is a part of the `allegro` package. Please see LICENSE and README at the root for information on using it.
import math
from e3nn import o3

from nequip.data import AtomicDataDict
from nequip.model import model_builder
from nequip.nn import (
    SequentialGraphNetwork,
    ScalarMLP,
    AtomwiseReduce,
    PerTypeScaleShift,
    ForceStressOutput,
)

from nequip.nn.embedding import (
    EdgeLengthNormalizer,
    AddRadialCutoffToData,
    PolynomialCutoff,
)
from allegro.nn import (
    TwoBodySphericalHarmonicTensorEmbed,
    EdgewiseReduce,
    Allegro_Module,
)
from nequip.utils import RankedLogger

from hydra.utils import instantiate
from typing import Sequence, Union, Optional, Dict


logger = RankedLogger(__name__, rank_zero_only=True)


def _allegro_docstring(header: str) -> str:
    """Generate common docstring for Allegro models with customizable header."""
    return f"""{header}

    Args:
        seed (int): seed for reproducibility
        model_dtype (str): ``float32`` or ``float64``
        r_max (float): cutoff radius
        per_edge_type_cutoff (Dict): one can optionally specify cutoffs for each edge type [must be smaller than ``r_max``] (default ``None``)
        type_names (Sequence[str]): list of atom type names
        l_max (int): maximum order :math:`\\ell` to use in spherical harmonics embedding, 1 is baseline (fast), 2 is more accurate, but slower, 3 highly accurate but slow
        parity (bool): whether to include features with odd mirror parity (default ``True``)
        radial_chemical_embed: an Allegro-compatible two-body radial-chemical embedding module, e.g. :class:`allegro.nn.TwoBodyBesselScalarEmbed`
        two_body_mlp_hidden_layers_depth (int): number of hidden layers of two-body MLP (default ``1``)
        two_body_mlp_hidden_layers_width (int): depth of hidden layers of two-body MLP
        two_body_mlp_nonlinearity (str): ``silu``, ``mish``, ``gelu``, or ``None`` (default ``silu``)
        scalar_embed_output_dim (int): output dimension of the scalar embedding module (default ``None`` will use ``num_scalar_features``)
        num_layers (int): number of Allegro layers
        num_scalar_features (int): multiplicity of scalar features in the Allegro layers
        num_tensor_features (int): multiplicity of tensor features in the Allegro layers
        allegro_mlp_hidden_layers_depth (int): number of hidden layers in the Allegro scalar MLPs (default ``1``)
        allegro_mlp_hidden_layers_width (int): width of hidden layers in the Allegro scalar MLPs (reasonable to set it to be the same as ``num_scalar_features``)
        allegro_mlp_nonlinearity (str): ``silu``, ``mish``, ``gelu``, or ``None`` (default ``silu``)
        tp_path_channel_coupling (bool): whether Allegro tensor product weights couple the paths with the channels or not, ``True`` is expected to be more expressive than ``False`` (default ``True``)
        readout_mlp_hidden_layers_depth (int): number of hidden layers in the readout MLP (default ``1``)
        readout_mlp_hidden_layers_width (int): width of hidden layers in the readout MLP (reasonable to set it to be the same as ``num_scalar_features``)
        readout_mlp_nonlinearity (str): ``silu``, ``mish``, ``gelu``, or ``None`` (default ``silu``)
        avg_num_neighbors (float): used to normalize edge sums for better numerics (default ``None``)
        per_type_energy_scales (float/List[float]): per-atom energy scales, which could be derived from the force RMS of the data (default ``None``)
        per_type_energy_shifts (float/List[float]): per-atom energy shifts, which should generally be isolated atom reference energies or estimated from average pre-atom energies of the data (default ``None``)
        per_type_energy_scales_trainable (bool): whether the per-atom energy scales are trainable (default ``False``)
        per_type_energy_shifts_trainable (bool): whether the per-atom energy shifts are trainable (default ``False``)
        pair_potential (torch.nn.Module): additional pair potential term, e.g. :class:``nequip.nn.pair_potential.ZBL`` (default ``None``)
    """


@model_builder
def AllegroEnergyModel(
    l_max: int,
    parity: bool = True,
    **kwargs,
):
    irreps_edge_sh = repr(o3.Irreps.spherical_harmonics(l_max, p=-1))
    # set tensor_track_allowed_irreps
    # note that it is treated as a set, so order doesn't really matter
    if parity:
        # we want all irreps up to lmax
        tensor_track_allowed_irreps = o3.Irreps(
            [(1, (this_l, p)) for this_l in range(l_max + 1) for p in (1, -1)]
        )
    else:
        # we want only irreps that show up in the original SH
        tensor_track_allowed_irreps = irreps_edge_sh

    return FullAllegroEnergyModel(
        irreps_edge_sh=irreps_edge_sh,
        tensor_track_allowed_irreps=tensor_track_allowed_irreps,
        **kwargs,
    )


# assign docstrings using the shared function
AllegroEnergyModel.__doc__ = _allegro_docstring(
    "Allegro model that predicts energies only."
)


@model_builder
def AllegroModel(**kwargs):
    return ForceStressOutput(AllegroEnergyModel(**kwargs))


# assign docstring for the force+energy model
AllegroModel.__doc__ = _allegro_docstring(
    "Allegro model that predicts energies and forces (and stresses if cell is provided)."
)


@model_builder
def FullAllegroEnergyModel(
    r_max: float,
    type_names: Sequence[str],
    # irreps
    irreps_edge_sh: Union[int, str, o3.Irreps],
    tensor_track_allowed_irreps: Union[str, o3.Irreps],
    # scalar embed
    radial_chemical_embed: Dict,
    radial_chemical_embed_dim: Optional[int] = None,
    per_edge_type_cutoff: Optional[Dict[str, Union[float, Dict[str, float]]]] = None,
    # scalar embed MLP
    scalar_embed_mlp_hidden_layers_depth: int = 1,
    scalar_embed_mlp_hidden_layers_width: int = 64,
    scalar_embed_mlp_nonlinearity: int = "silu",
    # allegro layers
    num_layers: int = 2,
    num_scalar_features: int = 64,
    num_tensor_features: int = 16,
    allegro_mlp_hidden_layers_depth: int = 1,
    allegro_mlp_hidden_layers_width: int = 64,
    allegro_mlp_nonlinearity: Optional[str] = "silu",
    tp_path_channel_coupling: bool = True,
    # readout
    readout_mlp_hidden_layers_depth: int = 1,
    readout_mlp_hidden_layers_width: int = 32,
    readout_mlp_nonlinearity: Optional[str] = "silu",
    # edge sum normalization
    avg_num_neighbors: Optional[float] = None,
    # allegro layers defaults
    weight_individual_irreps: bool = True,
    # per atom energy params
    per_type_energy_scales: Optional[Union[float, Sequence[float]]] = None,
    per_type_energy_shifts: Optional[Union[float, Sequence[float]]] = None,
    per_type_energy_scales_trainable: Optional[bool] = False,
    per_type_energy_shifts_trainable: Optional[bool] = False,
    pair_potential: Optional[Dict] = None,
    # weight initialization and normalization
    forward_normalize: bool = True,
):
    # === two-body scalar embedding ===
    edge_norm = EdgeLengthNormalizer(
        r_max=r_max,
        type_names=type_names,
        per_edge_type_cutoff=per_edge_type_cutoff,
    )
    radial_chemical_embed_module = instantiate(
        radial_chemical_embed,
        type_names=type_names,
        module_output_dim=(
            num_scalar_features
            if radial_chemical_embed_dim is None
            else radial_chemical_embed_dim
        ),
        forward_weight_init=forward_normalize,
        scalar_embed_field=AtomicDataDict.EDGE_EMBEDDING_KEY,
        irreps_in=edge_norm.irreps_out,
    )
    # ^ note that this imposes a contract with two-body scalar embedding modules
    # i.e. they must have `type_names`, `module_output_dim`, `scalar_embed_field`, `irreps_in`

    scalar_embed_mlp = ScalarMLP(
        output_dim=num_scalar_features,
        hidden_layers_depth=scalar_embed_mlp_hidden_layers_depth,
        hidden_layers_width=scalar_embed_mlp_hidden_layers_width,
        nonlinearity=scalar_embed_mlp_nonlinearity,
        bias=False,
        forward_weight_init=forward_normalize,
        field=AtomicDataDict.EDGE_EMBEDDING_KEY,
        out_field=AtomicDataDict.EDGE_EMBEDDING_KEY,
        irreps_in=radial_chemical_embed_module.irreps_out,
    )

    # === two-body tensor embedding ===
    tensor_embed = TwoBodySphericalHarmonicTensorEmbed(
        irreps_edge_sh=irreps_edge_sh,
        num_tensor_features=num_tensor_features,
        forward_weight_init=forward_normalize,
        scalar_embedding_in_field=AtomicDataDict.EDGE_EMBEDDING_KEY,
        tensor_basis_out_field=AtomicDataDict.EDGE_ATTRS_KEY,
        tensor_embedding_out_field=AtomicDataDict.EDGE_FEATURES_KEY,
        irreps_in=scalar_embed_mlp.irreps_out,
    )

    # === allegro module ===
    allegro = Allegro_Module(
        num_layers=num_layers,
        num_scalar_features=num_scalar_features,
        num_tensor_features=num_tensor_features,
        tensor_track_allowed_irreps=tensor_track_allowed_irreps,
        avg_num_neighbors=avg_num_neighbors,
        # MLP
        latent_kwargs={
            "hidden_layers_depth": allegro_mlp_hidden_layers_depth,
            "hidden_layers_width": allegro_mlp_hidden_layers_width,
            "nonlinearity": allegro_mlp_nonlinearity,
            "bias": False,
            "forward_weight_init": forward_normalize,
        },
        tp_path_channel_coupling=tp_path_channel_coupling,
        # best to use defaults for these
        weight_individual_irreps=weight_individual_irreps,
        # fields
        tensor_basis_in_field=AtomicDataDict.EDGE_ATTRS_KEY,
        tensor_features_in_field=AtomicDataDict.EDGE_FEATURES_KEY,
        scalar_in_field=AtomicDataDict.EDGE_EMBEDDING_KEY,
        scalar_out_field=AtomicDataDict.EDGE_FEATURES_KEY,
        irreps_in=tensor_embed.irreps_out,
    )

    modules = {
        "edge_norm": edge_norm,
        "radial_chemical_embed": radial_chemical_embed_module,
        "scalar_embed_mlp": scalar_embed_mlp,
        "tensor_embed": tensor_embed,
        "allegro": allegro,
    }

    # === allegro readout ===
    edge_readout = ScalarMLP(
        output_dim=1,
        hidden_layers_depth=readout_mlp_hidden_layers_depth,
        hidden_layers_width=readout_mlp_hidden_layers_width,
        nonlinearity=readout_mlp_nonlinearity,
        bias=False,
        forward_weight_init=forward_normalize,
        field=AtomicDataDict.EDGE_FEATURES_KEY,
        out_field=AtomicDataDict.EDGE_ENERGY_KEY,
        irreps_in=allegro.irreps_out,
    )
    edge_eng_sum = EdgewiseReduce(
        field=AtomicDataDict.EDGE_ENERGY_KEY,
        out_field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
        factor=1.0 / math.sqrt(2 * avg_num_neighbors),
        # ^ factor of 2 to normalize dE/dr_i which includes both contributions from dE/dr_ij and every other derivative against r_ji
        irreps_in=edge_readout.irreps_out,
    )

    # === per type scale shift ===
    per_type_energy_scale_shift = PerTypeScaleShift(
        type_names=type_names,
        field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
        out_field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
        scales=per_type_energy_scales,
        shifts=per_type_energy_shifts,
        scales_trainable=per_type_energy_scales_trainable,
        shifts_trainable=per_type_energy_shifts_trainable,
        irreps_in=edge_eng_sum.irreps_out,
    )

    modules.update(
        {
            "edge_readout": edge_readout,
            "edge_eng_sum": edge_eng_sum,
            "per_type_energy_scale_shift": per_type_energy_scale_shift,
        }
    )

    # === pair potentials ===
    prev_irreps_out = per_type_energy_scale_shift.irreps_out
    if pair_potential is not None:

        # case where model doesn't have edge cutoffs up to this point, but pair potential required
        if AtomicDataDict.EDGE_CUTOFF_KEY not in prev_irreps_out:
            cutoff = AddRadialCutoffToData(
                cutoff=PolynomialCutoff(6),
                irreps_in=prev_irreps_out,
            )
            prev_irreps_out = cutoff.irreps_out
            modules.update({"cutoff": cutoff})

        pair_potential = instantiate(
            pair_potential,
            type_names=type_names,
            irreps_in=prev_irreps_out,
        )
        prev_irreps_out = pair_potential.irreps_out
        modules.update({"pair_potential": pair_potential})

    # === sum to total energy ===
    total_energy_sum = AtomwiseReduce(
        irreps_in=prev_irreps_out,
        reduce="sum",
        field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
        out_field=AtomicDataDict.TOTAL_ENERGY_KEY,
    )
    modules.update({"total_energy_sum": total_energy_sum})

    # === finalize model ===
    return SequentialGraphNetwork(modules)


@model_builder
def FullAllegroModel(**kwargs):
    return ForceStressOutput(FullAllegroEnergyModel(**kwargs))
