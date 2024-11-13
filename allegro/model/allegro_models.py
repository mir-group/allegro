from e3nn import o3

from nequip.data import AtomicDataDict
from nequip.model import model_builder
from nequip.nn import (
    SequentialGraphNetwork,
    AtomwiseReduce,
    PerTypeScaleShift,
    ForceStressOutput,
)

from nequip.nn.embedding import (
    PolynomialCutoff,
    EdgeLengthNormalizer,
    BesselEdgeLengthEncoding,
)
from allegro.nn import (
    TwoBodySphericalHarmonicTensorEmbdedding,
    ProductTypeEmbedding,
    EdgewiseEnergySum,
    Allegro_Module,
    ScalarMLP,
)
from hydra.utils import instantiate
from typing import Sequence, Union, Optional, Dict


@model_builder
def AllegroEnergyModel(
    l_max: int,
    parity_setting: str,
    **kwargs,
):
    assert parity_setting in ("o3_full", "o3_restricted", "so3")
    irreps_edge_sh = repr(
        o3.Irreps.spherical_harmonics(l_max, p=(1 if parity_setting == "so3" else -1))
    )
    # set tensor_track_allowed_irreps
    # note that it is treated as a set, so order doesn't really matter
    if parity_setting == "o3_full":
        # we want all irreps up to lmax
        tensor_track_allowed_irreps = o3.Irreps(
            [(1, (this_l, p)) for this_l in range(l_max + 1) for p in (1, -1)]
        )
    else:
        # for so3 or o3_restricted, we want only irreps that show up in the original SH
        tensor_track_allowed_irreps = irreps_edge_sh

    return FullAllegroEnergyModel(
        irreps_edge_sh=irreps_edge_sh,
        tensor_track_allowed_irreps=tensor_track_allowed_irreps,
        **kwargs,
    )


@model_builder
def AllegroModel(**kwargs):
    """
    Args:
        l_max (int): maximum order l to use in spherical harmonics embedding, 1 is baseline (fast), 2 is more accurate, but slower, 3 highly accurate but slow
        parity_setting (str): whether to include parity symmetry equivariance; options are ``o3_full``, ``o3_restricted``, ``so3``
    """
    return ForceStressOutput(AllegroEnergyModel(**kwargs))


@model_builder
def FullAllegroEnergyModel(
    r_max: float,
    type_names: Sequence[str],
    # irreps
    irreps_edge_sh: Union[int, str, o3.Irreps],
    tensor_track_allowed_irreps: Union[str, o3.Irreps],
    # two body embedding
    two_body_embedding_dim: int,
    two_body_mlp_hidden_layer_depth: int,
    two_body_mlp_hidden_layer_width: int,
    # allegro layers
    num_layers: int,
    num_scalar_features: int,
    num_tensor_features: int,
    allegro_mlp_hidden_layer_depth: int,
    allegro_mlp_hidden_layer_width: int,
    # readout
    readout_mlp_hidden_layer_depth: int,
    readout_mlp_hidden_layer_width: int,
    # edge length encoding
    per_edge_type_cutoff: Optional[Dict[str, Union[float, Dict[str, float]]]] = None,
    num_bessels: int = 8,
    bessel_trainable: bool = False,
    polynomial_cutoff_p: int = 6,
    radial_basis_mlp_kwargs: Dict = {},
    # edge sum normalization
    avg_num_neighbors: Optional[float] = None,
    # allegro layers defaults
    tensors_mixing_mode: str = "p",
    tensor_track_weight_init: str = "uniform",
    weight_individual_irreps: bool = True,
    latent_resnet: bool = True,
    # per atom energy params
    per_type_energy_scales: Optional[Union[float, Sequence[float]]] = None,
    per_type_energy_shifts: Optional[Union[float, Sequence[float]]] = None,
    per_type_energy_scales_trainable: Optional[bool] = False,
    per_type_energy_shifts_trainable: Optional[bool] = False,
    pair_potential: Optional[Dict] = None,
):
    # === two-body scalar encoding ===
    edge_norm = EdgeLengthNormalizer(
        r_max=r_max,
        type_names=type_names,
        per_edge_type_cutoff=per_edge_type_cutoff,
    )
    bessel_encode = BesselEdgeLengthEncoding(
        num_bessels=num_bessels,
        trainable=bessel_trainable,
        cutoff=PolynomialCutoff(polynomial_cutoff_p),
        irreps_in=edge_norm.irreps_out,
    )
    typeembed = ProductTypeEmbedding(
        type_names=type_names,
        initial_embedding_dim=two_body_embedding_dim,
        radial_features_in_field=AtomicDataDict.EDGE_EMBEDDING_KEY,
        edge_embed_out_field=AtomicDataDict.EDGE_EMBEDDING_KEY,
        irreps_in=bessel_encode.irreps_out,
    )

    twobody_mlp = ScalarMLP(
        # input dims from previous module
        field=AtomicDataDict.EDGE_EMBEDDING_KEY,
        # output dims consistent with Allegro scalar dims
        mlp_output_dim=num_scalar_features,
        mlp_hidden_layer_depth=two_body_mlp_hidden_layer_depth,
        mlp_hidden_layer_width=two_body_mlp_hidden_layer_width,
        irreps_in=typeembed.irreps_out,
    )

    # === two-body tensor encoding ===
    tensor_embed = TwoBodySphericalHarmonicTensorEmbdedding(
        irreps_edge_sh=irreps_edge_sh,
        num_tensor_features=num_tensor_features,
        scalar_embedding_in_field=AtomicDataDict.EDGE_EMBEDDING_KEY,
        tensor_basis_out_field=AtomicDataDict.EDGE_ATTRS_KEY,
        tensor_embedding_out_field=AtomicDataDict.EDGE_FEATURES_KEY,
        irreps_in=twobody_mlp.irreps_out,
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
            "mlp_hidden_layer_depth": allegro_mlp_hidden_layer_depth,
            "mlp_hidden_layer_width": allegro_mlp_hidden_layer_width,
        },
        # best to use defaults for these
        tensors_mixing_mode=tensors_mixing_mode,
        tensor_track_weight_init=tensor_track_weight_init,
        weight_individual_irreps=weight_individual_irreps,
        # fields
        tensor_basis_in_field=AtomicDataDict.EDGE_ATTRS_KEY,
        tensor_features_in_field=AtomicDataDict.EDGE_FEATURES_KEY,
        scalar_in_field=AtomicDataDict.EDGE_EMBEDDING_KEY,
        scalar_out_field=AtomicDataDict.EDGE_FEATURES_KEY,
        irreps_in=tensor_embed.irreps_out,
    )

    # === allegro readout ===
    readout = ScalarMLP(
        mlp_output_dim=1,
        mlp_hidden_layer_depth=readout_mlp_hidden_layer_depth,
        mlp_hidden_layer_width=readout_mlp_hidden_layer_width,
        mlp_nonlinearity="silu" if readout_mlp_nonlinear else None,
        field=AtomicDataDict.EDGE_FEATURES_KEY,
        out_field=AtomicDataDict.EDGE_ENERGY_KEY,
        irreps_in=allegro.irreps_out,
    )

    # === edge -> atom ===
    edge_eng_sum = EdgewiseEnergySum(
        avg_num_neighbors=avg_num_neighbors,
        irreps_in=readout.irreps_out,
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

    modules = {
        "edge_norm": edge_norm,
        "bessel_encode": bessel_encode,
        "typeembed": typeembed,
        "twobody_mlp": twobody_mlp,
        "tensor_embed": tensor_embed,
        "allegro": allegro,
        "readout": readout,
        "edge_eng_sum": edge_eng_sum,
        "per_type_energy_scale_shift": per_type_energy_scale_shift,
    }

    # === pair potentials ===
    prev_irreps_out = per_type_energy_scale_shift.irreps_out
    if pair_potential is not None:
        pair_potential = instantiate(
            pair_potential, type_names=type_names, irreps_in=prev_irreps_out
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
