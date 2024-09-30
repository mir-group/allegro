import logging

from e3nn import o3

from nequip.data import AtomicDataDict

from nequip.nn import SequentialGraphNetwork, AtomwiseReduce

from nequip.nn.embedding import SphericalHarmonicEdgeAttrs

from allegro.nn import (
    NormalizedBasis,
    ProductTypeEmbedding,
    AllegroBesselBasis,
    EdgewiseEnergySum,
    Allegro_Module,
    ScalarMLP,
)
from allegro._keys import EDGE_FEATURES, EDGE_ENERGY


def _allegro_config_preprocess(config, initialize: bool):

    # Handle simple irreps
    if "l_max" in config:
        l_max = int(config["l_max"])
        parity_setting = config["parity"]
        assert parity_setting in ("o3_full", "o3_restricted", "so3")
        irreps_edge_sh = repr(
            o3.Irreps.spherical_harmonics(
                l_max, p=(1 if parity_setting == "so3" else -1)
            )
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
        # check consistant
        assert config.get("irreps_edge_sh", irreps_edge_sh) == irreps_edge_sh
        assert (
            config.get("tensor_track_allowed_irreps", tensor_track_allowed_irreps)
            == tensor_track_allowed_irreps
        )
        config["irreps_edge_sh"] = irreps_edge_sh
        config["tensor_track_allowed_irreps"] = tensor_track_allowed_irreps


def Allegro(config, initialize: bool):
    logging.debug("Building Allegro model...")

    _allegro_config_preprocess(config, initialize=initialize)

    layers = {
        # -- Encode --
        # Get various edge invariants
        "radial_basis": (
            NormalizedBasis
            if config.get("use_original_normalized_basis", False)
            else AllegroBesselBasis
        ),
        "typeembed": (
            ProductTypeEmbedding,
            dict(
                initial_scalar_embedding_dim=config.get(
                    "initial_scalar_embedding_dim",
                    # sane default to the MLP that comes next
                    config["two_body_latent_mlp_latent_dimensions"][0],
                ),
            ),
        ),
        # Get edge tensors
        "spharm": SphericalHarmonicEdgeAttrs,
        # The core allegro model:
        "allegro": (
            Allegro_Module,
            dict(
                field=AtomicDataDict.EDGE_ATTRS_KEY,  # initial input is the edge SH
                edge_invariant_field=AtomicDataDict.EDGE_EMBEDDING_KEY,
            ),
        ),
        "edge_eng": (
            ScalarMLP,
            dict(field=EDGE_FEATURES, out_field=EDGE_ENERGY, mlp_output_dimension=1),
        ),
        # Sum edgewise energies -> per-atom energies:
        "edge_eng_sum": EdgewiseEnergySum,
        # Sum system energy:
        "total_energy_sum": (
            AtomwiseReduce,
            dict(
                reduce="sum",
                field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
                out_field=AtomicDataDict.TOTAL_ENERGY_KEY,
            ),
        ),
    }

    model = SequentialGraphNetwork.from_parameters(shared_params=config, layers=layers)

    return model
