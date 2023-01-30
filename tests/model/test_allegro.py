import pytest

from nequip.data import AtomicDataDict
from nequip.utils.unittests.model_tests import BaseEnergyModelTests

COMMON_CONFIG = {
    "avg_num_neighbors": 5.0,  # very approximate to keep numerics sane
    "r_max": 4,
    "num_bessels_per_basis": 4,
    "num_bases": 4,
    "num_types": 3,
    "types_names": ["H", "C", "O"],
    "num_tensor_features": 4,
    "two_body_latent_mlp_latent_dimensions": [32],
    "latent_mlp_latent_dimensions": [32, 32],
    "env_embed_mlp_latent_dimensions": [],
    "edge_eng_mlp_latent_dimensions": [8],
}
# TODO: test so3 mode when can pass down option to assert equivariance to ignore parity
minimal_config1 = dict(
    l_max=1,
    parity="o3_full",
    num_layers=1,
    **COMMON_CONFIG,
)
minimal_config2 = dict(
    l_max=3,
    parity="o3_full",
    num_layers=2,
    **COMMON_CONFIG,
)
minimal_config3 = dict(
    l_max=2,
    parity="o3_restricted",
    num_layers=4,
    **COMMON_CONFIG,
)
minimal_config4 = dict(
    l_max=3,
    parity="o3_full",
    num_layers=3,
    latent_resnet=False,
    **COMMON_CONFIG,
)
minimal_config5 = dict(
    l_max=3,
    parity="o3_full",
    num_layers=3,
    latent_resnet=True,
    tensors_mixing_mode="uvv",
    **COMMON_CONFIG,
)
minimal_config6 = dict(
    l_max=4,
    parity="o3_full",
    num_layers=2,
    tensors_mixing_mode="p",
    **COMMON_CONFIG,
)


class TestAllegro(BaseEnergyModelTests):
    @pytest.fixture
    def strict_locality(self):
        return True

    @pytest.fixture(
        params=[
            minimal_config1,
            minimal_config2,
            minimal_config3,
            minimal_config4,
            minimal_config5,
            minimal_config6,
        ],
        scope="class",
    )
    def base_config(self, request):
        return request.param

    @pytest.fixture(
        params=[
            (
                ["allegro.model.Allegro", "ForceOutput"],
                [
                    AtomicDataDict.TOTAL_ENERGY_KEY,
                    AtomicDataDict.PER_ATOM_ENERGY_KEY,
                    AtomicDataDict.FORCE_KEY,
                    AtomicDataDict.EDGE_FEATURES_KEY,
                ],
            ),
            (
                ["allegro.model.Allegro", "StressForceOutput"],
                [
                    AtomicDataDict.TOTAL_ENERGY_KEY,
                    AtomicDataDict.PER_ATOM_ENERGY_KEY,
                    AtomicDataDict.FORCE_KEY,
                    AtomicDataDict.STRESS_KEY,
                    AtomicDataDict.VIRIAL_KEY,
                ],
            ),
        ],
        scope="class",
    )
    def config(self, request, base_config):
        config = base_config.copy()
        builder, out_fields = request.param
        config = config.copy()
        config["model_builders"] = builder
        return config, out_fields
