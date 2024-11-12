import pytest

from nequip.utils.unittests.model_tests import BaseEnergyModelTests

BASIC_INFO = {
    "_target_": "allegro.model.AllegroModel",
    "seed": 123,
    "type_names": ["H", "C", "O"],
    "r_max": 4.0,
}

COMMON_CONFIG = {
    "num_bessels": 4,
    "avg_num_neighbors": 5.0,  # very approximate to keep numerics sane
    "num_tensor_features": 4,
    "two_body_latent_kwargs": {
        "mlp_latent_dimensions": [32],
    },
    "latent_kwargs": {
        "mlp_latent_dimensions": [32, 32],
    },
    "env_embed_kwargs": {
        "mlp_latent_dimensions": [],
    },
    "edge_eng_kwargs": {
        "mlp_latent_dimensions": [8],
    },
    **BASIC_INFO,
}
# TODO: test so3 mode when can pass down option to assert equivariance to ignore parity
minimal_config1 = dict(
    l_max=1,
    parity_setting="o3_full",
    num_layers=1,
    **COMMON_CONFIG,
)
minimal_config2 = dict(
    l_max=3,
    parity_setting="o3_full",
    num_layers=2,
    per_edge_type_cutoff={"H": 2.0, "C": {"H": 4.0, "C": 3.5, "O": 3.7}, "O": 3.9},
    **COMMON_CONFIG,
)
minimal_config3 = dict(
    l_max=2,
    parity_setting="o3_restricted",
    num_layers=4,
    **COMMON_CONFIG,
)
minimal_config4 = dict(
    l_max=3,
    parity_setting="o3_full",
    num_layers=3,
    latent_resnet=False,
    **COMMON_CONFIG,
)
minimal_config5 = dict(
    l_max=3,
    parity_setting="o3_full",
    num_layers=3,
    latent_resnet=True,
    tensors_mixing_mode="uvvp",
    **COMMON_CONFIG,
)
minimal_config6 = dict(
    l_max=4,
    parity_setting="o3_full",
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
    def config(self, request):
        config = request.param
        config = config.copy()
        return config
