import pytest
from nequip.utils.unittests.model_tests import BaseEnergyModelTests


COMMON_CONFIG = {
    "_target_": "allegro.model.AllegroModel",
    "seed": 123,
    "type_names": ["H", "C", "O"],
    "r_max": 4.0,
    "avg_num_neighbors": 5.0,  # very approximate to keep numerics sane
    "num_layers": 2,
    "l_max": 2,
    "num_scalar_features": 32,
    "num_tensor_features": 4,
    "allegro_mlp_hidden_layer_depth": 2,
    "allegro_mlp_hidden_layer_width": 32,
    "readout_mlp_hidden_layer_depth": 1,
    "readout_mlp_hidden_layer_width": 8,
}

minimal_config0 = dict(
    **COMMON_CONFIG,
)
minimal_config1 = dict(
    per_edge_type_cutoff={"H": 2.0, "C": {"H": 4.0, "C": 3.5, "O": 3.7}, "O": 3.9},
    **COMMON_CONFIG,
)
minimal_config2 = dict(
    tensors_mixing_mode="uvvp",
    **COMMON_CONFIG,
)

BESSEL_CONFIG = {
    "_target_": "allegro.nn.TwoBodyBesselScalarEmbed",
    "num_bessels": 4,
    "two_body_embedding_dim": 8,
    "two_body_mlp_hidden_layer_depth": 1,
    "two_body_mlp_hidden_layer_width": 32,
}

SPLINE_CONFIG = {
    "_target_": "allegro.nn.TwoBodySplineScalarEmbed",
    "spline_grid": 5,
    "spline_span": 3,
}


class TestAllegro(BaseEnergyModelTests):
    @pytest.fixture
    def strict_locality(self):
        return True

    @pytest.fixture(
        params=[BESSEL_CONFIG, SPLINE_CONFIG],
        scope="class",
    )
    def scalar_embed_config(self, request):
        return request.param

    # TODO: test so3 mode when can pass down option to assert equivariance to ignore parity
    @pytest.fixture(
        params=["o3_full", "o3_restricted"],
        scope="class",
    )
    def parity_setting(self, request):
        return request.param

    @pytest.fixture(
        params=[True, False],
        scope="class",
    )
    def scatter_features(self, request):
        return request.param

    @pytest.fixture(
        params=[True, False],
        scope="class",
    )
    def node_readout(self, request):
        return request.param

    @pytest.fixture(
        params=[
            minimal_config0,
            minimal_config1,
            # minimal_config2,
        ],
        scope="class",
    )
    def config(
        self,
        request,
        scalar_embed_config,
        parity_setting,
        scatter_features,
        node_readout,
    ):
        config = request.param
        config = config.copy()
        config.update({"scalar_embed": scalar_embed_config})
        config.update({"parity_setting": parity_setting})
        config.update({"scatter_features": scatter_features})
        config.update({"node_readout": node_readout})
        return config
