import pytest
from nequip.utils.unittests.model_tests import BaseEnergyModelTests


COMMON_CONFIG = {
    "_target_": "allegro.model.AllegroModel",
    "seed": 123,
    "type_names": ["H", "C", "O"],
    "r_max": 4.0,
    "avg_num_neighbors": 5.0,  # very approximate to keep numerics sane
    "radial_chemical_embed_dim": 8,
    "scalar_embed_mlp_hidden_layers_depth": 1,
    "scalar_embed_mlp_hidden_layers_width": 32,
    "num_layers": 2,
    "l_max": 2,
    "num_scalar_features": 32,
    "num_tensor_features": 4,
    "allegro_mlp_hidden_layers_depth": 2,
    "allegro_mlp_hidden_layers_width": 32,
    "readout_mlp_hidden_layers_depth": 1,
    "readout_mlp_hidden_layers_width": 8,
}

minimal_config0 = dict(
    **COMMON_CONFIG,
)
minimal_config1 = dict(
    per_edge_type_cutoff={"H": 2.0, "C": {"H": 4.0, "C": 3.5, "O": 3.7}, "O": 3.9},
    **COMMON_CONFIG,
)


BESSEL_CONFIG = {
    "_target_": "allegro.nn.TwoBodyBesselScalarEmbed",
    "num_bessels": 4,
}

SPLINE_CONFIG = {
    "_target_": "allegro.nn.TwoBodySplineScalarEmbed",
    "num_splines": 8,
    "spline_span": 6,
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

    @pytest.fixture(
        # only test True case to save time
        params=[True],  # False
        scope="class",
    )
    def parity(self, request):
        return request.param

    @pytest.fixture(
        params=[True, False],
        scope="class",
    )
    def tp_path_channel_coupling(self, request):
        return request.param

    @pytest.fixture(
        params=[
            minimal_config0,
            minimal_config1,
        ],
        scope="class",
    )
    def config(
        self,
        request,
        scalar_embed_config,
        parity,
        tp_path_channel_coupling,
    ):
        config = request.param
        config = config.copy()
        config.update({"radial_chemical_embed": scalar_embed_config})
        config.update({"parity": parity})
        config.update({"tp_path_channel_coupling": tp_path_channel_coupling})
        return config
