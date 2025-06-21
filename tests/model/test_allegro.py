import pytest
from nequip.utils.unittests.model_tests import BaseEnergyModelTests
from nequip.utils.versions import _TORCH_GE_2_6

try:
    import triton  # noqa: F401

    _TRITON_INSTALLED = True
except ImportError:
    _TRITON_INSTALLED = False


COMMON_CONFIG = {
    "_target_": "allegro.model.AllegroModel",
    "seed": 123,
    "type_names": ["H", "C", "O"],
    "r_max": 4.0,
    "avg_num_neighbors": 20.0,
    "radial_chemical_embed_dim": 16,
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
    "num_bessels": 8,
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

    @pytest.fixture(scope="class")
    def nequip_compile_tol(self, model_dtype):
        return {"float32": 5e-5, "float64": 1e-10}[model_dtype]

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

    @pytest.fixture(
        scope="class",
        params=[None]
        + (["enable_TritonContracter"] if _TORCH_GE_2_6 and _TRITON_INSTALLED else []),
    )
    def nequip_compile_acceleration_modifiers(self, request):
        """Test acceleration modifiers in nequip-compile workflows."""
        if request.param is None:
            return None

        def modifier_handler(mode, device):
            if request.param == "enable_TritonContracter":

                if mode == "torchscript":
                    pytest.skip(
                        "TritonContracter tests skipped for TorchScript compilation mode"
                    )

                if device == "cpu":
                    pytest.skip("TritonContracter tests skipped for CPU")

                return ["enable_TritonContracter"]
            else:
                raise ValueError(f"Unknown modifier: {request.param}")

        return modifier_handler
