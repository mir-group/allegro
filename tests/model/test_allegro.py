import pytest
import torch
import copy
from nequip.utils.unittests.model_tests import BaseEnergyModelTests
from nequip.utils.versions import _TORCH_GE_2_6

try:
    import triton  # noqa: F401

    _TRITON_INSTALLED = True
except ImportError:
    _TRITON_INSTALLED = False

try:
    import cuequivariance  # noqa: F401
    import cuequivariance_torch  # noqa: F401

    _CUEQUIVARIANCE_INSTALLED = True
except ImportError:
    _CUEQUIVARIANCE_INSTALLED = False


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
        + (["enable_TritonContracter"] if _TORCH_GE_2_6 and _TRITON_INSTALLED else [])
        + (
            ["enable_CuEquivarianceContracter"]
            if _TORCH_GE_2_6 and _CUEQUIVARIANCE_INSTALLED
            else []
        ),
    )
    def nequip_compile_acceleration_modifiers(self, request):
        """Test acceleration modifiers in nequip-compile workflows."""
        if request.param is None:
            return None

        def modifier_handler(mode, device, model_dtype):
            if request.param == "enable_TritonContracter":

                if mode == "torchscript":
                    pytest.skip(
                        "TritonContracter tests skipped for TorchScript compilation mode"
                    )

                if device == "cpu":
                    pytest.skip("TritonContracter tests skipped for CPU")

                return ["enable_TritonContracter"]

            elif request.param == "enable_CuEquivarianceContracter":

                if device == "cpu":
                    pytest.skip("CuEquivarianceContracter tests skipped for CPU")

                # TODO: sort this out
                if mode == "aotinductor" and model_dtype == "float64":
                    pytest.skip(
                        "CuEquivarianceContracter tests skipped for AOTI and float64 due to known issue"
                    )

                return ["enable_CuEquivarianceContracter"]

            else:
                raise ValueError(f"Unknown modifier: {request.param}")

        return modifier_handler

    @pytest.fixture(
        scope="class",
        params=[None]
        + (
            ["enable_CuEquivarianceContracter"]
            if _TORCH_GE_2_6 and _CUEQUIVARIANCE_INSTALLED
            else []
        ),
    )
    def train_time_compile_acceleration_modifiers(self, request):
        """Test acceleration modifiers in train-time compile workflows."""
        if request.param is None:
            return None

        def modifier_handler(device):
            if request.param == "enable_CuEquivarianceContracter":

                if device == "cpu":
                    pytest.skip("CuEquivarianceContracter tests skipped for CPU")

                return [{"modifier": "enable_CuEquivarianceContracter"}]
            else:
                raise ValueError(f"Unknown modifier: {request.param}")

        return modifier_handler

    @pytest.mark.skipif(
        not (_TORCH_GE_2_6 and _TRITON_INSTALLED),
        reason="TritonContracter requires torch >= 2.6 and triton",
    )
    def test_triton_contracter_consistency(
        self, model, model_test_data, device, nequip_compile_tol
    ):
        """Test that TritonContracter-enabled model produces consistent results with original model."""
        if device == "cpu":
            pytest.skip("TritonContracter tests skipped for CPU")

        original_model, config, _ = model

        # create TritonContracter-enabled model
        triton_config = copy.deepcopy(config)
        triton_config = {
            "_target_": "nequip.model.modify",
            "modifiers": [{"modifier": "enable_TritonContracter"}],
            "model": triton_config,
        }
        triton_model = self.make_model(triton_config, device=device)
        triton_model.eval()
        triton_model.load_state_dict(original_model.state_dict())

        # test
        original_output = original_model(model_test_data.copy())
        triton_output = triton_model(model_test_data.copy())
        for key in ["atomic_energy", "total_energy", "forces"]:
            if key in original_output and key in triton_output:
                assert torch.allclose(
                    original_output[key],
                    triton_output[key],
                    rtol=nequip_compile_tol,
                    atol=nequip_compile_tol,
                ), f"Outputs differ for key {key}: max diff = {torch.max(torch.abs(original_output[key] - triton_output[key])).item()}"

    @pytest.mark.skipif(
        not (_TORCH_GE_2_6 and _CUEQUIVARIANCE_INSTALLED),
        reason="CuEquivarianceContracter requires cuequivariance",
    )
    def test_cuequivariance_contracter_consistency(
        self, model, model_test_data, device, nequip_compile_tol
    ):
        """Test that CuEquivarianceContracter-enabled model produces consistent results with original model."""
        original_model, config, _ = model

        # create CuEquivariance-enabled model
        cueq_config = copy.deepcopy(config)
        cueq_config = {
            "_target_": "nequip.model.modify",
            "modifiers": [{"modifier": "enable_CuEquivarianceContracter"}],
            "model": cueq_config,
        }
        cueq_model = self.make_model(cueq_config, device=device)
        cueq_model.load_state_dict(original_model.state_dict())

        # test
        self.compare_output_and_gradients(
            original_model, cueq_model, model_test_data, nequip_compile_tol
        )
