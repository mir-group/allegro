from typing import List, Optional, Union

import torch

from e3nn import o3
from e3nn.util.jit import compile_mode
from e3nn.util.codegen import CodeGenMixin

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin

from allegro.utils import to_int_list


@compile_mode("script")
class ScalarMLP(GraphModuleMixin, torch.nn.Module):
    """Apply an MLP to some scalar field."""

    field: str
    out_field: str

    def __init__(
        self,
        mlp_output_dim: Optional[int],
        mlp_hidden_layer_dims: Optional[Union[List[int], str]] = None,
        mlp_hidden_layer_depth: Optional[int] = None,
        mlp_hidden_layer_width: Optional[int] = None,
        mlp_nonlinearity: Optional[str] = "silu",
        mlp_bias: bool = False,
        mlp_bfloat16: bool = False,
        field: str = AtomicDataDict.NODE_FEATURES_KEY,
        out_field: Optional[str] = None,
        irreps_in=None,
    ):
        super().__init__()
        self.field = field
        self.out_field = out_field if out_field is not None else field
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[self.field],
        )

        assert len(self.irreps_in[self.field]) == 1
        assert self.irreps_in[self.field][0].ir == (0, 1)  # scalars
        self._module = ScalarMLPFunction(
            mlp_input_dim=self.irreps_in[self.field][0].mul,
            mlp_hidden_layer_dims=mlp_hidden_layer_dims,
            mlp_hidden_layer_depth=mlp_hidden_layer_depth,
            mlp_hidden_layer_width=mlp_hidden_layer_width,
            mlp_output_dim=mlp_output_dim,
            mlp_nonlinearity=mlp_nonlinearity,
            mlp_bias=mlp_bias,
            mlp_bfloat16=mlp_bfloat16,
        )
        self.irreps_out[self.out_field] = o3.Irreps(
            [(self._module.out_features, (0, 1))]
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data[self.out_field] = self._module(data[self.field])
        return data


class ScalarMLPFunction(CodeGenMixin, torch.nn.Module):
    """Module implementing an MLP according to provided options."""

    in_features: int
    out_features: int
    num_layers: int
    use_bfloat16: bool
    is_nonlinear: bool

    def __init__(
        self,
        mlp_input_dim: Optional[int],
        mlp_output_dim: Optional[int],
        mlp_hidden_layer_dims: Optional[Union[List[int], str]] = None,
        mlp_hidden_layer_depth: Optional[int] = None,
        mlp_hidden_layer_width: Optional[int] = None,
        mlp_extra_output_dim: int = 0,
        mlp_nonlinearity: Optional[str] = "silu",
        mlp_bias: bool = False,
        mlp_bfloat16: bool = False,
    ):
        super().__init__()

        # === handle nonlinearity ===
        nonlinearity = {
            None: torch.nn.Identity,
            "null": torch.nn.Identity,
            "None": torch.nn.Identity,
            "silu": torch.nn.SiLU,
            "mish": torch.nn.Mish,
            "gelu": torch.nn.GELU,
        }[mlp_nonlinearity]
        self.is_nonlinear = False  # updated below

        # === handle MLP dimensions ===
        err_msg = "either `mlp_hidden_layer_dims` OR `mlp_hidden_layer_depth` and `mlp_hidden_layer_width` must be provided, but not both."
        if mlp_hidden_layer_dims is None:
            assert (
                mlp_hidden_layer_depth is not None
                and mlp_hidden_layer_width is not None
            ), err_msg
            mlp_hidden_layer_dims = mlp_hidden_layer_depth * [mlp_hidden_layer_width]
        else:
            assert (
                mlp_hidden_layer_depth is None and mlp_hidden_layer_width is None
            ), err_msg
            mlp_hidden_layer_dims = to_int_list(mlp_hidden_layer_dims)
        self.dims = (
            ([mlp_input_dim] if mlp_input_dim is not None else [])
            + mlp_hidden_layer_dims
            + ([mlp_output_dim] if mlp_output_dim is not None else [])
        )
        self.dims[-1] += mlp_extra_output_dim
        assert (
            len(self.dims) >= 2
        ), f"`ScalarMLPFunction must have >= 2 dimensions (input and output), but found {self.dims}"
        num_layers = len(self.dims) - 1
        self.num_layers = num_layers
        self.in_features = self.dims[0]
        self.out_features = self.dims[-1]

        self.mlp = torch.nn.Sequential()
        for layer, (h_in, h_out) in enumerate(zip(self.dims, self.dims[1:])):
            # === instantiate `Linear` with no bias ===
            linear_layer = torch.nn.Linear(
                h_in,
                h_out,
                bias=mlp_bias,
                dtype=torch.bfloat16 if mlp_bfloat16 else None,
            )
            # === weight initialization ===
            # normalize with output dim, i.e. normalize backwards pass for forces
            if nonlinearity is None or layer == self.num_layers - 1:
                # no nonlinearity or last layer -> use `linear` gain (i.e. 1)
                init_nonlinearity = "linear"
            else:
                # even though we use SiLU, we just use the ReLU gain
                init_nonlinearity = "relu"
            torch.nn.init.kaiming_uniform_(
                linear_layer.weight, mode="fan_out", nonlinearity=init_nonlinearity
            )

            self.mlp.append(linear_layer)
            if layer != self.num_layers - 1:
                self.mlp.append(nonlinearity())
                if not self.is_nonlinear:
                    self.is_nonlinear = not isinstance(nonlinearity, torch.nn.Identity)

        self.use_bfloat16 = mlp_bfloat16

    def forward(self, x):
        if self.use_bfloat16:
            return self.mlp(x.to(torch.bfloat16)).to(x.dtype)
        else:
            return self.mlp(x)
