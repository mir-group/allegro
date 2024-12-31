from typing import List, Optional, Union
import math

import torch
from torch import fx

from e3nn import o3
from e3nn.util.jit import compile_mode
from e3nn.util.codegen import CodeGenMixin

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin
from nequip.nn.nonlinearities import ShiftedSoftPlus

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
        mlp_bfloat16: bool = False,
    ):
        super().__init__()

        # === handle nonlinearity ===
        nonlinearity = {
            None: None,
            # flexible parsing
            "null": None,
            "None": None,
            "silu": torch.nn.functional.silu,
            "ssp": ShiftedSoftPlus,
        }[mlp_nonlinearity]
        self.is_nonlinear = False  # updated in codegen below

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

        # === code ===
        params = {}
        graph = fx.Graph()
        tracer = fx.proxy.GraphAppendingTracer(graph)

        def Proxy(n):
            return fx.Proxy(n, tracer=tracer)

        base = torch.nn.Module()

        # make weights
        for layer, (h_in, h_out) in enumerate(zip(self.dims, self.dims[1:])):
            w = torch.empty(h_in, h_out)
            # normalize with output dim, i.e. normalize backwards pass for forces
            if nonlinearity is None or layer == self.num_layers - 1:
                # i.e. kaiming with no gain since no nonlinearity or last layer
                bound = math.sqrt(3 / h_out)
                torch.nn.init.uniform_(w, a=-bound, b=bound)
            else:
                # use kaiming uniform (see note on w.T)
                # https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_uniform_
                # even though we use SiLU, we just use the ReLU gain
                torch.nn.init.kaiming_uniform_(w.T, mode="fan_out", nonlinearity="relu")
            if mlp_bfloat16:
                w = w.to(torch.bfloat16)
            params[f"_weight_{layer}"] = w

        # generate code
        features = Proxy(graph.placeholder("x"))
        weights = [
            Proxy(graph.get_attr(f"_weight_{layer}"))
            for layer in range(len(self.dims) - 1)
        ]

        if (len(weights) > 1) and (nonlinearity is None):
            # we can special case since the whole thing is linear
            # we don't special case the linear projection case since:
            #  1) addmm can fuse the scalar multiply
            #  2) multi_dot doesn't support the identity case
            # Originally, we have:
            # ((x @ W1) @ W2 ) @ W3 = (W3 @ (W2 @ (W1 @ x)))^T = x @ (W3 @ W2 @ W1)^T
            total_weight = torch.linalg.multi_dot(weights)
            #            = torch.linalg.multi_dot([w.T for w in weights[::-1]]).T
            features = torch.mm(features, total_weight)
        else:
            # generate normal full MLP code
            for layer, w in enumerate(weights):
                features = features @ w
                # generate nonlinearity code
                if nonlinearity is not None and layer < num_layers - 1:
                    features = nonlinearity(features)
                    self.is_nonlinear = True  # one nonlinearity applied means the whole MLP is nonlinear

        graph.output(features.node)

        for pname, p in params.items():
            setattr(base, pname, torch.nn.Parameter(p))

        self._codegen_register({"_forward": fx.GraphModule(base, graph)})
        self.use_bfloat16 = mlp_bfloat16

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(\n  dims: {self.dims}\n  nonlinear: {self.is_nonlinear}\n)"

    def forward(self, x):
        if self.use_bfloat16:
            return self._forward(x.to(torch.bfloat16)).to(x.dtype)
        else:
            return self._forward(x)
