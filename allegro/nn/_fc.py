from typing import List, Optional
import math

import torch
from torch import fx

from e3nn import o3
from e3nn.util.jit import compile_mode
from e3nn.util.codegen import CodeGenMixin
from e3nn.math import normalize2mom

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin
from nequip.nn.nonlinearities import ShiftedSoftPlus


@compile_mode("script")
class ScalarMLP(GraphModuleMixin, torch.nn.Module):
    """Apply an MLP to some scalar field."""

    field: str
    out_field: str

    def __init__(
        self,
        mlp_latent_dimensions: List[int],
        mlp_output_dimension: Optional[int],
        mlp_nonlinearity: Optional[str] = "silu",
        mlp_initialization: str = "uniform",
        mlp_bias: bool = True,
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
        in_dim = self.irreps_in[self.field][0].mul
        self._module = ScalarMLPFunction(
            mlp_input_dimension=in_dim,
            mlp_latent_dimensions=mlp_latent_dimensions,
            mlp_output_dimension=mlp_output_dimension,
            mlp_nonlinearity=mlp_nonlinearity,
            mlp_initialization=mlp_initialization,
            mlp_bias=mlp_bias,
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

    def __init__(
        self,
        mlp_input_dimension: Optional[int],
        mlp_latent_dimensions: List[int],
        mlp_output_dimension: Optional[int],
        mlp_nonlinearity: Optional[str] = "silu",
        mlp_initialization: str = "normal",
        mlp_bias: bool = True,
    ):
        super().__init__()
        nonlinearity = {
            None: None,
            "silu": torch.nn.functional.silu,
            "ssp": ShiftedSoftPlus,
        }[mlp_nonlinearity]
        if nonlinearity is not None:
            nonlin_const = normalize2mom(nonlinearity).cst
        else:
            nonlin_const = 1.0

        dimensions = (
            ([mlp_input_dimension] if mlp_input_dimension is not None else [])
            + mlp_latent_dimensions
            + ([mlp_output_dimension] if mlp_output_dimension is not None else [])
        )
        assert len(dimensions) >= 2  # Must have input and output dim
        num_layers = len(dimensions) - 1

        self.in_features = dimensions[0]
        self.out_features = dimensions[-1]

        # Code
        params = {}
        graph = fx.Graph()
        tracer = fx.proxy.GraphAppendingTracer(graph)

        def Proxy(n):
            return fx.Proxy(n, tracer=tracer)

        features = Proxy(graph.placeholder("x"))
        norm_from_last: float = 1.0

        base = torch.nn.Module()
        if not mlp_bias:
            base.register_buffer("_bias_dummy", torch.as_tensor(0.0))
            bias = Proxy(graph.get_attr("_bias_dummy"))

        for layer, (h_in, h_out) in enumerate(zip(dimensions, dimensions[1:])):
            # make weights
            w = torch.empty(h_in, h_out)

            if mlp_initialization == "normal":
                w.normal_()
            elif mlp_initialization == "uniform":
                # these values give < x^2 > = 1
                w.uniform_(-math.sqrt(3), math.sqrt(3))
            elif mlp_initialization == "orthogonal":
                # this rescaling gives < x^2 > = 1
                torch.nn.init.orthogonal_(w, gain=math.sqrt(max(w.shape)))
            else:
                raise NotImplementedError(
                    f"Invalid mlp_initialization {mlp_initialization}"
                )

            # generate code
            params[f"_weight_{layer}"] = w
            w = Proxy(graph.get_attr(f"_weight_{layer}"))
            if mlp_bias:
                params[f"_bias_{layer}"] = torch.zeros(1, h_out)
                bias = Proxy(graph.get_attr(f"_bias_{layer}"))
            # computes beta*bias + alpha*(features @ w)
            features = torch.addmm(
                bias,
                features,
                w,
                beta=1 if mlp_bias else 0,
                # include any nonlinearity normalization from previous layers
                # we want to compute nonlin_alpha * nonlin(Wx + b) at each layer
                # at the second+ layer, this can be written as
                # nonlin(W(nonlin_alpha * x) + b)
                alpha=(norm_from_last / math.sqrt(float(h_in))),
            )

            # generate nonlinearity code
            if nonlinearity is not None and layer < num_layers - 1:
                features = nonlinearity(features)
                # add the normalization const in next layer
                norm_from_last = nonlin_const

        graph.output(features.node)

        for pname, p in params.items():
            setattr(base, pname, torch.nn.Parameter(p))

        self._codegen_register({"_forward": fx.GraphModule(base, graph)})

    def forward(self, x):
        return self._forward(x)
