from typing import List, Optional, Union
import math
import operator
import functools

import torch
from torch import fx

from e3nn import o3
from e3nn.util.jit import compile_mode
from e3nn.util.codegen import CodeGenMixin
from e3nn.math import normalize2mom

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin
from nequip.nn.nonlinearities import ShiftedSoftPlus

from ._misc import _init_weight


def _to_int_list(arg):
    # converts str: "64 64 64" to List[int]: [64, 64, 64]
    # or int: 64 to List[int]: [64]
    # to simplify parsing of list inputs when using 3rd party code
    # e.g. to pass inputs to wandb sweep configs
    if isinstance(arg, str):
        return [int(x) for x in arg.split()]
    elif isinstance(arg, int):
        return [arg]
    else:
        return arg


@compile_mode("script")
class ScalarMLP(GraphModuleMixin, torch.nn.Module):
    """Apply an MLP to some scalar field."""

    field: str
    out_field: str

    def __init__(
        self,
        mlp_latent_dimensions: Union[List[int], str],
        mlp_output_dimension: Optional[int],
        mlp_nonlinearity: Optional[str] = "silu",
        mlp_initialization: str = "uniform",
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

        mlp_latent_dimensions = _to_int_list(mlp_latent_dimensions)

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
        mlp_input_dimension: Optional[int],
        mlp_latent_dimensions: Union[List[int], str],
        mlp_output_dimension: Optional[int],
        mlp_nonlinearity: Optional[str] = "silu",
        mlp_initialization: str = "uniform",
        mlp_bias: bool = False,
        mlp_bfloat16: bool = False,
    ):
        super().__init__()
        assert not mlp_bias  # guard against accidents
        nonlinearity = {
            None: None,
            "null": None,
            "None": None,
            "silu": torch.nn.functional.silu,
            "ssp": ShiftedSoftPlus,
        }[mlp_nonlinearity]
        if nonlinearity is not None:
            nonlin_const = normalize2mom(nonlinearity).cst
        else:
            nonlin_const = 1.0

        mlp_latent_dimensions = _to_int_list(mlp_latent_dimensions)

        dimensions = (
            ([mlp_input_dimension] if mlp_input_dimension is not None else [])
            + mlp_latent_dimensions
            + ([mlp_output_dimension] if mlp_output_dimension is not None else [])
        )
        assert len(dimensions) >= 2  # Must have input and output dim
        num_layers = len(dimensions) - 1
        self.num_layers = num_layers

        self.in_features = dimensions[0]
        self.out_features = dimensions[-1]
        self.is_nonlinear = False  # updated in codegen below

        # Code
        params = {}
        graph = fx.Graph()
        tracer = fx.proxy.GraphAppendingTracer(graph)

        def Proxy(n):
            return fx.Proxy(n, tracer=tracer)

        base = torch.nn.Module()

        # make weights
        for layer, (h_in, h_out) in enumerate(zip(dimensions, dimensions[1:])):
            w = torch.empty(h_in, h_out)
            _init_weight(w, mlp_initialization, allow_orthogonal=True)
            if mlp_bfloat16:
                w = w.to(torch.bfloat16)
            params[f"_weight_{layer}"] = w
            if mlp_bias:
                b = torch.zeros(1, h_out)
                if mlp_bfloat16:
                    b = b.to(torch.bfloat16)
                params[f"_bias_{layer}"] = b

        # generate code
        features = Proxy(graph.placeholder("x"))
        weights = [
            Proxy(graph.get_attr(f"_weight_{layer}"))
            for layer in range(len(dimensions) - 1)
        ]
        if mlp_bias:
            biases = [
                Proxy(graph.get_attr(f"_bias_{layer}"))
                for layer in range(len(dimensions) - 1)
            ]
        else:
            base.register_buffer(
                "_bias_dummy",
                torch.as_tensor(
                    0.0,
                    dtype=torch.bfloat16 if mlp_bfloat16 else torch.get_default_dtype(),
                ),
            )
            biases = [Proxy(graph.get_attr("_bias_dummy"))] * (len(dimensions) - 1)

        if (len(weights) > 1) and (not mlp_bias) and (nonlinearity is None):
            # we can special case since the whole thing is linear
            # we don't special case the linear projection case since:
            #  1) addmm can fuse the scalar multiply
            #  2) multi_dot doesn't support the identity case
            norm_constant = 1.0 / functools.reduce(
                operator.mul, [math.sqrt(float(d)) for d in dimensions[:-1]]
            )
            # matmul is linear
            weights[0] = weights[0] * norm_constant
            # apply the first layer first
            # Originally, we have:
            # ((x @ W1) @ W2 ) @ W3 = (W3 @ (W2 @ (W1 @ x)))^T = x @ (W3 @ W2 @ W1)^T
            total_weight = torch.linalg.multi_dot(weights)
            #            = torch.linalg.multi_dot([w.T for w in weights[::-1]]).T
            features = torch.mm(features, total_weight)
        else:
            # generate normal full MLP code
            norm_from_last: float = 1.0
            for layer, (h_in, h_out, w, bias) in enumerate(
                zip(dimensions, dimensions[1:], weights, biases)
            ):
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
                    self.is_nonlinear = True  # one nonlinearity applied means the whole MLP is nonlinear
                    # add the normalization const in next layer
                    norm_from_last = nonlin_const

        graph.output(features.node)

        for pname, p in params.items():
            setattr(base, pname, torch.nn.Parameter(p))

        self._codegen_register({"_forward": fx.GraphModule(base, graph)})
        self.use_bfloat16 = mlp_bfloat16

    def forward(self, x):
        if self.use_bfloat16:
            return self._forward(x.to(torch.bfloat16)).to(x.dtype)
        else:
            return self._forward(x)
