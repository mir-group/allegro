from typing import Optional, List
import math
import functools

import torch
from torch_runstats.scatter import scatter

from e3nn import o3
from e3nn.util.jit import compile_mode

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin
from nequip.utils.tp_utils import tp_path_exists

from ._fc import ScalarMLPFunction
from .. import _keys
from ._strided import Contracter, MakeWeightedChannels, Linear
from .cutoffs import cosine_cutoff, polynomial_cutoff


@compile_mode("script")
class Allegro_Module(GraphModuleMixin, torch.nn.Module):
    # saved params
    num_layers: int
    field: str
    out_field: str
    num_types: int
    env_embed_mul: int
    weight_numel: int
    latent_resnet: bool

    # internal values
    _env_builder_w_index: List[int]
    _env_builder_n_irreps: int
    _input_pad: int

    def __init__(
        self,
        # required params
        num_layers: int,
        num_types: int,
        r_max: float,
        avg_num_neighbors: Optional[float] = None,
        # cutoffs
        r_start_cos_ratio: float = 0.8,
        PolynomialCutoff_p: float = 6,
        per_layer_cutoffs: Optional[List[float]] = None,
        cutoff_type: str = "polynomial",
        # general hyperparameters:
        field: str = AtomicDataDict.EDGE_ATTRS_KEY,
        edge_invariant_field: str = AtomicDataDict.EDGE_EMBEDDING_KEY,
        node_invariant_field: str = AtomicDataDict.NODE_ATTRS_KEY,
        env_embed_multiplicity: int = 32,
        linear_after_env_embed: bool = False,
        nonscalars_include_parity: bool = True,
        # MLP parameters:
        two_body_latent=ScalarMLPFunction,
        two_body_latent_kwargs={},
        env_embed=ScalarMLPFunction,
        env_embed_kwargs={},
        latent=ScalarMLPFunction,
        latent_kwargs={},
        latent_resnet: bool = True,
        latent_resnet_coefficients: Optional[List[float]] = None,
        latent_resnet_coefficients_learnable: bool = False,
        latent_out_field: Optional[str] = _keys.EDGE_FEATURES,
        # Performance parameters:
        pad_to_alignment: int = 1,
        sparse_mode: Optional[str] = None,
        # Other:
        irreps_in=None,
    ):
        super().__init__()
        SCALAR = o3.Irrep("0e")  # define for convinience

        # save parameters
        assert (
            num_layers >= 1
        )  # zero layers is "two body", but we don't need to support that fallback case
        self.num_layers = num_layers
        self.nonscalars_include_parity = nonscalars_include_parity
        self.field = field
        self.latent_out_field = latent_out_field
        self.edge_invariant_field = edge_invariant_field
        self.node_invariant_field = node_invariant_field
        self.latent_resnet = latent_resnet
        self.env_embed_mul = env_embed_multiplicity
        self.r_start_cos_ratio = r_start_cos_ratio
        self.polynomial_cutoff_p = float(PolynomialCutoff_p)
        self.cutoff_type = cutoff_type
        assert cutoff_type in ("cosine", "polynomial")
        self.avg_num_neighbors = avg_num_neighbors
        self.linear_after_env_embed = linear_after_env_embed
        self.num_types = num_types

        # set up irreps
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[
                self.field,
                self.edge_invariant_field,
                self.node_invariant_field,
            ],
        )

        # for normalization of env embed sums
        # one per layer
        self.register_buffer(
            "env_sum_normalizations",
            # dividing by sqrt(N)
            torch.as_tensor([avg_num_neighbors] * num_layers).rsqrt(),
        )

        latent = functools.partial(latent, **latent_kwargs)
        env_embed = functools.partial(env_embed, **env_embed_kwargs)

        self.latents = torch.nn.ModuleList([])
        self.env_embed_mlps = torch.nn.ModuleList([])
        self.tps = torch.nn.ModuleList([])
        self.linears = torch.nn.ModuleList([])
        self.env_linears = torch.nn.ModuleList([])

        # Embed to the spharm * it as mul
        input_irreps = self.irreps_in[self.field]
        # this is not inherant, but no reason to fix right now:
        assert all(mul == 1 for mul, ir in input_irreps)
        env_embed_irreps = o3.Irreps([(1, ir) for _, ir in input_irreps])
        assert (
            env_embed_irreps[0].ir == SCALAR
        ), "env_embed_irreps must start with scalars"
        self._input_pad = (
            int(math.ceil(env_embed_irreps.dim / pad_to_alignment)) * pad_to_alignment
        ) - env_embed_irreps.dim
        self.register_buffer("_zero", torch.zeros(1, 1))

        # Initially, we have the B(r)Y(\vec{r})-projection of the edges,
        # embeded by a Linear.
        arg_irreps = env_embed_irreps

        # - begin irreps -
        # start to build up the irreps for the iterated TPs
        tps_irreps = [arg_irreps]

        for layer_idx in range(num_layers):
            # Create higher order terms cause there are more TPs coming
            if layer_idx == 0:
                # Add parity irreps
                ir_out = []
                for (mul, ir) in env_embed_irreps:
                    if self.nonscalars_include_parity:
                        # add both parity options
                        ir_out.append((1, (ir.l, 1)))
                        ir_out.append((1, (ir.l, -1)))
                    else:
                        # add only the parity option seen in the inputs
                        ir_out.append((1, ir))

                ir_out = o3.Irreps(ir_out)

            if layer_idx == self.num_layers - 1:
                # ^ means we're doing the last layer
                # No more TPs follow this, so only need scalars
                ir_out = o3.Irreps([(1, (0, 1))])

            # Prune impossible paths
            ir_out = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in ir_out
                    if tp_path_exists(arg_irreps, env_embed_irreps, ir)
                ]
            )

            # the argument to the next tensor product is the output of this one
            arg_irreps = ir_out
            tps_irreps.append(ir_out)
        # - end build irreps -

        # == Remove unneeded paths ==
        out_irreps = tps_irreps[-1]
        new_tps_irreps = [out_irreps]
        for arg_irreps in reversed(tps_irreps[:-1]):
            new_arg_irreps = []
            for mul, arg_ir in arg_irreps:
                for _, env_ir in env_embed_irreps:
                    if any(i in out_irreps for i in arg_ir * env_ir):
                        # arg_ir is useful: arg_ir * env_ir has a path to something we want
                        new_arg_irreps.append((mul, arg_ir))
                        # once its useful once, we keep it no matter what
                        break
            new_arg_irreps = o3.Irreps(new_arg_irreps)
            new_tps_irreps.append(new_arg_irreps)
            out_irreps = new_arg_irreps

        assert len(new_tps_irreps) == len(tps_irreps)
        tps_irreps = list(reversed(new_tps_irreps))
        del new_tps_irreps

        assert tps_irreps[-1].lmax == 0

        tps_irreps_in = tps_irreps[:-1]
        tps_irreps_out = tps_irreps[1:]
        del tps_irreps

        # Environment builder:
        self._env_weighter = MakeWeightedChannels(
            irreps_in=input_irreps,
            multiplicity_out=env_embed_multiplicity,
            pad_to_alignment=pad_to_alignment,
        )

        self._n_scalar_outs = []

        # == Build TPs ==
        for layer_idx, (arg_irreps, out_irreps) in enumerate(
            zip(tps_irreps_in, tps_irreps_out)
        ):
            # Make the env embed linear
            if self.linear_after_env_embed:
                self.env_linears.append(
                    Linear(
                        [(env_embed_multiplicity, ir) for _, ir in env_embed_irreps],
                        [(env_embed_multiplicity, ir) for _, ir in env_embed_irreps],
                        shared_weights=True,
                        internal_weights=True,
                    )
                )
            else:
                self.env_linears.append(torch.nn.Identity())
            # Make TP
            tmp_i_out: int = 0
            instr = []
            n_scalar_outs: int = 0
            full_out_irreps = []
            for i_out, (_, ir_out) in enumerate(out_irreps):
                for i_1, (_, ir_1) in enumerate(arg_irreps):
                    for i_2, (_, ir_2) in enumerate(env_embed_irreps):
                        if ir_out in ir_1 * ir_2:
                            if ir_out == SCALAR:
                                n_scalar_outs += 1
                            instr.append((i_1, i_2, tmp_i_out))
                            full_out_irreps.append((env_embed_multiplicity, ir_out))
                            tmp_i_out += 1
            full_out_irreps = o3.Irreps(full_out_irreps)
            self._n_scalar_outs.append(n_scalar_outs)
            assert all(ir == SCALAR for _, ir in full_out_irreps[:n_scalar_outs])
            tp = Contracter(
                irreps_in1=o3.Irreps(
                    [(env_embed_multiplicity, ir) for _, ir in arg_irreps]
                ),
                irreps_in2=o3.Irreps(
                    [(env_embed_multiplicity, ir) for _, ir in env_embed_irreps]
                ),
                irreps_out=o3.Irreps(
                    [(env_embed_multiplicity, ir) for _, ir in full_out_irreps]
                ),
                instructions=instr,
                connection_mode="uuu",
                shared_weights=False,
                has_weight=False,
                pad_to_alignment=pad_to_alignment,
                sparse_mode=sparse_mode,
            )
            self.tps.append(tp)
            # we extract the scalars from the first irrep of the tp
            assert out_irreps[0].ir == SCALAR

            # Make env embed mlp
            generate_n_weights = (
                self._env_weighter.weight_numel
            )  # the weight for the edge embedding
            if layer_idx == 0:
                # also need weights to embed the edge itself
                # this is because the 2 body latent is mixed in with the first layer
                # in terms of code
                generate_n_weights += self._env_weighter.weight_numel

            # the linear acts after the extractor
            self.linears.append(
                Linear(
                    full_out_irreps,
                    [(env_embed_multiplicity, ir) for _, ir in out_irreps],
                    shared_weights=True,
                    internal_weights=True,
                    pad_to_alignment=pad_to_alignment,
                )
            )

            if layer_idx == 0:
                # at the first layer, we have no invariants from previous TPs
                self.latents.append(
                    two_body_latent(
                        mlp_input_dimension=(
                            (
                                # Node invariants for center and neighbor (chemistry)
                                2 * self.irreps_in[self.node_invariant_field].num_irreps
                                # Plus edge invariants for the edge (radius).
                                + self.irreps_in[self.edge_invariant_field].num_irreps
                            )
                        ),
                        mlp_output_dimension=None,
                        **two_body_latent_kwargs,
                    )
                )
            else:
                self.latents.append(
                    latent(
                        mlp_input_dimension=(
                            (
                                # the embedded latent invariants from the previous layer(s)
                                self.latents[-1].out_features
                                # and the invariants extracted from the last layer's TP:
                                # above, we already appended the n_scalar_out for the new TP for
                                # the layer we are building right now. So, we need -2
                                # to get the n_scalar_out for the previous TP, which are what we are actually integrating:
                                # in forward(), the `latent` is called _first_ before the TP
                                # of this layer we are building.
                                + env_embed_multiplicity * self._n_scalar_outs[-2]
                            )
                        ),
                        mlp_output_dimension=None,
                    )
                )
            # the env embed MLP takes the last latent's output as input
            # and outputs enough weights for the env embedder
            self.env_embed_mlps.append(
                env_embed(
                    mlp_input_dimension=self.latents[-1].out_features,
                    mlp_output_dimension=generate_n_weights,
                )
            )

        # For the final layer, we specialize:
        # we don't need to propagate nonscalars, so there is no TP
        # thus we only need the latent:
        self.final_latent = latent(
            mlp_input_dimension=self.latents[-1].out_features
            # here we use self._n_scalar_outs[-1] since we haven't appended anything to it
            # so it is still the correct n_scalar_outs for the previous (and last) TP
            + env_embed_multiplicity * self._n_scalar_outs[-1],
            mlp_output_dimension=None,
        )
        # - end build modules -

        # - layer resnet update weights -
        if latent_resnet_coefficients is None:
            # We initialize to zeros, which under exp() all go to ones
            latent_resnet_coefficients_params = torch.zeros(num_layers + 1)
        else:
            latent_resnet_coefficients = torch.as_tensor(
                latent_resnet_coefficients, dtype=torch.get_default_dtype()
            )
            assert latent_resnet_coefficients.min() > 0.0
            # dividing out a common factor doesn't affect the final normalized coefficients
            # and it keeps the numerics / gradients saner
            latent_resnet_coefficients /= latent_resnet_coefficients.min()
            # invert desired coefficients into params:
            latent_resnet_coefficients_params = torch.log(latent_resnet_coefficients)
        assert latent_resnet_coefficients_params.shape == (
            num_layers + 1,
        ), f"There must be {num_layers + 1} layer resnet update ratios, one for the two-body latent and each following layer"
        if latent_resnet_coefficients_learnable:
            self._latent_resnet_coefficients_params = torch.nn.Parameter(
                latent_resnet_coefficients_params
            )
        else:
            self.register_buffer(
                "_latent_resnet_coefficients_params", latent_resnet_coefficients_params
            )

        # - Per-layer cutoffs -
        if per_layer_cutoffs is None:
            per_layer_cutoffs = torch.full((num_layers + 1,), r_max)
        self.register_buffer("per_layer_cutoffs", torch.as_tensor(per_layer_cutoffs))
        assert torch.all(self.per_layer_cutoffs <= r_max)
        assert self.per_layer_cutoffs.shape == (
            num_layers + 1,
        ), "Must be one per-layer cutoff for layer 0 and every layer for a total of {num_layers} cutoffs (the first applies to the two body latent, which is 'layer 0')"
        assert (
            self.per_layer_cutoffs[1:] <= self.per_layer_cutoffs[:-1]
        ).all(), "Per-layer cutoffs must be equal or decreasing"
        assert (
            self.per_layer_cutoffs.min() > 0
        ), "Per-layer cutoffs must be >0. To remove higher layers entirely, lower `num_layers`."
        self._latent_dim = self.final_latent.out_features
        self.register_buffer("_zero", torch.as_tensor(0.0))

        self.irreps_out.update(
            {
                self.latent_out_field: o3.Irreps(
                    [(self.final_latent.out_features, (0, 1))]
                ),
            }
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        """Evaluate.

        :param data: AtomicDataDict.Type
        :return: AtomicDataDict.Type
        """
        edge_center = data[AtomicDataDict.EDGE_INDEX_KEY][0]
        edge_neighbor = data[AtomicDataDict.EDGE_INDEX_KEY][1]

        edge_attr = data[self.field]
        # pad edge_attr
        if self._input_pad > 0:
            edge_attr = torch.cat(
                (
                    edge_attr,
                    self._zero.expand(len(edge_attr), self._input_pad),
                ),
                dim=-1,
            )

        edge_length = data[AtomicDataDict.EDGE_LENGTH_KEY]
        num_edges: int = len(edge_attr)
        edge_invariants = data[self.edge_invariant_field]
        node_invariants = data[self.node_invariant_field]
        # pre-declare variables as Tensors for TorchScript
        scalars = self._zero
        coefficient_old = scalars
        coefficient_new = scalars
        # Initialize state
        latents = torch.zeros(
            (num_edges, self._latent_dim),
            dtype=edge_attr.dtype,
            device=edge_attr.device,
        )
        active_edges = torch.arange(
            num_edges,
            device=edge_attr.device,
        )

        # For the first layer, we use the input invariants:
        # The center and neighbor invariants and edge invariants
        latent_inputs_to_cat = [
            node_invariants[edge_center],
            node_invariants[edge_neighbor],
            edge_invariants,
        ]
        # The nonscalar features. Initially, the edge data.
        features = edge_attr

        layer_index: int = 0
        # precompute the exp() and cumsum for each layer
        latent_coefficients = self._latent_resnet_coefficients_params.exp()
        latent_coefficients_cumsum = latent_coefficients.cumsum(dim=0)

        # Vectorized precompute per layer cutoffs
        if self.cutoff_type == "cosine":
            cutoff_coeffs_all = cosine_cutoff(
                edge_length,
                self.per_layer_cutoffs,
                r_start_cos_ratio=self.r_start_cos_ratio,
            )
        elif self.cutoff_type == "polynomial":
            cutoff_coeffs_all = polynomial_cutoff(
                edge_length, self.per_layer_cutoffs, p=self.polynomial_cutoff_p
            )
        else:
            # This branch is unreachable (cutoff type is checked in __init__)
            # But TorchScript doesn't know that, so we need to make it explicitly
            # impossible to make it past so it doesn't throw
            # "cutoff_coeffs_all is not defined in the false branch"
            assert False, "Invalid cutoff type"

        # !!!! REMEMBER !!!! update final layer if update the code in main loop!!!
        # This goes through layer0, layer1, ..., layer_max-1
        for latent, env_embed_mlp, env_linear, tp, linear in zip(
            self.latents, self.env_embed_mlps, self.env_linears, self.tps, self.linears
        ):
            # Determine which edges are still in play
            cutoff_coeffs = cutoff_coeffs_all[layer_index]
            prev_mask = cutoff_coeffs[active_edges] > 0
            active_edges = (cutoff_coeffs > 0).nonzero().squeeze(-1)

            # Compute latents
            new_latents = latent(torch.cat(latent_inputs_to_cat, dim=-1)[prev_mask])
            # Apply cutoff, which propagates through to everything else
            new_latents = cutoff_coeffs[active_edges].unsqueeze(-1) * new_latents

            if self.latent_resnet and layer_index > 0:
                # previous normalization denominator / new normalization denominator
                # ^ cancels the old normalization, and ^ applies new
                # sqrt accounts for stdev vs variance
                coefficient_old = (
                    latent_coefficients_cumsum[layer_index - 1]
                    / latent_coefficients_cumsum[layer_index]
                ).sqrt()
                # just take the coefficient for the new latents
                coefficient_new = (
                    latent_coefficients[layer_index]
                    / latent_coefficients_cumsum[layer_index]
                ).sqrt()
                # Residual update
                # Note that it only runs when there are latents to resnet with, so not at the first layer
                # index_add adds only to the edges for which we have something to contribute
                latents = torch.index_add(
                    coefficient_old * latents,
                    0,
                    active_edges,
                    coefficient_new * new_latents,
                )
            else:
                # Normal (non-residual) update
                # index_copy replaces, unlike index_add
                latents = torch.index_copy(latents, 0, active_edges, new_latents)

            # From the latents, compute the weights for active edges:
            weights = env_embed_mlp(latents[active_edges])
            w_index: int = 0

            if layer_index == 0:
                # embed initial edge
                env_w = weights.narrow(-1, w_index, self._env_weighter.weight_numel)
                w_index += self._env_weighter.weight_numel
                features = self._env_weighter(
                    features[prev_mask], env_w
                )  # features is edge_attr
            else:
                # just take the previous features that we still need
                features = features[prev_mask]

            # Extract weights for the environment builder
            env_w = weights.narrow(-1, w_index, self._env_weighter.weight_numel)
            w_index += self._env_weighter.weight_numel

            # Build the local environments
            # This local environment should only be a sum over neighbors
            # who are within the cutoff of the _current_ layer
            # Those are the active edges, which are the only ones we
            # have weights for (env_w) anyway.
            # So we mask out the edges in the sum:
            local_env_per_edge = scatter(
                self._env_weighter(edge_attr[active_edges], env_w),
                edge_center[active_edges],
                dim=0,
            )
            if self.env_sum_normalizations.ndim < 2:
                # it's a scalar per layer
                norm_const = self.env_sum_normalizations[layer_index]
            else:
                # it's per type
                # get shape [N_atom, 1] for broadcasting
                norm_const = self.env_sum_normalizations[
                    layer_index, data[AtomicDataDict.ATOM_TYPE_KEY]
                ].unsqueeze(-1)
            local_env_per_edge = local_env_per_edge * norm_const
            local_env_per_edge = env_linear(local_env_per_edge)
            # Copy to get per-edge
            # Large allocation, but no better way to do this:
            local_env_per_edge = local_env_per_edge[edge_center[active_edges]]

            # Now do the TP
            # recursively tp current features with the environment embeddings
            features = tp(features, local_env_per_edge)

            # Get invariants
            # features has shape [z][mul][k]
            # we know scalars are first
            scalars = features[:, :, : self._n_scalar_outs[layer_index]].reshape(
                features.shape[0], -1
            )

            # do the linear
            features = linear(features)

            # For layer2+, use the previous latents and scalars
            # This makes it deep
            latent_inputs_to_cat = [
                latents[active_edges],
                scalars,
            ]

            # increment counter
            layer_index += 1

        # - final layer -
        # due to TorchScript limitations, we have to
        # copy and repeat the code here --- no way to
        # escape the final iteration of the loop early
        cutoff_coeffs = cutoff_coeffs_all[layer_index]
        prev_mask = cutoff_coeffs[active_edges] > 0
        active_edges = (cutoff_coeffs > 0).nonzero().squeeze(-1)
        new_latents = self.final_latent(
            torch.cat(latent_inputs_to_cat, dim=-1)[prev_mask]
        )
        new_latents = cutoff_coeffs[active_edges].unsqueeze(-1) * new_latents
        if self.latent_resnet:
            coefficient_old = (
                latent_coefficients_cumsum[layer_index - 1]
                / latent_coefficients_cumsum[layer_index]
            ).sqrt()
            coefficient_new = (
                latent_coefficients[layer_index]
                / latent_coefficients_cumsum[layer_index]
            ).sqrt()
            latents = torch.index_add(
                coefficient_old * latents,
                0,
                active_edges,
                coefficient_new * new_latents,
            )
        else:
            latents = torch.index_copy(latents, 0, active_edges, new_latents)
        # - end final layer -

        # final latents
        data[self.latent_out_field] = latents

        return data
