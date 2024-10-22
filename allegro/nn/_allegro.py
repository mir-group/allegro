from typing import Optional, List
import math
import functools
import warnings

import torch

from e3nn import o3
from e3nn.util.jit import compile_mode

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin, scatter, tp_path_exists

from ._fc import ScalarMLPFunction
from ._strided import Contracter, MakeWeightedChannels, Linear


@compile_mode("script")
class Allegro_Module(GraphModuleMixin, torch.nn.Module):
    # saved params
    num_layers: int
    field: str
    out_field: str
    num_types: int
    num_tensor_features: int
    weight_numel: int
    latent_resnet: bool
    self_edge_tensor_product: bool

    # internal values
    _env_builder_w_index: List[int]
    _env_builder_n_irreps: int
    _env_sum_constant: float

    def __init__(
        self,
        # required hyperparameters:
        num_layers: int,
        type_names: List[str],
        r_max: float,
        num_tensor_features: int,
        tensor_track_allowed_irreps: o3.Irreps,
        avg_num_neighbors: Optional[float] = None,
        # optional hyperparameters:
        field: str = AtomicDataDict.EDGE_ATTRS_KEY,
        edge_invariant_field: str = AtomicDataDict.EDGE_EMBEDDING_KEY,
        self_edge_tensor_product: bool = False,
        tensors_mixing_mode: str = "p",
        tensor_track_weight_init: str = "uniform",
        weight_individual_irreps: bool = True,
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
        latent_out_field: Optional[str] = AtomicDataDict.EDGE_FEATURES_KEY,
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
        self.tensor_track_allowed_irreps = o3.Irreps(tensor_track_allowed_irreps)
        assert set(mul for mul, ir in self.tensor_track_allowed_irreps) == {1}
        self.field = field
        self.latent_out_field = latent_out_field
        self.edge_invariant_field = edge_invariant_field
        self.latent_resnet = latent_resnet
        self.num_tensor_features = num_tensor_features
        self.avg_num_neighbors = avg_num_neighbors
        self.num_types = len(type_names)
        self.self_edge_tensor_product = self_edge_tensor_product

        assert tensors_mixing_mode in ("uuulin", "uuup", "uvvp", "p")
        tp_tensors_mixing_mode = {
            "uuulin": "uuu",
            "uuup": "uuu",
            "uvvp": "uvv",
            "p": "p",
        }[tensors_mixing_mode]
        internal_weight_tp = tensors_mixing_mode != "uuulin"

        self.register_buffer("r_max", torch.as_tensor(float(r_max)))
        assert not any(
            k["mlp_bias"]
            for k in (two_body_latent_kwargs, latent_kwargs, env_embed_kwargs)
        )

        # set up irreps
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[
                self.field,
                self.edge_invariant_field,
            ],
        )

        latent = functools.partial(latent, **latent_kwargs)
        env_embed = functools.partial(env_embed, **env_embed_kwargs)

        self.latents = torch.nn.ModuleList([])
        self.env_embed_mlps = torch.nn.ModuleList([])
        self.tps = torch.nn.ModuleList([])
        self.linears = torch.nn.ModuleList([])

        # Embed to the spharm * it as mul
        input_irreps = self.irreps_in[self.field]
        # this is not inherant, but no reason to fix right now:
        assert all(mul == 1 for mul, ir in input_irreps)
        env_embed_irreps = o3.Irreps([(1, ir) for _, ir in input_irreps])
        assert (
            env_embed_irreps[0].ir == SCALAR
        ), "env_embed_irreps must start with scalars"
        self.register_buffer("_zero", torch.zeros(1, 1))

        # Initially, we have the B(r)Y(\vec{r})-projection of the edges,
        # embeded by a Linear.
        arg_irreps = env_embed_irreps

        # - begin irreps -
        # start to build up the irreps for the iterated TPs
        tps_irreps = [arg_irreps]

        for layer_idx in range(num_layers):
            if layer_idx == self.num_layers - 1:
                # ^ means we're doing the last layer
                # No more TPs follow this, so only need scalars
                ir_out = o3.Irreps([(1, (0, 1))])
            else:
                # allow everything allowed
                ir_out = self.tensor_track_allowed_irreps

            # Prune impossible paths, leaving only allowed irreps that can be constructed:
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
        del ir_out, layer_idx, arg_irreps
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
        del new_tps_irreps, new_arg_irreps, arg_irreps

        assert tps_irreps[-1].lmax == 0

        tps_irreps_in = tps_irreps[:-1]
        tps_irreps_out = tps_irreps[1:]
        del tps_irreps

        # Environment builder:
        # For weighting the initial edge features:
        self._edge_weighter = MakeWeightedChannels(
            irreps_in=input_irreps,
            multiplicity_out=num_tensor_features,
            weight_individual_irreps=weight_individual_irreps,
        )
        # - normalization -
        # we divide the env embed sums by sqrt(N) to normalize
        # note that if self_edge_tensor_product = False, then the number of neighbors being summed is one smaller
        env_sum_constant = 1.0
        if avg_num_neighbors is not None:
            env_sum_constant = 1.0 / math.sqrt(
                avg_num_neighbors - (0 if self.self_edge_tensor_product else 1)
            )
        self._env_weighter = MakeWeightedChannels(
            irreps_in=input_irreps,
            multiplicity_out=num_tensor_features,
            weight_individual_irreps=weight_individual_irreps,
            alpha=env_sum_constant,
        )
        del env_sum_constant

        self._n_scalar_outs = []

        # == Build TPs ==
        for layer_idx, (arg_irreps, out_irreps) in enumerate(
            zip(tps_irreps_in, tps_irreps_out)
        ):
            # Make TP
            tmp_i_out: int = 0
            instr = []
            n_scalar_outs: int = 0
            full_out_irreps = out_irreps if internal_weight_tp else []
            for i_out, (_, ir_out) in enumerate(out_irreps):
                for i_1, (_, ir_1) in enumerate(arg_irreps):
                    for i_2, (_, ir_2) in enumerate(env_embed_irreps):
                        if ir_out in ir_1 * ir_2:
                            if internal_weight_tp:
                                if ir_out == SCALAR:
                                    n_scalar_outs = 1
                                instr.append((i_1, i_2, i_out))
                            else:
                                if ir_out == SCALAR:
                                    n_scalar_outs += 1
                                instr.append((i_1, i_2, tmp_i_out))
                                full_out_irreps.append((num_tensor_features, ir_out))
                                tmp_i_out += 1
            full_out_irreps = o3.Irreps(full_out_irreps)
            del tmp_i_out
            self._n_scalar_outs.append(n_scalar_outs)
            assert all(ir == SCALAR for _, ir in full_out_irreps[:n_scalar_outs])
            tp = Contracter(
                irreps_in1=o3.Irreps(
                    [(num_tensor_features, ir) for _, ir in arg_irreps]
                ),
                irreps_in2=o3.Irreps(
                    [(num_tensor_features, ir) for _, ir in env_embed_irreps]
                ),
                irreps_out=o3.Irreps(
                    [(num_tensor_features, ir) for _, ir in full_out_irreps]
                ),
                instructions=instr,
                connection_mode=tp_tensors_mixing_mode,
                shared_weights=internal_weight_tp,
                has_weight=internal_weight_tp,
                internal_weights=internal_weight_tp,
                initialization=tensor_track_weight_init,
            )
            self.tps.append(tp)
            del tp
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
                generate_n_weights += self._edge_weighter.weight_numel

            # the linear acts after the extractor
            self.linears.append(
                Linear(
                    full_out_irreps,
                    [(num_tensor_features, ir) for _, ir in out_irreps],
                    shared_weights=True,
                    internal_weights=True,
                    initialization=tensor_track_weight_init,
                )
                if not internal_weight_tp
                else torch.nn.Identity()
            )

            if layer_idx == 0:
                # at the first layer, we have no invariants from previous TPs
                self.latents.append(
                    two_body_latent(
                        mlp_input_dimension=(
                            (
                                # initial edge invariants for the edge (radial-chemical embedding).
                                self.irreps_in[self.edge_invariant_field].num_irreps
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
                                (
                                    self.latents[-1].out_features
                                    if self.latent_resnet
                                    else sum(mlp.out_features for mlp in self.latents)
                                )
                                # and the invariants extracted from the last layer's TP:
                                # above, we already appended the n_scalar_out for the new TP for
                                # the layer we are building right now. So, we need -2
                                # to get the n_scalar_out for the previous TP, which are what we are actually integrating:
                                # in forward(), the `latent` is called _first_ before the TP
                                # of this layer we are building.
                                + num_tensor_features * self._n_scalar_outs[-2]
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
            del generate_n_weights

        # For the final layer, we specialize:
        # we don't need to propagate tensors, so there is no TP
        # thus we only need the latent:
        self.final_latent = latent(
            mlp_input_dimension=(
                self.latents[-1].out_features
                if self.latent_resnet
                else sum(mlp.out_features for mlp in self.latents)
            )
            # here we use self._n_scalar_outs[-1] since we haven't appended anything to it
            # so it is still the correct n_scalar_outs for the previous (and last) TP
            + num_tensor_features * self._n_scalar_outs[-1],
            mlp_output_dimension=None,
        )
        # - end build modules -
        for l in self.latents + [self.final_latent]:
            if not l.is_nonlinear:
                warnings.warn(
                    "Latent MLP is linear. Nonlinear latent MLPs are strongly recommended and using linear ones may significantly affect accuracy in some systems. Ensure two_body_latent_mlp_latent_dimensions and latent_mlp_latent_dimensions are at least two entries long."
                )

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
        num_atoms: int = len(data[AtomicDataDict.POSITIONS_KEY])

        edge_attr = data[self.field]
        edge_invariants = data[self.edge_invariant_field]
        # pre-declare variables as Tensors for TorchScript
        scalars = self._zero
        coefficient_old = self._zero
        coefficient_new = self._zero
        latents = self._zero

        # For the first layer, we use the input edge invariants
        latent_inputs_to_cat = [edge_invariants]
        # The nonscalar features. Initially, the edge data.
        features = edge_attr

        layer_index: int = 0
        # precompute the exp() and cumsum for each layer
        # note that because our coefficients are exp() over sums,
        # this is just a cummulative softmax-- so we can use the typical
        # numerical tricks to help stability
        # a shift to all coefficients => constant factor after exp => cancels with denominator
        # helps prevent dividing large / large
        latent_coefficients = self._latent_resnet_coefficients_params
        latent_coefficients = (latent_coefficients - latent_coefficients.max()).exp()
        # add 1e-12 so that we never divide by zero (though that is extremely unlikely)
        latent_coefficients_cumsum = latent_coefficients.cumsum(dim=0) + 1e-12

        prev_latents_for_next = []

        # !!!! REMEMBER !!!! update final layer if update the code in main loop!!!
        # This goes through layer0, layer1, ..., layer_max-1
        for latent, env_embed_mlp, tp, linear in zip(
            self.latents, self.env_embed_mlps, self.tps, self.linears
        ):
            # Compute latents
            new_latents = latent(torch.cat(latent_inputs_to_cat, dim=-1))

            if self.latent_resnet and layer_index > 0:
                # previous normalization denominator / new normalization denominator
                # ^ cancels the old normalization, and ^ applies new
                # sqrt accounts for stdev vs variance
                # at the 2nd layer the cumsum is just the first coefficient, so this multiplies the
                # previous latents (which hadn't been multiplied by anything) by coeff_0 / coeff_0 + coeff_1
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
                # Note that it only runs when there are latents to resnet with
                latents = coefficient_old * latents + coefficient_new * new_latents
                prev_latents_for_next = [latents]
            else:
                # Normal (non-residual) update
                latents = new_latents
                prev_latents_for_next.append(latents)

            # From the latents, compute the weights for active edges:
            weights = env_embed_mlp(latents)
            w_index: int = 0

            if layer_index == 0:
                # embed initial edge
                env_w = weights.narrow(-1, w_index, self._edge_weighter.weight_numel)
                w_index += self._edge_weighter.weight_numel
                features = self._edge_weighter(features, env_w)  # features is edge_attr

            # Extract weights for the environment builder
            env_w = weights.narrow(-1, w_index, self._env_weighter.weight_numel)
            w_index += self._env_weighter.weight_numel

            # Build the local environments
            # This local environment is a sum over neighbors
            # We apply the normalization constant to the env_w weights here, since
            # everything here before the TP is linear and the env_w is likely smallest
            # since it only contains the scalars.
            # It is applied via _env_weighter's alpha
            env_w_edges = self._env_weighter(edge_attr, env_w)
            local_env_per_edge = scatter(
                env_w_edges,
                edge_center,
                dim=0,
                dim_size=num_atoms,
            )
            # make it per edge
            local_env_per_edge = torch.index_select(local_env_per_edge, 0, edge_center)

            if not self.self_edge_tensor_product:
                # subtract out the current edge from each env sum
                # sum_i{x_i} - x_j = sum_{i != j}{x_i}
                # this gives for each edge a sum over all _other_ edges sharing the center
                # i.e.  env_ij = sum_{k for k != j}{edge_ik}
                local_env_per_edge = local_env_per_edge - env_w_edges

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
            latent_inputs_to_cat = prev_latents_for_next + [scalars]

            # increment counter
            layer_index += 1

        # - final layer -
        # due to TorchScript limitations, we have to
        # copy and repeat the code here --- no way to
        # escape the final iteration of the loop early
        new_latents = self.final_latent(torch.cat(latent_inputs_to_cat, dim=-1))
        if self.latent_resnet:
            coefficient_old = (
                latent_coefficients_cumsum[layer_index - 1]
                / latent_coefficients_cumsum[layer_index]
            ).sqrt()
            coefficient_new = (
                latent_coefficients[layer_index]
                / latent_coefficients_cumsum[layer_index]
            ).sqrt()
            latents = coefficient_old * latents + coefficient_new * new_latents
        else:
            latents = new_latents
        # - end final layer -

        # final latents
        data[self.latent_out_field] = latents

        return data
