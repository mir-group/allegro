from typing import Optional
import math
import functools

import torch

from e3nn import o3
from e3nn.util.jit import compile_mode

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin, tp_path_exists

from ._fc import ScalarMLPFunction
from ._strided import Contracter, MakeWeightedChannels


@compile_mode("script")
class Allegro_Module(GraphModuleMixin, torch.nn.Module):
    """Allegro layers."""

    def __init__(
        self,
        # required hyperparameters:
        num_layers: int,
        num_scalar_features: int,
        num_tensor_features: int,
        tensor_track_allowed_irreps: o3.Irreps,
        # optional hyperparameters:
        avg_num_neighbors: Optional[float] = None,
        tensors_mixing_mode: str = "p",
        tensor_track_weight_init: str = "uniform",
        weight_individual_irreps: bool = True,
        scatter_features: bool = True,
        # ^ scatter V_ik, alternative is to scatter embedded environment w_ij Y_ij
        # MLP parameters:
        latent=ScalarMLPFunction,
        latent_kwargs={},
        # bookkeeping:
        tensor_basis_in_field: Optional[str] = AtomicDataDict.EDGE_ATTRS_KEY,
        tensor_features_in_field: Optional[str] = AtomicDataDict.EDGE_FEATURES_KEY,
        scalar_in_field: Optional[str] = AtomicDataDict.EDGE_EMBEDDING_KEY,
        scalar_out_field: Optional[str] = AtomicDataDict.EDGE_FEATURES_KEY,
        irreps_in=None,
    ):
        super().__init__()
        SCALAR = o3.Irrep("0e")  # define for convenience

        # === early sanity checks ===
        assert (
            num_layers >= 1
        )  # zero layers is "two body", but we don't need to support that fallback case
        assert not any(k.get("mlp_bias", False) for k in (latent_kwargs,))

        assert (
            avg_num_neighbors is not None
        ), "`avg_num_neighbors` must be set for Allegro models, but `avg_num_neighbors=None` found"

        # === save parameters ===
        self.num_layers = num_layers
        self.num_scalar_features = num_scalar_features
        self.num_tensor_features = num_tensor_features
        self.scatter_features = scatter_features
        self.tensor_track_allowed_irreps = o3.Irreps(tensor_track_allowed_irreps)
        assert set(mul for mul, ir in self.tensor_track_allowed_irreps) == {1}

        # == tensor mixing mode ==
        assert tensors_mixing_mode in ("uuup", "uvvp", "p")
        tp_tensors_mixing_mode = {
            "uuup": "uuu",
            "uvvp": "uvv",
            "p": "p",
        }[tensors_mixing_mode]
        internal_weight_tp = True

        # == bookkeeping parameters ==
        self.tensor_basis_in_field = tensor_basis_in_field
        self.tensor_features_in_field = tensor_features_in_field
        self.scalar_in_field = scalar_in_field
        self.scalar_out_field = scalar_out_field

        # == set up irreps ==
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[
                self.tensor_basis_in_field,
                self.tensor_features_in_field,
                self.scalar_in_field,
            ],
        )
        scalar_input_dim: int = self.irreps_in[self.scalar_in_field].num_irreps

        latent = functools.partial(latent, **latent_kwargs)
        self.latents = torch.nn.ModuleList([])
        self.tps = torch.nn.ModuleList([])

        # Embed to the spharm * it as mul
        input_irreps = self.irreps_in[self.tensor_basis_in_field]
        # this is not inherant, but no reason to fix right now:
        assert all(mul == 1 for mul, ir in input_irreps)
        env_embed_irreps = o3.Irreps([(1, ir) for _, ir in input_irreps])
        assert (
            env_embed_irreps[0].ir == SCALAR
        ), "env_embed_irreps must start with scalars"

        arg_irreps = env_embed_irreps

        # === begin irreps ===
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
        # --- end build irreps ---

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

        # === edge `scatter` normalization ===
        env_sum_constant = 1.0
        if avg_num_neighbors is not None:
            # we divide the env embed sums by sqrt(N) to normalize
            env_sum_constant = 1.0 / math.sqrt(avg_num_neighbors)

        # === env weighter ===
        self._env_weighter = MakeWeightedChannels(
            irreps_in=input_irreps,
            multiplicity_out=self.num_tensor_features,
            weight_individual_irreps=weight_individual_irreps,
        )

        # === first layer linear projection ===
        # hardcode linear projection: twobody features -> twobody features + env weights
        self.first_layer_env_embed_projection = latent(
            mlp_input_dim=scalar_input_dim,
            mlp_hidden_layer_dims=[],
            mlp_hidden_layer_depth=None,
            mlp_hidden_layer_width=None,
            mlp_output_dim=self.num_scalar_features + self._env_weighter.weight_numel,
            mlp_nonlinearity=None,
        )
        assert not self.first_layer_env_embed_projection.is_nonlinear

        # === Build TPs and latents ===
        self._n_scalar_outs = []
        for layer_idx, (arg_irreps, out_irreps) in enumerate(
            zip(tps_irreps_in, tps_irreps_out)
        ):
            # irin2 is the scattered feature
            if self.scatter_features:
                irin1 = env_embed_irreps
                irin2 = arg_irreps
            else:
                irin1 = arg_irreps
                irin2 = env_embed_irreps

            tmp_i_out: int = 0
            instr = []
            n_scalar_outs: int = 0
            full_out_irreps = out_irreps if internal_weight_tp else []
            for i_out, (_, ir_out) in enumerate(out_irreps):
                for i_1, (_, ir_1) in enumerate(irin1):
                    for i_2, (_, ir_2) in enumerate(irin2):
                        if ir_out in ir_1 * ir_2:
                            if internal_weight_tp:
                                if ir_out == SCALAR:
                                    n_scalar_outs = 1
                                instr.append((i_1, i_2, i_out))
                            else:
                                if ir_out == SCALAR:
                                    n_scalar_outs += 1
                                instr.append((i_1, i_2, tmp_i_out))
                                full_out_irreps.append(
                                    (self.num_tensor_features, ir_out)
                                )
                                tmp_i_out += 1
            full_out_irreps = o3.Irreps(full_out_irreps)
            del tmp_i_out
            self._n_scalar_outs.append(n_scalar_outs)
            assert all(ir == SCALAR for _, ir in full_out_irreps[:n_scalar_outs])
            tp = Contracter(
                irreps_in1=o3.Irreps(
                    [(self.num_tensor_features, ir) for _, ir in irin1]
                ),
                irreps_in2=o3.Irreps(
                    [(self.num_tensor_features, ir) for _, ir in irin2]
                ),
                irreps_out=o3.Irreps(
                    [(self.num_tensor_features, ir) for _, ir in full_out_irreps]
                ),
                instructions=instr,
                connection_mode=tp_tensors_mixing_mode,
                initialization=tensor_track_weight_init,
                scatter_factor=env_sum_constant,
            )
            self.tps.append(tp)
            del tp
            # we extract the scalars from the first irrep of the tp
            assert out_irreps[0].ir == SCALAR

            self.latents.append(
                latent(
                    mlp_input_dim=(
                        # initial two-body scalar features +
                        # all scalar features from previous layer(s) (densenet structure)
                        self.num_scalar_features * (layer_idx + 1)
                        # scalars extracted from the this layer's TP:
                        # each layer executes a TP and then a latent, so the last entry in _n_scalar_outs corresponds to this layer's TP
                        + self.num_tensor_features * self._n_scalar_outs[-1]
                    ),
                    mlp_output_dim=(
                        # number of scalar features
                        self.num_scalar_features
                        # env weighter for next layer's TP (except in last layer)
                        + (
                            self._env_weighter.weight_numel
                            if layer_idx < self.num_layers - 1
                            else 0
                        )
                    ),
                )
            )

        # --- end build modules ----

        self.irreps_out.update(
            {
                self.scalar_out_field: o3.Irreps(
                    [(self.num_scalar_features * (self.num_layers + 1), (0, 1))]
                ),
            }
        )

    def extra_repr(self) -> str:
        msg = "  {:29} : {}\n".format("num layers", self.num_layers)
        msg += "  {:29} : {}\n".format("num scalar features", self.num_scalar_features)
        msg += "  {:29} : {}\n".format("num tensor features", self.num_tensor_features)
        msg += "  {:29} : {}\n".format(
            "scalar output dim", self.irreps_out[self.scalar_out_field].num_irreps
        )
        msg += "  {:29} : {}\n".format(
            "per-layer num tensor->scalar", self._n_scalar_outs
        )
        return msg

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        edge_center = data[AtomicDataDict.EDGE_INDEX_KEY][0]
        num_atoms: int = AtomicDataDict.num_nodes(data)

        # unweighted two-body tensor basis (i.e. spherical harmonics)
        tensor_basis = data[self.tensor_basis_in_field]
        # weighted two-body tensor features
        tensor_features = data[self.tensor_features_in_field]
        # two-body scalar embedding
        twobody_scalar_embed = data[self.scalar_in_field]

        # compute zeroth layer projection and slice for
        # - two-body feature
        # - the env embedding weights in the first layer
        projection = self.first_layer_env_embed_projection(twobody_scalar_embed)
        twobody_scalar_features = torch.narrow(
            projection, -1, 0, self.num_scalar_features
        )
        accumulated_scalar_features = [twobody_scalar_features]
        env_w = torch.narrow(
            projection, -1, self.num_scalar_features, self._env_weighter.weight_numel
        )

        layer_index: int = 0
        for latent, tp in zip(self.latents, self.tps):
            # === Env Weight & TP ===
            env_w_edges = self._env_weighter(tensor_basis, env_w)
            # depending on `self.scatter_features`, scatter `env_w_edges` or `tensor_features`, and TP them together
            # second input irreps is the one that is scattered
            if self.scatter_features:
                irin1 = env_w_edges
                irin2 = tensor_features
            else:
                irin1 = tensor_features
                irin2 = env_w_edges
            tensor_features = tp(irin1, irin2, edge_center, num_atoms)

            # Extract invariants from tensor track
            # features has shape [z][mul][k], where scalars are first
            scalars = tensor_features[:, :, : self._n_scalar_outs[layer_index]].reshape(
                tensor_features.shape[0],
                tensor_features.shape[1] * self._n_scalar_outs[layer_index],
            )

            # === Compute Latents & Slice ===
            latents = latent(torch.cat(accumulated_scalar_features + [scalars], dim=-1))
            # slice to
            # 1. propagated latents
            # 2. env embed weights for next layer (but not for last layer)

            # accumulate scalar features
            accumulated_scalar_features.append(
                torch.narrow(latents, -1, 0, self.num_scalar_features)
            )
            # env embed weights
            if layer_index < self.num_layers - 1:
                env_w = torch.narrow(
                    latents,
                    -1,
                    self.num_scalar_features,
                    self._env_weighter.weight_numel,
                )

            # == increment counter ==
            layer_index += 1

        # save accumulated scalar features
        data[self.scalar_out_field] = torch.cat(accumulated_scalar_features, dim=-1)
        return data
