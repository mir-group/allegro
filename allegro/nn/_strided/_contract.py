from typing import List, Optional, Tuple
from math import sqrt

import torch
from torch import fx

from e3nn import o3
from e3nn.util.jit import compile
from e3nn.util import prod
from e3nn.o3 import Instruction

from opt_einsum_fx import jitable, optimize_einsums, EfficientShapeProp

from ._layout import StridedLayout
from .._misc import _init_weight


def codegen_strided_tensor_product_forward(
    irreps_in1: o3.Irreps,
    in1_var: List[float],
    irreps_in2: o3.Irreps,
    in2_var: List[float],
    irreps_out: o3.Irreps,
    out_var: List[float],
    instructions: List[Instruction],
    normalization: str = "component",
    shared_weights: bool = False,
    internal_weights: bool = False,
    initialization: str = "uniform",
    specialized_code: bool = True,
    pad_to_alignment: int = 1,
) -> Optional[fx.GraphModule]:
    """Returns None if strided doesn't make sense for this TP."""
    # TODO padding
    # Check if irreps can be strided
    try:
        layout_in1 = StridedLayout(irreps_in1, pad_to_multiple=pad_to_alignment)
        layout_in2 = StridedLayout(irreps_in2, pad_to_multiple=pad_to_alignment)
        layout_out = StridedLayout(irreps_out, pad_to_multiple=pad_to_alignment)
    except ValueError:
        # one cannot be strided
        return None

    # check the instructions
    assert specialized_code

    connection_mode = instructions[0].connection_mode
    if not all(ins.connection_mode == connection_mode for ins in instructions):
        return None

    has_weight = instructions[0].has_weight
    if not all(ins.has_weight == has_weight for ins in instructions):
        return None
    if not has_weight:
        assert connection_mode == "uuu"  # for now

    if internal_weights:
        assert shared_weights
        assert has_weight

    # TODO: sort insturctions?

    # Make the big w3j
    w3j_index = []
    w3j_values = []

    for ins_i, ins in enumerate(instructions):
        mul_ir_in1 = layout_in1.base_irreps[ins.i_in1]
        mul_ir_in2 = layout_in2.base_irreps[ins.i_in2]
        mul_ir_out = layout_out.base_irreps[ins.i_out]

        assert mul_ir_in1.ir.p * mul_ir_in2.ir.p == mul_ir_out.ir.p
        assert (
            abs(mul_ir_in1.ir.l - mul_ir_in2.ir.l)
            <= mul_ir_out.ir.l
            <= mul_ir_in1.ir.l + mul_ir_in2.ir.l
        )

        if mul_ir_in1.dim == 0 or mul_ir_in2.dim == 0 or mul_ir_out.dim == 0:
            raise ValueError

        this_w3j = o3.wigner_3j(mul_ir_in1.ir.l, mul_ir_in2.ir.l, mul_ir_out.ir.l)
        this_w3j_index = this_w3j.nonzero()
        w3j_values.append(
            this_w3j[this_w3j_index[:, 0], this_w3j_index[:, 1], this_w3j_index[:, 2]]
        )

        # Normalize the path through its w3j entries
        # TODO: path_weight
        # TODO: in and out var
        if normalization == "component":
            w3j_norm_term = 2 * mul_ir_out.ir.l + 1
        if normalization == "norm":
            w3j_norm_term = (2 * mul_ir_in1.ir.l + 1) * (2 * mul_ir_in2.ir.l + 1)
        alpha = sqrt(
            ins.path_weight  # per-path weight
            * out_var[ins.i_out]  # enforce output variance
            * w3j_norm_term
            / sum(
                in1_var[i.i_in1]
                * in2_var[i.i_in2]
                * {
                    "uvw": (layout_in1.mul * layout_in2.mul),
                    "uvu": layout_in2.mul,
                    "uvv": layout_in1.mul,
                    "uuw": layout_in1.mul,
                    "uuu": 1,
                    "uvuv": 1,
                    # p means just to weight paths, so no change to normalization per-path
                    "p": 1,
                }[connection_mode]
                for i in instructions
                if i.i_out == ins.i_out
            )
        )
        w3j_values[-1].mul_(alpha)

        this_w3j_index[:, 0] += layout_in1.base_irreps[: ins.i_in1].dim
        this_w3j_index[:, 1] += layout_in2.base_irreps[: ins.i_in2].dim
        this_w3j_index[:, 2] += layout_out.base_irreps[: ins.i_out].dim
        # Now need to flatten the index to be for [pk][ij]
        w3j_index.append(
            torch.cat(
                (
                    (ins_i if ins.has_weight else 0)  # unweighted all go in first path
                    * layout_out.base_dim
                    + this_w3j_index[:, 2].unsqueeze(-1),
                    this_w3j_index[:, 0].unsqueeze(-1) * layout_in2.base_dim
                    + this_w3j_index[:, 1].unsqueeze(-1),
                ),
                dim=1,
            )
        )

    num_paths: int = len(instructions) if has_weight else 1

    w3j = torch.sparse_coo_tensor(
        indices=torch.cat(w3j_index, dim=0).t(),
        values=torch.cat(w3j_values, dim=0),
        size=(
            num_paths * layout_out.base_dim,
            layout_in1.base_dim * layout_in2.base_dim,
        ),
    ).coalesce()

    # w3j is k,i,j, so this is whether, for nonzero entries,
    # the i index is always equal to the j index. If so, then
    # it is diagonal and we can eliminate the j dimension
    # in this case we are only taking diagonal (i == j)
    # entries from the outer product; but those values are just
    # the direct multiplication of the two tensors, eliminating
    # the need for the outer product.
    # obviously this only makes sense if they have the same size as well
    # this is more or less a test of whether this TP is an inner product
    w3j_i_indexes = torch.div(
        w3j.indices()[1], layout_in1.base_dim, rounding_mode="floor"
    )
    w3j_j_indexes = w3j.indices()[1] % layout_in1.base_dim
    w3j_is_ij_diagonal = (layout_in1.base_dim == layout_in2.base_dim) and torch.all(
        w3j_i_indexes == w3j_j_indexes
    )
    if w3j_is_ij_diagonal:
        # change the w3j to eliminate the dimension
        # now its just k,i
        w3j = torch.sparse_coo_tensor(
            indices=torch.stack((w3j.indices()[0], w3j_i_indexes)),
            values=w3j.values(),
            size=(
                num_paths * layout_out.base_dim,
                layout_in1.base_dim,
            ),
        )

    # in dense, must shape it for einsum:
    if w3j_is_ij_diagonal:
        kij_shape = (
            layout_out.base_dim,
            layout_in1.base_dim,
        )
    else:
        kij_shape = (
            layout_out.base_dim,
            layout_in1.base_dim,
            layout_in2.base_dim,
        )
    w3j = (
        w3j.to_dense()
        .reshape(((num_paths,) if num_paths > 1 else tuple()) + kij_shape)
        .contiguous()
    )
    del kij_shape

    # Generate the mixer
    u, v, w = "uuu" if connection_mode == "p" else connection_mode
    if has_weight:
        weight_label = {"uvw": "uvw", "uuu": "u", "uvv": "uv", "p": ""}[connection_mode]

        z = "" if shared_weights else "z"

        weight_shape = {
            "uvw": (layout_in1.mul, layout_in2.mul, layout_out.mul),
            "uuu": (layout_in1.mul,),
            "uvv": (layout_in1.mul, layout_in2.mul),
            "p": tuple(),
        }[connection_mode]
        if connection_mode == "p":
            assert num_paths > 1
        if num_paths > 1:
            # ^ if there's only one weighted path, the einsum simplifies without the p dimension
            weight_label = weight_label + "p"
            weight_shape = weight_shape + (num_paths,)
        if not shared_weights:
            weight_shape = (-1,) + weight_shape
    else:
        weight_shape = tuple()

    # generate actual code
    graph_out = fx.Graph()
    base_module = torch.nn.Module()
    tracer = fx.proxy.GraphAppendingTracer(graph_out)

    def Proxy(n):
        return fx.Proxy(n, tracer=tracer)

    # = Function definitions =
    x1s_out = Proxy(graph_out.placeholder("x1", torch.Tensor))
    x2s_out = Proxy(graph_out.placeholder("x2", torch.Tensor))
    if has_weight:
        if internal_weights:
            ws_out = Proxy(graph_out.get_attr("w", torch.Tensor))
            base_module.w = torch.nn.Parameter(torch.randn(weight_shape))
            _init_weight(base_module.w, initialization=initialization)
        else:
            ws_out = Proxy(graph_out.placeholder("w", torch.Tensor))
            ws_out = ws_out.reshape(weight_shape)

    w3j_proxy = Proxy(graph_out.get_attr("_big_w3j"))

    if shared_weights and connection_mode == "p":
        # special case
        assert has_weight
        # we can precontract weights and w3j to remove the `p` dimension
        # weights are `p`, w3j is pkij
        # non diagonal
        if w3j_is_ij_diagonal:
            x1s_out = x1s_out.reshape(-1, layout_in1.base_dim)
            x2s_out = x2s_out.reshape(-1, layout_in2.base_dim)
            # ikp @ p -> ik (mv)
            # zi,zi->zi     (mul)
            # zi @ ik -> zk  (mm)
            # w3j is pki here
            w3j = w3j.permute(2, 1, 0).reshape(-1, num_paths).contiguous()  # => [ik]p
            # contract with p weights, ikp @ p -> ik (mv)
            ww3j_proxy = torch.mv(w3j_proxy, ws_out).view(
                layout_in1.base_dim, layout_out.base_dim
            )  # => ik
            out = x1s_out * x2s_out  # zi
            out = torch.mm(out, ww3j_proxy).view(
                -1, layout_out.mul, layout_out.base_dim
            )  # zk
        else:
            x1s_out = x1s_out.reshape(-1, layout_in1.base_dim)
            x2s_out = x2s_out.reshape(-1, layout_in2.base_dim, 1)
            # ikjp @ p -> ikj (mv)
            # zi @ i[kj] -> z[kj]  (mm)
            # zkj @ zj1 -> zk1 (bmm)
            # w3j is pkij here
            w3j = (
                w3j.permute(2, 1, 3, 0).reshape(-1, num_paths).contiguous()
            )  # => [ikj]p
            # contract with p weights, ikjp @ p -> ikj (mv)
            ww3j_proxy = torch.mv(w3j_proxy, ws_out).view(
                layout_in1.base_dim, layout_out.base_dim * layout_in2.base_dim
            )  # i[kj]
            out = torch.mm(x1s_out, ww3j_proxy).view(
                -1, layout_out.base_dim, layout_in2.base_dim
            )  # z[kj] -> zkj
            out = torch.bmm(out, x2s_out).reshape(
                -1, layout_out.mul, layout_out.base_dim
            )  # [zu]k1 -> zuk
    else:
        # convert to strided
        x1s_out = x1s_out.reshape(-1, layout_in1.mul, layout_in1.base_dim)
        x2s_out = x2s_out.reshape(-1, layout_in2.mul, layout_in2.base_dim)
        # do the einsum
        # has shape zwk
        j = "i" if w3j_is_ij_diagonal else "j"
        ij = "i" if w3j_is_ij_diagonal else "ij"
        if has_weight:
            p = "p" if num_paths > 1 else ""

            if shared_weights:
                # for shared weights, we can precontract weights and w3j so they can be frozen together
                # this is usually advantageous for inference, since the weights would have to be
                # multiplied in anyway at some point
                ww3j_proxy = torch.einsum(
                    f"{weight_label},{p}k{ij}->{weight_label.rstrip('p')}k{ij}",
                    ws_out,
                    w3j_proxy,
                )
                # we use minimal opt_einsum_fx later without einsum fusion, so this is safe
                out = torch.einsum(
                    f"z{u}i,z{v}{j},{weight_label.rstrip('p')}k{ij}->z{w}k",
                    x1s_out,
                    x2s_out,
                    ww3j_proxy,
                )
            else:
                # use einsum for the full contract
                einstr = f"{z}{weight_label},z{u}i,z{v}{j},{p}k{ij}->z{w}k"
                out = torch.einsum(einstr, ws_out, x1s_out, x2s_out, w3j_proxy)
        else:
            # use einsum for the full contract
            einstr = f"z{u}i,z{v}{j},{'p' if num_paths > 1 else ''}k{ij}->z{w}k"
            out = torch.einsum(einstr, x1s_out, x2s_out, w3j_proxy)

    graph_out.output(out.node)

    # check graphs
    graph_out.lint()

    # Make GraphModules
    base_module.register_buffer("_big_w3j", w3j)
    graphmod_out = fx.GraphModule(base_module, graph_out, class_name="tp_forward")

    if True:  # optimize_einsums
        # Note that for our einsums, we can optimize _once_ for _any_ batch dimension
        # and still get the right path for _all_ batch dimensions.
        # This is because our einsums are essentially of the form:
        #    zuvw,ijk,zuvij->zwk    OR     uvw,ijk,zuvij->zwk
        # In the first case, all but one operands have the batch dimension
        #    => The first contraction gains the batch dimension
        #    => All following contractions have batch dimension
        #    => All possible contraction paths have cost that scales linearly in batch size
        #    => The optimal path is the same for all batch sizes
        # For the second case, this logic follows as long as the first contraction is not between the first two operands. Since those two operands do not share any indexes, contracting them first is a rare pathological case. See
        # https://github.com/dgasmith/opt_einsum/issues/158
        # for more details.
        #
        # We use float32 and zeros to save memory and time, since opt_einsum_fx looks only at traced shapes, not values or dtypes.
        batchdim = 4
        example_inputs = (
            torch.zeros((batchdim, layout_in1.dim)),
            torch.zeros((batchdim, layout_in2.dim)),
            torch.zeros(
                1 if shared_weights else batchdim,
                sum(prod(ins.path_shape) for ins in instructions if ins.has_weight),
            ).squeeze(0),
        )
        if internal_weights:
            example_inputs = example_inputs[:-1]
        # graphmod_out = jitable(optimize_einsums_full(graphmod_out, example_inputs))
        # We do a minimal einsum optimization, since all codegen above only contains
        # either one einsum or intentionally two separate einsums, so we can avoid
        # einsum fusion, and all scalars are already rolled into the w3j, so
        # we can skip scalar fusion.
        # Shape propagation
        sp = EfficientShapeProp(graphmod_out)
        sp.run(*example_inputs)
        # Optimize einsums
        graphmod_out.graph = jitable(optimize_einsums(graphmod_out.graph))
        graphmod_out.recompile()

    graphmod_out.weight_shape = weight_shape
    graphmod_out._dim_in1 = layout_in1.base_dim
    graphmod_out._dim_in2 = layout_in2.base_dim
    graphmod_out._dim_out = layout_out.base_dim
    graphmod_out._mul_out = layout_out.mul
    graphmod_out.weight_numel = abs(prod(weight_shape))
    graphmod_out.irreps_in1 = str(irreps_in1)
    graphmod_out.irreps_in2 = str(irreps_in2)
    graphmod_out.irreps_out = str(irreps_out)
    graphmod_out.connection_mode = connection_mode
    graphmod_out.has_weight = has_weight
    graphmod_out.instructions = [
        (ins.i_in1, ins.i_in2, ins.i_out) for ins in instructions
    ]

    return graphmod_out


def Contracter(
    irreps_in1,
    irreps_in2,
    irreps_out,
    instructions: List[Tuple[int, int, int]],
    has_weight: bool,
    connection_mode: str,
    pad_to_alignment: int = 1,
    shared_weights: bool = False,
    internal_weights: bool = False,
    initialization: str = "uniform",
):
    irreps_in1 = o3.Irreps(irreps_in1)
    assert all(mul == irreps_in1[0].mul for mul, ir in irreps_in1)
    irreps_in2 = o3.Irreps(irreps_in2)
    assert all(mul == irreps_in2[0].mul for mul, ir in irreps_in2)
    irreps_out = o3.Irreps(irreps_out)
    assert all(mul == irreps_out[0].mul for mul, ir in irreps_out)

    mod = codegen_strided_tensor_product_forward(
        irreps_in1,
        [1.0 for _ in irreps_in1],
        irreps_in2,
        [1.0 for _ in irreps_in2],
        irreps_out,
        [1.0 for _ in irreps_out],
        instructions=[
            Instruction(
                i_in1,
                i_in2,
                i_out,
                connection_mode,
                has_weight,
                1.0,
                {
                    "uvw": (
                        irreps_in1[i_in1].mul,
                        irreps_in2[i_in2].mul,
                        irreps_out[i_out].mul,
                    ),
                    "uvu": (irreps_in1[i_in1].mul, irreps_in2[i_in2].mul),
                    "uvv": (irreps_in1[i_in1].mul, irreps_in2[i_in2].mul),
                    "uuw": (irreps_in1[i_in1].mul, irreps_out[i_out].mul),
                    "uuu": (irreps_in1[i_in1].mul,),
                    "uvuv": (
                        irreps_in1[i_in1].mul,
                        irreps_in2[i_in2].mul,
                    ),
                    "p": tuple(),
                }[connection_mode],
            )
            for i_in1, i_in2, i_out in instructions
        ],
        shared_weights=shared_weights,
        internal_weights=internal_weights,
        initialization=initialization,
        pad_to_alignment=pad_to_alignment,
    )
    if mod is None:
        raise ValueError("Couldn't use strided for given layout")
    mod = compile(mod)
    return mod


def contracter_paths(contracter) -> List[Tuple[o3.Irrep, o3.Irrep, o3.Irrep]]:
    """Return the irreps combined by each path/instruction in ``contracter``.

    Args:
        contracter

    Returns:
        paths: a list of tuples of (irrep_in1, irrep_in2, irrep_out)
    """
    irreps_in1 = o3.Irreps(contracter.irreps_in1)
    irreps_in2 = o3.Irreps(contracter.irreps_in2)
    irreps_out = o3.Irreps(contracter.irreps_out)
    return [
        (
            irreps_in1[tp_ins[0]].ir,
            irreps_in2[tp_ins[1]].ir,
            irreps_out[tp_ins[2]].ir,
        )
        for tp_ins in contracter.instructions
    ]
