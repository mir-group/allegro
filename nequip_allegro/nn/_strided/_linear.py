from typing import List, Optional, Tuple, NamedTuple
from math import sqrt

import torch
from torch import fx

from e3nn import o3
from e3nn.util.jit import compile
from e3nn.o3._tensor_product._codegen import _sum_tensors

from opt_einsum_fx import jitable, optimize_einsums_full

from ._layout import StridedLayout


class Instruction(NamedTuple):
    i_in: int
    i_out: int
    path_shape: tuple


def codegen_strided_linear(
    irreps_in: o3.Irreps,
    irreps_out: o3.Irreps,
    instructions: List[Instruction],
    normalization: str = "component",
    internal_weights: bool = False,
    shared_weights: bool = False,
    pad_to_alignment: int = 1,
) -> Optional[fx.GraphModule]:
    """Returns None if strided doesn't make sense for this TP."""
    # Check if irreps can be strided
    try:
        layout_in = StridedLayout(irreps_in, pad_to_multiple=pad_to_alignment)
        layout_out = StridedLayout(irreps_out, pad_to_multiple=pad_to_alignment)
    except ValueError:
        # one cannot be strided
        return None

    assert normalization == "component"

    if internal_weights:
        assert shared_weights

    # group instructions by output
    ins_per_output: List[List[Instruction]] = [
        [ins for ins in instructions if ins.i_out == i]
        for i in range(len(layout_out.base_irreps))
    ]
    ins_group_irrep_slice: List[Tuple[int, int]] = []
    # check that each output is a mix of sequential irreps
    for ins_group in ins_per_output:
        if len(ins_group) == 0:
            ins_group_irrep_slice.append(None)
            continue
        i_ins = set(ins.i_in for ins in ins_group)
        ins_group_irrep_slice.append((min(i_ins), max(i_ins)))
        min_i_in, max_i_in = ins_group_irrep_slice[-1]
        assert i_ins == set(range(min_i_in, 1 + max_i_in))
        assert all(
            layout_in.base_irreps[min_i_in] == layout_in.base_irreps[i]
            for i in range(min_i_in, max_i_in + 1)
        ), "All mixed irreps must be the same"
        assert all(ins.i_out == ins_group[0].i_out for ins in ins_group)

    # TODO: split bad groups into multiple groups

    # generate actual code
    graph_out = fx.Graph()
    tracer = fx.proxy.GraphAppendingTracer(graph_out)

    def Proxy(n):
        return fx.Proxy(n, tracer=tracer)

    # = Function definitions =
    x = Proxy(graph_out.placeholder("x", torch.Tensor))
    x = x.reshape(-1, layout_in.mul, layout_in.base_dim)
    if internal_weights:
        ws = Proxy(graph_out.get_attr("w", torch.Tensor))
    else:
        ws = Proxy(graph_out.placeholder("w", torch.Tensor))

    outs = [[] for _ in range(len(layout_out.base_irreps))]

    z = "" if shared_weights else "z"

    w_index: int = 0
    for ins_grp_i, (ins_grp, ins_grp_ins) in enumerate(
        zip(ins_per_output, ins_group_irrep_slice)
    ):
        if len(ins_grp) == 0:
            continue
        # for a given group, which mixes a consecutive set of irreps of the same irrep,
        # we can reduce it to a rectangular operation:
        to_mix = x[
            :,
            :,
            layout_in.base_irreps[: ins_grp_ins[0]]
            .dim : layout_in.base_irreps[: ins_grp_ins[1] + 1]
            .dim,
        ]  # index the i dim in z u i
        # ^ has i index ranging over ins_grp_ins inputs *of same irrep*, so we can rectangularize with a new "n" dimension:
        n: int = 1 + ins_grp_ins[1] - ins_grp_ins[0]
        to_mix = to_mix.reshape(
            -1, layout_in.mul, n, layout_in.base_irreps[ins_grp_ins[0]].ir.dim
        )  # z u n i
        n_weight = layout_in.mul * n * layout_out.mul
        if shared_weights:
            this_w = ws[w_index : w_index + n_weight].reshape(
                layout_out.mul, layout_in.mul, n
            )
        else:
            this_w = ws[:, w_index : w_index + n_weight].reshape(
                -1, layout_out.mul, layout_in.mul, n
            )
        outs[ins_grp[0].i_out].append(torch.einsum(f"{z}vun,zuni->zvi", this_w, to_mix))
        w_index += n_weight

    outs = [
        _sum_tensors(
            o,
            shape=(
                x.shape[0],
                layout_out.mul,
                layout_out.base_irreps[i].dim,
            ),
            like=x,
        )
        for i, (ins_grp, o) in enumerate(zip(ins_per_output, outs))
    ]
    outs = [
        (
            (1.0 / sqrt(sum(layout_in.mul for ins in instructions if ins.i_out == i)))
            * out
        )
        if len(ins_per_output[i]) > 0
        else out
        for i, out in enumerate(outs)
    ]
    if len(outs) > 1:
        out = torch.cat(outs, dim=-1)
    else:
        out = outs[0]

    # pad output
    padding: int = layout_out.base_dim - layout_out.base_irreps.dim
    if padding > 0:
        out = torch.nn.functional.pad(
            out,
            (0, padding),
        )

    graph_out.output(out.node)

    # check graphs
    graph_out.lint()

    # Make GraphModules
    # By putting the constants in a Module rather than a dict,
    # we force FX to copy them as buffers instead of as attributes.
    #
    # FX seems to have resolved this issue for dicts in 1.9, but we support all the way back to 1.8.0.
    constants_root = torch.nn.Module()
    if internal_weights:
        constants_root.w = torch.nn.Parameter(torch.randn(w_index))
    graphmod_out = fx.GraphModule(
        constants_root, graph_out, class_name="linear_forward"
    )

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
        # TODO: consider the impact maximum intermediate result size on this logic
        #         \- this is the `memory_limit` option in opt_einsum
        # TODO: allow user to choose opt_einsum parameters?
        #
        # We use float32 and zeros to save memory and time, since opt_einsum_fx looks only at traced shapes, not values or dtypes.
        batchdim = 4
        example_inputs = (
            torch.zeros((batchdim, layout_in.dim)),
            torch.zeros((tuple() if shared_weights else (batchdim,)) + (w_index,)),
        )
        graphmod_out = jitable(optimize_einsums_full(graphmod_out, example_inputs))

    graphmod_out.weight_numel = w_index
    graphmod_out.dim_in = layout_in.base_dim

    return graphmod_out


def Linear(
    irreps_in,
    irreps_out,
    shared_weights: Optional[bool] = None,
    internal_weights: bool = False,
    instructions: Optional[List[Tuple[int, int]]] = None,
    pad_to_alignment: int = 1,
):
    irreps_in = o3.Irreps(irreps_in)
    irreps_out = o3.Irreps(irreps_out)
    # == Instructions ==
    if instructions is None:
        # By default, make all possible connections
        instructions = [
            (i_in, i_out)
            for i_in, (_, ir_in) in enumerate(irreps_in)
            for i_out, (_, ir_out) in enumerate(irreps_out)
            if ir_in == ir_out
        ]
        # note that "empty" instructions to/from empty irreps are dealt with in the codegen

    instructions = [
        Instruction(i_in=e[0], i_out=e[1], path_shape=None) for e in instructions
    ]

    mod = codegen_strided_linear(
        irreps_in,
        irreps_out,
        instructions=instructions,
        shared_weights=shared_weights,
        internal_weights=internal_weights,
        pad_to_alignment=pad_to_alignment,
    )

    if mod is None:
        raise ValueError
    return compile(mod)
