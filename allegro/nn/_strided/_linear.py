from typing import List, Optional, Tuple, NamedTuple
from math import sqrt

import torch
from torch import fx

from e3nn import o3
from e3nn.o3._tensor_product._codegen import _sum_tensors

from opt_einsum_fx import jitable, optimize_einsums_full

from nequip.utils.compile import conditional_torchscript_jit
from ._layout import StridedLayout
from .._misc import _init_weight


class Instruction(NamedTuple):
    i_in: int
    i_out: int


def codegen_strided_linear(
    irreps_in: o3.Irreps,
    irreps_out: o3.Irreps,
    instructions: List[Instruction],
    normalization: str = "component",
    internal_weights: bool = False,
    shared_weights: bool = False,
    initialization: str = "uniform",
    alpha: float = 1.0,
) -> Optional[fx.GraphModule]:
    """Returns None if strided doesn't make sense for this TP."""
    # Check if irreps can be strided
    try:
        layout_in = StridedLayout(irreps_in)
        layout_out = StridedLayout(irreps_out)
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
            (
                (
                    1.0
                    / sqrt(sum(layout_in.mul for ins in instructions if ins.i_out == i))
                )
                * out
                * alpha
            )
            if len(ins_per_output[i]) > 0
            else out
        )
        for i, out in enumerate(outs)
    ]
    if len(outs) > 1:
        out = torch.cat(outs, dim=-1)
    else:
        out = outs[0]

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
        _init_weight(constants_root.w, initialization=initialization)
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
    graphmod_out.mul_in = layout_in.mul
    graphmod_out.dim_out = layout_out.base_dim
    graphmod_out.mul_out = layout_out.mul
    graphmod_out.irreps_in = str(irreps_in)
    graphmod_out.irreps_out = str(irreps_out)
    graphmod_out.instructions = [tuple(ins) for ins in instructions]
    graphmod_out._ins_group_irrep_slice = ins_group_irrep_slice

    return graphmod_out


def Linear(
    irreps_in,
    irreps_out,
    shared_weights: Optional[bool] = None,
    internal_weights: bool = False,
    initialization: str = "uniform",
    instructions: Optional[List[Tuple[int, int]]] = None,
    alpha: float = 1.0,
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

    instructions = [Instruction(i_in=e[0], i_out=e[1]) for e in instructions]

    mod = codegen_strided_linear(
        irreps_in,
        irreps_out,
        instructions=instructions,
        shared_weights=shared_weights,
        internal_weights=internal_weights,
        initialization=initialization,
        alpha=alpha,
    )

    if mod is None:
        raise ValueError
    return conditional_torchscript_jit(mod)


def weights_per_instruction(
    linear,
) -> Tuple[o3.Irreps, o3.Irreps, List[Tuple[int, int]], List[torch.Tensor]]:
    """Extract the per-instruction weights from a `Linear`.

    Args:
        linear

    Returns:
        irreps_in
        irreps_out
        instructions: list of tuples (in_index, out_index)
        weights: list of per-instruction weight tensors of shape (mul_out, mul_in)
    """
    irreps_in = o3.Irreps(linear.irreps_in)
    irreps_out = o3.Irreps(linear.irreps_out)
    instructions = linear.instructions

    layout_in = StridedLayout(irreps_in)
    layout_out = StridedLayout(irreps_out)

    weights = []
    ws = linear.w.detach()

    # in the code for efficiency the weight shape is vun,
    # where n is over instructions within an "instruction group"
    # i.e. the closest packed weight dimension goes over multiple instructions
    # we unpack that here
    # adapted from _linear.py#L103-L111
    w_index: int = 0
    n: int
    for ins_grp_ins in linear._ins_group_irrep_slice:
        if ins_grp_ins is None:
            # nothing goes to this output
            n = 0
        else:
            n = 1 + ins_grp_ins[1] - ins_grp_ins[0]
        n_weight = layout_in.mul * n * layout_out.mul
        this_w = ws[w_index : w_index + n_weight].reshape(
            layout_out.mul, layout_in.mul, n
        )
        for i in range(n):
            weights.append(this_w[:, :, i])
        w_index += n_weight

    weights = torch.vstack([w.unsqueeze(0) for w in weights])
    assert weights.shape == (len(instructions), layout_out.mul, layout_in.mul)

    return irreps_in, irreps_out, instructions, weights


def weights_from_per_instruction(linear, weights) -> torch.Tensor:
    """Re-pack the per-instruction weights from a `Linear`.

    Args:
        linear
        weights: list of per-instruction weight tensors of shape (mul_out, mul_in)

    Returns:
        weight: a flattened, packed weight tensor that can be assigned to `linear.w` or used in a state dict
    """
    instructions = linear.instructions

    assert len(weights) == len(instructions)
    assert frozenset(w.shape for w in weights) == {(linear.mul_out, linear.mul_in)}

    # need to group them back into instruction groups
    to_cat = []
    for ins_grp_ins in linear._ins_group_irrep_slice:
        if ins_grp_ins is None:
            # Nothing goes to this output
            continue
        to_cat.append(
            torch.cat(
                [
                    w.unsqueeze(-1)
                    for w, ins in zip(weights, instructions)
                    if ins[0] in range(ins_grp_ins[0], ins_grp_ins[1] + 1)
                ],
                dim=-1,
            )
        )

    return torch.cat([w.view(-1) for w in to_cat])
