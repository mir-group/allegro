# This file is a part of the `allegro` package. Please see LICENSE and README at the root for information on using it.
import torch

from torch.library import triton_op, wrap_triton
import triton
import triton.language as tl

from ._contract import Contracter
from ._lexsort import lexsort


TORCH_TRITON_DTYPE_MAPPER = {
    torch.float64: tl.float64,
    torch.float32: tl.float32,
    torch.float16: tl.float16,
}


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_B": 16, "BLOCK_DIM": 16}, num_warps=4, num_stages=2)
    ],
    key=["BATCH", "XDIM", "YDIM", "OUTDIM", "UMAX", "NNZ"],
)
@triton.jit
def tensor_product_p_kernel(
    # Pointers to matrices
    x_ptr,
    y_ptr,
    output_ptr,
    # Pointers to sparse data
    indptr_ptr,
    x_idx_ptr,
    y_idx_ptr,
    p_to_nnz_mapper_ptr,
    # CG vals
    vals_ptr,
    # Weights
    weights_ptr,
    # Matrix dimensions
    BATCH,
    XDIM,
    YDIM,
    OUTDIM,
    UMAX,
    NNZ,
    # Strides
    x_stride_dim,
    x_stride_batch,
    y_stride_dim,
    y_stride_batch,
    output_stride_dim,
    output_stride_batch,
    # Grid level blocks
    BLOCK_B: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    output_dtype: tl.constexpr,
):

    # Program IDs remain the same
    pid_b = tl.program_id(0)
    pid_dim = tl.program_id(1)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_DIM, BLOCK_B), dtype=output_dtype)

    # Sparse iteration setup
    start_ptr = tl.load(
        indptr_ptr + ((pid_dim * BLOCK_DIM) + tl.arange(0, BLOCK_DIM)),
        mask=(((pid_dim * BLOCK_DIM) + tl.arange(0, BLOCK_DIM)) < OUTDIM),
    )
    end_ptr = tl.load(
        indptr_ptr + ((pid_dim * BLOCK_DIM) + tl.arange(0, BLOCK_DIM)) + 1,
        mask=(((pid_dim * BLOCK_DIM) + tl.arange(0, BLOCK_DIM)) < OUTDIM),
    )
    max_nnz = tl.max(end_ptr - start_ptr)

    # Process non-zero elements
    for p in tl.range(0, max_nnz, loop_unroll_factor=1):
        pos = start_ptr + p
        pos_mask = pos < end_ptr

        b = (pid_b * BLOCK_B) + tl.arange(0, BLOCK_B)

        b_mask = b < (BATCH * UMAX)

        x_idx = tl.load(x_idx_ptr + pos, mask=pos_mask, eviction_policy="evict_first")
        y_idx = tl.load(y_idx_ptr + pos, mask=pos_mask, eviction_policy="evict_first")
        w_idx = tl.load(
            p_to_nnz_mapper_ptr + pos, mask=pos_mask, eviction_policy="evict_first"
        )
        vals = tl.load(vals_ptr + pos, mask=pos_mask, eviction_policy="evict_first")

        # Create load mask
        load_mask = pos_mask[:, None] & b_mask[None, :]

        # Calculate input offsets
        x_offsets = b[None, :] * x_stride_batch + x_idx[:, None]

        y_offsets = b[None, :] * y_stride_batch + y_idx[:, None]

        # Load inputs and compute
        x = tl.load(x_ptr + x_offsets, mask=load_mask)
        y = tl.load(y_ptr + y_offsets, mask=load_mask)
        w = tl.load(weights_ptr + w_idx, mask=pos_mask)

        product = x * y * vals[:, None] * w[:, None]
        acc = tl.where(load_mask, acc + product, acc)

    # Calculate output offsets
    out_offsets = ((pid_b * BLOCK_B) + tl.arange(0, BLOCK_B))[
        None, :
    ] * output_stride_batch + ((pid_dim * BLOCK_DIM) + tl.arange(0, BLOCK_DIM))[
        :, None
    ] * output_stride_dim

    # Create combined mask for output
    full_mask = (((pid_b * BLOCK_B) + tl.arange(0, BLOCK_B)) < (BATCH * UMAX))[
        None, :
    ] & (((pid_dim * BLOCK_DIM) + tl.arange(0, BLOCK_DIM)) < OUTDIM)[:, None]

    tl.store(output_ptr + out_offsets, acc, mask=full_mask)


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_B": 8, "BLOCK_U": 8, "BLOCK_DIM": 16},
            num_warps=4,
            num_stages=3,
            maxnreg=128 if "nvidia" in torch.cuda.get_device_name(0).lower() else None,
        )
    ],
    key=["BATCH", "XDIM", "YDIM", "OUTDIM", "UMAX", "NNZ"],
)
@triton.jit
def tensor_product_up_kernel(
    # Pointers to matrices
    x_ptr,
    y_ptr,
    output_ptr,
    # Pointers to sparse data
    indptr_ptr,
    x_idx_ptr,
    y_idx_ptr,
    p_to_nnz_mapper_ptr,
    # CG vals
    vals_ptr,
    # Weights
    weights_ptr,
    # Matrix dimensions
    BATCH,
    XDIM,
    YDIM,
    OUTDIM,
    UMAX,
    NNZ,
    # Strides
    x_stride_dim,
    x_stride_u,
    x_stride_batch,
    y_stride_dim,
    y_stride_u,
    y_stride_batch,
    output_stride_dim,
    output_stride_u,
    output_stride_batch,
    vals_stride_dim,
    weight_stride_u,
    weight_stride_dim,
    # Block sizes
    BLOCK_B: tl.constexpr,
    BLOCK_U: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    output_dtype: tl.constexpr,
):

    # Calculate program IDs
    pid_b = tl.program_id(0)
    pid_u = tl.program_id(1)
    pid_dim = tl.program_id(2)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_DIM, BLOCK_U, BLOCK_B), dtype=output_dtype)

    # Sparse iteration setup
    start_ptr = tl.load(
        indptr_ptr + ((pid_dim * BLOCK_DIM) + tl.arange(0, BLOCK_DIM)),
        mask=(((pid_dim * BLOCK_DIM) + tl.arange(0, BLOCK_DIM)) < OUTDIM),
    )
    end_ptr = tl.load(
        indptr_ptr + ((pid_dim * BLOCK_DIM) + tl.arange(0, BLOCK_DIM)) + 1,
        mask=(((pid_dim * BLOCK_DIM) + tl.arange(0, BLOCK_DIM)) < OUTDIM),
    )
    max_nnz = tl.max(end_ptr - start_ptr)

    # Process non-zero elements
    for p in tl.range(0, max_nnz, loop_unroll_factor=2):
        pos = start_ptr + p
        pos_mask = pos < end_ptr

        b = (pid_b * BLOCK_B) + tl.arange(0, BLOCK_B)
        u = (pid_u * BLOCK_U) + tl.arange(0, BLOCK_U)

        b_mask = b < BATCH
        u_mask = u < UMAX

        x_idx = tl.load(x_idx_ptr + pos, mask=pos_mask, eviction_policy="evict_first")
        y_idx = tl.load(y_idx_ptr + pos, mask=pos_mask, eviction_policy="evict_first")
        w_idx = tl.load(
            p_to_nnz_mapper_ptr + pos, mask=pos_mask, eviction_policy="evict_first"
        )

        # Load values
        vals = tl.load(vals_ptr + pos, mask=pos_mask)

        # Create load mask
        load_mask = (
            pos_mask[:, None, None] & u_mask[None, :, None] & b_mask[None, None, :]
        )
        load_mask_w = pos_mask[:, None] & u_mask[None, :]

        # Calculate input offsets
        x_offsets = (
            b[None, None, :] * x_stride_batch
            + u[None, :, None] * x_stride_u
            + x_idx[:, None, None]
        )

        y_offsets = (
            b[None, None, :] * y_stride_batch
            + u[None, :, None] * y_stride_u
            + y_idx[:, None, None]
        )

        w_offsets = u[None, :] * weight_stride_u + w_idx[:, None] * weight_stride_dim
        # Load inputs and compute
        x = tl.load(x_ptr + x_offsets, mask=load_mask)
        y = tl.load(y_ptr + y_offsets, mask=load_mask)
        w = tl.load(weights_ptr + w_offsets, mask=load_mask_w)

        vals = tl.broadcast_to(vals[:, None, None], (BLOCK_DIM, BLOCK_U, BLOCK_B))
        w = tl.broadcast_to(w[:, :, None], (BLOCK_DIM, BLOCK_U, BLOCK_B))

        product = x * y * vals * w
        acc = tl.where(load_mask, acc + product, acc)

    # Store results
    # Calculate output offsets
    out_offsets = (
        ((pid_b * BLOCK_B) + tl.arange(0, BLOCK_B))[None, None, :] * output_stride_batch
        + ((pid_u * BLOCK_U) + tl.arange(0, BLOCK_U))[None, :, None] * output_stride_u
        + ((pid_dim * BLOCK_DIM) + tl.arange(0, BLOCK_DIM))[:, None, None]
        * output_stride_dim
    )

    # Create combined mask for output
    full_mask = (
        (((pid_b * BLOCK_B) + tl.arange(0, BLOCK_B)) < BATCH)[None, None, :]
        & (((pid_u * BLOCK_U) + tl.arange(0, BLOCK_U)) < UMAX)[None, :, None]
        & (((pid_dim * BLOCK_DIM) + tl.arange(0, BLOCK_DIM)) < OUTDIM)[:, None, None]
    )

    tl.store(output_ptr + out_offsets, acc, mask=full_mask)


def _metadata_helper(dim, coo, coovalue, p_to_nnz_values):
    sort_order = lexsort((coo[:, 2], coo[:, 1], coo[:, 0]))
    sorted_coo = coo[sort_order]

    # Create indptr with torch.long first, then validate and cast to int16
    indptr = torch.zeros(dim + 1, dtype=torch.long)
    indices = sorted_coo[:, 0].long() + 1
    src = torch.ones_like(indices, dtype=torch.long)
    indptr = indptr.clone().scatter_add_(0, indices, src)
    del src, indices
    indptr = torch.cumsum(indptr, dim=0)

    # Assert that all values fit within int16 range
    assert torch.all(
        indptr <= torch.iinfo(torch.int16).max
    ), "Values in indptr exceed int16 max value"
    assert torch.all(
        indptr >= torch.iinfo(torch.int16).min
    ), "Values in indptr exceed int16 min value"

    # Cast to int16 after validation
    indptr = indptr.to(dtype=torch.int16)

    l1s = sorted_coo[:, 1].contiguous()
    l2s = sorted_coo[:, 2].contiguous()

    vals = coovalue[sort_order].contiguous()
    p_to_nnz_mapper = p_to_nnz_values[sort_order].contiguous()
    return indptr, l1s, l2s, vals, p_to_nnz_mapper


def _initialize_metadata(w3j):

    assert len(w3j.shape) == 4
    P, I, J, K = w3j.shape

    # push all the non-zeros from pijk -> ijk which is what we do in the weight computation
    w3j_sum = torch.sum(w3j, dim=0)
    coo = torch.nonzero(w3j_sum)
    coovalue = w3j_sum[tuple(coo.t())]

    # process `w3j` to `p_to_nnz_mapper`
    # Create with torch.long first
    p_to_nnz_mapper = torch.full_like(w3j_sum, fill_value=-1, dtype=torch.long)
    for p in range(w3j.size(0)):
        nzidx = torch.nonzero(w3j[p], as_tuple=True)
        p_to_nnz_mapper[nzidx[0], nzidx[1], nzidx[2]] = p
    del nzidx, w3j_sum, w3j

    # Assert that all values fit within int16 range
    assert torch.all(
        p_to_nnz_mapper <= torch.iinfo(torch.int16).max
    ), "Values in p_to_nnz_mapper exceed int16 max value"
    assert torch.all(
        p_to_nnz_mapper >= torch.iinfo(torch.int16).min
    ), "Values in p_to_nnz_mapper exceed int16 min value"

    # Cast to int16 after validation
    p_to_nnz_mapper = p_to_nnz_mapper.to(dtype=torch.int16)

    p_to_nnz_mapper_nzidx = torch.nonzero(p_to_nnz_mapper >= 0)
    p_to_nnz_values = p_to_nnz_mapper[tuple(p_to_nnz_mapper_nzidx.t())]

    del p_to_nnz_mapper, p_to_nnz_mapper_nzidx

    # forward
    indptr_fwd, l1s_fwd, l2s_fwd, vals_fwd, p_to_nnz_mapper_fwd = _metadata_helper(
        K, coo[:, [2, 0, 1]], coovalue, p_to_nnz_values
    )

    # backward wrt input1
    indptr_bwd1, ks_bwd1, l2s_bwd1, vals_bwd1, p_to_nnz_mapper_bwd1 = _metadata_helper(
        I, coo[:, [0, 2, 1]], coovalue, p_to_nnz_values
    )

    # backward wrt input2
    indptr_bwd2, ks_bwd2, l1s_bwd2, vals_bwd2, p_to_nnz_mapper_bwd2 = _metadata_helper(
        J, coo[:, [1, 2, 0]], coovalue, p_to_nnz_values
    )

    return (
        indptr_fwd,
        l1s_fwd,
        l2s_fwd,
        indptr_bwd1,
        ks_bwd1,
        l2s_bwd1,
        indptr_bwd2,
        ks_bwd2,
        l1s_bwd2,
        vals_fwd,
        vals_bwd1,
        vals_bwd2,
        p_to_nnz_mapper_fwd,
        p_to_nnz_mapper_bwd1,
        p_to_nnz_mapper_bwd2,
    )


@triton_op("mylib::allegro", mutates_args={})
def _triton_kernel_allegro(
    mode: str,
    # Pointers to matrices
    x: torch.Tensor,
    y: torch.Tensor,
    # Pointers to sparse data
    indptr: torch.Tensor,
    x_idx: torch.Tensor,
    y_idx: torch.Tensor,
    # CG values and W idx
    vals: torch.Tensor,
    p_to_nnz_mapper: torch.Tensor,
    # Weights
    weights: torch.Tensor,
    # Matrix dimensions
    OUTDIM: int,
    XDIM: int,
    YDIM: int,
    NNZ: int,
    # Acc dtype
    output_dtype: torch.dtype,
) -> torch.Tensor:

    output_dtype = TORCH_TRITON_DTYPE_MAPPER[output_dtype]

    BATCH = x.shape[0]
    UMAX = x.shape[1]

    if mode == "p":
        output = torch.empty((BATCH * UMAX, OUTDIM), dtype=x.dtype, device=x.device)
        grid = lambda META: (  # noqa: E731
            triton.cdiv(BATCH * UMAX, META["BLOCK_B"]),
            triton.cdiv(OUTDIM, META["BLOCK_DIM"]),
        )

        x = x.reshape(BATCH * UMAX, -1)
        y = y.reshape(BATCH * UMAX, -1)

    elif mode == "up":
        output = torch.empty((BATCH, UMAX, OUTDIM), dtype=x.dtype, device=x.device)
        grid = lambda META: (  # noqa: E731
            triton.cdiv(BATCH, META["BLOCK_B"]),
            triton.cdiv(UMAX, META["BLOCK_U"]),
            triton.cdiv(OUTDIM, META["BLOCK_DIM"]),
        )
    else:
        raise ValueError("Invalid mode")

    if mode == "p":
        wrap_triton(tensor_product_p_kernel)[grid](
            x_ptr=x,
            y_ptr=y,
            output_ptr=output,
            indptr_ptr=indptr,
            x_idx_ptr=x_idx,
            y_idx_ptr=y_idx,
            p_to_nnz_mapper_ptr=p_to_nnz_mapper,
            vals_ptr=vals,
            weights_ptr=weights,
            BATCH=BATCH,
            OUTDIM=OUTDIM,
            XDIM=XDIM,
            YDIM=YDIM,
            NNZ=NNZ,
            UMAX=UMAX,
            output_stride_dim=output.stride(1),
            output_stride_batch=output.stride(0),
            x_stride_dim=x.stride(1),
            x_stride_batch=x.stride(0),
            y_stride_dim=y.stride(1),
            y_stride_batch=y.stride(0),
            output_dtype=output_dtype,
        )

    elif mode == "up":
        wrap_triton(tensor_product_up_kernel)[grid](
            x_ptr=x,
            y_ptr=y,
            output_ptr=output,
            indptr_ptr=indptr,
            x_idx_ptr=x_idx,
            y_idx_ptr=y_idx,
            p_to_nnz_mapper_ptr=p_to_nnz_mapper,
            vals_ptr=vals,
            weights_ptr=weights,
            BATCH=BATCH,
            OUTDIM=OUTDIM,
            XDIM=XDIM,
            YDIM=YDIM,
            UMAX=UMAX,
            NNZ=NNZ,
            output_stride_dim=output.stride(2),
            output_stride_u=output.stride(1),
            output_stride_batch=output.stride(0),
            x_stride_dim=x.stride(2),
            x_stride_u=x.stride(1),
            x_stride_batch=x.stride(0),
            y_stride_dim=y.stride(2),
            y_stride_u=y.stride(1),
            y_stride_batch=y.stride(0),
            vals_stride_dim=vals.stride(0),
            weight_stride_dim=weights.stride(1),
            weight_stride_u=weights.stride(0),
            output_dtype=output_dtype,
        )
    else:
        raise ValueError("Invalid mode")
    return output.reshape(BATCH, UMAX, OUTDIM)


@triton_op("triton::flashallegro_forward", mutates_args={})
def _flashallegro_forward(
    input1: torch.Tensor,
    input2: torch.Tensor,
    mode: str,
    indptr_fwd: torch.Tensor,
    indptr_bwd1: torch.Tensor,
    indptr_bwd2: torch.Tensor,
    l1s_fwd: torch.Tensor,
    l2s_fwd: torch.Tensor,
    vals_fwd: torch.Tensor,
    p_to_nnz_mapper_fwd: torch.Tensor,
    ks_bwd1: torch.Tensor,
    l2s_bwd1: torch.Tensor,
    vals_bwd1: torch.Tensor,
    p_to_nnz_mapper_bwd1: torch.Tensor,
    ks_bwd2: torch.Tensor,
    l1s_bwd2: torch.Tensor,
    vals_bwd2: torch.Tensor,
    p_to_nnz_mapper_bwd2: torch.Tensor,
    weights: torch.Tensor,
    OUTDIM: int,
    XDIM: int,
    YDIM: int,
    NNZ: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    return _triton_kernel_allegro(
        mode,
        input1,
        input2,
        indptr_fwd,
        l1s_fwd,
        l2s_fwd,
        vals_fwd,
        p_to_nnz_mapper_fwd,
        weights,
        OUTDIM,
        XDIM,
        YDIM,
        NNZ,
        output_dtype,
    )


def _flashallegro_setup_context(ctx, inputs, output):
    (
        input1,
        input2,
        mode,
        indptr_fwd,
        indptr_bwd1,
        indptr_bwd2,
        l1s_fwd,
        l2s_fwd,
        vals_fwd,
        p_to_nnz_mapper_fwd,
        ks_bwd1,
        l2s_bwd1,
        vals_bwd1,
        p_to_nnz_mapper_bwd1,
        ks_bwd2,
        l1s_bwd2,
        vals_bwd2,
        p_to_nnz_mapper_bwd2,
        weights,
        OUTDIM,
        XDIM,
        YDIM,
        NNZ,
        output_dtype,
    ) = inputs

    ctx.save_for_backward(
        input1,
        input2,
        indptr_bwd1,
        indptr_bwd2,
        ks_bwd1,
        l2s_bwd1,
        vals_bwd1,
        p_to_nnz_mapper_bwd1,
        ks_bwd2,
        l1s_bwd2,
        vals_bwd2,
        p_to_nnz_mapper_bwd2,
        weights,
    )
    ctx.mode = mode
    ctx.OUTDIM = OUTDIM
    ctx.XDIM = XDIM
    ctx.YDIM = YDIM
    ctx.NNZ = NNZ
    ctx.output_dtype = output_dtype


def _flashallegro_backward(ctx, grad_output):
    (
        input1,
        input2,
        indptr_bwd1,
        indptr_bwd2,
        ks_bwd1,
        l2s_bwd1,
        vals_bwd1,
        p_to_nnz_mapper_bwd1,
        ks_bwd2,
        l1s_bwd2,
        vals_bwd2,
        p_to_nnz_mapper_bwd2,
        weights,
    ) = ctx.saved_tensors
    mode, OUTDIM, XDIM, YDIM, NNZ, output_dtype = (
        ctx.mode,
        ctx.OUTDIM,
        ctx.XDIM,
        ctx.YDIM,
        ctx.NNZ,
        ctx.output_dtype,
    )

    grad_input1 = _triton_kernel_allegro(
        mode,
        grad_output,
        input2,
        indptr_bwd1,
        ks_bwd1,
        l2s_bwd1,
        vals_bwd1,
        p_to_nnz_mapper_bwd1,
        weights,
        XDIM,
        OUTDIM,
        YDIM,
        NNZ,
        output_dtype,
    )
    grad_input2 = _triton_kernel_allegro(
        mode,
        grad_output,
        input1,
        indptr_bwd2,
        ks_bwd2,
        l1s_bwd2,
        vals_bwd2,
        p_to_nnz_mapper_bwd2,
        weights,
        YDIM,
        OUTDIM,
        XDIM,
        NNZ,
        output_dtype,
    )

    return (
        grad_input1,
        grad_input2,
        None,  # mode
        None,  # indptr_fwd
        None,  # indptr_bwd1
        None,  # indptr_bwd2
        None,  # l1s_fwd
        None,  # l2s_fwd
        None,  # vals_fwd
        None,  # p_to_nnz_mapper_fwd
        None,  # ks_bwd1
        None,  # l2s_bwd1
        None,  # vals_bwd1
        None,  # p_to_nnz_mapper_bwd1
        None,  # ks_bwd2
        None,  # l1s_bwd2
        None,  # vals_bwd2
        None,  # p_to_nnz_mapper_bwd2
        None,  # weights
        None,  # OUTDIM
        None,  # XDIM
        None,  # YDIM
        None,  # NNZ
        None,  # output_dtype
    )


_flashallegro_forward.register_autograd(
    _flashallegro_backward, setup_context=_flashallegro_setup_context
)


class TritonContracter(Contracter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # for now precomputing the number of non-zeros from ww3j
        self.NNZ = int(torch.count_nonzero(self.w3j).item())
        self.mode = "up" if self.path_channel_coupling else "p"

        (
            indptr_fwd,
            l1s_fwd,
            l2s_fwd,
            indptr_bwd1,
            ks_bwd1,
            l2s_bwd1,
            indptr_bwd2,
            ks_bwd2,
            l1s_bwd2,
            vals_fwd,
            vals_bwd1,
            vals_bwd2,
            p_to_nnz_mapper_fwd,
            p_to_nnz_mapper_bwd1,
            p_to_nnz_mapper_bwd2,
        ) = _initialize_metadata(self.w3j)

        self.register_buffer("indptr_fwd", indptr_fwd, persistent=False)
        self.register_buffer("l1s_fwd", l1s_fwd, persistent=False)
        self.register_buffer("l2s_fwd", l2s_fwd, persistent=False)

        self.register_buffer("indptr_bwd1", indptr_bwd1, persistent=False)
        self.register_buffer("ks_bwd1", ks_bwd1, persistent=False)
        self.register_buffer("l2s_bwd1", l2s_bwd1, persistent=False)

        self.register_buffer("indptr_bwd2", indptr_bwd2, persistent=False)
        self.register_buffer("ks_bwd2", ks_bwd2, persistent=False)
        self.register_buffer("l1s_bwd2", l1s_bwd2, persistent=False)

        self.register_buffer("vals_fwd", vals_fwd, persistent=False)
        self.register_buffer("vals_bwd1", vals_bwd1, persistent=False)
        self.register_buffer("vals_bwd2", vals_bwd2, persistent=False)

        self.register_buffer(
            "p_to_nnz_mapper_fwd", p_to_nnz_mapper_fwd, persistent=False
        )
        self.register_buffer(
            "p_to_nnz_mapper_bwd1", p_to_nnz_mapper_bwd1, persistent=False
        )
        self.register_buffer(
            "p_to_nnz_mapper_bwd2", p_to_nnz_mapper_bwd2, persistent=False
        )

    def _contract(self, x1, x2):
        # runtime conditions for triggering kernel code path
        if x1.is_cuda and not self.training:
            return torch.ops.triton.flashallegro_forward(
                x1,
                x2,
                self.mode,
                self.indptr_fwd,
                self.indptr_bwd1,
                self.indptr_bwd2,
                self.l1s_fwd,
                self.l2s_fwd,
                self.vals_fwd,
                self.p_to_nnz_mapper_fwd,
                self.ks_bwd1,
                self.l2s_bwd1,
                self.vals_bwd1,
                self.p_to_nnz_mapper_bwd1,
                self.ks_bwd2,
                self.l1s_bwd2,
                self.vals_bwd2,
                self.p_to_nnz_mapper_bwd2,
                self.weights,
                self.base_dim_out,
                self.base_dim1,
                self.base_dim2,
                self.NNZ,
                x1.dtype,
            )
        else:
            return super()._contract(x1, x2)
