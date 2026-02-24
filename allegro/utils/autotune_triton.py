#!/usr/bin/env python
# This file is a part of the `allegro` package. Please see LICENSE and README at the root for information on using it.

"""Minimal autotuning rig for TritonContracter."""

import torch
from ase.build import bulk, make_supercell
from e3nn import o3

from nequip.data.transforms import NeighborListTransform
from nequip.data.ase import from_ase
from nequip.data import AtomicDataDict
from nequip.nn.utils import with_edge_vectors_
from nequip.utils import torch_default_dtype

from allegro.nn._strided import Contracter
from allegro.nn._strided._flashallegro import TritonContracter

_CUEQ_AVAILABLE = False
try:
    from allegro.nn._strided._cueq_contracter import CuEquivarianceContracter
    import cuequivariance_torch  # noqa: F401

    _CUEQ_AVAILABLE = True
except ImportError:
    CuEquivarianceContracter = None


def create_nacl_supercell(supercell_size=10):
    """Create a large NaCl supercell for realistic benchmarking.

    Args:
        supercell_size: Size of supercell in each dimension

    Returns:
        AtomicDataDict with edge information
    """
    atoms = bulk(name="NaCl", crystalstructure="rocksalt", a=5.64)
    atoms = make_supercell(
        atoms,
        [[supercell_size, 0, 0], [0, supercell_size, 0], [0, 0, supercell_size]],
    )
    atoms.rattle(stdev=0.1)

    r_max = 6.0
    data = NeighborListTransform(r_max=r_max)(from_ase(atoms))
    data = with_edge_vectors_(data, with_lengths=True)

    return data


# hardcoded challenging configurations for benchmarking
CHALLENGING_CONFIGS = [
    {
        "name": "mul64-coupled",
        "irreps_in1": o3.Irreps("1x0e+1x1o+1x2e+1x3o"),
        "irreps_in2": o3.Irreps("1x0e+1x1o+1x2e+1x3o"),
        "irreps_out": o3.Irreps("1x0e+1x1o+1x2e+1x3o"),
        "mul": 64,
        "path_channel_coupling": True,
    },
    {
        "name": "mul64-uncoupled",
        "irreps_in1": o3.Irreps("1x0e+1x1o+1x2e+1x3o"),
        "irreps_in2": o3.Irreps("1x0e+1x1o+1x2e+1x3o"),
        "irreps_out": o3.Irreps("1x0e+1x1o+1x2e+1x3o"),
        "mul": 64,
        "path_channel_coupling": False,
    },
    {
        "name": "mul128-coupled",
        "irreps_in1": o3.Irreps("1x0e+1x1o+1x2e+1x3o"),
        "irreps_in2": o3.Irreps("1x0e+1x1o+1x2e+1x3o"),
        "irreps_out": o3.Irreps("1x0e+1x1o+1x2e+1x3o"),
        "mul": 128,
        "path_channel_coupling": True,
    },
    {
        "name": "mul128-uncoupled",
        "irreps_in1": o3.Irreps("1x0e+1x1o+1x2e+1x3o"),
        "irreps_in2": o3.Irreps("1x0e+1x1o+1x2e+1x3o"),
        "irreps_out": o3.Irreps("1x0e+1x1o+1x2e+1x3o"),
        "mul": 128,
        "path_channel_coupling": False,
    },
]


def benchmark_forward(contracter, input1, input2, scatter_idxs, warmup=3, n_iter=10):
    """Benchmark forward pass."""
    # warmup
    for _ in range(warmup):
        _ = contracter(input1, input2, scatter_idxs)
        torch.cuda.synchronize()

    # benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(n_iter):
        _ = contracter(input1, input2, scatter_idxs)
    end_event.record()

    torch.cuda.synchronize()
    total_time_ms = start_event.elapsed_time(end_event)

    return total_time_ms / n_iter


def benchmark_backward(contracter, input1, input2, scatter_idxs, warmup=3, n_iter=10):
    """Benchmark full forward+backward pass.

    Returns:
        time in ms for forward+backward
    """
    # warmup
    for _ in range(warmup):
        input1.grad = None
        input2.grad = None
        out = contracter(input1, input2, scatter_idxs)
        grad_out = torch.randn_like(out)
        out.backward(grad_out)
        torch.cuda.synchronize()

    # benchmark full forward+backward
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(n_iter):
        input1.grad = None
        input2.grad = None
        out = contracter(input1, input2, scatter_idxs)
        grad_out = torch.randn_like(out)
        out.backward(grad_out)
    end_event.record()

    torch.cuda.synchronize()
    time_fwd_bwd_ms = start_event.elapsed_time(end_event) / n_iter

    return time_fwd_bwd_ms


def autotune(
    supercell_size=10,
    dtype=torch.float32,
    device="cuda",
    with_cueq=False,
):
    """Run autotuning for TritonContracter.

    Args:
        supercell_size: Size of NaCl supercell
        dtype: Data type for tensors
        device: Device to run on
        with_cueq: If True, also benchmark CuEquivariance (default: False)
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for autotuning")

    print("=" * 80)
    print("TritonContracter Autotuning")
    print("=" * 80)

    print(f"\nBenchmarking {len(CHALLENGING_CONFIGS)} configuration(s):")
    for i, cfg in enumerate(CHALLENGING_CONFIGS, 1):
        print(f"  {i}. {cfg['name']}")
        print(
            f"     irreps: {cfg['irreps_in1']} x {cfg['irreps_in2']} -> {cfg['irreps_out']}"
        )
        print(
            f"     mul: {cfg['mul']}, path_channel_coupling: {cfg['path_channel_coupling']}"
        )

    # create test data
    print(
        f"\nCreating NaCl supercell ({supercell_size}x{supercell_size}x{supercell_size})..."
    )
    with torch_default_dtype(dtype):
        data = create_nacl_supercell(supercell_size)
        data = AtomicDataDict.to_(data, device)

    num_nodes = AtomicDataDict.num_nodes(data)
    num_edges = AtomicDataDict.num_edges(data)
    scatter_idxs = data[AtomicDataDict.EDGE_INDEX_KEY][1]

    print(f"  num_nodes: {num_nodes}")
    print(f"  num_edges: {num_edges}")

    # benchmark each model configuration
    all_results = {}
    for model_config in CHALLENGING_CONFIGS:
        config_name = model_config["name"]
        print(f"\n{'=' * 80}")
        print(f"Configuration: {config_name}")
        print(f"{'=' * 80}")

        # create inputs
        irreps_in1 = model_config["irreps_in1"]
        irreps_in2 = model_config["irreps_in2"]
        mul = model_config["mul"]

        # input1 is edge-indexed, input2 is node-indexed
        input1 = irreps_in1.randn(num_edges, mul, -1, dtype=dtype, device=device)
        input2 = irreps_in2.randn(num_nodes, mul, -1, dtype=dtype, device=device)
        input1.requires_grad_(True)
        input2.requires_grad_(True)

        # create contracter - need base contracter for instructions
        contracter_config = {k: v for k, v in model_config.items() if k != "name"}

        # create base contracter to get instructions
        c_base = Contracter(**contracter_config).to(device=device)

        # create triton contracter with instructions from base
        contracter = (
            TritonContracter(**contracter_config, instructions=c_base.instructions)
            .to(device)
            .eval()
        )

        # copy weights from base
        with torch.no_grad():
            contracter.weights.copy_(c_base.weights)

        # benchmark triton
        try:
            print("Triton:")
            fwd_time_triton = benchmark_forward(
                contracter,
                input1,
                input2,
                scatter_idxs,
                warmup=5,
                n_iter=20,
            )

            fwd_bwd_time_triton = benchmark_backward(
                contracter,
                input1,
                input2,
                scatter_idxs,
                warmup=5,
                n_iter=20,
            )

            print(f"  Forward:          {fwd_time_triton:.3f} ms")
            print(f"  Forward+Backward: {fwd_bwd_time_triton:.3f} ms")

            all_results[config_name] = {
                "triton_forward": fwd_time_triton,
                "triton_fwd+bwd": fwd_bwd_time_triton,
            }
        except Exception as e:
            print(f"  FAILED: {e}")
            all_results[config_name] = {
                "triton_forward": None,
                "triton_fwd+bwd": None,
            }

        # benchmark cuequivariance if requested and available
        if with_cueq and _CUEQ_AVAILABLE:
            try:
                print("CuEquivariance:")
                cueq_contracter = (
                    CuEquivarianceContracter(
                        **contracter_config, instructions=c_base.instructions
                    )
                    .to(device)
                    .eval()
                )
                with torch.no_grad():
                    cueq_contracter.weights.copy_(c_base.weights)

                fwd_time_cueq = benchmark_forward(
                    cueq_contracter,
                    input1,
                    input2,
                    scatter_idxs,
                    warmup=5,
                    n_iter=20,
                )

                fwd_bwd_time_cueq = benchmark_backward(
                    cueq_contracter,
                    input1,
                    input2,
                    scatter_idxs,
                    warmup=5,
                    n_iter=20,
                )

                print(f"  Forward:          {fwd_time_cueq:.3f} ms")
                print(f"  Forward+Backward: {fwd_bwd_time_cueq:.3f} ms")

                all_results[config_name]["cueq_forward"] = fwd_time_cueq
                all_results[config_name]["cueq_fwd+bwd"] = fwd_bwd_time_cueq
            except Exception as e:
                print(f"  FAILED: {e}")
                all_results[config_name]["cueq_forward"] = None
                all_results[config_name]["cueq_fwd+bwd"] = None

    # report summary
    print(f"\n{'=' * 80}")
    print("Summary")
    print(f"{'=' * 80}")
    if with_cueq and _CUEQ_AVAILABLE:
        print(
            f"{'Config':<25} {'Triton Fwd':>12} {'Triton F+B':>12} {'CuEq Fwd':>12} {'CuEq F+B':>12}"
        )
        print("-" * 80)
        for config_name, times in all_results.items():
            triton_fwd = (
                f"{times['triton_forward']:.3f}"
                if times["triton_forward"] is not None
                else "FAILED"
            )
            triton_fb = (
                f"{times['triton_fwd+bwd']:.3f}"
                if times["triton_fwd+bwd"] is not None
                else "FAILED"
            )
            cueq_fwd = (
                f"{times['cueq_forward']:.3f}"
                if times.get("cueq_forward") is not None
                else "FAILED"
            )
            cueq_fb = (
                f"{times['cueq_fwd+bwd']:.3f}"
                if times.get("cueq_fwd+bwd") is not None
                else "FAILED"
            )
            print(
                f"{config_name:<25} "
                f"{triton_fwd:>12}  "
                f"{triton_fb:>12}  "
                f"{cueq_fwd:>12}  "
                f"{cueq_fb:>12}"
            )
    else:
        print(f"{'Config':<30} {'Triton Fwd':>12} {'Triton F+B':>12}")
        print("-" * 80)
        for config_name, times in all_results.items():
            triton_fwd = (
                f"{times['triton_forward']:.3f}"
                if times["triton_forward"] is not None
                else "FAILED"
            )
            triton_fb = (
                f"{times['triton_fwd+bwd']:.3f}"
                if times["triton_fwd+bwd"] is not None
                else "FAILED"
            )
            print(f"{config_name:<30} {triton_fwd:>12}  {triton_fb:>12}")
    print(f"{'=' * 80}")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Autotune TritonContracter")
    parser.add_argument(
        "--supercell-size",
        type=int,
        default=10,
        help="size of NaCl supercell (default: 10)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float64"],
        help="data type (default: float32)",
    )
    parser.add_argument(
        "--with-cueq",
        action="store_true",
        help="also benchmark CuEquivariance (default: False)",
    )

    args = parser.parse_args()

    dtype = torch.float32 if args.dtype == "float32" else torch.float64

    autotune(
        supercell_size=args.supercell_size,
        dtype=dtype,
        with_cueq=args.with_cueq,
    )
