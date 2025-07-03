# CuEquivariance Acceleration

[CuEquivariance](https://github.com/NVIDIA/cuEquivariance), developed by NVIDIA, provides GPU-accelerated tensor product operations for equivariant neural networks.
This integration accelerates Allegro models during both training and inference.

**Requirements:**

- [PyTorch](https://pytorch.org/) >= 2.6
- CUDA-compatible GPU
- [cuequivariance](https://github.com/NVIDIA/cuEquivariance) library installed: 

```bash
pip install cuequivariance-torch cuequivariance-ops-torch-cu12
```

## Training with CuEquivariance

To enable CuEquivariance acceleration during training, use the model modifier in your configuration file. This replaces standard tensor product operations with optimized GPU kernels:

```yaml
training_module:
  _target_: allegro.train.EMALightningModule
  
  # ... other training module configurations ...
  
  model:
    _target_: nequip.model.modify
    modifiers:
      - modifier: enable_CuEquivarianceContracter
    model:
      _target_: allegro.model.AllegroModel

      # ... your standard Allegro model configuration ...

      compile_mode: compile
      # ^ CuEquivariance composes with train-time compilation
```

CuEquivariance composes with [`torch.compile`](https://pytorch.org/docs/stable/generated/torch.compile.html), and can be used in conjunction with [train-time compilation](https://nequip.readthedocs.io/en/latest/guide/accelerations/pt2_compilation.html).

## Inference with CuEquivariance

For inference, you can compile your trained model with CuEquivariance acceleration enabled using `nequip-compile`:

### AOT Inductor Compilation

```{warning}
AOT Inductor compilation with CuEquivariance is **not supported** for `float64` models (i.e., `model_dtype: float64`). Use TorchScript compilation instead for double precision models.
```

```bash
nequip-compile \
    path/to/model.ckpt \
    path/to/compiled_model.nequip.pt2 \
    --device cuda \
    --mode aotinductor \
    --target ase \
    --modifiers enable_CuEquivarianceContracter
```

### TorchScript Compilation

```bash
nequip-compile \
    path/to/model.ckpt \
    path/to/compiled_model.nequip.pth \
    --device cuda \
    --mode torchscript \
    --target ase \
    --modifiers enable_CuEquivarianceContracter
```

To use the compiled model, you must import `cuequivariance_torch` before loading:

```python
import cuequivariance_torch
from nequip.ase import NequIPCalculator

# Load the compiled model
calc = NequIPCalculator.from_compiled_model(
    "path/to/compiled_model.nequip.pt2/pth",
    device="cuda"
)

# Use with ASE atoms object
atoms.calc = calc
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
```

If cuequivariance is not imported before model loading, this error will be thrown: `RuntimeError: Could not find schema for cuequivariance_ops::tensor_product_uniform_1d_jit`
