# Accelerations

## Custom Triton Tensor Product Kernel

In [https://arxiv.org/abs/2504.16068](https://arxiv.org/abs/2504.16068), we introduced a custom tensor product kernel that accelerates Allegro inference when incorporated during `nequip-compile`.
To use this acceleration, one must specify an additional flag `--modifiers enable_TritonContracter` during `nequip-compile`, i.e.

```bash
nequip-compile \
    path/to/ckpt_file/or/package_file \
    path/to/compiled_model.nequip.pt2 \
    --device [cpu|cuda] \
    --mode aotinductor \
    --target pair_allegro \
    --modifiers enable_TritonContracter
```

Note that `--target pair_allegro` means that one is compiling the model for use in LAMMPS.
This acceleration can also be used for other target integrations, e.g. for Python inference, one can use `--target ase` to be used with the [NequIP framework's ASE calculator](https://nequip.readthedocs.io/en/latest/api/ase.html).