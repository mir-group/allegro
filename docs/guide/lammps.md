# LAMMPS Integration

## Compilation

As usual with the NequIP framework, one must compile the model for use in LAMMPS. The command for compiling a TorchScript model is the same for Allegro as it is for NequIP models:
```bash
nequip-compile \
--input-path path/to/ckpt_file/or/package_file \
--output-path path/to/compiled_model.nequip.pth \
--device (cpu/cuda) \
--mode torchscript
```
The command for compiling an AOT Inductor Allegro model to be used in LAMMPS, however, requires the use of `--target pair_allegro`:
```bash
nequip-compile \
--input-path path/to/ckpt_file/or/package_file \
--output-path path/to/compiled_model.nequip.pt2 \
--device (cpu/cuda) \
--mode aotinductor \
--target pair_allegro
```

## LAMMPS Pair Style