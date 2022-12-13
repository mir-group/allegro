# Allegro

This package implements the Allegro E(3)-equivariant machine-learning interatomic potential (https://arxiv.org/abs/2204.05249).

![Allegro logo](./logo.png)

In particular, `allegro` implements the Allegro model as an **extension package** to the [NequIP package](https://github.com/mir-group/nequip).


## Installation
`allegro` requires the `nequip` package and its dependencies; please see the [NequIP installation instructions](https://github.com/mir-group/nequip#installation) for details.

Once `nequip` is installed, you can install `allegro` from source by running:
```bash
git clone --depth 1 https://github.com/mir-group/allegro.git
cd allegro
pip install .
```

## Tutorial
The best way to learn how to use Allegro is through the [Colab Tutorial](https://colab.research.google.com/drive/1yq2UwnET4loJYg_Fptt9kpklVaZvoHnq). This will run entirely on Google's cloud virtual machine, you do not need to install or run anything locally.

## Usage
Allegro models are trained, evaluated, deployed, etc. identically to NequIP models using the `nequip-*` commands. See the [NequIP README](https://github.com/mir-group/nequip#usage) for details.

The key difference between using an Allegro and NequIP model is in the options used to define the model. We provide two Allegro config files analogous to those in `nequip`:
 - [`configs/minimal.yaml`](`configs/minimal.yaml`): A minimal example of training a toy model on force data.
 - [`configs/example.yaml`](`configs/example.yaml`): Training a more realistic model on forces and energies. **Start here for real models!**

The key option that tells `nequip` to build an Allegro model is the `model_builders` option, which we set to:
```yaml
model_builders:
 - allegro.model.Allegro
 # the typical model builders from `nequip` are still used to wrap the core Allegro energy model:
 - PerSpeciesRescale
 - ForceOutput
 - RescaleEnergyEtc
```

## LAMMPS Integration

We offer a LAMMPS plugin [`pair_allegro`](https://github.com/mir-group/pair_allegro) to use Allegro models in LAMMPS simulations, including support for Kokkos acceleration and MPI and parallel simulations. Please see the [`pair_allegro`](https://github.com/mir-group/pair_allegro) repository for more details.

## References and citing

The Allegro model and the theory behind it is described in our pre-print:

> *Learning Local Equivariant Representations for Large-Scale Atomistic Dynamics* <br/>
> Albert Musaelian, Simon Batzner, Anders Johansson, Lixin Sun, Cameron J. Owen, Mordechai Kornbluth, Boris Kozinsky <br/>
> https://arxiv.org/abs/2204.05249 <br/>
> https://doi.org/10.48550/arXiv.2204.05249

The implementation of Allegro is built on NequIP [1], our framework for E(3)-equivariant interatomic potentials, and e3nn, [2] a general framework for building E(3)-equivariant neural networks. If you use this repository in your work, please consider citing the NequIP code [1] and e3nn [3] as well:

 1. https://github.com/mir-group/nequip
 2. https://e3nn.org
 3. https://doi.org/10.5281/zenodo.3724963

## Contact, questions, and contributing

If you have questions, please don't hesitate to reach out to batzner[at]g[dot]harvard[dot]edu and albym[at]seas[dot]harvard[dot]edu.

If you find a bug or have a proposal for a feature, please post it in the [Issues](https://github.com/mir-group/allegro/issues).
If you have a question, topic, or issue that isn't obviously one of those, try our [GitHub Disucssions](https://github.com/mir-group/allegro/discussions).

**If your post is related to the general NequIP framework/package, please post in the issues/discussion on [that repository](https://github.com/mir-group/nequip).** Discussions on this repository should be specific to the `allegro` package and Allegro model.

If you want to contribute to the code, please read [`CONTRIBUTING.md`](https://github.com/mir-group/nequip/blob/main/CONTRIBUTING.md) from the `nequip` repository; this repository follows all the same processes.
