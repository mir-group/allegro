# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Most recent change on the bottom.



## [0.6.3]

## Fixed
- `enable_TritonContracter` restores weights correctly now

## [0.6.2]

### Added
- Exposed `AllegroEnergyModel` in docs

## [0.6.1]

### Removed
- `scatter_features` option (undocumented feature)

## [0.6.0]

### Changed
- bound `nequip` version >=0.8.0
- [Breaking] mechanism for using custom Triton TP kernel

## [0.5.1]

### Changed
- bound `nequip` version under 0.8.0


## [0.5.0]

[Breaking] Breaking changes to simplify the user interface for using Allegro models, including refactoring modules and removing hyperparameter options.

### Added
- Allegro model docs

### Changed
- [Breaking] refactored scalar MLP out of Bessel embedding
- [Breaking] Allegro hyperparameter defaults

### Removed
- [Breaking] `so3` argument for `parity_setting` (only `o3_full` and `o3_restricted` allowed now)
- [Breaking] turn `parity_setting` into a bool called `parity` (`true` corresponds to `o3_full`; `false` corresponds to `o3_restricted`)
- [Breaking] `scalar_embed_output_dim` is no longer a hyperparameter (and is fixed as `num_scalar_features` in the Allegro model)

## [0.4.0]

[Breaking] Major breaking changes wrt previous versions due to significant restructuring and refactoring for compatibility with `nequip` 0.7.0. Model checkpoints from previous versions will not be compatible with this version.

## [0.3.0]
### Added
- Hyperparameters `tensors_mixing_mode`, `weight_individual_irreps`, `typexbasis_mode`, `tensor_track_weight_init`, `self_edge_tensor_product`
- New Bessel basis with hyperparameters `num_bessels_per_basis`, `bessel_frequency_cutoff`, `per_edge_type_cutoff`

### Changed
- Resnet normalization scheme
- [Breaking] `latent_resnet: false` enables skip connections
- [Breaking] Default for `norm_basis_mean_shift` changed from `True` -> `False`
- [Breaking] Rename `env_embed_mul` -> `num_tensor_features`
- [Breaking] Replaced `nonscalars_include_parity` with `tensor_track_allowed_irreps`

### Fixed
- `norm_basis_mean_shift=False` is now correct

### Removed
- [Breaking] Hyperparameters `embed_initial_edge`, `cutoff_type`, `per_layer_cutoffs`, `mlp_batchnorm`, `mlp_dropout`, `per_edge_species_scale`, `linear_after_env_embed`

## [0.2.0]

## [Unreleased]

### Fixed
- Typo of `latent_resent` -> `latent_resnet`