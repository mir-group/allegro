# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Most recent change on the bottom.

## [0.4.1]

### Changed
- [Breaking] refactored scalar MLP out of Bessel embedding

### Removed
- [Breaking] `so3` argument for `parity_setting` (only `o3_full` and `o3_restricted` allowed now)

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