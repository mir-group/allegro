# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Most recent change on the bottom.

## [0.3.0]
### Added
- `tensors_mixing_mode`

### Changed
- Resnet normalization scheme
- [Breaking] Default `norm_basis_mean_shift` `True` -> `False`
- [Breaking] Rename `env_embed_mul` -> `num_tensor_features`
- [Breaking] Replaced `nonscalars_include_parity` with `tensor_track_allowed_irreps`

### Fixed
- `norm_basis_mean_shift=False`

### Removed
- [Breaking] Hyperparameters `embed_initial_edge`, `cutoff_type`, `per_layer_cutoffs`, `mlp_batchnorm`, `mlp_dropout`, `per_edge_species_scale`, `linear_after_env_embed`

## [Unreleased]

### Fixed
- Typo of `latent_resent` -> `latent_resnet`