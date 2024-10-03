# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Most recent change on the bottom.

## Unreleased

### Added
- Allegro readout module
- MLPs can be configured with `hidden_layer_depth` and `hidden_layer_width` as an alternative to an explicit `hidden_layer_dims` list

### Changed
- [Breaking] default `tensors_mixing_mode` = `p`
- [Breaking] initial two-body embedding and `allegro` modules resturctured
- [Breaking] MLP params use `hidden_layer` instead of `latent`

### Removed
- [Breaking] `latent_resent` (latents have access to concatenation of previous layers' latents by default)
- [Breaking] remove `env_embed_mlp`
- [Breaking] remove `tensors_mixing_mode` = `uuulin`
- [Breaking] remove `self_edge_tensor_product` as a hyperparameter (fix behavior to be `self_edge_tensor_product=True`)

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