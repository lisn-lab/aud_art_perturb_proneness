# HGF computational modelling

This folder contains the pinned Julia environment and the hierarchical Gaussian
filter models used to produce the HGF parameters and trial-level trajectories
consumed by the R reports.

## Folder guide

| Path | Purpose |
| --- | --- |
| `experiment_1/run_model.jl` | Main Experiment 1 fitting and extraction entry point |
| `experiment_2/run_model.jl` | Main Experiment 2 fitting and extraction entry point |
| `model_definition/create_agent.jl` | Shared HGF and response-model definition |
| `experiment_*/functions` | Functions called by the corresponding main script |
| `experiment_*/validation` | Checks for saved fits and state extraction |
| `experiment_2/utilities` | Recovery utilities specific to Experiment 2 |
| `utilities` | Shared post-fit recovery utility |
| `validation` | Cross-experiment validation of recovered trajectories |

Files under `functions`, `utilities`, and `validation` are not alternative main
pipelines. Start with `run_model.jl` for a complete fit.

## Prepare Julia

The environment requires Julia 1.12.x and is locked by `Project.toml` and
`Manifest.toml`. From the project root, run:

```sh
julia +1.12 analysis_scripts/computational_modelling_hgf/setup_environment.jl
```

The recorded environment was created with Julia 1.12.6, ActionModels 0.6.6,
and HierarchicalGaussianFiltering 0.6.2.

## Refit a model

The canonical fitted outputs are already included in
`model_outputs/computational_modelling_hgf`. Each main script refuses to
overwrite an existing canonical output set, so move the corresponding existing
output folder to a safe archive location before a full refit.

A full refit runs four chains with 1,000 adaptation and 1,000 retained
iterations per chain. The supplied parameter tables contain 100 fitted
participant-condition cells for Experiment 1 and 510 for Experiment 2. Wall
time depends on the available processors and was not benchmarked for this
public package.

```sh
julia +1.12 analysis_scripts/computational_modelling_hgf/experiment_1/run_model.jl
julia +1.12 analysis_scripts/computational_modelling_hgf/experiment_2/run_model.jl
```

Each run reads its public HGF input from `analysis_data/hgf_inputs` and writes
the fitted chain, diagnostics, parameter estimates, fitted input rows,
trial-level trajectories, and provenance record to its experiment folder under
`model_outputs/computational_modelling_hgf`.
