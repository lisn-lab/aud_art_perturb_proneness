# Recompute Experiment 1 HGF diagnostics from the saved chain.

script_dir = @__DIR__
experiment_dir = normpath(joinpath(script_dir, ".."))
project_dir = normpath(joinpath(experiment_dir, ".."))
project_root = normpath(joinpath(project_dir, "..", ".."))
output_dir = joinpath(project_root, "model_outputs", "computational_modelling_hgf", "experiment_1")

import Pkg
Pkg.activate(project_dir)

using HDF5, MCMCChainsStorage, MCMCChains
include(joinpath(experiment_dir, "functions", "diagnostics.jl"))

chain_path = joinpath(output_dir, "fitting_results.h5")
isfile(chain_path) || error("No fitting_results.h5 found in $(output_dir).")

println("Loading $(chain_path)")
posterior_chains = h5open(chain_path, "r") do file
    read(file, Chains)
end

diagnostic_result = check_fit_diagnostics(
    posterior_chains;
    output_dir = output_dir,
)
diagnostic_result.passed || error(
    "The saved fit does not pass the documented diagnostic criteria.",
)
