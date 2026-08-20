# =============================================================================
# Resume Experiment 2 output extraction from an already saved posterior chain
# =============================================================================

script_dir = @__DIR__
experiment_dir = normpath(joinpath(script_dir, ".."))
project_dir = normpath(joinpath(experiment_dir, ".."))
project_root = normpath(joinpath(project_dir, "..", ".."))
import Pkg
Pkg.activate(project_dir)

(VERSION.major == 1 && VERSION.minor == 12) || error(
    "This project requires Julia 1.12.x; the current runtime is $(VERSION).",
)

using ActionModels
using CSV, DataFrames
using Distributions
using HDF5, MCMCChainsStorage
using HierarchicalGaussianFiltering, LogExpFunctions
using MCMCChains
using Printf, Statistics

include(joinpath(project_dir, "model_definition", "create_agent.jl"))
include(joinpath(experiment_dir, "functions", "diagnostics.jl"))
include(joinpath(experiment_dir, "functions", "output_provenance.jl"))
include(joinpath(experiment_dir, "functions", "trajectory_replay.jl"))
include(joinpath(experiment_dir, "functions", "trajectory_extraction.jl"))

output_dir = joinpath(project_root, "model_outputs", "computational_modelling_hgf", "experiment_2")
chain_path = joinpath(output_dir, "fitting_results.h5")
input_path = joinpath(output_dir, "fitted_input_rows.csv")
environment_path = joinpath(output_dir, "run_environment.toml")
all(isfile.([chain_path, input_path, environment_path])) || error(
    "Saved chains, fitted input, or run-environment record is missing.",
)

println("Loading saved chains")
posterior_chains = h5open(chain_path, "r") do file
    read(file, Chains)
end

diagnostic_result = check_fit_diagnostics(
    posterior_chains;
    output_dir = output_dir,
    max_divergence_fraction = 0.005,
)
diagnostic_result.passed || error(
    "The saved fit exceeds the documented diagnostic tolerance.",
)
diagnostic_result.accepted_with_warning && @warn(
    "Proceeding under the documented small-divergence tolerance.",
)

data = CSV.read(input_path, DataFrame)
disallowmissing!(data)
cells = collect(groupby(data, [:subject_nr, :condition]))
length(cells) == 510 || error("Expected 510 fitted participant-condition cells.")
all(nrow(cell) == 80 for cell in cells) || error("Expected 80 rows per fitted cell.")

priors = Dict(
    "prior_posterior_weight" =>
        truncated(Normal(0.72646851, 1), lower = 0),
    "action_noise" =>
        truncated(Normal(0.29350739, 1), lower = 0),
    ("xprob", "volatility") =>
        truncated(Normal(-5.1682685, 1), upper = -0.5),
    ("xprob", "initial_precision") =>
        LogNormal(0, 1),
)
parameter_keys = collect(keys(priors))
agent_ids = collect(1:length(cells))
parameter_draws = extract_parameter_draws(
    posterior_chains,
    agent_ids,
    parameter_keys,
)
estimates_df = summarise_parameter_draws(
    parameter_draws,
    data,
    parameter_keys,
)

estimates_path = joinpath(output_dir, "parameter_estimates_df.csv")
CSV.write(estimates_path, estimates_df)
println("Parameter estimates overwritten at $(estimates_path)")

divergences = vec(divergence_mask(posterior_chains))
sensitivity = DataFrame(
    parameter = String[],
    max_divergence_filtered_shift_sd = Float64[],
    max_leave_one_chain_shift_sd = Float64[],
)
for parameter_name in sort(collect(keys(parameter_draws)); by = string)
    draws = parameter_draws[parameter_name]
    filtered_shifts = Float64[]
    leave_one_chain_shifts = Float64[]
    for cell_index in axes(draws, 1)
        cell_draws = @view draws[cell_index, :, :]
        all_values = vec(cell_draws)
        all_median = median(all_values)
        posterior_sd = std(all_values)
        filtered_median = median(all_values[.!divergences])
        leave_one_chain_medians = [
            median(vec(cell_draws[:, setdiff(axes(draws, 3), chain)]))
            for chain in axes(draws, 3)
        ]
        push!(
            filtered_shifts,
            abs(filtered_median - all_median) / posterior_sd,
        )
        push!(
            leave_one_chain_shifts,
            maximum(abs.(leave_one_chain_medians .- all_median)) / posterior_sd,
        )
    end
    push!(
        sensitivity,
        (
            string(parameter_name),
            maximum(filtered_shifts),
            maximum(leave_one_chain_shifts),
        ),
    )
end

sensitivity_path = joinpath(output_dir, "diagnostic_sensitivity.csv")
CSV.write(sensitivity_path, sensitivity)
max_filtered_shift =
    maximum(sensitivity.max_divergence_filtered_shift_sd)
max_leave_one_chain_shift =
    maximum(sensitivity.max_leave_one_chain_shift_sd)

supplement_path = joinpath(output_dir, "fitting_diagnostics_supplement.txt")
open(supplement_path, "w") do io
    println(io, "Experiment 2 HGF fitting diagnostics")
    println(io)
    println(io, "The model estimated four parameters independently for each of 510 participant-condition cells (2,040 fitted parameters in total). Four NUTS chains each retained 1,000 post-warm-up draws after 1,000 adaptation iterations. The target acceptance probability was 0.99.")
    println(io)
    @printf(io, "All fitted parameters met the convergence criteria: the maximum rank-normalised R-hat was %.4f, the minimum bulk effective sample size was %.1f, and the minimum tail effective sample size was %.1f. Energy mixing was satisfactory in every chain (E-BFMI range %.3f-%.3f), and the maximum observed tree depth was %.0f.\n",
            diagnostic_result.max_rhat,
            diagnostic_result.min_bulk_ess,
            diagnostic_result.min_tail_ess,
            minimum(diagnostic_result.ebfmi),
            maximum(diagnostic_result.ebfmi),
            diagnostic_result.max_tree_depth)
    println(io)
    @printf(io, "There were %d divergent transitions among %d retained transitions (%.2f%%; chain counts: %s). The fit was therefore accepted with a warning under a post-fit tolerance of at most %.2f%% divergent transitions, rather than classified as a divergence-free fit. During these transitions, the maximum Hamiltonian energy error exceeded 1,000 or became infinite. The model deliberately assigns zero probability to HGF proposals that produce an invalid negative state precision; the divergences are consistent with occasional Hamiltonian trajectories entering this rejection region in the 2,040-dimensional joint parameterisation. Inspection did not identify a single participant or condition responsible for the divergences.\n",
            diagnostic_result.divergences,
            diagnostic_result.total_transitions,
            100 * diagnostic_result.divergence_fraction,
            join(diagnostic_result.divergences_by_chain, ", "),
            100 * diagnostic_result.max_divergence_fraction)
    println(io)
    @printf(io, "Sensitivity checks were performed on the participant-condition posterior medians used in subsequent analyses. Removing every iteration marked as divergent changed no median by more than %.3f posterior standard deviations. Recomputing the medians after omitting each complete chain in turn changed no median by more than %.3f posterior standard deviations. All retained draws, including the endpoints of divergence-marked transitions, were used in the reported estimates; removing marked draws was used only as a sensitivity analysis. These checks indicate that the small number of divergences did not materially affect the extracted point estimates, although the fit is reported transparently as accepted with a divergence warning.\n",
            max_filtered_shift,
            max_leave_one_chain_shift)
    println(io)
    println(io, "Complete parameter-level diagnostics are provided in diagnostics_table.csv, and the parameter-specific sensitivity results are provided in diagnostic_sensitivity.csv.")
end
println("Supplement diagnostics written to $(supplement_path)")

model_proxy = (args = (agent_ids = agent_ids,),)
trajectories = extract_observed_trajectories(
    model_proxy,
    parameter_draws,
    estimates_df,
    data;
    output_dir = output_dir,
    project_dir = project_dir,
    environment_path = environment_path,
    input_path = input_path,
    chain_path = chain_path,
    estimates_path = estimates_path,
)

println("Recovered $(nrow(trajectories)) trajectory rows from the saved fit")
