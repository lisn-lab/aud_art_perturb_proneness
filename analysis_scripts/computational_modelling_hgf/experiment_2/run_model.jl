# =============================================================================
# Experiment 2: fit the binary HGF and extract trial-level trajectories
# =============================================================================
#
# Run this file from beginning to end to fit the full Experiment 2 dataset.
# In VS Code, restart the Julia REPL and use `Julia: Execute File in REPL`; the
# individual `# %%` cells depend on earlier cells and must otherwise run in order.
# From the project root, run
# `julia +1.12 analysis_scripts/computational_modelling_hgf/experiment_2/run_model.jl`
# in a prepared project environment.
#
# Free parameters per subject-condition cell
# -------------------------------------------------
# prior_posterior_weight         action-model prior-to-sensory weight
# action_noise                   action-model response noise
# xprob.volatility               tonic probability-state volatility
# xprob.initial_precision        strength of the initial probability belief
#
# Fixed parameters
# -------------------------------------------------
# xprob.initial_mean = 0         no directional initial signal expectation
# xvol.volatility = -6           fixed meta-volatility
#
# The first trial has the model's initialised probability expectation because
# Experiment 2 has no preceding conditioning phase. Initial precision influences
# early updating but is distinct from the recovered trial-level state means.
# =============================================================================


# %% 1. Activate the pinned project and load packages

script_dir = @__DIR__
project_dir = normpath(joinpath(script_dir, ".."))
project_root = normpath(joinpath(script_dir, "..", "..", ".."))
import Pkg
Pkg.activate(project_dir)

(VERSION.major == 1 && VERSION.minor == 12) || error(
    "This project requires Julia 1.12.x; the current runtime is $(VERSION).",
)

using ActionModels
using CSV, DataFrames
using Distributed
using Distributions
using HDF5, MCMCChainsStorage
using HierarchicalGaussianFiltering, LogExpFunctions
using Random
using Turing

input_filename = "experiment_2_hgf.csv"
input_path = joinpath(project_root, "analysis_data", "hgf_inputs", input_filename)
output_dir = joinpath(project_root, "model_outputs", "computational_modelling_hgf", "experiment_2")
isfile(input_path) || error("Input CSV not found: $(input_path)")
cd(script_dir)

include(joinpath(script_dir, "functions", "diagnostics.jl"))
include(joinpath(script_dir, "functions", "output_provenance.jl"))
include(joinpath(script_dir, "functions", "trajectory_replay.jl"))
include(joinpath(script_dir, "functions", "trajectory_extraction.jl"))

println("Step 1/7: project, packages, paths, and helper functions loaded")
println("  Julia: $(VERSION)")
println("  input: $(input_path)")
println("  output directory: $(output_dir)")


# %% 2. Define the run and inspect the input DataFrame

fit_seed = 666
n_adapts = 1000
n_retained = 1000
target_accept = 0.99
n_chains = 4

assert_no_existing_outputs(output_dir)

data = CSV.read(input_path, DataFrame)
disallowmissing!(data)

required_columns = ["detectprob", "resp_binary", "subject_nr", "condition"]
all(required_columns .∈ Ref(names(data))) || error(
    "Input CSV does not contain all required columns: $(required_columns)",
)
nrow(data) > 0 || error("Input CSV contains no data rows")

# Apply the eight declared zero-hit exclusions before fitting. Use the same
# response-invariance screen as Experiment 1: exclude participants who used one
# binary response on at least 95% of their 480 trials. In the current data this
# rule identifies only subject 313, who is already a declared exclusion.
declared_excluded_subjects =
    Set([1, 112, 208, 209, 303, 313, 321, 404])
response_invariance_threshold = 0.95
subject_row_counts = [nrow(subject) for subject in groupby(data, :subject_nr)]
all(subject_row_counts .== 480) || error(
    "Experiment 2 response-invariance screening expects 480 trials per participant.",
)
response_invariance = combine(
    groupby(data, :subject_nr),
    :resp_binary =>
        (responses -> max(sum(responses), length(responses) - sum(responses)) / length(responses)) =>
        :dominant_response_proportion,
)
near_invariant_subjects = Set(
    response_invariance.subject_nr[
        response_invariance.dominant_response_proportion .>=
        response_invariance_threshold
    ],
)
excluded_subjects =
    union(declared_excluded_subjects, near_invariant_subjects)
data = data[[!(subject in excluded_subjects) for subject in data.subject_nr], :]
length(unique(data.subject_nr)) == 85 || error(
    "Experiment 2 participant filtering should retain 85 participants.",
)
nrow(data) > 0 || error("Participant filtering removed all Experiment 2 rows")

fitted_input_path = joinpath(output_dir, "fitted_input_rows.csv")
CSV.write(fitted_input_path, data)

cells = groupby(data, [:subject_nr, :condition])
cell_row_counts = [nrow(cell) for cell in cells]
all(cell_row_counts .== 80) || error(
    "Experiment 2 expects 80 trials per subject-condition cell.",
)

environment_path = write_run_environment(
    output_dir = output_dir,
    project_dir = project_dir,
    intended_chains = n_chains,
)

println("Step 2/7: filtered fitted input and run environment written")
println("  declared exclusions: $(sort(collect(declared_excluded_subjects)))")
println("  near-invariant responders (>= $(response_invariance_threshold)): $(sort(collect(near_invariant_subjects)))")
println("  all excluded subjects: $(sort(collect(excluded_subjects)))")
println("  fitted participants: $(length(unique(data.subject_nr)))")
println("  data: $(nrow(data)) rows x $(ncol(data)) columns")
println("  fitted cells: $(length(cells)) subject_nr x condition groups")
println("  rows per cell: $(sort(unique(cell_row_counts)))")


# %% 3. Start local workers and construct the fitted model

external_workers = nprocs() - 1
workers_to_add = max(0, n_chains - external_workers)
added_worker_ids = workers_to_add == 0 ?
    Int[] : addprocs(workers_to_add; exeflags = "--project=$(project_dir)")

@everywhere begin
    import Pkg
    Pkg.activate($project_dir)
    cd($script_dir)
    using ActionModels, Turing, HierarchicalGaussianFiltering, LogExpFunctions
    using Distributions, CSV, DataFrames

    include(joinpath($project_dir, "model_definition", "create_agent.jl"))
    agent = create_agent("binary_3level")
    data = CSV.read($fitted_input_path, DataFrame)

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

    model = create_model(
        agent,
        priors,
        data;
        grouping_cols = [:subject_nr, :condition],
        input_cols = [:detectprob],
        action_cols = [:resp_binary],
        check_parameter_rejections = true,
    )
end

println("Step 3/7: model constructed in the main process and $(n_chains) workers")


# %% 4. Fit four NUTS chains

sampler = NUTS(
    n_adapts,
    target_accept;
    adtype = AutoReverseDiff(; compile = true),
)

println("Step 4/7: fitting $(n_chains) chains")
println("  $(n_adapts) adaptation + $(n_retained) retained iterations per chain")
flush(stdout)

posterior_chains = try
    sample(
        MersenneTwister(fit_seed),
        model,
        sampler,
        MCMCDistributed(),
        n_retained,
        n_chains;
        check_model = true,
    )
finally
    isempty(added_worker_ids) || rmprocs(added_worker_ids)
end

@assert !(posterior_chains isa Function)
println("  posterior_chains: $(size(posterior_chains)) = iterations x quantities x chains")


# %% 5. Save the chains and enforce the diagnostics gate

chain_path = joinpath(output_dir, "fitting_results.h5")
h5open(chain_path, "w") do file
    write(file, posterior_chains)
end
println("Step 5/7: complete chains written to $(chain_path)")

diagnostic_result =
    check_fit_diagnostics(
        posterior_chains;
        output_dir = output_dir,
        max_divergence_fraction = 0.005,
    )
diagnostic_result.passed || error(
    "The fit failed diagnostics. Chains and diagnostic reports were saved; parameter and trajectory tables were not written.",
)
diagnostic_result.accepted_with_warning && @warn(
    "The fit is being used under the documented small-divergence tolerance; see diagnostics.txt.",
)


# %% 6. Extract posterior-median parameter estimates

# `keys(priors)` is the runtime order used by ActionModels to construct the
# stored `parameters[parameter_index, cell_index]` chain names. Reading those
# labelled columns directly avoids reevaluating the full likelihood merely to
# recover parameter draws.
parameter_keys = collect(keys(priors))
parameter_draws = extract_parameter_draws(
    posterior_chains,
    model.args.agent_ids,
    parameter_keys,
)
estimates_df = summarise_parameter_draws(parameter_draws, data, parameter_keys)

estimates_path = joinpath(output_dir, "parameter_estimates_df.csv")
CSV.write(estimates_path, estimates_df)

println("Step 6/7: parameter estimates written to $(estimates_path)")
println("  parameter arrays: $(length(parameter_draws)) x cells x draws x chains")
println("  estimates_df: $(nrow(estimates_df)) rows x $(ncol(estimates_df)) columns")


# %% 7. Extract faithful observed-response trajectories

trajectories = extract_observed_trajectories(
    model,
    parameter_draws,
    estimates_df,
    data;
    output_dir = output_dir,
    project_dir = project_dir,
    environment_path = environment_path,
    input_path = fitted_input_path,
    chain_path = chain_path,
    estimates_path = estimates_path,
)

println("Step 7/7: trajectory extraction completed")
println("  trajectories: $(nrow(trajectories)) rows x $(ncol(trajectories)) columns")
println("Experiment 2 HGF pipeline completed successfully")
flush(stdout)
