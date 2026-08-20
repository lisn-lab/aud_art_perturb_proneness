# =============================================================================
# Experiment 1: fit the binary HGF and extract trial-level trajectories
# =============================================================================
#
# Run this file from beginning to end to fit the full Experiment 1 dataset.
# In VS Code, restart the Julia REPL and use `Julia: Execute File in REPL`; the individual `# %%` cells depend on earlier cells and must otherwise be run in order.
# From the project root, run
# `julia +1.12 analysis_scripts/computational_modelling_hgf/experiment_1/run_model.jl`.
# The script activates the pinned project environment itself.
#
# Main objects and their R analogies
# ----------------------------------
# data              DataFrame: rows x columns, like an R data.frame or tibble
# model             Turing/DynamicPPL model: model definition plus data
# posterior_chains  Chains: iteration x sampled quantity x chain
# agent_parameters  AxisArray: cell x parameter x draw x chain
# estimates_df      DataFrame: one posterior-median parameter row per cell
# trajectories      DataFrame: one posterior-median state row per trial
#
# The helper files define functions but do not run separate analyses. This file
# remains the only entry point for fitting, diagnostics, and state extraction.
# =============================================================================


# %% 1. Load packages and helper functions

# `Pkg.activate` selects the Project.toml and Manifest.toml two folders above this file before any analysis package is loaded.
script_dir = @__DIR__
project_dir = normpath(joinpath(script_dir, ".."))
project_root = normpath(joinpath(script_dir, "..", "..", ".."))
import Pkg
Pkg.activate(project_dir)

# Install any package sources or binary artifacts missing from this machine at the exact versions recorded in Manifest.toml.
Pkg.instantiate()

# `using` makes a package's exported functions available in this Julia session.
using ActionModels
using CSV, DataFrames
using Distributed
using Distributions
using HDF5, MCMCChainsStorage
using HierarchicalGaussianFiltering, LogExpFunctions
using Random
using Turing

# `@__DIR__` is replaced by the directory containing this script. `normpath` resolves `..` without depending on the folder from which Julia was launched.
input_filename = "experiment_1_hgf_with_conditioning.csv"
input_path = joinpath(project_root, "analysis_data", "hgf_inputs", input_filename)
output_dir = joinpath(project_root, "model_outputs", "computational_modelling_hgf", "experiment_1")
isfile(input_path) || error("Input CSV not found: $(input_path)")
cd(script_dir)

# `include(path)` evaluates another Julia file in this session. These files only
# define named functions. In VS Code, F12 or Cmd-click on a function call should
# navigate to its definition after the Julia extension indexes the workspace.
include(joinpath(script_dir, "functions", "diagnostics.jl"))
include(joinpath(script_dir, "functions", "output_provenance.jl"))
include(joinpath(script_dir, "functions", "trajectory_replay.jl"))
include(joinpath(script_dir, "functions", "trajectory_extraction.jl"))

println("Step 1/7: packages, paths, and helper functions loaded")
println("  input: $(input_path)")
println("  output directory: $(output_dir)")


# %% 2. Define the run and inspect the input DataFrame

# These values completely specify the retained MCMC sample. Adaptation is NUTS
# warm-up used to tune the sampler and is discarded. Retained iterations are the
# posterior draws kept after warm-up. The target acceptance of 0.99 requests
# conservative NUTS steps to reduce divergent transitions.
fit_seed = 666
n_adapts = 1000
n_retained = 1000
target_accept = 0.99
n_chains = 4

# Refuse to mix files from separate runs. Archive an existing complete or partial
# output set before starting another fit.
assert_no_existing_outputs(output_dir)

# `CSV.read(path, DataFrame)` is comparable to `readr::read_csv(path)` in R. `fitted_input_rows.csv` preserves the exact DataFrame subsequently loaded by
# every fitting worker and retained with the fitted outputs.
data = CSV.read(input_path, DataFrame)
disallowmissing!(data)

required_columns = ["detectprob", "binary_resp", "SubjID", "condition", "training"]
all(required_columns .∈ Ref(names(data))) || error("Input CSV does not contain all required columns: $(required_columns)")
nrow(data) > 0 || error("Input CSV contains no complete data rows")

# Apply participant-level quality exclusions before fitting. Subjects 9, 20,
# and 21 failed the imagery-vividness criterion, subject 22 failed the hit-rate
# criterion, and subject 52 exceeded the rapid-response criterion. The
# additional response-invariance rule excludes participants who used the same
# binary response on at least 95% of their 200 non-conditioning task trials.
declared_excluded_subjects = Set([9, 20, 21, 22, 52])
response_invariance_threshold = 0.95
task_data = data[data.training .== 0, :]
task_row_counts = [nrow(subject) for subject in groupby(task_data, :SubjID)]
all(task_row_counts .== 200) || error(
    "Experiment 1 response-invariance screening expects 200 task trials per participant.",
)
response_invariance = combine(
    groupby(task_data, :SubjID),
    :binary_resp =>
        (responses -> max(sum(responses), length(responses) - sum(responses)) / length(responses)) =>
        :dominant_response_proportion,
)
near_invariant_subjects = Set(
    response_invariance.SubjID[
        response_invariance.dominant_response_proportion .>=
        response_invariance_threshold
    ],
)
excluded_subjects =
    union(declared_excluded_subjects, near_invariant_subjects)
data = data[[!(subject in excluded_subjects) for subject in data.SubjID], :]
length(unique(data.SubjID)) == 50 || error(
    "Experiment 1 participant filtering should retain 50 participants.",
)
nrow(data) > 0 || error("Participant filtering removed all Experiment 1 rows")

fitted_input_path = joinpath(output_dir, "fitted_input_rows.csv")
CSV.write(fitted_input_path, data)
cells = groupby(data, [:SubjID, :condition])
cell_row_counts = [nrow(cell) for cell in cells]  # A GroupedDataFrame can be iterated over explicitly but cannot be used with dot broadcasting.

println("Step 2/7: filtered fitted input written to $(fitted_input_path)")
println("  declared exclusions: $(sort(collect(declared_excluded_subjects)))")
println("  near-invariant responders (>= $(response_invariance_threshold)): $(sort(collect(near_invariant_subjects)))")
println("  all excluded subjects: $(sort(collect(excluded_subjects)))")
println("  fitted participants: $(length(unique(data.SubjID)))")
println("  data object: DataFrame with $(nrow(data)) rows x $(ncol(data)) columns")
println("  fitted cells: $(length(cells)) SubjID x condition groups")
println("  rows per cell: $(sort(unique(cell_row_counts)))")


# %% 3. Start local workers and construct the fitted model

# The main Julia process coordinates the analysis. One additional local process
# per chain performs sampling. These workers are operating-system processes on
# this computer, not remote servers.
external_workers = nprocs() - 1
workers_to_add = max(0, n_chains - external_workers)
added_worker_ids = workers_to_add == 0 ? Int[] : addprocs(workers_to_add; exeflags = "--project=$(project_dir)")

# `@everywhere begin ... end` evaluates the enclosed statements in the main
# process and in every worker. Each process has separate memory, so each needs
# its own packages, model definition, input data, and model object.
@everywhere begin
    cd($script_dir)
    using ActionModels, Turing, HierarchicalGaussianFiltering, LogExpFunctions
    using Distributions, CSV, DataFrames

    # This shared file defines create_agent() for both experiments.
    include(joinpath($project_dir, "model_definition", "create_agent.jl"))

    # `agent` is a template object containing the HGF and action model. Creating it does not fit anything or process any trials.
    agent = create_agent("binary_3level")
    # Every worker reloads the exact cleaned table saved in Step 2. Thus the file retained with the outputs is the data object used by the likelihood.
    data = CSV.read($fitted_input_path, DataFrame)

    # A Dict is a key-value object, comparable to a named list in R. These are
    # the three free parameters fitted independently in each cell. Parameters not
    # listed here retain their create_agent.jl defaults.
    priors = Dict(
        "prior_posterior_weight" => truncated(Normal(0.72646851, 1), lower = 0),
        "action_noise" => truncated(Normal(0.29350739, 1), lower = 0),
        ("xprob", "volatility") => truncated(Normal(-5.1682685, 1), upper = -0.5),
    )

    model = create_model(
        agent,
        priors,
        data;
        grouping_cols = [:SubjID, :condition],
        input_cols = [:detectprob],
        action_cols = [:binary_resp],

        # Some parameter proposals make the custom action probability invalid.
        # This converts the model's deliberate RejectParameters exception into
        # probability zero, so NUTS rejects that proposal. Other errors propagate.
        check_parameter_rejections = true,
    )
end

println("Step 3/7: model constructed in the main process and $(n_chains) workers")


# %% 4. Fit four NUTS chains

# NUTS is Turing's No-U-Turn Hamiltonian Monte Carlo sampler. ReverseDiff
# calculates the gradients NUTS needs. `compile=true` spends setup time compiling
# a reusable derivative tape so repeated model evaluations are faster.
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
        # MersenneTwister is a deterministic pseudo-random-number generator.
        # Supplying a seed fixes its starting random stream in this environment.
        MersenneTwister(fit_seed),
        model,
        sampler,

        # MCMCDistributed assigns the independent chains to local workers.
        MCMCDistributed(),
        n_retained,
        n_chains;

        # Turing performs its generic preflight model check before sampling.
        check_model = true,
    )
finally
    # Remove only the worker processes added by this run. If you want to rerun
    # the fitting cell in the same REPL, rerun Step 3 first to recreate workers.
    isempty(added_worker_ids) || rmprocs(added_worker_ids)
end

println("  posterior_chains object: $(size(posterior_chains)) = iterations x quantities x chains")


# %% 5. Save the complete chains and enforce the diagnostics gate

# HDF5 is a binary data format. MCMCChainsStorage teaches HDF5 how to preserve a
# labelled Chains object, including its NUTS internals, rather than saving only a
# plain numeric matrix.
chain_path = joinpath(output_dir, "fitting_results.h5")
h5open(chain_path, "w") do file
    write(file, posterior_chains)
end
println("Step 5/7: complete chains written to $(chain_path)")

# This writes Rhat, bulk ESS, and divergence reports. Extraction stops if any
# Rhat exceeds 1.01, any bulk ESS is below 400, or any divergence occurred.
diagnostic_result = check_fit_diagnostics(posterior_chains; output_dir = output_dir)
diagnostic_result.passed || error(
    "The fit failed diagnostics. Chains and diagnostic reports were saved; parameter and trajectory tables were not written.",
)


# %% 6. Extract posterior-median parameter estimates

# `extract_quantities` returns a labelled four-dimensional AxisArray:
# cell x parameter x retained draw x chain. `get_estimates` takes the median over
# its draw and chain dimensions, producing one DataFrame row per fitted cell.
agent_parameters = extract_quantities(model, posterior_chains)
estimates_df = get_estimates(agent_parameters)

# ActionModels reconstructs grouping identifiers as text. Convert them to the
# integer types used by the input DataFrame before checking cell alignment.
estimates_df.SubjID = [
    parse(eltype(data.SubjID), string(value)) for value in estimates_df.SubjID
]
estimates_df.condition = [
    parse(eltype(data.condition), string(value)) for value in estimates_df.condition
]

estimates_path = joinpath(output_dir, "parameter_estimates_df.csv")
CSV.write(estimates_path, estimates_df)

println("Step 6/7: parameter estimates written to $(estimates_path)")
println("  agent_parameters: $(size(agent_parameters)) = cells x parameters x draws x chains")
println("  estimates_df: $(nrow(estimates_df)) rows x $(ncol(estimates_df)) columns")


# %% 7. Extract observed-response trajectories from the same posterior draws

# This function loops over every fitted cell and posterior draw. It installs one
# parameter draw, resets the agent, and replays the actual detectprob and
# binary_resp sequences in fitted-likelihood order. It never samples responses.
trajectories = extract_observed_trajectories(
    model,
    agent_parameters,
    estimates_df,
    data;
    output_dir = output_dir,
    input_path = fitted_input_path,
    chain_path = chain_path,
    estimates_path = estimates_path,
)

println("Step 7/7: trajectory extraction completed")
println("  trajectories: $(nrow(trajectories)) rows x $(ncol(trajectories)) columns")
println("Experiment 1 HGF pipeline completed successfully")
flush(stdout)
