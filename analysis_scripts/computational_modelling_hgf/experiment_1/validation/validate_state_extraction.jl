# =============================================================================
# One-subject validation of faithful versus incorrect state extraction
# =============================================================================
#
# This script does not fit an MCMC model and does not write canonical outputs.
# It tests the exact replay_observed_responses!() function used by run_model.jl.
#
# The test demonstrates four facts:
#   1. Faithful replay stores every observed response exactly.
#   2. Faithful replay is unchanged when the RNG seed changes because it never
#      samples responses.
#   3. Incorrect give_inputs!() replay can repeat under the same seed, so merely
#      obtaining identical output twice does not prove correctness.
#   4. Incorrect give_inputs!() replay changes under a different seed and its
#      sampled response history differs from the participant's history.
# =============================================================================


# %% 1. Load packages, model definition, and canonical replay function

using ActionModels, CSV, DataFrames, Random
using HierarchicalGaussianFiltering, LogExpFunctions

script_dir = @__DIR__
experiment_dir = normpath(joinpath(script_dir, ".."))
hgf_root = normpath(joinpath(experiment_dir, ".."))
project_root = normpath(joinpath(hgf_root, "..", ".."))
input_path = joinpath(project_root, "analysis_data", "hgf_inputs", "experiment_1_hgf_with_conditioning.csv")
include(joinpath(hgf_root, "model_definition", "create_agent.jl"))
include(joinpath(experiment_dir, "functions", "trajectory_replay.jl"))


# %% 2. Select one subject and define fixed parameters

# Change this integer to inspect another participant. The first participant has
# one weak-prior and one strong-prior condition, so both trial lengths are tested.
validation_subject = 1

data = CSV.read(input_path, DataFrame)
subject_data = data[data.SubjID .== validation_subject, :]
nrow(subject_data) > 0 || error("Subject $(validation_subject) was not found")
subject_cells = collect(groupby(subject_data, [:SubjID, :condition]))

# Fixed values are sufficient because this test concerns response replay, not
# parameter estimation. The canonical fit repeats the same operation for every
# posterior draw with that draw's fitted values installed.
validation_parameters = Dict(
    "prior_posterior_weight" => 0.72646851,
    "action_noise" => 0.29350739,
    ("xprob", "volatility") => -5.1682685,
)

println("Validating subject $(validation_subject) across $(length(subject_cells)) conditions")


# %% 3. Define small inspection helpers

"""
    new_validation_agent() -> Agent

Create a fresh Agent, install the fixed validation parameters, and enable state
history. A fresh object prevents one run's history leaking into another run.
"""
function new_validation_agent()
    agent = create_agent("binary_3level")
    set_parameters!(agent, validation_parameters)
    set_save_history!(agent, true)
    return agent
end


"""
    state_snapshot(agent, n_trials) -> DataFrame

Convert one Agent history into a trial-level DataFrame for exact comparisons.
This is comparable to constructing a small R tibble from several model vectors.
"""
function state_snapshot(agent, n_trials::Int)
    return DataFrame(
        response = Int.(get_history(agent, "action")[2:end]),
        belief = Float64.(get_history(agent, "belief")[2:end]),
        xprob_posterior_mean = Float64.(
            get_history(agent, ("xprob", "posterior_mean"))[1:n_trials],
        ),
        xvol_posterior_mean = Float64.(
            get_history(agent, ("xvol", "posterior_mean"))[1:n_trials],
        ),
    )
end


"""
    faithful_snapshot(inputs, observed_responses; seed) -> DataFrame

Run the same observed-response replay function used by run_model.jl. The seed is
changed deliberately to demonstrate that this path does not consume randomness.
"""
function faithful_snapshot(inputs, observed_responses; seed::Int)
    Random.seed!(seed)
    agent = new_validation_agent()
    replay_observed_responses!(agent, inputs, observed_responses)
    return state_snapshot(agent, length(inputs))
end


"""
    incorrect_sampled_snapshot(inputs; seed) -> DataFrame

Negative control only. `give_inputs!()` calls `rand()` on each response
distribution and feeds those sampled responses back into the HGF. This is the
incorrect method previously used for fitted-state extraction.
"""
function incorrect_sampled_snapshot(inputs; seed::Int)
    Random.seed!(seed)
    agent = new_validation_agent()
    reset!(agent)
    give_inputs!(agent, inputs)
    return state_snapshot(agent, length(inputs))
end


# %% 4. Run faithful and deliberately incorrect extraction for each condition

validation_summary = DataFrame(
    condition = Int[],
    rows = Int[],
    faithful_response_mismatches = Int[],
    faithful_equal_across_different_seeds = Bool[],
    incorrect_equal_with_same_seed = Bool[],
    incorrect_response_mismatches = Int[],
    incorrect_changes_with_different_seed = Bool[],
    maximum_xprob_difference_from_faithful = Float64[],
)

for cell_data in subject_cells
    condition = first(cell_data.condition)
    inputs = Float64.(cell_data.detectprob)
    observed_responses = Int.(cell_data.binary_resp)

    # Positive control: different seeds must give identical faithful histories.
    faithful_a = faithful_snapshot(inputs, observed_responses; seed = 101)
    faithful_b = faithful_snapshot(inputs, observed_responses; seed = 202)

    # Negative control A: the same seed can make an incorrect stochastic method
    # repeat exactly. This is why same-seed reproducibility alone is insufficient.
    incorrect_same_seed_a = incorrect_sampled_snapshot(inputs; seed = 303)
    incorrect_same_seed_b = incorrect_sampled_snapshot(inputs; seed = 303)

    # Negative control B: changing the seed exposes its stochastic response path.
    incorrect_different_seed = incorrect_sampled_snapshot(inputs; seed = 404)

    faithful_mismatches = sum(faithful_a.response .!= observed_responses)
    incorrect_mismatches =
        sum(incorrect_same_seed_a.response .!= observed_responses)
    faithful_seed_invariant = isequal(faithful_a, faithful_b)
    incorrect_same_seed_repeats =
        isequal(incorrect_same_seed_a, incorrect_same_seed_b)
    incorrect_seed_sensitive =
        !isequal(incorrect_same_seed_a, incorrect_different_seed)
    maximum_xprob_difference = maximum(abs.(
        faithful_a.xprob_posterior_mean .-
        incorrect_same_seed_a.xprob_posterior_mean,
    ))

    @assert faithful_mismatches == 0
    @assert faithful_seed_invariant
    @assert incorrect_same_seed_repeats
    @assert incorrect_mismatches > 0
    @assert incorrect_seed_sensitive
    @assert maximum_xprob_difference > 0

    push!(
        validation_summary,
        (
            condition = condition,
            rows = nrow(cell_data),
            faithful_response_mismatches = faithful_mismatches,
            faithful_equal_across_different_seeds = faithful_seed_invariant,
            incorrect_equal_with_same_seed = incorrect_same_seed_repeats,
            incorrect_response_mismatches = incorrect_mismatches,
            incorrect_changes_with_different_seed = incorrect_seed_sensitive,
            maximum_xprob_difference_from_faithful = maximum_xprob_difference,
        ),
    )
end


# %% 5. Inspect the validation result

show(validation_summary; allcols = true, allrows = true)
println()
println()
println("PASS: faithful replay used every observed response and was seed-invariant")
println("PASS: the negative control repeated with the same seed despite being wrong")
println("PASS: changing the negative-control seed changed its sampled history")
