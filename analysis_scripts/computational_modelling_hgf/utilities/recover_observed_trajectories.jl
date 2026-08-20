using HDF5
using ActionModels, HierarchicalGaussianFiltering, LogExpFunctions
using CSV, DataFrames, Statistics
using SHA

experiment = parse(Int, get(ENV, "HGF_EXPERIMENT", "0"))
experiment in (1, 2) || error("Set HGF_EXPERIMENT to 1 or 2")

hgf_root = normpath(joinpath(@__DIR__, ".."))
project_root = normpath(joinpath(hgf_root, "..", ".."))
if experiment == 1
    fit_root = joinpath(project_root, "model_outputs", "computational_modelling_hgf", "experiment_1")
    data_path = joinpath(project_root, "analysis_data", "hgf_inputs", "experiment_1_hgf_with_conditioning.csv")
    subject_col = :SubjID
    response_col = :binary_resp
    parameter_specs = [
        ("prior_posterior_weight", :prior_posterior_weight, 1),
        ("action_noise", :action_noise, 2),
        (("xprob", "volatility"), :xprob__volatility, 3),
    ]
else
    fit_root = joinpath(project_root, "model_outputs", "computational_modelling_hgf", "experiment_2")
    data_path = joinpath(project_root, "analysis_data", "hgf_inputs", "experiment_2_hgf.csv")
    subject_col = :subject_nr
    response_col = :resp_binary
    parameter_specs = [
        (("xprob", "initial_precision"), :xprob__initial_precision, 1),
        ("prior_posterior_weight", :prior_posterior_weight, 2),
        ("action_noise", :action_noise, 3),
        (("xprob", "volatility"), :xprob__volatility, 4),
    ]
end

include(joinpath(hgf_root, "model_definition", "create_agent.jl"))

data = CSV.read(data_path, DataFrame)
estimates = CSV.read(joinpath(fit_root, "parameter_estimates_df.csv"), DataFrame)
cell_map = [
    (estimates[i, subject_col], estimates.condition[i])
    for i in 1:nrow(estimates)
]

h5path = joinpath(fit_root, "fitting_results.h5")
n_samples, n_chains = h5open(h5path, "r") do f
    size(read(f["parameters"]["parameters[1, 1]"]))
end

samples_per_chain = min(
    n_samples,
    parse(Int, get(ENV, "HGF_RECOVERY_SAMPLES_PER_CHAIN", string(n_samples))),
)
sample_indices = samples_per_chain == n_samples ?
    collect(1:n_samples) :
    unique(round.(Int, range(1, n_samples; length = samples_per_chain)))
max_cells = min(
    length(cell_map),
    parse(Int, get(ENV, "HGF_RECOVERY_MAX_CELLS", string(length(cell_map)))),
)
output_name = get(ENV, "HGF_RECOVERY_OUTPUT", "estimated_state_trajectories.csv")
output_path = isabspath(output_name) ? output_name : joinpath(fit_root, output_name)
total_draws = length(sample_indices) * n_chains
canonical_output = normpath(output_path) == normpath(joinpath(fit_root, "estimated_state_trajectories.csv"))

if canonical_output && length(sample_indices) != n_samples
    error("Canonical recovery requires every posterior sample. Use a noncanonical HGF_RECOVERY_OUTPUT for subsampled tests.")
end
if canonical_output && max_cells != length(cell_map)
    error("Canonical recovery requires every fitted cell. Use a noncanonical HGF_RECOVERY_OUTPUT for partial tests.")
end

states_to_extract = [
    ("xbin", "prediction_mean"),
    ("xbin", "value_prediction_error"),
    ("xprob", "value_prediction_error"),
    ("xprob", "precision_prediction_error"),
    ("xprob", "posterior_precision"),
    ("xprob", "posterior_mean"),
    ("xvol", "value_prediction_error"),
    ("xvol", "posterior_precision"),
    ("xvol", "posterior_mean"),
]
state_cols = Symbol.(join.(states_to_extract, "__"))
output_cols = vcat(:belief, state_cols, :signal_expectation)

println("Experiment $(experiment): $(length(cell_map)) cells")
println("Using $(length(sample_indices)) samples per chain x $(n_chains) chains = $(total_draws) draws per cell")
println("Writing $(max_cells) cells to $(output_path)")
flush(stdout)

rows = DataFrame()
agent = create_agent("binary_3level")

h5open(h5path, "r") do f
    parameters = f["parameters"]

    for cell_idx in 1:max_cells
        subject, condition = cell_map[cell_idx]
        cell_data = data[
            (data[!, subject_col] .== subject) .& (data.condition .== condition),
            :,
        ]
        inputs = Vector{Float64}(cell_data.detectprob)
        responses = Vector{Int}(cell_data[!, response_col])
        n_t = length(inputs)

        if experiment == 1
            expected_training = condition < 3 ? 1 : 20
            @assert sum(cell_data.training .== 1) == expected_training
            @assert sum(cell_data.training .== 0) == 100
            @assert all(cell_data.detectprob[cell_data.training .== 1] .== 1)
            @assert all(cell_data[!, response_col][cell_data.training .== 1] .== 1)
        else
            @assert n_t == 80
        end

        parameter_draws = Dict{Int,Matrix{Float64}}()
        for (_, estimate_col, parameter_idx) in parameter_specs
            draws = read(parameters["parameters[$parameter_idx, $cell_idx]"])
            @assert isapprox(median(draws), estimates[cell_idx, estimate_col])
            parameter_draws[parameter_idx] = draws
        end

        # Preallocate one dense posterior array per cell. The earlier
        # vector-of-vectors implementation produced billions of allocations
        # in Experiment 2 while storing the same values.
        values = fill(NaN, length(output_cols), n_t, total_draws)
        params = Dict{Any,Float64}()
        draw_idx = 0

        for chain_idx in 1:n_chains, sample_idx in sample_indices
            draw_idx += 1
            for (parameter_key, _, parameter_idx) in parameter_specs
                params[parameter_key] = parameter_draws[parameter_idx][sample_idx, chain_idx]
            end

            set_parameters!(agent, params)
            reset!(agent)

            # Replay the likelihood exactly. This action model uses the
            # participant's previous response to update the HGF. give_inputs!
            # would sample a new action history and recover different states.
            for (input, observed_response) in zip(inputs, responses)
                agent.action_model(agent, input)
                update_states!(agent, "action", observed_response)
            end

            histories = [get_history(agent, state) for state in states_to_extract]
            belief_history = get_history(agent, "belief")

            for t in 1:n_t
                # Agent-level histories include an initial state; belief[t+1]
                # is the belief used for the action on input row t.
                belief = belief_history[t + 1]
                if !ismissing(belief) && isfinite(belief)
                    values[1, t, draw_idx] = Float64(belief)
                end

                for state_idx in eachindex(states_to_extract)
                    history = histories[state_idx]
                    state_value = t <= length(history) ? history[t] : missing
                    if !ismissing(state_value) && isfinite(state_value)
                        values[state_idx + 1, t, draw_idx] = Float64(state_value)
                    end
                end

                probability_state = histories[6][t]
                if !ismissing(probability_state) && isfinite(probability_state)
                    values[end, t, draw_idx] = 1 / (1 + exp(-Float64(probability_state)))
                end
            end
        end
        @assert draw_idx == total_draws

        medians = Matrix{Union{Missing,Float64}}(undef, length(output_cols), n_t)
        for col_idx in eachindex(output_cols), t in 1:n_t
            finite_values = filter(isfinite, @view values[col_idx, t, :])
            medians[col_idx, t] = isempty(finite_values) ?
                missing : median(finite_values)
        end

        for t in 1:n_t
            phase = experiment == 1 ? cell_data.training[t] : 0
            previous_response = t == 1 ? missing : responses[t - 1]
            identifiers = experiment == 1 ?
                (SubjID = subject, condition = condition) :
                (subject_nr = subject, condition = condition)
            push!(
                rows,
                merge(
                    identifiers,
                    (
                        timestep = t - 1,
                        current_detectprob = inputs[t],
                        current_response = responses[t],
                        previous_response = previous_response,
                        training = phase,
                    ),
                    NamedTuple{Tuple(output_cols)}(
                        Tuple(medians[col_idx, t] for col_idx in eachindex(output_cols)),
                    ),
                );
                promote = true,
            )
        end

        if cell_idx % 5 == 0 || cell_idx == max_cells
            println("cell $(cell_idx) / $(max_cells)")
            flush(stdout)
        end
    end
end

@assert nrow(rows) == sum(
    sum((data[!, subject_col] .== subject) .& (data.condition .== condition))
    for (subject, condition) in cell_map[1:max_cells]
)
CSV.write(output_path, rows)
println("Wrote $(nrow(rows)) faithful observed-response rows to $(output_path)")

file_sha256(path) = open(path, "r") do io
    bytes2hex(SHA.sha256(io))
end

provenance_path = replace(output_path, r"\.csv$" => ".provenance.toml")
open(provenance_path, "w") do io
    println(io, "experiment = $(experiment)")
    println(io, "julia_version = \"$(VERSION)\"")
    println(io, "replay = \"observed responses in fitted-likelihood order\"")
    println(io, "summary = \"posterior median across all retained draws\"")
    println(io, "samples_per_chain = $(length(sample_indices))")
    println(io, "chains = $(n_chains)")
    println(io, "draws_per_cell = $(total_draws)")
    println(io, "cells = $(max_cells)")
    println(io, "rows = $(nrow(rows))")
    println(io, "input_file = \"$(relpath(data_path, hgf_root))\"")
    println(io, "input_sha256 = \"$(file_sha256(data_path))\"")
    println(io, "chains_file = \"$(relpath(h5path, hgf_root))\"")
    println(io, "chains_sha256 = \"$(file_sha256(h5path))\"")
    estimates_path = joinpath(fit_root, "parameter_estimates_df.csv")
    println(io, "parameter_estimates_file = \"$(relpath(estimates_path, hgf_root))\"")
    println(io, "parameter_estimates_sha256 = \"$(file_sha256(estimates_path))\"")
    println(io, "output_sha256 = \"$(file_sha256(output_path))\"")
end
println("Wrote recovery provenance to $(provenance_path)")
