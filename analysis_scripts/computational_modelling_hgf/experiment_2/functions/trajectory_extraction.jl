# =============================================================================
# Experiment 2 posterior trajectory extraction and long-format output
# =============================================================================
#
# `parameter_draws` is a Dict of three-dimensional arrays:
# parameter name -> participant-condition cell x retained draw x chain. Each
# output row corresponds to one original trial and posterior-median states.
# =============================================================================

using ActionModels, CSV, DataFrames, MCMCChains, Statistics


"""Convert an Agent parameter key to the flattened CSV/output column name."""
function fitted_parameter_symbol(parameter_key)
    return parameter_key isa Tuple ?
        Symbol(join(parameter_key, "__")) : Symbol(parameter_key)
end


"""
    extract_parameter_draws(posterior_chains, agent_ids, parameter_keys) -> Dict

Read fitted parameter matrices directly from the labelled Chains object without
calling ActionModels.extract_quantities(), which reevaluates the full likelihood
for every draw. Each returned array has dimensions cell x draw x chain.

`parameter_keys` must be `collect(keys(priors))` from the exact Dict used to
construct the fitted model. That runtime order is what defines the first index
in the stored name `parameters[parameter_index, cell_index]`.
"""
function extract_parameter_draws(
    posterior_chains,
    agent_ids,
    parameter_keys,
)
    n_samples, _, n_chains = size(posterior_chains)
    n_cells = length(agent_ids)
    available_names = Set(Symbol.(
        MCMCChains.names(MCMCChains.get_sections(posterior_chains, :parameters)),
    ))
    parameter_draws = Dict{Symbol,Array{Float64,3}}()

    for (parameter_index, parameter_key) in enumerate(parameter_keys)
        parameter_name = fitted_parameter_symbol(parameter_key)
        raw_names = [
            Symbol("parameters[$(parameter_index), $(cell_index)]")
            for cell_index in 1:n_cells
        ]
        all(raw_names .∈ Ref(available_names)) || error(
            "The saved chain is missing columns for $(parameter_name).",
        )

        # MCMCChains returns draw x cell x chain. Reorder it to the common
        # cell x draw x chain shape used by summaries and trajectory replay.
        selected_chains = posterior_chains[:, raw_names, :]
        selected = Array(selected_chains.value)  # Preserve draw x cell x chain; Array(selected_chains) concatenates chains into a matrix.
        parameter_draws[parameter_name] =
            permutedims(selected, (2, 1, 3))
        @assert size(parameter_draws[parameter_name]) ==
                (n_cells, n_samples, n_chains)
    end

    expected_names = Set((
        :prior_posterior_weight,
        :action_noise,
        :xprob__volatility,
        :xprob__initial_precision,
    ))
    Set(keys(parameter_draws)) == expected_names || error(
        "Unexpected fitted parameter set: $(collect(keys(parameter_draws)))",
    )
    return parameter_draws
end


"""Summarise each cell and fitted parameter by its all-draw posterior median."""
function summarise_parameter_draws(
    parameter_draws,
    data::DataFrame,
    parameter_keys,
)
    cells = collect(groupby(data, [:subject_nr, :condition]))
    estimates = DataFrame(
        subject_nr = [first(cell.subject_nr) for cell in cells],
        condition = [first(cell.condition) for cell in cells],
    )

    for parameter_key in parameter_keys
        parameter_name = fitted_parameter_symbol(parameter_key)
        draws = parameter_draws[parameter_name]
        estimates[!, parameter_name] = [
            median(vec(@view draws[cell_index, :, :]))
            for cell_index in eachindex(cells)
        ]
    end
    return estimates
end


"""
    extract_observed_trajectories(model, parameter_draws, estimates_df, data; ...)
        -> DataFrame

Replay each participant-condition cell at every retained posterior draw, using
the observed responses rather than simulated actions. Write one posterior-median
state row per original Experiment 2 trial.
"""
function extract_observed_trajectories(
    model,
    parameter_draws,
    estimates_df::DataFrame,
    data::DataFrame;
    output_dir::AbstractString,
    project_dir::AbstractString,
    environment_path::AbstractString,
    input_path::AbstractString,
    chain_path::AbstractString,
    estimates_path::AbstractString,
)
    agent_ids = model.args.agent_ids
    parameter_names = collect(keys(parameter_draws))
    expected_parameter_names = Set((
        :prior_posterior_weight,
        :action_noise,
        :xprob__volatility,
        :xprob__initial_precision,
    ))
    @assert Set(parameter_names) == expected_parameter_names

    n_cells, n_samples, n_chains = size(first(values(parameter_draws)))
    @assert all(size(draws) == (n_cells, n_samples, n_chains)
                for draws in values(parameter_draws))
    @assert n_cells == length(agent_ids)
    sample_ids = 1:n_samples
    chain_ids = 1:n_chains
    total_draws = n_samples * n_chains

    cells = collect(groupby(data, [:subject_nr, :condition]))
    @assert length(cells) == length(agent_ids) == nrow(estimates_df)

    states = trajectory_states()
    state_columns = Symbol.(join.(states, "__"))
    output_columns = vcat(:belief, state_columns, :signal_expectation)
    rows = DataFrame()

    trajectory_agent = create_agent("binary_3level")
    set_save_history!(trajectory_agent, true)

    estimate_columns = Dict(
        :prior_posterior_weight => :prior_posterior_weight,
        :action_noise => :action_noise,
        :xprob__volatility => :xprob__volatility,
        :xprob__initial_precision => :xprob__initial_precision,
    )

    for (cell_index, (agent_id, cell_data)) in enumerate(zip(agent_ids, cells))
        subject = first(cell_data.subject_nr)
        condition = first(cell_data.condition)

        @assert all(cell_data.subject_nr .== subject)
        @assert all(cell_data.condition .== condition)
        @assert estimates_df.subject_nr[cell_index] == subject
        @assert estimates_df.condition[cell_index] == condition

        inputs = Float64.(cell_data.detectprob)
        observed_responses = Int.(cell_data.resp_binary)
        n_trials = length(inputs)
        @assert n_trials == 80

        # Verify that the summary table and posterior array identify the same
        # cell and use the same posterior-median convention.
        for parameter_name in parameter_names
            draws = vec(@view parameter_draws[parameter_name][cell_index, :, :])
            estimate = estimates_df[cell_index, estimate_columns[parameter_name]]
            @assert isapprox(median(draws), estimate)
        end

        # One cell at a time bounds memory use and also supports unequal cell
        # lengths should the design change in a later dataset.
        values = fill(NaN, length(output_columns), n_trials, total_draws)
        fitted_parameters = Dict{Any,Float64}()
        draw_index = 0

        for chain_id in chain_ids, sample_id in sample_ids
            draw_index += 1
            for parameter_name in parameter_names
                fitted_parameters[trajectory_parameter_key(parameter_name)] = Float64(
                    parameter_draws[parameter_name][cell_index, sample_id, chain_id],
                )
            end
            set_parameters!(trajectory_agent, fitted_parameters)
            replay_observed_responses!(
                trajectory_agent,
                inputs,
                observed_responses,
            )
            values[:, :, draw_index] =
                current_draw_values(trajectory_agent, n_trials)
        end
        @assert draw_index == total_draws

        posterior_medians = Matrix{Union{Missing,Float64}}(
            undef,
            length(output_columns),
            n_trials,
        )
        for output_index in eachindex(output_columns), trial in 1:n_trials
            draw_values = @view values[output_index, trial, :]
            finite_values = filter(isfinite, draw_values)
            posterior_medians[output_index, trial] = isempty(finite_values) ?
                missing : median(finite_values)
        end

        for trial in 1:n_trials
            previous_response = trial == 1 ? missing : observed_responses[trial - 1]
            identifiers = (
                source_row = parentindices(cell_data)[1][trial],
                subject_nr = subject,
                condition = condition,
                timestep = trial - 1,
                current_detectprob = inputs[trial],
                current_response = observed_responses[trial],
                previous_response = previous_response,
                training = 0,
            )
            state_values = NamedTuple{Tuple(output_columns)}(
                Tuple(
                    posterior_medians[output_index, trial]
                    for output_index in eachindex(output_columns)
                ),
            )
            push!(rows, merge(identifiers, state_values); promote = true)
        end

        if cell_index % 5 == 0 || cell_index == length(cells)
            println("Extracted trajectories for cell $(cell_index) / $(length(cells))")
            flush(stdout)
        end
    end

    @assert nrow(rows) == nrow(data)
    @assert sort(rows.source_row) == collect(1:nrow(data))
    sort!(rows, :source_row)

    trajectory_path = joinpath(output_dir, "estimated_state_trajectories.csv")
    CSV.write(trajectory_path, rows)

    provenance_path = write_trajectory_provenance(
        output_dir = output_dir,
        project_dir = project_dir,
        environment_path = environment_path,
        input_path = input_path,
        chain_path = chain_path,
        estimates_path = estimates_path,
        trajectory_path = trajectory_path,
        samples_per_chain = length(sample_ids),
        n_chains = length(chain_ids),
        draws_per_cell = total_draws,
        n_cells = length(cells),
        n_rows = nrow(rows),
    )

    println("Trajectories written to $(trajectory_path)")
    println("Trajectory provenance written to $(provenance_path)")
    return rows
end
