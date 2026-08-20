# =============================================================================
# Posterior trajectory extraction and long-format output
# =============================================================================
#
# R mental model
# --------------
# `agent_parameters` is a labelled four-dimensional AxisArray:
#
#     participant-condition cell x parameter x draw x chain
#
# `cells` is comparable to `group_split(data, SubjID, condition)`.
# `rows` becomes a long-format DataFrame with one row per original trial.
#
# Julia syntax used below
# -----------------------
# `collect(x)`      materialises an iterable as a Vector, like `as.list(x)`.
# `enumerate(x)`    supplies both an integer position and each value.
# `@assert x`       stops if x is false, like `stopifnot(x)` in R.
# `Dict(k => v)`    creates a key-value lookup, similar to a named R list.
# `@view array[...]` refers to an array slice without copying it.
# `condition ? a : b` is a compact `if ... else` expression.
# =============================================================================

using ActionModels, CSV, DataFrames, Statistics


"""
    extract_observed_trajectories(model, agent_parameters, estimates_df, data; ...)
        -> DataFrame

Replay every participant-condition cell at every retained posterior draw, take
the posterior median of each state on each trial, and write the canonical
long-format trajectory table.

Inputs
------
- `model`: the DynamicPPL/Turing model. It supplies the fitted cell identifiers.
- `agent_parameters`: four-dimensional AxisArray with axes cell x parameter x
  retained draw x chain.
- `estimates_df`: DataFrame with one posterior-median parameter row per cell.
- `data`: input DataFrame with one row per conditioning or task trial.
- paths: strings identifying the output directory and source files.

Output
------
A DataFrame with exactly as many rows as `data`. It is also written to
`estimated_state_trajectories.csv`. Each state column contains the median across
all retained draws and chains for the corresponding cell and trial.
"""
function extract_observed_trajectories(
    model,
    agent_parameters,
    estimates_df::DataFrame,
    data::DataFrame;
    output_dir::AbstractString,
    input_path::AbstractString,
    chain_path::AbstractString,
    estimates_path::AbstractString,
)
    # AxisArrays attach names to array dimensions. `collect` converts each axis
    # into an ordinary Vector that can be inspected and looped over.
    agent_axis, parameter_axis, sample_axis, chain_axis = agent_parameters.axes
    agent_ids = collect(agent_axis)
    parameter_names = collect(parameter_axis)
    sample_ids = collect(sample_axis)
    chain_ids = collect(chain_axis)
    total_draws = length(sample_ids) * length(chain_ids)

    # Assertions are executable checks, not statistical tests. They stop the
    # script rather than allowing mislabelled cells or parameters into output.
    @assert agent_ids == model.args.agent_ids
    @assert Set(parameter_names) == Set((
        :prior_posterior_weight,
        :action_noise,
        :xprob__volatility,
    ))

    # `groupby` creates a lazy grouped DataFrame. `collect` materialises its
    # participant-condition groups as a Vector of SubDataFrame objects.
    cells = collect(groupby(data, [:SubjID, :condition]))
    @assert length(cells) == length(agent_ids) == nrow(estimates_df)

    states = trajectory_states()
    state_columns = Symbol.(join.(states, "__"))
    output_columns = vcat(:belief, state_columns, :signal_expectation)
    rows = DataFrame()

    trajectory_agent = create_agent("binary_3level")
    set_save_history!(trajectory_agent, true)

    # `enumerate(zip(...))` supplies both the cell number and corresponding
    # fitted agent ID/data subset.
    for (cell_index, (agent_id, cell_data)) in enumerate(zip(agent_ids, cells))
        subject = first(cell_data.SubjID)
        condition = first(cell_data.condition)

        @assert all(cell_data.SubjID .== subject)
        @assert all(cell_data.condition .== condition)
        @assert estimates_df.SubjID[cell_index] == subject
        @assert estimates_df.condition[cell_index] == condition

        # These are concrete numeric Vectors in input-row order. They are the
        # actual stimulus-strength and participant-response sequences used by
        # replay_observed_responses!().
        inputs = Float64.(cell_data.detectprob)
        observed_responses = Int.(cell_data.binary_resp)
        n_trials = length(inputs)

        # Validate the design rather than deleting conditioning rows by timestep.
        # Conditions 1-2 are weak prior with 1 conditioning row. Conditions 3-4
        # are strong prior with 20. Every cell then has 100 task rows. All
        # conditioning rows must contain signal input 1 and response 1.
        expected_conditioning_rows = condition < 3 ? 1 : 20
        @assert sum(cell_data.training .== 1) == expected_conditioning_rows
        @assert sum(cell_data.training .== 0) == 100
        @assert all(cell_data.detectprob[cell_data.training .== 1] .== 1)
        @assert all(cell_data.binary_resp[cell_data.training .== 1] .== 1)

        # Confirm that the parameter summary table and posterior array refer to
        # the same cells and draws before those draws are used for trajectories.
        estimate_columns = Dict(
            :prior_posterior_weight => :prior_posterior_weight,
            :action_noise => :action_noise,
            :xprob__volatility => :xprob__volatility,
        )
        for parameter_name in parameter_names
            draws = vec(Array(agent_parameters[agent_id, parameter_name, :, :]))
            estimate = estimates_df[cell_index, estimate_columns[parameter_name]]
            @assert isapprox(median(draws), estimate)
        end

        # Store one cell at a time. The third dimension holds all posterior
        # draws, avoiding the package error caused by forcing all cells to have
        # the first cell's trial count.
        values = fill(NaN, length(output_columns), n_trials, total_draws)
        fitted_parameters = Dict{Any,Float64}()
        draw_index = 0

        for chain_id in chain_ids, sample_id in sample_ids
            draw_index += 1

            # Install the three values from one cell x draw x chain combination.
            for parameter_name in parameter_names
                fitted_parameters[trajectory_parameter_key(parameter_name)] = Float64(
                    agent_parameters[agent_id, parameter_name, sample_id, chain_id],
                )
            end
            set_parameters!(trajectory_agent, fitted_parameters)

            # This is the only replay function used by the canonical extraction.
            # It resets the agent, pairs CSV inputs and observed responses with
            # zip(), and never samples a response.
            replay_observed_responses!(
                trajectory_agent,
                inputs,
                observed_responses,
            )
            values[:, :, draw_index] =
                current_draw_values(trajectory_agent, n_trials)
        end
        @assert draw_index == total_draws

        # For every output variable and trial, collapse the draw dimension to one
        # posterior median. With the canonical fit this summarises 1,000 draws x
        # 4 chains = 4,000 state values into one reported value.
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

        # Convert this cell's matrix into long-format DataFrame rows. `push!`
        # appends one NamedTuple, analogous to adding one named record in R.
        for trial in 1:n_trials
            previous_response = trial == 1 ? missing : observed_responses[trial - 1]
            identifiers = (
                source_row = parentindices(cell_data)[1][trial],
                SubjID = subject,
                condition = condition,
                timestep = trial - 1,
                current_detectprob = inputs[trial],
                current_response = observed_responses[trial],
                previous_response = previous_response,
                training = cell_data.training[trial],
            )
            state_values = NamedTuple{Tuple(output_columns)}(
                Tuple(
                    posterior_medians[output_index, trial] for
                    output_index in eachindex(output_columns)
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
