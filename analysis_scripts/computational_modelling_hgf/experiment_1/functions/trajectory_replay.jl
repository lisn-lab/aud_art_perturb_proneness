# =============================================================================
# Replay one participant-condition history for one posterior draw
# =============================================================================
#
# This file contains the critical distinction between faithful state extraction
# and response simulation. It never calls `rand()` or `give_inputs!()`.
#
# Julia syntax used below
# -----------------------
# `f!(x)`       indicates by convention that f modifies the existing object x.
# `zip(a, b)`   pairs a[1] with b[1], a[2] with b[2], and so on.
# `@assert x`   stops if x is false, like `stopifnot(x)` in R.
# `1:n`         is the integer sequence from 1 through n.
# `eachindex(x)` returns valid positions for x.
# `missing`     is Julia's equivalent of R's NA.
# =============================================================================

using ActionModels


"""
    trajectory_parameter_key(parameter_name::Symbol) -> String or Tuple

Translate the flattened name stored on the fitted parameter array back to the
key used by the Agent.

For example, ActionModels stores the nested HGF parameter
`("xprob", "volatility")` as the single array-axis label
`:xprob__volatility`. `set_parameters!()` requires the original tuple.
"""
function trajectory_parameter_key(parameter_name::Symbol)
    key_lookup = Dict(
        :prior_posterior_weight => "prior_posterior_weight",
        :action_noise => "action_noise",
        :xprob__volatility => ("xprob", "volatility"),
    )
    haskey(key_lookup, parameter_name) ||
        error("Unexpected fitted parameter: $(parameter_name)")
    return key_lookup[parameter_name]
end


"""
    trajectory_states() -> Vector{Tuple{String,String}}

Return the ordered HGF state names extracted from every posterior draw. This is
comparable to an R character vector of requested variable names, except nested
HGF names are represented by two-part tuples.
"""
function trajectory_states()
    return [
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
end


"""
    replay_observed_responses!(agent, inputs, observed_responses) -> Agent

Reset one Agent and replay the participant's actual trial sequence in the exact
order used by the fitted likelihood.

Inputs
------
- `agent`: an ActionModels Agent with one posterior draw already installed.
- `inputs`: a numeric Vector containing `detectprob`, one value per data row.
- `observed_responses`: an integer Vector containing `binary_resp`, one value per
  corresponding data row.

Output
------
The same Agent object, modified in place. Its histories now contain the states
produced by the observed response sequence.

This function contains no random sampling. Changing an RNG seed cannot change
its output when its agent, parameters, inputs, and responses are unchanged.
"""
function replay_observed_responses!(agent, inputs, observed_responses)
    @assert length(inputs) == length(observed_responses)

    # Clear the states and histories left by the previous posterior draw. The
    # fitted parameter values installed with set_parameters!() are retained.
    reset!(agent)

    # `zip` guarantees that each stimulus-strength input is paired with the
    # participant response from the same CSV row.
    for (input, observed_response) in zip(inputs, observed_responses)
        # CRITICAL LINE 1: calculate the current response distribution. The
        # action model first updates the HGF with the response stored from the
        # preceding row, matching ActionModels.agent_models() during fitting.
        agent.action_model(agent, input)

        # CRITICAL LINE 2: store the participant's observed response, not a
        # simulated draw. On the next iteration it becomes the HGF input.
        update_states!(agent, "action", observed_response)
    end

    # Confirm that the stored action history, excluding its initial missing
    # entry, exactly reproduces the observed response vector.
    @assert Int.(get_history(agent, "action")[2:end]) == observed_responses
    return agent
end


"""
    current_draw_values(agent, n_trials) -> Matrix{Float64}

Extract the requested histories for one posterior draw as a numeric matrix:

    output variable x trial

Rows contain belief, nine HGF states, and signal expectation. Columns correspond
to the original CSV rows for one participant-condition cell.
"""
function current_draw_values(agent, n_trials::Int)
    states = trajectory_states()
    values = fill(NaN, length(states) + 2, n_trials)
    state_histories = [get_history(agent, state) for state in states]
    belief_history = get_history(agent, "belief")

    for trial in 1:n_trials
        # Agent-level belief history has an initial entry at position 1. The
        # belief calculated for CSV row `trial` is therefore at `trial + 1`.
        belief = belief_history[trial + 1]
        if !ismissing(belief) && isfinite(belief)
            values[1, trial] = Float64(belief)
        end

        # HGF histories already use position `trial` for the state available to
        # the action model on that CSV row. No extra shift is applied here.
        for state_index in eachindex(states)
            state_value = state_histories[state_index][trial]
            if !ismissing(state_value) && isfinite(state_value)
                values[state_index + 1, trial] = Float64(state_value)
            end
        end

        # State 6 is xprob posterior mean in log-odds. The logistic transform is
        # the signal expectation used by the action model on the current row.
        probability_state = state_histories[6][trial]
        if !ismissing(probability_state) && isfinite(probability_state)
            values[end, trial] = 1 / (1 + exp(-Float64(probability_state)))
        end
    end

    return values
end
