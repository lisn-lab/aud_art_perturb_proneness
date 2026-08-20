# =============================================================================
# Faithful replay of one Experiment 2 participant-condition history
# =============================================================================

using ActionModels


"""Translate flattened fitted-parameter names back to Agent parameter keys."""
function trajectory_parameter_key(parameter_name::Symbol)
    key_lookup = Dict(
        :prior_posterior_weight => "prior_posterior_weight",
        :action_noise => "action_noise",
        :xprob__volatility => ("xprob", "volatility"),
        :xprob__initial_precision => ("xprob", "initial_precision"),
    )
    haskey(key_lookup, parameter_name) ||
        error("Unexpected fitted parameter: $(parameter_name)")
    return key_lookup[parameter_name]
end


"""Return the HGF states retained for every posterior draw and task trial."""
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

Reset one Agent and replay the participant's observed responses in the same
order as the fitted likelihood. This function never samples a response.
"""
function replay_observed_responses!(agent, inputs, observed_responses)
    @assert length(inputs) == length(observed_responses)
    reset!(agent)

    for (input, observed_response) in zip(inputs, observed_responses)
        agent.action_model(agent, input)
        update_states!(agent, "action", observed_response)
    end

    @assert Int.(get_history(agent, "action")[2:end]) == observed_responses
    return agent
end


"""Extract belief and HGF histories as an output-variable x trial matrix."""
function current_draw_values(agent, n_trials::Int)
    states = trajectory_states()
    values = fill(NaN, length(states) + 2, n_trials)
    state_histories = [get_history(agent, state) for state in states]
    belief_history = get_history(agent, "belief")

    for trial in 1:n_trials
        belief = belief_history[trial + 1]
        if !ismissing(belief) && isfinite(belief)
            values[1, trial] = Float64(belief)
        end

        for state_index in eachindex(states)
            state_value = state_histories[state_index][trial]
            if !ismissing(state_value) && isfinite(state_value)
                values[state_index + 1, trial] = Float64(state_value)
            end
        end

        probability_state = state_histories[6][trial]
        if !ismissing(probability_state) && isfinite(probability_state)
            values[end, trial] = 1 / (1 + exp(-Float64(probability_state)))
        end
    end

    return values
end
