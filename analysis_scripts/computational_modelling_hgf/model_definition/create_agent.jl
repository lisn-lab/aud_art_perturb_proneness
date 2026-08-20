using ActionModels, HierarchicalGaussianFiltering, LogExpFunctions

#### Create Model ####
#Create conditioned hallucination action model
update_hgf_binary_conditioned_hallucination = function (agent::Agent, input::Real)

    ## SETUP ##
    #Extract action model parameters
    prior_posterior_weight = agent.parameters["prior_posterior_weight"]
    action_noise = agent.parameters["action_noise"]

    #Transform the action noise to action precision
    action_noise > 0 || throw(
        RejectParameters("Action noise must be greater than zero."),
    )
    action_precision = action_noise^(-1)

    #Extract the HGF
    hgf = agent.substruct

    ## UPDATE HGF ##
    #Participant's own previous action is the input to the HGF
    hgf_input = get_states(agent, "action")

    #If it is not the first trial
    if !ismissing(hgf_input)
        #Update the HGF
        update_hgf!(hgf, hgf_input)
    end

    ## CALCULATE ACTION PROBABILITY ##
    #The input is the stimulus strength
    stimulus_strength = input

    #Calculate the prediction for the current trial
    xprob_posterior_mean = get_states(hgf, ("xprob", "posterior_mean"))
    xbin_prediction_mean = 1 / (1 + exp(-xprob_posterior_mean)) #NOTE: COUPLING STRENGTH DACTIVATED HERE

    #Get the belief as a weighting between stimulus strength and the prediction
    belief =
        xbin_prediction_mean +
        1 / (1 + prior_posterior_weight) * (stimulus_strength - xbin_prediction_mean)

    #The belief must be a probability before it is converted to response log odds.
    if !(0 <= belief <= 1)
        throw(
            RejectParameters(
                "With these parameters and inputs, belief should be between 0 and 1.",
            ),
        )
    end

    #This is algebraically identical to the previous unit-square sigmoid:
    # belief^precision / (belief^precision + (1-belief)^precision).
    # Calculating the Bernoulli likelihood directly from the log odds prevents
    # both powered terms from underflowing to zero when action noise is small.
    action_logit = LogExpFunctions.logit(belief) * action_precision
    isnan(action_logit) && throw(
        RejectParameters("The action log odds became NaN."),
    )
    distribution = Distributions.BernoulliLogit(action_logit)


    ## FINALIZE ##
    #Save the belief
    update_states!(agent, "belief", belief)

    #Return the action distribution
    return distribution
end

#### Create agent ####
function create_agent(hgf_string = "binary_3level")

    #Set the action model
    action_model = update_hgf_binary_conditioned_hallucination

    #Set the default parameters
    parameter_defaults = Dict(
        ("xprob", "volatility") => -5.1682685,
        "prior_posterior_weight" => 0.72646851,
        "action_noise" => 0.29350739,
    )
    if hgf_string == "binary_3level"
        parameter_defaults[("xvol", "volatility")] = -6
    end

    #Set the HGF
    hgf = premade_hgf(hgf_string, parameter_defaults, verbose = false)

    #Set parameters
    parameters = Dict("action_noise" => 1, "prior_posterior_weight" => 1)
    #Set states
    states = Dict("belief" => missing)

    #Create the agent
    agent =
        init_agent(action_model; substruct = hgf, parameters = parameters, states = states)

    return agent
end
