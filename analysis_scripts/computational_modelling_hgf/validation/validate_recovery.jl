using CSV, DataFrames, SHA, Statistics, TOML

hgf_root = normpath(joinpath(@__DIR__, ".."))
project_root = normpath(joinpath(hgf_root, "..", ".."))

file_sha256(path) = open(path, "r") do io
    bytes2hex(SHA.sha256(io))
end

function validate_provenance(trajectory_path, input_path, expected_cells, expected_rows)
    provenance_path = replace(trajectory_path, r"\.csv$" => ".provenance.toml")
    isfile(provenance_path) || error("Missing recovery provenance: $(provenance_path)")
    provenance = TOML.parsefile(provenance_path)
    @assert provenance["summary"] == "posterior median across all retained draws"
    @assert provenance["samples_per_chain"] == 1000
    @assert provenance["chains"] == 4
    @assert provenance["draws_per_cell"] == 4000
    @assert provenance["cells"] == expected_cells
    @assert provenance["rows"] == expected_rows
    @assert provenance["input_sha256"] == file_sha256(input_path)
    @assert provenance["output_sha256"] == file_sha256(trajectory_path)
    return provenance_path
end

function validate_experiment(experiment, trajectory_path)
    if experiment == 1
        subject_col = :SubjID
        response_col = :binary_resp
    else
        subject_col = :subject_nr
        response_col = :resp_binary
    end
    experiment_dir = experiment == 1 ? "experiment_1" : "experiment_2"
    fit_root = joinpath(project_root, "model_outputs", "computational_modelling_hgf", experiment_dir)
    data_path = joinpath(fit_root, "fitted_input_rows.csv")
    estimates = CSV.read(joinpath(fit_root, "parameter_estimates_df.csv"), DataFrame)
    data = CSV.read(data_path, DataFrame)
    trajectories = CSV.read(trajectory_path, DataFrame)

    @assert nrow(trajectories) == nrow(data)
    @assert nrow(estimates) == length(groupby(data, [subject_col, :condition]))
    required_states = [
        :belief,
        :xprob__posterior_mean,
        :xprob__posterior_precision,
        :xvol__posterior_mean,
        :xvol__posterior_precision,
        :signal_expectation,
    ]
    @assert all(state -> state in propertynames(trajectories), required_states)
    @assert all(state -> all(.!ismissing.(trajectories[!, state])), required_states)
    # Each column is summarised across posterior draws before CSV export. For an
    # even number of draws, the median of the logistic transform can differ
    # slightly from the logistic transform of the median because the median is
    # the mean of the two central values.
    signal_expectation_difference = abs.(
        trajectories.signal_expectation .-
        1 ./ (1 .+ exp.(-trajectories.xprob__posterior_mean))
    )
    maximum_signal_expectation_difference = maximum(signal_expectation_difference)
    @assert maximum_signal_expectation_difference < 1e-6

    maximum_input_difference = 0.0
    maximum_initial_precision_difference = 0.0
    for row in eachrow(estimates)
        subject = row[subject_col]
        condition = row.condition
        input_cell = data[
            (data[!, subject_col] .== subject) .& (data.condition .== condition),
            :,
        ]
        trajectory_cell = trajectories[
            (trajectories[!, subject_col] .== subject) .&
            (trajectories.condition .== condition),
            :,
        ]
        @assert nrow(input_cell) == nrow(trajectory_cell)
        @assert trajectory_cell.timestep == collect(0:(nrow(trajectory_cell) - 1))
        @assert trajectory_cell.current_response == input_cell[!, response_col]
        @assert ismissing(trajectory_cell.previous_response[1])
        @assert trajectory_cell.previous_response[2:end] == input_cell[1:end-1, response_col]
        maximum_input_difference = max(
            maximum_input_difference,
            maximum(abs.(trajectory_cell.current_detectprob .- input_cell.detectprob)),
        )

        if experiment == 1
            expected_training = condition < 3 ? 1 : 20
            @assert sum(trajectory_cell.training .== 1) == expected_training
            @assert sum(trajectory_cell.training .== 0) == 100
            @assert trajectory_cell.xprob__posterior_precision[1] == 1
        else
            @assert nrow(trajectory_cell) == 80
            initial_difference = abs(
                trajectory_cell.xprob__posterior_precision[1] - row.xprob__initial_precision,
            )
            maximum_initial_precision_difference = max(
                maximum_initial_precision_difference,
                initial_difference,
            )
        end
    end
    @assert maximum_input_difference < 1e-12
    if experiment == 2
        @assert maximum_initial_precision_difference < 1e-12
    end

    provenance_path = validate_provenance(
        trajectory_path,
        data_path,
        nrow(estimates),
        nrow(data),
    )

    println("Experiment $(experiment) validation passed")
    println("  cells: $(nrow(estimates))")
    println("  rows: $(nrow(trajectories))")
    println("  maximum input difference: $(maximum_input_difference)")
    println("  maximum signal-expectation transform difference: $(maximum_signal_expectation_difference)")
    if experiment == 1
        task_first = combine(
            groupby(trajectories[trajectories.training .== 0, :], [subject_col, :condition]),
            first,
        )
        weak = task_first.signal_expectation[task_first.condition .< 3]
        strong = task_first.signal_expectation[task_first.condition .>= 3]
        println("  first task-trial expectation, weak-prior mean: $(mean(weak))")
        println("  first task-trial expectation, strong-prior mean: $(mean(strong))")
    else
        println("  maximum initial-precision difference: $(maximum_initial_precision_difference)")
    end
    println("  provenance: $(provenance_path)")
end

exp1_path = get(
    ENV,
    "HGF_EXP1_TRAJECTORIES",
    joinpath(project_root, "model_outputs", "computational_modelling_hgf", "experiment_1", "estimated_state_trajectories.csv"),
)
exp2_path = get(
    ENV,
    "HGF_EXP2_TRAJECTORIES",
    joinpath(project_root, "model_outputs", "computational_modelling_hgf", "experiment_2", "estimated_state_trajectories.csv"),
)

validate_experiment(1, exp1_path)
validate_experiment(2, exp2_path)
